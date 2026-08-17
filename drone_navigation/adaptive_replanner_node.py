import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os

from drone_navigation.ra_coverage import RavenReplanner

class AdaptiveReplannerNode(Node):
    def __init__(self):
        super().__init__('adaptive_replanner')
        self.get_logger().info('Initializing Adaptive Replanner Node (RRA)...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.declare_parameter('n_waypoints_fa', 20)
        self.declare_parameter('replan_interval_sec', 15.0) # corresponds to 150 steps at 10Hz
        
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.n_waypoints_fa = self.get_parameter('n_waypoints_fa').get_parameter_value().integer_value
        self.replan_interval = self.get_parameter('replan_interval_sec').get_parameter_value().double_value
        
        # Internal state variables
        self.global_waypoints = None # shape: (n_drones, n_waypoints_fa, 3)
        self.active_waypoints = None # shape: (n_drones, n_waypoints_fa, 3)
        self.drone_positions = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.coverage_grid = np.zeros((100, 100), dtype=np.float32)
        self.current_wp_idx = [0] * self.n_drones
        self.step_count = 0
        
        # Publishers
        self.active_waypoints_pub = self.create_publisher(Float32MultiArray, '/drones/active_waypoints', 10)
        
        # Subscribers
        self.global_wps_sub = self.create_subscription(Float32MultiArray, '/drones/global_waypoints', self.global_wps_callback, 10)
        self.fleet_state_sub = self.create_subscription(Float32MultiArray, '/whiteboard/fleet_state', self.fleet_state_callback, 10)
        self.coverage_map_sub = self.create_subscription(Float32MultiArray, '/whiteboard/coverage_map', self.coverage_map_callback, 10)
        
        # Timers
        # Timer to run tracking & arrival check at 10Hz
        self.track_timer = self.create_timer(0.1, self.tracking_callback)
        # Timer to trigger RRA replanning periodically
        self.replan_timer = self.create_timer(self.replan_interval, self.trigger_replan)
        
        self.get_logger().info('Adaptive Replanner Node started.')

    def global_wps_callback(self, msg: Float32MultiArray):
        """Receive the initial waypoints planned by the Global Planner."""
        expected_size = self.n_drones * self.n_waypoints_fa * 3
        if len(msg.data) == expected_size:
            wps = np.array(msg.data, dtype=np.float32).reshape(self.n_drones, self.n_waypoints_fa, 3)
            self.global_waypoints = wps.copy()
            # Initialize active waypoints as a copy of global planned waypoints
            if self.active_waypoints is None:
                self.active_waypoints = wps.copy()
                self.publish_active_waypoints()
                self.get_logger().info('Initialized active waypoints from global planner.')
        else:
            self.get_logger().warn(f'Received global waypoints size {len(msg.data)}, expected {expected_size}')

    def fleet_state_callback(self, msg: Float32MultiArray):
        """Update estimated drone positions from the whiteboard registry."""
        # msg.data layout: flat array of [x, y, z, vx, vy, vz, battery, cov] per drone (size 8 * n_drones)
        n_features = 8
        if len(msg.data) >= self.n_drones * n_features:
            for i in range(self.n_drones):
                offset = i * n_features
                self.drone_positions[i] = msg.data[offset:offset+3]

    def coverage_map_callback(self, msg: Float32MultiArray):
        """Update local copy of the fused coverage map from the whiteboard registry."""
        if len(msg.data) == 10000:
            self.coverage_grid = np.array(msg.data, dtype=np.float32).reshape(100, 100)

    def tracking_callback(self):
        """Check if drones have arrived at their current waypoints and advance indices."""
        if self.active_waypoints is None:
            return
            
        arrival_radius = 3.0
        max_wp = self.n_waypoints_fa - 1
        
        for i in range(self.n_drones):
            if self.current_wp_idx[i] >= max_wp:
                continue
            
            target = self.active_waypoints[i, self.current_wp_idx[i]]
            pos = self.drone_positions[i]
            dist = np.linalg.norm(pos - target)
            
            if dist < arrival_radius:
                self.current_wp_idx[i] += 1
                self.get_logger().info(f'Drone {i} reached waypoint {self.current_wp_idx[i]-1}. Advancing to {self.current_wp_idx[i]}')
                self.publish_active_waypoints()
                
        # Simple step counter simulation
        self.step_count += 1

    def trigger_replan(self):
        """Periodically run the Raven Roosting Algorithm (RRA) to adjust remaining waypoints."""
        if self.active_waypoints is None or self.global_waypoints is None:
            return
            
        # Determine remaining waypoints to optimize (we optimize based on the minimum remaining waypoints across drones)
        min_rem_wps = min(self.n_waypoints_fa - idx for idx in self.current_wp_idx)
        
        if min_rem_wps <= 2:
            self.get_logger().info('Drones are close to the final goals. Skipping RRA replan.')
            return

        self.get_logger().info(f'Triggering RRA Replanning (remaining waypoints: {min_rem_wps})...')
        
        # Initialize replanner
        ra = RavenReplanner(
            n_drones=self.n_drones,
            max_iter=20, # keeping iterations reasonable for performance
            n_ravens=15
        )
        
        # Run replanning
        # RRA takes current positions, coverage grid, and remaining waypoints count
        try:
            new_wps = ra.replan(
                current_positions=self.drone_positions,
                coverage_grid=(self.coverage_grid > 0.5), # bool mask
                n_remaining_waypoints=min_rem_wps
            )
            
            # Update active waypoints list: replace remaining segments with the new planned coordinates
            for i in range(self.n_drones):
                idx = self.current_wp_idx[i]
                rem_count = self.n_waypoints_fa - idx
                # Fill in from idx to end
                take = min(new_wps.shape[1], rem_count)
                self.active_waypoints[i, idx:idx+take] = new_wps[i, :take]
                
            self.get_logger().info('Active waypoints updated via RRA.')
            self.publish_active_waypoints()
            
        except Exception as e:
            self.get_logger().error(f'RRA replanning failed: {str(e)}')

    def publish_active_waypoints(self):
        """Publish updated waypoints list and the current target indices."""
        if self.active_waypoints is None:
            return
            
        # Layout: we publish a Float32MultiArray containing:
        # - Waypoint coordinates: flat float array (size n_drones * n_waypoints * 3)
        # - Current target index per drone: appended at the end of data (size n_drones)
        # Total size = n_drones * n_waypoints * 3 + n_drones
        coords = self.active_waypoints.flatten().tolist()
        indices = [float(idx) for idx in self.current_wp_idx]
        
        msg = Float32MultiArray()
        msg.data = coords + indices
        self.active_waypoints_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveReplannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

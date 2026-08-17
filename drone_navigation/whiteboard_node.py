import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan, Range, Imu
import numpy as np
import os

class WhiteboardNode(Node):
    def __init__(self):
        super().__init__('whiteboard')
        self.get_logger().info('Initializing Central Whiteboard Node...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        
        # Load Terrain map for Z position lookup from ultrasonic height AGL
        self.terrain = None
        terrain_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'terrain.npy')
        )
        if os.path.exists(terrain_path):
            self.terrain = np.load(terrain_path)
            self.get_logger().info(f'Loaded terrain map from {terrain_path} for altitude lookup.')
        else:
            self.get_logger().warn(f'Terrain file not found at {terrain_path}. Flat terrain (z=0) assumed.')
            self.terrain = np.zeros((100, 100), dtype=np.float32)

        # Fused State registry for the drones:
        # Each drone state: [x, y, z, vx, vy, vz, battery, coverage_ratio]
        self.fused_states = np.zeros((self.n_drones, 8), dtype=np.float32)
        
        # Initialize default position to starts
        starts = [
            [4.5, 4.5, 6.0], [4.5, 6.5, 6.0], [6.5, 4.5, 6.0],
            [6.5, 6.5, 6.0], [5.5, 4.5, 6.0], [5.5, 6.5, 6.0]
        ]
        for i in range(self.n_drones):
            self.fused_states[i, :3] = starts[i % len(starts)]
            self.fused_states[i, 6] = 200.0 # battery capacity

        # Fused global coverage grid (100x100)
        self.fused_coverage = np.zeros((100, 100), dtype=np.float32)
        
        # Obstacle storage (Lidar detected obstacle locations)
        self.detected_obstacles = [] # list of [x, y, z]

        # Fused state publishers
        self.fleet_state_pub = self.create_publisher(Float32MultiArray, '/whiteboard/fleet_state', 10)
        self.coverage_map_pub = self.create_publisher(Float32MultiArray, '/whiteboard/coverage_map', 10)
        
        # Subscribers
        self.states_sub = self.create_subscription(Float32MultiArray, '/drones/states', self.raw_states_callback, 10)
        self.map_sub = self.create_subscription(Float32MultiArray, '/map/coverage', self.raw_map_callback, 10)
        
        # Per-drone sensor subscriptions
        self.lidar_subs = []
        self.camera_subs = []
        self.range_subs = []
        self.imu_subs = []
        
        # Keep track of local sensor readings for Kalman updates
        self.last_imu_accel = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.last_range = np.zeros(self.n_drones, dtype=np.float32)
        
        for i in range(self.n_drones):
            # Capture scope variables using default arguments in lambda
            self.lidar_subs.append(
                self.create_subscription(LaserScan, f'/drone_{i}/lidar', 
                                      lambda msg, idx=i: self.lidar_callback(msg, idx), 10)
            )
            self.camera_subs.append(
                self.create_subscription(Float32MultiArray, f'/drone_{i}/camera', 
                                      lambda msg, idx=i: self.camera_callback(msg, idx), 10)
            )
            self.range_subs.append(
                self.create_subscription(Range, f'/drone_{i}/range', 
                                      lambda msg, idx=i: self.range_callback(msg, idx), 10)
            )
            self.imu_subs.append(
                self.create_subscription(Imu, f'/drone_{i}/imu', 
                                      lambda msg, idx=i: self.imu_callback(msg, idx), 10)
            )
            
        # State publisher timer (updates whiteboard registry at 10Hz)
        self.pub_timer = self.create_timer(0.1, self.publish_whiteboard)
        self.get_logger().info('Central Whiteboard Node successfully started.')

    def raw_states_callback(self, msg: Float32MultiArray):
        """GPS-like global state observation from simulator. Used to correct predictions."""
        # msg.data contains flattened observation vector (14 elements per drone)
        # Each drone: [pos_norm(3), vel_norm(3), battery_norm(1), coverage_ratio(1), rel_centroid(3), wind(3)]
        n_features = 14
        if len(msg.data) >= self.n_drones * n_features:
            for i in range(self.n_drones):
                offset = i * n_features
                # Denormalize GPS positions and velocities
                x_gps = msg.data[offset] * 100.0
                y_gps = msg.data[offset+1] * 100.0
                z_gps = msg.data[offset+2] * 15.0
                vx_gps = msg.data[offset+3] * 1.8
                vy_gps = msg.data[offset+4] * 1.8
                vz_gps = msg.data[offset+5] * 1.8
                battery = msg.data[offset+6] * 200.0
                cov = msg.data[offset+7]
                
                # Kalman Filter Measurement Update (Fusing GPS pose with local IMU/Range updates)
                # GPS measurement covariance is higher than Range, but it provides absolute XY reference
                # Update position
                self.fused_states[i, 0] = 0.8 * self.fused_states[i, 0] + 0.2 * x_gps
                self.fused_states[i, 1] = 0.8 * self.fused_states[i, 1] + 0.2 * y_gps
                # Z is heavily weighted towards Ultrasonic Range measurements
                self.fused_states[i, 2] = 0.6 * self.fused_states[i, 2] + 0.4 * z_gps
                
                # Update velocity
                self.fused_states[i, 3] = 0.7 * self.fused_states[i, 3] + 0.3 * vx_gps
                self.fused_states[i, 4] = 0.7 * self.fused_states[i, 4] + 0.3 * vy_gps
                self.fused_states[i, 5] = 0.7 * self.fused_states[i, 5] + 0.3 * vz_gps
                
                # Battery and coverage ratio update
                self.fused_states[i, 6] = battery
                self.fused_states[i, 7] = cov

    def raw_map_callback(self, msg: Float32MultiArray):
        """Subscribe to the environment's true map to initialize base whiteboard coverage."""
        if len(msg.data) == 10000:
            map_data = np.array(msg.data, dtype=np.float32).reshape(100, 100)
            self.fused_coverage = np.maximum(self.fused_coverage, map_data)

    def lidar_callback(self, msg: LaserScan, idx: int):
        """LiDAR Global sensing callback: Extracts local obstacles and maps them to absolute coordinates."""
        drone_pos = self.fused_states[idx, :3]
        angle = msg.angle_min
        new_obstacles = []
        for r in msg.ranges:
            # If range is less than max range (10m), an obstacle is detected!
            if r < 9.5:
                # Local coordinate projection
                lx = r * np.cos(angle)
                ly = r * np.sin(angle)
                # Global coordinate projection
                gx = drone_pos[0] + lx
                gy = drone_pos[1] + ly
                gz = drone_pos[2] # LiDAR scan plane
                new_obstacles.append([gx, gy, gz])
            angle += msg.angle_increment
            
        # Update/Fuse obstacles list on the whiteboard (avoiding duplicates)
        for obs in new_obstacles:
            if not any(np.linalg.norm(np.array(obs) - np.array(existing)) < 1.0 for existing in self.detected_obstacles):
                self.detected_obstacles.append(obs)
                
        # Limit obstacles registry size to keep memory bound
        if len(self.detected_obstacles) > 200:
            self.detected_obstacles = self.detected_obstacles[-200:]

    def camera_callback(self, msg: Float32MultiArray, idx: int):
        """Camera Global sensing callback: Extracts visual coverage and updates the whiteboard map."""
        # Camera feature vector layout:
        # [terrain_height, dx, dy, coverage_frac, rc_rel_x, rc_rel_y, rc_rel_z, obs_rel_x, obs_rel_y, obs_rel_z]
        if len(msg.data) >= 10:
            drone_pos = self.fused_states[idx, :3]
            coverage_frac = msg.data[3]
            
            # The camera observes an area around the drone. If coverage fraction is high,
            # we fill in cells in the whiteboard's map under the drone's FOV (5x5 footprint)
            xi, yi = int(np.clip(drone_pos[0], 0, 99)), int(np.clip(drone_pos[1], 0, 99))
            x0, x1 = max(0, xi - 2), min(100, xi + 3)
            y0, y1 = max(0, yi - 2), min(100, yi + 3)
            # Fusing camera visual verification of coverage
            self.fused_coverage[x0:x1, y0:y1] = 1.0

    def range_callback(self, msg: Range, idx: int):
        """Ultrasonic range callback: Local sensing altitude AGL."""
        self.last_range[idx] = msg.range
        drone_xy = self.fused_states[idx, :2]
        
        # Look up terrain height directly below estimated drone XY
        xi = int(np.clip(drone_xy[0], 0, 99))
        yi = int(np.clip(drone_xy[1], 0, 99))
        terrain_z = self.terrain[xi, yi] if self.terrain is not None else 0.0
        
        # Fused Altitude = Terrain Height + Ultrasonic AGL measurement
        fused_alt = terrain_z + msg.range
        
        # Fuse with high weight due to high ultrasonic reliability
        self.fused_states[idx, 2] = 0.3 * self.fused_states[idx, 2] + 0.7 * fused_alt

    def imu_callback(self, msg: Imu, idx: int):
        """IMU local sensing callback: Tracks linear acceleration for dead reckoning."""
        # IMU linear acceleration contains gravity (z += 9.81)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z - 9.81  # subtract gravity
        
        self.last_imu_accel[idx] = [ax, ay, az]
        
        # Integrate acceleration to predict velocity and position (Dead reckoning prediction step)
        dt = 0.1 # timer frequency
        self.fused_states[idx, 3] += ax * dt # vx
        self.fused_states[idx, 4] += ay * dt # vy
        self.fused_states[idx, 5] += az * dt # vz
        
        # Predict position
        self.fused_states[idx, 0] += self.fused_states[idx, 3] * dt + 0.5 * ax * dt**2
        self.fused_states[idx, 1] += self.fused_states[idx, 4] * dt + 0.5 * ay * dt**2
        self.fused_states[idx, 2] += self.fused_states[idx, 5] * dt + 0.5 * az * dt**2

    def publish_whiteboard(self):
        """Periodically publish the fused states and map from the whiteboard registry."""
        # Publish fleet states
        fleet_msg = Float32MultiArray()
        fleet_msg.data = self.fused_states.flatten().tolist()
        self.fleet_state_pub.publish(fleet_msg)
        
        # Publish fused coverage grid
        map_msg = Float32MultiArray()
        map_msg.data = self.fused_coverage.flatten().tolist()
        self.coverage_map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WhiteboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

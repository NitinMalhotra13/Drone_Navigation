import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import os

class LowLevelControllerNode(Node):
    def __init__(self):
        super().__init__('low_level_controller')
        self.get_logger().info('Initializing Low Level Controller Node...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.declare_parameter('n_waypoints_fa', 12)
        self.declare_parameter('use_ppo', True)
        
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.n_waypoints_fa = self.get_parameter('n_waypoints_fa').get_parameter_value().integer_value
        self.use_ppo = self.get_parameter('use_ppo').get_parameter_value().bool_value
        
        # Internal state variables
        self.fused_states = np.zeros((self.n_drones, 8), dtype=np.float32) # [x, y, z, vx, vy, vz, battery, cov]
        self.active_waypoints = None # shape: (n_drones, n_waypoints_fa, 3)
        self.current_wp_idx = [0] * self.n_drones
        
        # Command publisher
        self.cmd_pub = self.create_publisher(Float32MultiArray, '/drones/commands', 10)
        
        # Gazebo command publishers
        self.gazebo_cmd_pubs = []
        for i in range(self.n_drones):
            self.gazebo_cmd_pubs.append(
                self.create_publisher(Twist, f'/drone_{i}/cmd_vel', 10)
            )
        
        # Subscribers
        self.fleet_state_sub = self.create_subscriber(Float32MultiArray, '/whiteboard/fleet_state', self.fleet_state_callback, 10)
        self.active_wps_sub = self.create_subscriber(Float32MultiArray, '/drones/active_waypoints', self.active_wps_callback, 10)
        
        # Load PPO Models if enabled
        self.model = None
        self.vecnorm = None
        if self.use_ppo:
            self.load_ppo_models()

        # Timer to run control loop at 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Low Level Controller Node started.')

    def load_ppo_models(self):
        """Attempt to load trained PPO policy model and VecNormalize statistics."""
        src_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(src_dir, '..', 'models', 'ppo_multi_drone_final.zip'))
        vec_path = os.path.normpath(os.path.join(src_dir, '..', 'models', 'multi_drone_vecnorm.pkl'))
        
        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            self.get_logger().warn(f'PPO Model files not found. Fallback to Proportional Control.')
            self.use_ppo = False
            return
            
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from drone_navigation.multi_drone_coverage_env import MultiDroneCoverageEnv
            
            # PPO expects a 6-drone env configuration
            raw_env = DummyVecEnv([lambda: MultiDroneCoverageEnv(n_drones=6)])
            self.vecnorm = VecNormalize.load(vec_path, raw_env)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
            
            self.model = PPO.load(model_path, env=self.vecnorm)
            self.get_logger().info('Successfully loaded PPO model and VecNormalize stats.')
        except Exception as e:
            self.get_logger().error(f'Failed to load PPO packages/models: {str(e)}. Falling back to Proportional Control.')
            self.use_ppo = False

    def fleet_state_callback(self, msg: Float32MultiArray):
        """Receive fused state estimate from whiteboard."""
        n_features = 8
        if len(msg.data) >= self.n_drones * n_features:
            for i in range(self.n_drones):
                offset = i * n_features
                self.fused_states[i] = msg.data[offset:offset+n_features]

    def active_wps_callback(self, msg: Float32MultiArray):
        """Receive the active waypoints and waypoint progress indices from the replanner."""
        expected_size = self.n_drones * self.n_waypoints_fa * 3 + self.n_drones
        if len(msg.data) == expected_size:
            # Squeeze coordinates
            coords_size = self.n_drones * self.n_waypoints_fa * 3
            coords = np.array(msg.data[:coords_size], dtype=np.float32)
            self.active_waypoints = coords.reshape(self.n_drones, self.n_waypoints_fa, 3)
            
            # Squeeze current waypoint index progress
            self.current_wp_idx = [int(idx) for idx in msg.data[coords_size:]]
        else:
            self.get_logger().warn(f'Received active waypoints size {len(msg.data)}, expected {expected_size}')

    def control_loop(self):
        """Run control calculations and publish drone commands at 10Hz."""
        if self.active_waypoints is None:
            # No waypoints planned yet, hold / hover
            self.hover()
            return
            
        # Target waypoint for each drone (size: n_drones x 3)
        targets = np.zeros((self.n_drones, 3), dtype=np.float32)
        max_wp = self.n_waypoints_fa - 1
        for i in range(self.n_drones):
            wp_idx = min(self.current_wp_idx[i], max_wp)
            targets[i] = self.active_waypoints[i, wp_idx]
            
        if self.use_ppo and self.model is not None and self.vecnorm is not None:
            # PPO Low level control (with 3-drone to 6-drone observation padding)
            action = self.compute_ppo_action(targets)
        else:
            # Proportional control fallback
            action = self.compute_proportional_action(targets)
            
        # Publish actions
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.cmd_pub.publish(msg)
        
        # Publish to Gazebo cmd_vel topics (scaled by max speed)
        for i in range(self.n_drones):
            twist = Twist()
            offset = i * 3
            twist.linear.x = float(action[offset]) * 1.8
            twist.linear.y = float(action[offset+1]) * 1.8
            twist.linear.z = float(action[offset+2]) * 1.8
            self.gazebo_cmd_pubs[i].publish(twist)

    def compute_ppo_action(self, targets: np.ndarray) -> np.ndarray:
        """Padd observations dynamically if n_drones < 6 to match 6-drone PPO requirement, else construct directly."""
        cov_ratio = self.fused_states[0, 7] # shared coverage ratio
        centroid = self.fused_states[:, :3].mean(axis=0) # centroid of active drones
        
        parts = []
        # 1. actual drones
        for i in range(self.n_drones):
            pos = self.fused_states[i, :3]
            vel = self.fused_states[i, 3:6]
            battery = self.fused_states[i, 6]
            
            pos_n = pos / np.array([100.0, 100.0, 15.0], dtype=np.float32)
            vel_n = vel / 1.8
            bat_n = np.array([battery / 200.0], dtype=np.float32)
            cov_n = np.array([cov_ratio], dtype=np.float32)
            rel_c = (pos - centroid) / 100.0
            wind_n = np.zeros(3, dtype=np.float32) # Wind is set to zero for simplicity in control obs
            
            part = np.concatenate([pos_n, vel_n, bat_n, cov_n, rel_c, wind_n])
            parts.append(part)
            
        # 2. if n_drones < 6, pad with virtual/dummy drones to reach 6 drones
        if self.n_drones < 6:
            n_dummy = 6 - self.n_drones
            dummy_starts = [[5.0, 5.0, 6.0], [5.0, 7.0, 6.0], [7.0, 5.0, 6.0], [7.0, 7.0, 6.0], [6.0, 5.0, 6.0], [6.0, 7.0, 6.0]]
            for i in range(n_dummy):
                pos_dummy = np.array(dummy_starts[i % len(dummy_starts)], dtype=np.float32)
                pos_n = pos_dummy / np.array([100.0, 100.0, 15.0], dtype=np.float32)
                vel_n = np.zeros(3, dtype=np.float32)
                bat_n = np.array([1.0], dtype=np.float32)
                cov_n = np.array([cov_ratio], dtype=np.float32)
                rel_c = np.zeros(3, dtype=np.float32)
                wind_n = np.zeros(3, dtype=np.float32)
                
                part = np.concatenate([pos_n, vel_n, bat_n, cov_n, rel_c, wind_n])
                parts.append(part)
                
        fused_obs = np.concatenate(parts).astype(np.float32).reshape(1, -1)
        
        # Predict using normalized observation wrapper
        norm_obs = self.vecnorm.normalize_obs(fused_obs)
        action, _states = self.model.predict(norm_obs, deterministic=True)
        
        action_flat = action.flatten()
        return action_flat[:self.n_drones * 3]

    def compute_proportional_action(self, targets: np.ndarray) -> np.ndarray:
        """Fallback proportional control logic (with slowdown buffer to prevent overshoot)."""
        action = np.zeros((self.n_drones, 3), dtype=np.float32)
        max_speed = 0.9 # nominal tracking speed
        
        for i in range(self.n_drones):
            pos = self.fused_states[i, :3]
            diff = targets[i] - pos
            dist = np.linalg.norm(diff) + 1e-9
            
            if dist < 8.0:
                speed_factor = max(0.35, dist / 8.0)
                action[i] = np.clip((diff / dist) * max_speed * speed_factor, -1.0, 1.0)
            else:
                action[i] = np.clip((diff / dist) * max_speed, -1.0, 1.0)
                
        return action.flatten()

    def hover(self):
        """Publish zero command velocities for all drones."""
        msg = Float32MultiArray()
        msg.data = [0.0] * (self.n_drones * 3)
        self.cmd_pub.publish(msg)
        
        # Also stop Gazebo models
        for i in range(self.n_drones):
            twist = Twist()
            self.gazebo_cmd_pubs[i].publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LowLevelControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

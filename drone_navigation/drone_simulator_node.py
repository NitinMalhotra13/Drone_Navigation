import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan, Range, Imu
import numpy as np
import time

from drone_navigation.multi_drone_coverage_env import MultiDroneCoverageEnv

class DroneSimulatorNode(Node):
    def __init__(self):
        super().__init__('drone_simulator')
        self.get_logger().info('Initializing Drone Simulator Node...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.declare_parameter('sim_rate_hz', 10.0)
        
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.sim_rate = self.get_parameter('sim_rate_hz').get_parameter_value().double_value
        
        # Initialize environment
        self.env = MultiDroneCoverageEnv(n_drones=self.n_drones, wind_enabled=True, thermal_enabled=True)
        self.obs, _ = self.env.reset()
        
        # Current action cache (last received action commands for 3 drones)
        # Size = n_drones * 3. Default to zero actions (hovering/stable)
        self.current_action = np.zeros(self.n_drones * 3, dtype=np.float32)
        
        # Publishers
        self.states_pub = self.create_publisher(Float32MultiArray, '/drones/states', 10)
        self.map_pub = self.create_publisher(Float32MultiArray, '/map/coverage', 10)
        
        # Per-drone publishers
        self.lidar_pubs = []
        self.camera_pubs = []
        self.range_pubs = []
        self.imu_pubs = []
        
        for i in range(self.n_drones):
            self.lidar_pubs.append(self.create_publisher(LaserScan, f'/drone_{i}/lidar', 10))
            self.camera_pubs.append(self.create_publisher(Float32MultiArray, f'/drone_{i}/camera', 10))
            self.range_pubs.append(self.create_publisher(Range, f'/drone_{i}/range', 10))
            self.imu_pubs.append(self.create_publisher(Imu, f'/drone_{i}/imu', 10))
            
        # Subscribers
        self.cmd_sub = self.create_subscriber(Float32MultiArray, '/drones/commands', self.cmd_callback, 10)
        
        # Timed loop
        self.timer = self.create_timer(1.0 / self.sim_rate, self.timer_callback)
        self.get_logger().info('Drone Simulator Node started.')

    def cmd_callback(self, msg: Float32MultiArray):
        """Cache the latest velocity/action command vector from the controller."""
        if len(msg.data) == self.n_drones * 3:
            self.current_action = np.array(msg.data, dtype=np.float32)
        else:
            self.get_logger().warn(f'Received commands of length {len(msg.data)}, expected {self.n_drones * 3}')

    def timer_callback(self):
        """Simulate one environment step and publish all sensor observations."""
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(self.current_action)
        self.obs = obs
        
        # Reset if terminal state is hit
        if terminated or truncated:
            self.get_logger().info('Simulation episode ended. Resetting environment...')
            self.obs, _ = self.env.reset()
            self.current_action = np.zeros(self.n_drones * 3, dtype=np.float32)
            return

        # 1. Publish raw observations
        obs_msg = Float32MultiArray()
        obs_msg.data = self.obs.tolist()
        self.states_pub.publish(obs_msg)

        # 2. Publish coverage map
        map_msg = Float32MultiArray()
        map_msg.data = self.env.coverage_grid.flatten().astype(np.float32).tolist()
        self.map_pub.publish(map_msg)

        # 3. Publish per-drone sensor topics
        stamp = self.get_clock().now().to_msg()
        for i in range(self.n_drones):
            # A. LiDAR scan
            lidar_data = self.env.get_lidar_scan(i)
            lidar_msg = LaserScan()
            lidar_msg.header.stamp = stamp
            lidar_msg.header.frame_id = f'drone_{i}_lidar_link'
            lidar_msg.angle_min = 0.0
            lidar_msg.angle_max = 2.0 * np.pi
            lidar_msg.angle_increment = 2.0 * np.pi / 36.0
            lidar_msg.time_increment = 0.0
            lidar_msg.scan_time = 1.0 / self.sim_rate
            lidar_msg.range_min = 0.1
            lidar_msg.range_max = 10.0
            lidar_msg.ranges = lidar_data.tolist()
            self.lidar_pubs[i].publish(lidar_msg)
            
            # B. Camera features
            cam_data = self.env.get_camera_image(i)
            cam_msg = Float32MultiArray()
            cam_msg.data = cam_data.tolist()
            self.camera_pubs[i].publish(cam_msg)
            
            # C. Ultrasonic Range
            range_val = self.env.get_ultrasonic_range(i)
            range_msg = Range()
            range_msg.header.stamp = stamp
            range_msg.header.frame_id = f'drone_{i}_range_link'
            range_msg.radiation_type = Range.ULTRASOUND
            range_msg.field_of_view = 0.3  # rad
            range_msg.min_range = 0.1
            range_msg.max_range = 10.0
            range_msg.range = range_val
            self.range_pubs[i].publish(range_msg)
            
            # D. IMU data
            imu_data = self.env.get_imu_data(i)
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = f'drone_{i}_imu_link'
            
            # Populate acceleration
            imu_msg.linear_acceleration.x = imu_data['linear_acceleration'][0]
            imu_msg.linear_acceleration.y = imu_data['linear_acceleration'][1]
            imu_msg.linear_acceleration.z = imu_data['linear_acceleration'][2]
            
            # Populate angular velocity
            imu_msg.angular_velocity.x = imu_data['angular_velocity'][0]
            imu_msg.angular_velocity.y = imu_data['angular_velocity'][1]
            imu_msg.angular_velocity.z = imu_data['angular_velocity'][2]
            
            # Euler to Quaternion conversion for orientation representation
            roll, pitch, yaw = imu_data['orientation']
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            imu_msg.orientation.w = cr * cp * cy + sr * sp * sy
            imu_msg.orientation.x = sr * cp * cy - cr * sp * sy
            imu_msg.orientation.y = cr * sp * cy + sr * cp * sy
            imu_msg.orientation.z = cr * cp * sy - sr * sp * cy
            
            self.imu_pubs[i].publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DroneSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

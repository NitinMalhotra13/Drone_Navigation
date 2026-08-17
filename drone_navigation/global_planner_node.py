import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
import numpy as np
import os

from drone_navigation.fa_coverage import FireflyPlanner

class GlobalPlannerNode(Node):
    def __init__(self):
        super().__init__('global_planner')
        self.get_logger().info('Initializing Global Planner Node (FA)...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.declare_parameter('n_waypoints', 20)
        self.declare_parameter('fa_iterations', 40)
        
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.n_waypoints = self.get_parameter('n_waypoints').get_parameter_value().integer_value
        self.fa_iterations = self.get_parameter('fa_iterations').get_parameter_value().integer_value
        
        # Publisher
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/drones/global_waypoints', 10)
        
        # Service
        self.srv = self.create_service(Trigger, '/plan_global_path', self.handle_replan_service)
        
        # Execute global planning once on startup
        self.plan_and_publish()

    def plan_and_publish(self):
        """Execute Firefly Algorithm to plan initial fanned-out corridors."""
        self.get_logger().info(f'Running Firefly Algorithm for {self.n_drones} drones ({self.fa_iterations} iterations)...')
        
        planner = FireflyPlanner(
            n_drones=self.n_drones,
            n_waypoints=self.n_waypoints,
            n_fireflies=30,
            max_iter=self.fa_iterations
        )
        
        planner.optimize(verbose=True)
        waypoints = planner.get_best_waypoints() # Shape: (n_drones, n_waypoints, 3)
        
        self.get_logger().info(f'Planning completed. Fused coverage: {planner.get_coverage_stats()["coverage_ratio"] * 100:.1f}%')
        
        # Publish planned waypoints
        msg = Float32MultiArray()
        msg.data = waypoints.flatten().tolist()
        self.waypoints_pub.publish(msg)
        self.get_logger().info('Published global waypoints to /drones/global_waypoints.')

    def handle_replan_service(self, request, response):
        """Service callback to re-trigger global planning."""
        try:
            self.plan_and_publish()
            response.success = True
            response.message = 'Successfully re-planned global path waypoints.'
        except Exception as e:
            response.success = False
            response.message = f'Failed to re-plan: {str(e)}'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

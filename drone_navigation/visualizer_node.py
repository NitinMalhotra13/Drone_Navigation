import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
import os

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        self.get_logger().info('Initializing Visualizer/Recorder Node...')
        
        # ROS Parameters
        self.declare_parameter('n_drones', 6)
        self.declare_parameter('output_path', 'models/drone_coverage_ros2.gif')
        
        self.n_drones = self.get_parameter('n_drones').get_parameter_value().integer_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        
        # Ensure absolute output path
        src_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.normpath(os.path.join(src_dir, '..', self.output_path))
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Data trackers for animation
        self.history_steps = []
        self.history_positions = [[] for _ in range(self.n_drones)]
        self.history_batteries = [[] for _ in range(self.n_drones)]
        self.history_coverage = []
        self.history_collisions = []
        self.history_path_len = []
        
        self.step_counter = 0
        self.global_waypoints = None
        self.fused_coverage = np.zeros((100, 100), dtype=np.float32)
        
        # Subscribers
        self.fleet_state_sub = self.create_subscriber(Float32MultiArray, '/whiteboard/fleet_state', self.fleet_state_callback, 10)
        self.coverage_map_sub = self.create_subscriber(Float32MultiArray, '/whiteboard/coverage_map', self.coverage_map_callback, 10)
        self.global_wps_sub = self.create_subscriber(Float32MultiArray, '/drones/global_waypoints', self.global_wps_callback, 10)
        
        # Register shutdown hook to write the video/animation when the node exits
        rclpy.get_default_context().on_shutdown(self.on_shutdown)
        self.get_logger().info('Visualizer Node started. Subscribed to whiteboard topic logs.')

    def global_wps_callback(self, msg: Float32MultiArray):
        """Receive the fanned-out waypoints from the planner."""
        if self.global_waypoints is None:
            self.global_waypoints = np.array(msg.data, dtype=np.float32).reshape(self.n_drones, -1, 3)

    def coverage_map_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 10000:
            self.fused_coverage = np.array(msg.data, dtype=np.float32).reshape(100, 100)

    def fleet_state_callback(self, msg: Float32MultiArray):
        """Accumulate fleet state frame history."""
        # msg.data layout: flat array of [x, y, z, vx, vy, vz, battery, cov] per drone (size 8 * n_drones)
        n_features = 8
        if len(msg.data) >= self.n_drones * n_features:
            self.step_counter += 1
            self.history_steps.append(self.step_counter)
            
            total_collisions = 0
            total_path_len = 0.0
            
            for i in range(self.n_drones):
                offset = i * n_features
                pos = msg.data[offset:offset+3]
                battery = msg.data[offset+6]
                cov_ratio = msg.data[offset+7]
                
                self.history_positions[i].append(pos)
                self.history_batteries[i].append(battery)
                
                # Compute path length approximate based on step integrations
                if len(self.history_positions[i]) > 1:
                    prev = self.history_positions[i][-2]
                    total_path_len += np.linalg.norm(np.array(pos) - np.array(prev))
                    
            # Record global metrics
            self.history_coverage.append(cov_ratio * 100.0)
            self.history_collisions.append(0) # placeholder for local fusion logging
            self.history_path_len.append(total_path_len)

    def on_shutdown(self):
        """Matplotlib video/GIF rendering execution on ROS shutdown."""
        self.get_logger().info('ROS Shutdown detected. Saving final flight video...')
        if len(self.history_steps) < 10:
            self.get_logger().warn('Too few frames captured to render video.')
            return
            
        try:
            self.render_and_save()
        except Exception as e:
            self.get_logger().error(f'Failed to render/save video: {str(e)}')

    def render_and_save(self):
        """Set up 2-panel figure dashboard and write GIF/MP4 file."""
        self.get_logger().info(f'Rendering {len(self.history_steps)} animation frames to {self.output_path}...')
        
        # We render a 2-panel dashboard: Left = 2D path tracking, Right = live metrics
        fig, (ax_map, ax_met) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0d0d1a')
        fig.suptitle('ROS2 Drone Coverage Simulation Run', color='white', fontsize=12, fontweight='bold')
        
        # Format axes
        for ax in (ax_map, ax_met):
            ax.set_facecolor('#131328')
            for spine in ax.spines.values():
                spine.set_color('#3a3a6a')
            ax.tick_params(colors='white')
            
        ax_map.set_title('Top-Down Drone Paths (Fused State)', color='white')
        ax_map.set_xlim(0, 100)
        ax_map.set_ylim(0, 100)
        ax_map.set_xlabel('X Grid', color='white')
        ax_map.set_ylabel('Y Grid', color='white')
        
        ax_met.set_title('Telemetry Metrics Over Time', color='white')
        ax_met.set_xlabel('Steps', color='white')
        ax_met.set_ylabel('% / Units', color='white')
        ax_met.set_xlim(0, len(self.history_steps))
        ax_met.set_ylim(0, 100)
        
        # Plot lines
        colors = ['red', 'cyan', 'lime']
        path_lines = []
        drone_dots = []
        for i in range(self.n_drones):
            line, = ax_map.plot([], [], color=colors[i], label=f'Drone {i}')
            dot, = ax_map.plot([], [], 'o', color=colors[i], markersize=6)
            path_lines.append(line)
            drone_dots.append(dot)
        ax_map.legend(labelcolor='white')
        
        cov_line, = ax_met.plot([], [], color='purple', label='Coverage %')
        bat_line, = ax_met.plot([], [], color='orange', label='Battery %')
        ax_met.legend(labelcolor='white')
        
        # Subsample frames to keep output file size small
        frame_skip = 5
        frame_indices = list(range(0, len(self.history_steps), frame_skip))
        
        def update_frame(frame_idx):
            # Update paths
            for i in range(self.n_drones):
                pts = np.array(self.history_positions[i][:frame_idx+1])
                if len(pts) > 0:
                    path_lines[i].set_data(pts[:, 0], pts[:, 1])
                    drone_dots[i].set_data([pts[-1, 0]], [pts[-1, 1]])
            
            # Update metrics
            steps = self.history_steps[:frame_idx+1]
            covs = self.history_coverage[:frame_idx+1]
            bats = [np.mean([self.history_batteries[d][s] for d in range(self.n_drones)]) / 2.0 for s in range(len(steps))]
            
            cov_line.set_data(steps, covs)
            bat_line.set_data(steps, bats)
            
            return path_lines + drone_dots + [cov_line, bat_line]
            
        anim = FuncAnimation(fig, update_frame, frames=frame_indices, blit=True)
        
        # Save as GIF (fallback-safe Pillow writer)
        writer = PillowWriter(fps=10)
        anim.save(self.output_path, writer=writer)
        plt.close(fig)
        self.get_logger().info(f'Flight animation video successfully saved to: {self.output_path}')

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

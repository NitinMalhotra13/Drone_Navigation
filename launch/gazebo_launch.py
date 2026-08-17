import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Share directory lookup
    try:
        pkg_share = get_package_share_directory('drone_navigation')
    except Exception:
        pkg_share = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    world_path = os.path.normpath(os.path.join(pkg_share, 'worlds', 'drone_coverage.world'))
    urdf_path = os.path.normpath(os.path.join(pkg_share, 'urdf', 'drone.urdf'))
    
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "false" to run Gazebo headless (gzserver only)'
    )
    
    # Path to standard gazebo_ros launcher
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'gui': LaunchConfiguration('gui')
        }.items()
    )
    
    # 6 drone starts mapped to Gazebo coordinate offsets
    starts = [
        (4.5, 4.5), (4.5, 6.5), (6.5, 4.5),
        (6.5, 6.5), (5.5, 4.5), (5.5, 6.5)
    ]
    
    spawn_drones = []
    for i, (sx, sy) in enumerate(starts):
        spawn_drones.append(
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name=f'spawn_drone_{i}',
                arguments=[
                    '-file', urdf_path,
                    '-entity', f'drone_{i}',
                    '-robot_namespace', f'drone_{i}',
                    '-x', str(sx),
                    '-y', str(sy),
                    '-z', '0.5'
                ],
                output='screen'
            )
        )
        
    # Fleet control nodes (excluding simulator, since Gazebo runs physics)
    whiteboard = Node(
        package='drone_navigation',
        executable='whiteboard',
        name='whiteboard',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    global_planner = Node(
        package='drone_navigation',
        executable='global_planner',
        name='global_planner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    adaptive_replanner = Node(
        package='drone_navigation',
        executable='adaptive_replanner',
        name='adaptive_replanner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    low_level_controller = Node(
        package='drone_navigation',
        executable='low_level_controller',
        name='low_level_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    dynamic_obstacles = Node(
        package='drone_navigation',
        executable='dynamic_obstacles',
        name='dynamic_obstacles_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        gui_arg,
        gazebo_launch,
        *spawn_drones,
        whiteboard,
        global_planner,
        adaptive_replanner,
        low_level_controller,
        dynamic_obstacles
    ])

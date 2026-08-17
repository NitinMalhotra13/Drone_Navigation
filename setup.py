from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*launch.[pxy][yma]*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/worlds', glob('worlds/*')),
        ('share/' + package_name + '/models', glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nitin',
    maintainer_email='nitin@example.com',
    description='Autonomous 3D Multi-Drone Area Coverage inside complex 3D environments under ROS2 framework',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator = drone_navigation.drone_simulator_node:main',
            'whiteboard = drone_navigation.whiteboard_node:main',
            'global_planner = drone_navigation.global_planner_node:main',
            'adaptive_replanner = drone_navigation.adaptive_replanner_node:main',
            'low_level_controller = drone_navigation.low_level_controller_node:main',
            'visualizer = drone_navigation.visualizer_node:main',
        ],
    },
)

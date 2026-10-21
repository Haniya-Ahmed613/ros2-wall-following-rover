import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_description = get_package_share_directory('diffbot_description')
    pkg_gazebo = get_package_share_directory('diffbot_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_description, 'urdf', 'diffbot.urdf.xacro')
    world_file = os.path.join(pkg_gazebo, 'worlds', 'wall_world.sdf')
    bridge_config = os.path.join(pkg_gazebo, 'config', 'bridge.yaml')
    robot_description = Command(['xacro ', xacro_file])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    spawn_entity = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', 'diffbot',
                   '-x', '-2.5', '-y', '-2.5', '-z', '0.1'],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output='screen'
    )

    rviz = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(pkg_description, 'rviz', 'diffbot.rviz')],
        output='screen'
    )

    return LaunchDescription([gz_sim, robot_state_publisher, spawn_entity, bridge, rviz])
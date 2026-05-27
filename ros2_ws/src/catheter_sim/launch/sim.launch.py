#!/usr/bin/env python3
"""
sim.launch.py — GZ Sim 8 + catheter arm launch file

Starts:
  1. GZ Sim with catheter_world.sdf
  2. robot_state_publisher (with processed XACRO)
  3. ros_gz_bridge for /clock, /joint_states, /tf
  4. Spawn robot model in GZ Sim
  5. joint_state_broadcaster controller spawner
  6. joint_trajectory_controller controller spawner
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro


def generate_launch_description():
    pkg = get_package_share_directory('catheter_sim')

    # ── Arguments ──────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = os.path.join(pkg, 'worlds', 'catheter_world.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'catheter_arm.urdf.xacro')

    # Process XACRO → URDF string
    robot_description_content = xacro.process_file(xacro_file).toxml()

    # ── GZ Sim ─────────────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}',
        }.items(),
    )

    # ── robot_state_publisher ──────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time,
        }],
    )

    # ── Spawn robot in GZ Sim ──────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_catheter_arm',
        arguments=[
            '-name', 'catheter_arm',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
        ],
        output='screen',
    )

    # ── ros_gz_bridge ──────────────────────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── Controller spawners ────────────────────────────────────────
    # Delay until robot is spawned and controllers are loaded
    joint_state_broadcaster_spawner = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ],
    )

    joint_trajectory_controller_spawner = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true',
        ),
        gz_sim,
        robot_state_publisher,
        bridge,
        TimerAction(period=2.0, actions=[spawn_robot]),
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
    ])

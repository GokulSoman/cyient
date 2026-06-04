#!/usr/bin/env python3
"""
full_system.launch.py — Complete catheter simulation system

Launches in sequence:
  1.   sim.launch.py       — GZ Sim + robot + controllers
  2.   image_publisher     — streams ultrasound images (3 s delay)
  3.   vessel_detector     — OpenCV perception node  (3 s delay)
  4.   target_planner      — pixel→3D coordinate map (3.5 s delay)
  5.   arm_controller      — MoveIt2 / fallback ctrl (5 s delay)
  6.   rviz2               — visualisation          (4 s delay)

Single command:
  ros2 launch catheter_sim full_system.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, TimerAction, DeclareLaunchArgument,
    GroupAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    pkg = get_package_share_directory('catheter_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # ── 1. Simulation (GZ Sim + robot + controllers) ───────────────
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'sim.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # ── 2. Image publisher ─────────────────────────────────────────
    image_dir = os.path.join(
        os.path.expanduser('~'),
        'github_repos', 'cyient', 'data', 'sample_images',
    )
    image_publisher = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='catheter_sim',
                executable='image_publisher',
                name='image_publisher',
                output='screen',
                parameters=[{
                    'image_dir': image_dir,
                    'publish_rate': 1.0,
                    'use_sim_time': use_sim_time,
                }],
            )
        ],
    )

    # ── 3. Vessel detector ─────────────────────────────────────────
    vessel_detector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='catheter_sim',
                executable='vessel_detector',
                name='vessel_detector',
                output='screen',
                parameters=[{
                    'debug_publish': True,
                    'use_sim_time': use_sim_time,
                }],
            )
        ],
    )

    # ── 4. Target planner ──────────────────────────────────────────
    target_planner = TimerAction(
        period=3.5,
        actions=[
            Node(
                package='catheter_sim',
                executable='target_planner',
                name='target_planner',
                output='screen',
                parameters=[{
                    'world_frame': 'world',
                    'use_sim_time': use_sim_time,
                }],
            )
        ],
    )

    # ── 5. Arm controller ──────────────────────────────────────────
    arm_controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='catheter_sim',
                executable='arm_controller',
                name='arm_controller',
                output='screen',
                parameters=[{
                    'move_group_name': 'arm',
                    'position_tolerance': 0.02,
                    'use_sim_time': use_sim_time,
                }],
            )
        ],
    )

    # ── 6. RViz2 ───────────────────────────────────────────────────
    rviz_config = os.path.join(pkg, 'config', 'catheter_sim.rviz')
    rviz = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
                parameters=[{'use_sim_time': use_sim_time}],
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock',
        ),
        sim_launch,
        image_publisher,
        vessel_detector,
        target_planner,
        arm_controller,
        rviz,
    ])

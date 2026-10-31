"""Launch file for the drone_agent pipeline.

Launches:
1. ros_gz_bridge — bridges Gazebo camera to ROS2
2. yolo_detector — runs YOLOv8 on camera images
3. brain_node — orchestrates LLM + PX4 control

Prerequisites (run in separate terminals):
- PX4 SITL: cd PX4-Autopilot && make px4_sitl gz_x500_mono_cam
- uXRCE-DDS: MicroXRCEAgent udp4 -p 8888
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        # --- Arguments ---
        DeclareLaunchArgument(
            'camera_topic', default_value='/camera',
            description='Gazebo camera topic name',
        ),
        DeclareLaunchArgument(
            'yolo_model', default_value='yolov8n.pt',
            description='YOLO model file (e.g. yolov8n.pt, yolov8s.pt)',
        ),
        DeclareLaunchArgument(
            'gt_overlay', default_value='false',
            description='Demo only: draw simulator ground-truth people/cars as detections',
        ),
        DeclareLaunchArgument(
            'llm_interval', default_value='7.0',
            description='Seconds between periodic LLM re-evaluation calls',
        ),

        # --- Gazebo 4-drone cameras & Ardupilot_Swarm 3D LiDAR bridge ---
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/camera@sensor_msgs/msg/Image@gz.msgs.Image',
                '/world/drone_demo/model/x500_mono_cam_0/link/mono_cam/base_link/sensor/imager/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/world/drone_demo/model/x500_mono_cam_1/link/mono_cam/base_link/sensor/imager/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/world/drone_demo/model/x500_mono_cam_2/link/mono_cam/base_link/sensor/imager/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/world/drone_demo/model/x500_mono_cam_3/link/mono_cam/base_link/sensor/imager/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '--ros-args',
                '-r', '/world/drone_demo/model/x500_mono_cam_0/link/mono_cam/base_link/sensor/imager/image:=/camera',
                '-r', '/world/drone_demo/model/x500_mono_cam_1/link/mono_cam/base_link/sensor/imager/image:=/uav_1/camera',
                '-r', '/world/drone_demo/model/x500_mono_cam_2/link/mono_cam/base_link/sensor/imager/image:=/uav_2/camera',
                '-r', '/world/drone_demo/model/x500_mono_cam_3/link/mono_cam/base_link/sensor/imager/image:=/uav_3/camera',
                '-r', '/lidar/points:=/iris1/cloud_in',
            ],
            output='screen',
        ),

        # --- YOLO detection node ---
        Node(
            package='drone_agent',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('yolo_model'),
                'confidence_threshold': 0.5,
                'camera_topic': LaunchConfiguration('camera_topic'),
                'skip_frames': 9,          # ~3 inferences/s per camera: YOLO on CPU at 10/s x4 starved PX4 (state estimate lost)
                'gt_overlay': ParameterValue(LaunchConfiguration('gt_overlay'), value_type=bool),
            }],
        ),

        # --- De-duplicated victim count from real YOLO boxes (/swarm/victims) ---
        Node(
            package='drone_agent',
            executable='victim_registry',
            name='victim_registry',
            output='screen',
        ),

        # --- Brain node ---
        Node(
            package='drone_agent',
            executable='brain_node',
            name='brain_node',
            output='screen',
            parameters=[{
                'llm_interval_sec': LaunchConfiguration('llm_interval'),
                'offboard_rate_hz': 25.0,
            }],
        ),

        # --- Live 2D Map + Annotated YOLO Camera GUI ---
        ExecuteProcess(
            cmd=[
                # lowest priority: the GUI must never steal CPU from PX4/Gazebo (a starved PX4 instance
                # lost its state estimate and flew away in one run)
                'nice', '-n', '12',
                '/usr/bin/python3',
                '/home/tushar/Desktop/swarm/LLM-controlled-drone/src/drone_agent/drone_agent/map_gui_node.py',
            ],
            output='screen',
        ),
    ])

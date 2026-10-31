#!/usr/bin/env python3
"""Live 4-Drone Decentralized Swarm 2D Tactical Map + YOLOv8 + Multi-Camera + 3D OctoMap GUI.

Features:
  [PAGE 1] 4-DRONE SWARM 2D TACTICAL MAP & MULTI-CAMERA VIEW:
           - Left Pane: 2D Tactical Map showing all 4 drones (UAV-1..4), heading arrows,
             assigned GNN/Heuristic task targets, and live Peer-to-Peer GNN communication links.
           - Right Pane: 2x2 QUAD-CAMERA LIVE MULTI-VIEW showing all 4 drone camera feeds
             simultaneously (UAV-1, UAV-2, UAV-3, UAV-4)!
             Clicking any drone camera quadrant (or using top buttons) expands that camera
             to full-view; clicking again returns to the 4-cam quad grid.
  [PAGE 2] WHOLE SCANNED SURROUNDING MAP (4-Drone Collaborative 3D LiDAR + OctoMap Engine):
           - Left Pane: Full-Area 2D Occupancy & Elevation Scanned Map (120m x 120m).
           - Right Pane: Interactive 3D OctoMap Voxel & PointCloud Viewer with all 4 drones,
             3D peer-to-peer GNN links, and 4-camera live PiP multi-view!
"""

import json
import math
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PilImage, ImageTk
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String

from drone_agent.rescue_coordinator import RescueCoordinator
from drone_agent.rover_nav import NavParams, Rover
from drone_agent.voxel_map import VoxelMap
from drone_agent.voxel_viewer import ROVER_COLORS, Scene, ViewOptions, VoxelViewer
from drone_agent.world_sim import Detection, SimDetector, World


def imgmsg_to_cv2_safe(bridge: CvBridge, msg: Image) -> np.ndarray | None:
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception:
        pass
    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = (msg.encoding or '').lower()
        if 'rgb' in enc:
            rgb = buf.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif 'bgr' in enc:
            return buf.reshape((msg.height, msg.width, 3))
        elif 'mono8' in enc or '8uc1' in enc:
            return cv2.cvtColor(buf.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2BGR)
        return buf.reshape((msg.height, msg.width, -1))
    except Exception:
        return None


# Distinct BGR colors for the 4 swarm drones
DRONE_COLORS_BGR = [
    (255, 240, 0),    # UAV-1: Bright Cyan
    (60, 255, 80),    # UAV-2: Neon Green
    (255, 95, 255),   # UAV-3: Magenta / Pink
    (30, 195, 255),   # UAV-4: Bright Orange / Gold
]


class MapGuiNode(Node):
    def __init__(self):
        super().__init__('map_gui_node')
        self.bridge = CvBridge()

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Primary drone (UAV-1) telemetry
        self.x = 3.0
        self.y = 0.0
        self.z = -9.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.armed = True
        self.nav_state = 0
        self.odom_received = False

        self.target_x = 10.0
        self.target_y = 7.0
        self.target_z = -9.0
        self.last_cmd = '4-Drone Decentralized Swarm GNN + Heuristic Whole-Area Scan'
        self.detections = []

        # ── 4-Drone Swarm State ───────────────────────────────────────────
        default_spawns = [
            (0, 'UAV-1', 3.0, 0.0, -9.0, 18.0, 18.0, 'NE Sector (GNN)'),
            (1, 'UAV-2', 0.0, 3.0, -10.5, 18.0, -18.0, 'NW Sector (GNN)'),
            (2, 'UAV-3', -3.0, 0.0, -12.0, -18.0, -18.0, 'SW Sector (GNN)'),
            (3, 'UAV-4', 0.0, -3.0, -13.5, -18.0, 18.0, 'SE Sector (GNN)'),
        ]
        self.swarm_drones: list[dict] = [
            {
                'id': did,
                'name': name,
                'x': sx,
                'y': sy,
                'z': sz,
                'vx': 0.0,
                'vy': 0.0,
                'vz': 0.0,
                'yaw': 0.0,
                'pitch': 0.0,
                'armed': True,
                'battery': 1.0,
                'target_x': tx,
                'target_y': ty,
                'target_z': sz,
                'task_label': lbl,
                'alloc_mode': 'GNN (GraphSAGE)',
                'gnn_score': 0.48,
            }
            for did, name, sx, sy, sz, tx, ty, lbl in default_spawns
        ]
        self.swarm_trails: list[deque] = [deque(maxlen=600) for _ in range(4)]
        self.comm_adj = [[1.0 if i != j else 0.0 for j in range(4)] for i in range(4)]
        self.allocator_status = 'GraphSAGE GNN + Heuristic Fallback'
        self.swarm_mode = 'swarm_scan'
        self.swarm_lock = threading.Lock()

        # ── 4-Drone Individual Camera Feeds ───────────────────────────────
        # Indices 0: UAV-1, 1: UAV-2, 2: UAV-3, 3: UAV-4
        self.drone_cameras: list[np.ndarray | None] = [None, None, None, None]
        self.yolo_annotated_frames: dict[int, np.ndarray | None] = {0: None, 1: None, 2: None, 3: None}
        self.last_yolo_img_times: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.yolo_annotated_frame: np.ndarray | None = None
        self.last_yolo_img_time = 0.0
        # Camera layout mode: 'quad' (2x2 grid showing all 4) or 0, 1, 2, 3 (single focus)
        self.cam_layout = 'quad'

        # ── Collaborative 4-Drone 3D LiDAR & OctoMap Voxel Engine State ───
        self.grid_res = 0.5
        self.grid_half_m = 60.0
        self.grid_size = int((self.grid_half_m * 2) / self.grid_res)  # 240x240
        self.occ_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.height_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.voxels_3d: dict[tuple[int, int, int], float] = {}
        self.drone_scan_endpoints: dict[int, list[tuple[float, float, float]]] = {
            0: [], 1: [], 2: [], 3: []
        }
        self.lidar_msg_count = 0
        self.scan_lock = threading.Lock()

        # 3D viewer camera angles
        self.view_azim = 38.0
        self.view_elev = 28.0
        self.auto_rotate_3d = True
        self.view_zoom = 1.0

        sat_raw = cv2.imread('/home/tushar/Desktop/swarm/LLM-controlled-drone/worlds/satellite_map.png')
        if sat_raw is not None:
            self.sat_map = cv2.convertScaleAbs(sat_raw, alpha=1.22, beta=24)
        else:
            self.sat_map = None

        # Publisher for user commands
        self.cmd_pub = self.create_publisher(String, '/user_command', 10)

        # Telemetry Subscriptions
        self.create_subscription(String, '/swarm/state', self._swarm_state_cb, 10)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, px4_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._status_cb, px4_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_cb, px4_qos)
        self.create_subscription(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self._sp_cb, 10)
        self.create_subscription(String, '/user_command', self._cmd_cb, 10)
        self.create_subscription(String, '/yolo/detections', self._det_cb, 10)

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 4-Drone Camera Feed Subscriptions (YOLO Annotated + Raw) ──────
        self.create_subscription(Image, '/yolo/image_annotated', lambda msg: self._img_yolo_cb(0, msg), img_qos)
        self.create_subscription(Image, '/uav_0/yolo/annotated', lambda msg: self._img_yolo_cb(0, msg), img_qos)
        self.create_subscription(Image, '/uav_1/yolo/annotated', lambda msg: self._img_yolo_cb(1, msg), img_qos)
        self.create_subscription(Image, '/uav_2/yolo/annotated', lambda msg: self._img_yolo_cb(2, msg), img_qos)
        self.create_subscription(Image, '/uav_3/yolo/annotated', lambda msg: self._img_yolo_cb(3, msg), img_qos)

        self.create_subscription(Image, '/camera', lambda msg: self._raw_cam_cb(0, msg), img_qos)
        self.create_subscription(Image, '/uav_0/camera', lambda msg: self._raw_cam_cb(0, msg), img_qos)
        self.create_subscription(Image, '/uav_1/camera', lambda msg: self._raw_cam_cb(1, msg), img_qos)
        self.create_subscription(Image, '/uav_2/camera', lambda msg: self._raw_cam_cb(2, msg), img_qos)
        self.create_subscription(Image, '/uav_3/camera', lambda msg: self._raw_cam_cb(3, msg), img_qos)

        # Ardupilot_Swarm LiDAR subscriptions
        self.create_subscription(PointCloud2, '/iris1/cloud_in', self._pc2_cb, img_qos)
        self.create_subscription(LaserScan, '/lidar', self._laserscan_cb, img_qos)

        # 5Hz collaborative 4-drone 3D LiDAR scanner
        self.world_boxes, self.world_cyls = self._get_world_obstacles()
        self.create_timer(0.2, self._collaborative_4drone_lidar_tick)

        # ── 3-D voxel map, rescue rovers and coordinator ──────────────────
        # The voxel map is fed by the same (simulated) LiDAR the 2-D map uses, plus any real PointCloud2/LaserScan.
        self.vmap = VoxelMap(res=0.5, half=self.grid_half_m)
        self.sim_world = World()          # same obstacle list as MapGuiNode._get_world_obstacles()
        self.nav_params = NavParams()
        self.rovers = [Rover(0, 'Rover 1', ROVER_COLORS[0], -2.0, 5.0, yaw=math.pi / 2),
                       Rover(1, 'Rover 2', ROVER_COLORS[1], 2.0, -5.0, yaw=-math.pi / 2)]
        self.coord = RescueCoordinator(self.vmap, self.sim_world, self.rovers, self.nav_params)
        self.sim_detector = SimDetector(self.sim_world)
        self.det_source = 'sim'                       # 'sim' | 'swarm' | 'both' | 'off'
        self.rescue_lock = threading.RLock()
        self.viewer = VoxelViewer(self.vmap, ViewOptions())
        self.viewer.cam.auto_orbit = False
        self.use_legacy_viewer = False
        self.drone_trails3d: list[deque] = [deque(maxlen=500) for _ in range(4)]
        self._last_trail_t = 0.0
        self._last_rescue_t = time.monotonic()
        self._last_det_t = 0.0
        self._lidar_tick_n = 0
        self.swarm_victims: list[dict] = []
        self.create_subscription(String, '/swarm/victims', self._swarm_victims_cb, 10)
        self.create_timer(0.05, self._rescue_tick)

        # Rover cameras: first-person views rendered from the voxel map (SIMULATED cameras: the rovers are not in Gazebo).
        # Rendered on a worker thread at ~4 Hz so the GUI and ROS callbacks never wait for them.
        self.rover_viewers = [VoxelViewer.first_person(self.viewer) for _ in self.rovers]
        self.rover_cam_frames: list = [None] * len(self.rovers)
        self._rover_cam_stop = False
        threading.Thread(target=self._rover_cam_loop, daemon=True).start()

        self.get_logger().info('4-Drone Swarm 2D Tactical Map + Multi-Camera + 3D OctoMap GUI initialized.')

    @staticmethod
    def _get_world_obstacles():
        boxes = [
            (26.0, 32.0, 14.0, 18.0, 11.5),     # Collapsed Fire Station NE
            (26.0, -32.0, 15.0, 15.0, 8.6),     # Collapsed Police Station NW
            (-24.0, 30.0, 12.0, 12.0, 6.0),     # Collapsed House SE
            (-24.0, -30.0, 12.0, 12.0, 6.0),    # Collapsed House SW
            (36.0, 0.0, 12.0, 14.0, 6.0),       # Collapsed House North
            (-36.0, 0.0, 14.0, 15.0, 8.6),      # Collapsed Police Station South
            (-8.0, 11.0, 5.0, 2.5, 2.4),        # Rescue Ambulance
            (3.0, 12.0, 4.4, 2.0, 1.5),         # Stranded Prius Car
            (-3.0, -11.0, 4.6, 2.2, 1.8),       # Damaged SUV
        ]
        cyls = [
            (8.0, 8.0, 0.6, 1.85),              # Walking Survivor #1
            (-8.0, -8.0, 0.6, 1.85),            # Walking Survivor #2
            (10.0, 4.0, 0.6, 1.90),             # Standing Survivor Male #1
            (6.0, -8.0, 0.6, 1.77),             # Standing Survivor Female #1
            (-5.0, 9.0, 0.6, 1.90),             # Standing Survivor Male #2
            (8.15, 2.36, 0.9, 0.6),             # Lying Victim Male #1
            (4.60, -6.78, 0.9, 0.6),            # Lying Victim Female #1
            (-8.54, -3.21, 0.9, 0.6),           # Lying Victim Male #2
            (-7.77, 6.58, 0.9, 0.6),            # Lying Victim Female #2
        ]
        return boxes, cyls

    def clear_scanned_map(self):
        with self.scan_lock:
            self.occ_grid.fill(0)
            self.height_grid.fill(0.0)
            self.voxels_3d.clear()
            for k in self.drone_scan_endpoints:
                self.drone_scan_endpoints[k].clear()
        self.reset_rescue(clear_map=True)
        self.get_logger().info('Cleared collaborative 4-drone OctoMap & occupancy grid.')

    def send_user_command(self, text: str):
        cmd_text = text.strip()
        if not cmd_text:
            return
        msg = String()
        msg.data = cmd_text
        self.cmd_pub.publish(msg)
        self.last_cmd = cmd_text
        self.get_logger().info(f'Sent user command from GUI: "{cmd_text}"')

    def _swarm_state_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            drones = data.get('drones', [])
            if not drones:
                return
            with self.swarm_lock:
                self.swarm_drones = drones
                self.comm_adj = data.get('comm_adj', self.comm_adj)
                self.allocator_status = data.get('allocator_status', self.allocator_status)
                self.swarm_mode = data.get('swarm_mode', self.swarm_mode)
                for idx, d in enumerate(drones[:4]):
                    dx, dy = float(d['x']), float(d['y'])
                    tr = self.swarm_trails[idx]
                    if not tr or math.hypot(dx - tr[-1][0], dy - tr[-1][1]) > 0.30:
                        tr.append((dx, dy))
                # Sync primary drone fields with UAV-1
                d0 = drones[0]
                self.x = float(d0['x'])
                self.y = float(d0['y'])
                self.z = float(d0['z'])
                self.vx = float(d0['vx'])
                self.vy = float(d0['vy'])
                self.vz = float(d0['vz'])
                self.yaw = float(d0['yaw'])
                self.armed = bool(d0['armed'])
                self.target_x = float(d0['target_x'])
                self.target_y = float(d0['target_y'])
                self.target_z = float(d0['target_z'])
        except Exception:
            pass

    def _odom_cb(self, msg: VehicleOdometry):
        self.odom_received = True
        q = msg.q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        self.pitch = math.asin(sinp)

    def _status_cb(self, msg: VehicleStatus):
        self.nav_state = int(msg.nav_state)

    def _sp_cb(self, msg: TrajectorySetpoint):
        pass

    def _cmd_cb(self, msg: String):
        self.last_cmd = msg.data.strip()

    def _det_cb(self, msg: String):
        try:
            self.detections = json.loads(msg.data)
        except Exception:
            pass

    def _img_yolo_cb(self, drone_idx: int, msg: Image):
        try:
            frame = imgmsg_to_cv2_safe(self.bridge, msg)
            if frame is None:
                return
            self.yolo_annotated_frames[drone_idx] = frame
            self.drone_cameras[drone_idx] = frame
            self.last_yolo_img_times[drone_idx] = time.time()
            if drone_idx == 0:
                self.yolo_annotated_frame = frame
                self.last_yolo_img_time = time.time()
        except Exception:
            pass

    def _raw_cam_cb(self, drone_idx: int, msg: Image):
        try:
            frame = imgmsg_to_cv2_safe(self.bridge, msg)
            if frame is None:
                return
            # Only use raw frame if no YOLO annotated frame arrived recently
            last_yolo = self.last_yolo_img_times.get(drone_idx, 0.0)
            if (time.time() - last_yolo) > 2.0 or self.yolo_annotated_frames.get(drone_idx) is None:
                self.drone_cameras[drone_idx] = frame
        except Exception:
            pass

    def _ned_to_grid(self, north: float, east: float) -> tuple[int, int]:
        gx = int((east + self.grid_half_m) / self.grid_res)
        gy = int((self.grid_half_m - north) / self.grid_res)
        return gx, gy

    def _integrate_drone_scan(
        self,
        drone_id: int,
        origin_n: float,
        origin_e: float,
        hit_points_ned: list[tuple[float, float, float]],
        free_endpoints_ned: list[tuple[float, float]],
        feed_voxel_map: bool = True,
    ):
        if feed_voxel_map:
            # real sensor data (PointCloud2 / LaserScan) goes into the probabilistic voxel map as well
            alt = max(0.5, -self.z) if drone_id == 0 else 1.0
            self.vmap.integrate_points((origin_n, origin_e, alt), hit_points_ned, free_endpoints_ned,
                                       agent_bit=1 << drone_id)
        with self.scan_lock:
            d_gx, d_gy = self._ned_to_grid(origin_n, origin_e)
            for fn, fe in free_endpoints_ned:
                e_gx, e_gy = self._ned_to_grid(fn, fe)
                cv2.line(self.occ_grid, (d_gx, d_gy), (e_gx, e_gy), 1, 1)

            new_endpoints = []
            for hn, he, hz_agl in hit_points_ned:
                if hz_agl < 0.35:
                    g_gx, g_gy = self._ned_to_grid(hn, he)
                    if 0 <= g_gx < self.grid_size and 0 <= g_gy < self.grid_size:
                        if self.occ_grid[g_gy, g_gx] == 0:
                            self.occ_grid[g_gy, g_gx] = 1
                    continue

                h_gx, h_gy = self._ned_to_grid(hn, he)
                cv2.line(self.occ_grid, (d_gx, d_gy), (h_gx, h_gy), 1, 1)
                if 0 <= h_gx < self.grid_size and 0 <= h_gy < self.grid_size:
                    self.occ_grid[h_gy, h_gx] = 2
                    if hz_agl > self.height_grid[h_gy, h_gx]:
                        self.height_grid[h_gy, h_gx] = hz_agl

                vx = int(round(hn / 1.0))
                vy = int(round(he / 1.0))
                vz = int(round(hz_agl / 1.0))
                if len(self.voxels_3d) < 9000:
                    self.voxels_3d[(vx, vy, vz)] = hz_agl
                new_endpoints.append((hn, he, hz_agl))

            self.drone_scan_endpoints[drone_id] = new_endpoints[:60]

    def _laserscan_cb(self, msg: LaserScan):
        self.lidar_msg_count += 1
        alt_agl = max(0.0, -self.z)
        angle = msg.angle_min
        hits, frees = [], []
        step = max(1, len(msg.ranges) // 60)
        for i in range(0, len(msg.ranges), step):
            r = msg.ranges[i]
            a_world = self.yaw - (angle + i * msg.angle_increment)
            if math.isnan(r) or math.isinf(r) or r >= msg.range_max * 0.97:
                rmax = min(30.0, float(msg.range_max))
                frees.append((self.x + rmax * math.cos(a_world), self.y + rmax * math.sin(a_world)))
            elif r > msg.range_min:
                hn = self.x + r * math.cos(a_world)
                he = self.y + r * math.sin(a_world)
                hits.append((hn, he, max(1.0, alt_agl)))
        self._integrate_drone_scan(0, self.x, self.y, hits, frees)

    def _pc2_cb(self, msg: PointCloud2):
        self.lidar_msg_count += 1
        try:
            pts = np.frombuffer(msg.data, dtype=np.float32)
            point_step_floats = msg.point_step // 4
            if point_step_floats < 3 or len(pts) < point_step_floats:
                return
            pts = pts.reshape(-1, point_step_floats)
            pts = pts[::max(1, len(pts) // 200), :3]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) == 0:
                return

            alt_agl = max(0.0, -self.z)
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            hits, frees = [], []
            for lx, ly, lz in pts:
                dist_xy = math.hypot(float(lx), float(ly))
                if dist_xy < 0.7 or dist_xy > 55.0:
                    continue
                wn = self.x + float(lx) * cy - float(ly) * sy
                we = self.y + float(lx) * sy + float(ly) * cy
                wz_agl = alt_agl + float(lz)
                if wz_agl > 0.4:
                    hits.append((wn, we, wz_agl))
                else:
                    frees.append((wn, we))
            self._integrate_drone_scan(0, self.x, self.y, hits, frees)
        except Exception:
            pass

    def _collaborative_4drone_lidar_tick(self):
        with self.swarm_lock:
            drones_snapshot = [dict(d) for d in self.swarm_drones]

        max_range = 28.0
        for d in drones_snapshot:
            did = int(d['id'])
            dx, dy, dz = float(d['x']), float(d['y']), float(d['z'])
            dyaw = float(d['yaw'])
            alt_agl = max(0.5, -dz)
            hits, frees = [], []

            for i in range(48):
                ang = dyaw + (i * 2.0 * math.pi / 48.0)
                ca, sa = math.cos(ang), math.sin(ang)
                ray_hit = False

                for r in np.arange(1.5, max_range, 1.2):
                    rn = dx + r * ca
                    re = dy + r * sa

                    for bn, be, sn, se, bh in self.world_boxes:
                        if abs(rn - bn) <= sn * 0.5 and abs(re - be) <= se * 0.5:
                            max_vis_h = min(bh, alt_agl + r * 0.45)
                            if max_vis_h >= 0.5:
                                for hz in np.arange(1.0, max_vis_h + 0.5, 2.0):
                                    hits.append((rn, re, float(hz)))
                                ray_hit = True
                            break
                    if ray_hit:
                        break

                    for cn, ce, crad, ch in self.world_cyls:
                        if math.hypot(rn - cn, re - ce) <= crad:
                            max_vis_h = min(ch, alt_agl + r * 0.45)
                            if max_vis_h >= 0.5:
                                for hz in np.arange(1.0, max_vis_h + 0.5, 2.0):
                                    hits.append((rn, re, float(hz)))
                                ray_hit = True
                            break
                    if ray_hit:
                        break

                if not ray_hit:
                    frees.append((dx + max_range * ca, dy + max_range * sa))

            self._integrate_drone_scan(did, dx, dy, hits, frees, feed_voxel_map=False)

            # dense scan of the same world for the voxel map (analytic ray casting, 192 rays, same visibility rule)
            rays = self.sim_world.scan(dx, dy, alt_agl, dyaw, n_rays=192, max_range=max_range)
            self.vmap.integrate_rays((dx, dy, alt_agl), rays, agent_bit=1 << did)
            if self._lidar_tick_n % 5 == 0:           # roofs are static: refresh them once a second
                self.vmap.integrate_surface(self.sim_world.roof_points(dx, dy, alt_agl, max_range))
        self._lidar_tick_n += 1

    # ══════════════════════════════════════════════════════════════════════════
    # RESCUE ROVERS: detections -> coordinator -> OctoMap-based navigation
    # ══════════════════════════════════════════════════════════════════════════
    def reset_rescue(self, clear_map: bool = False):
        with self.rescue_lock:
            if clear_map:
                self.vmap.clear()
            self.sim_world.clear_dynamic()
            starts = [(-2.0, 5.0, math.pi / 2), (2.0, -5.0, -math.pi / 2)]
            for r, (n, e, yaw) in zip(self.rovers, starts):
                r.n, r.e, r.yaw, r.v, r.w = n, e, yaw, 0.0, 0.0
                r.state, r.mission_id, r.target, r.route, r.route_idx = 'idle', None, None, [], 0
                r.trail.clear()
                r.replans = r.bumps = 0
            self.coord.missions.clear()
            self.coord.queue.clear()
            self.coord._by_survivor.clear()
            self.coord.transmissions.clear()
            self.sim_detector.seen.clear()

    def set_det_source(self, mode: str):
        self.det_source = mode

    def set_allow_unknown(self, allow: bool):
        self.nav_params.unknown_policy = 'penalize' if allow else 'forbid'

    def drop_obstacle_on_route(self) -> bool:
        """Demo/test: drop rubble on the route of the first navigating rover so it must replan."""
        with self.rescue_lock:
            for r in self.rovers:
                if r.state != 'navigating' or len(r.route) < 2:
                    continue
                pts = [r.pos()] + r.route[r.route_idx:]
                target_d, acc = 9.0, 0.0
                for a, b in zip(pts[:-1], pts[1:]):
                    seg = math.hypot(b[0] - a[0], b[1] - a[1])
                    if acc + seg >= target_d and seg > 1e-6:
                        f = (target_d - acc) / seg
                        self.sim_world.add_dynamic_box(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 3.0, 4.0, 1.4)
                        self.coord.log('obstacle_dropped', rover=r.id)
                        return True
                    acc += seg
        return False

    def _swarm_victims_cb(self, msg: String):
        try:
            rep = json.loads(msg.data)
        except Exception:
            return
        self.swarm_victims = [v for v in rep.get('victims', []) if v.get('confirmed')]

    def _real_detections(self, now: float) -> list:
        import zlib
        out = []
        for v in self.swarm_victims:
            vid = v.get('id')
            key = zlib.crc32('-'.join(vid).encode()) if isinstance(vid, list) else zlib.crc32(str(vid).encode())
            owner = int(vid[0]) if isinstance(vid, list) and vid and str(vid[0]).isdigit() else 0
            n, e = v['n_ned']
            out.append(Detection(1000 + key % 100000, float(n), float(e), min(0.99, 0.5 + float(v.get('obs', 0)) / 60.0),
                                 owner, now, 'swarm', 'YOLO victim'))
        return out

    def _drone_pos3(self, did: int):
        with self.swarm_lock:
            d = self.swarm_drones[did]
            return (float(d['x']), float(d['y']), max(0.0, -float(d['z'])))

    def _rescue_tick(self):
        now = time.monotonic()
        dt = min(0.2, now - self._last_rescue_t)
        self._last_rescue_t = now
        with self.swarm_lock:
            drones = [dict(d) for d in self.swarm_drones]
        if now - self._last_trail_t > 0.4:
            self._last_trail_t = now
            for i, d in enumerate(drones[:4]):
                self.drone_trails3d[i].append((float(d['x']), float(d['y']), max(0.0, -float(d['z']))))
        with self.rescue_lock:
            if now - self._last_det_t > 0.5 and self.det_source != 'off':
                self._last_det_t = now
                dets = []
                if self.det_source in ('sim', 'both'):
                    dets += self.sim_detector.step(drones, self.coord.t)
                if self.det_source in ('swarm', 'both'):
                    dets += self._real_detections(self.coord.t)
                for det in dets:
                    self.coord.on_detection(det, drone_pos=self._drone_pos3(det.drone_id) if det.drone_id < 4 else None)
            self.coord.tick(dt)

    def _build_scene(self) -> Scene:
        with self.swarm_lock:
            drones = [dict(d) for d in self.swarm_drones]
            comm_adj = [list(row) for row in self.comm_adj]
        sc = Scene(now=self.coord.t, hub=(0.0, 0.0))
        for i, d in enumerate(drones[:4]):
            sc.drones.append(dict(id=i, name=d['name'], n=float(d['x']), e=float(d['y']), alt=max(0.0, -float(d['z'])),
                                  yaw=float(d['yaw']), trail=list(self.drone_trails3d[i])))
        for i in range(len(sc.drones)):
            for j in range(i + 1, len(sc.drones)):
                if i < len(comm_adj) and j < len(comm_adj[i]) and comm_adj[i][j] > 0.5:
                    sc.links.append((i, j))
        with self.rescue_lock:
            routes = {r.id: [r.pos()] + r.route[r.route_idx:] for r in self.rovers if r.route and r.state in ('navigating', 'blocked')}
            for r in self.rovers:
                sc.rovers.append(dict(id=r.id, name=r.name, n=r.n, e=r.e, yaw=r.yaw, color=r.color, state=r.state,
                                      wheel=r.wheel_phase, trail=list(r.trail)[-300:], route=routes.get(r.id, [])))
            for m in self.coord.missions.values():
                sc.survivors.append(dict(id=m.survivor_id, name=m.name, n=m.n, e=m.e, conf=m.conf, drone_id=m.drone_id,
                                         state='rescued' if m.state == 'rescued' else 'detected', radius=3.0))
            for tr in self.coord.transmissions:
                item = dict(tr)
                if tr['kind'] == 'to_coordinator' and tr.get('drone') is not None and tr['drone'] < 4:
                    item['from'] = self._drone_pos3(tr['drone'])
                if tr['kind'] == 'to_rover':
                    rv = self.rovers[tr['rover']]
                    item['rover_n'], item['rover_e'] = rv.n, rv.e
                sc.transmissions.append(item)
            lines = []
            for m in list(self.coord.missions.values())[-5:]:
                rv = f'Rover {m.rover_id + 1}' if m.rover_id is not None else 'unassigned'
                lines.append(f'M{m.id} {m.name} conf {m.conf * 100:.0f}% UAV-{m.drone_id + 1} -> {rv}  {m.state}'
                             f'  {m.route_len:.0f} m' + (f'  ({m.note})' if m.note else ''))
            sc.stats = dict(self.coord.summary(), lines=lines)
        src = {'sim': 'SIMULATED survivor detector', 'swarm': 'real YOLO registry (/swarm/victims)',
               'both': 'simulated + real YOLO', 'off': 'detections OFF'}[self.det_source]
        policy = 'unknown ground allowed' if self.nav_params.unknown_policy == 'penalize' else 'known-free ground only'
        sc.source_label = f'DETECTIONS: {src} | MAP: simulated LiDAR on the world obstacle list | ROVER NAV: {policy}'
        return sc

    def focus_entity(self, kind: str, idx: int):
        sc = self._build_scene()
        try:
            if kind == 'drone':
                d = sc.drones[idx]
                self.viewer.cam.focus((d['n'], d['e'], d['alt']), dist=45.0)
            elif kind == 'rover':
                r = sc.rovers[idx]
                self.viewer.cam.focus((r['n'], r['e'], 0.5), dist=18.0)
            elif kind == 'survivor':
                s_ = sc.survivors[idx]
                self.viewer.cam.focus((s_['n'], s_['e'], 1.0), dist=20.0)
        except IndexError:
            pass

    def _rover_cam_loop(self):
        while not self._rover_cam_stop:
            time.sleep(0.4)
            try:
                sc = self._build_scene()
                for i, rv in enumerate(sc.rovers):
                    self.rover_viewers[i].geo = self.viewer.geo
                    self.rover_cam_frames[i] = self.rover_viewers[i].render_pov(self.viewer, 384, 240, sc, rv)
            except Exception:
                pass

    def _render_rover_cam(self, idx: int, w: int, h: int) -> np.ndarray:
        """Rover camera feed with badge and mission banner (first-person view rendered from the voxel map)."""
        w, h = max(140, int(w)), max(100, int(h))
        col = tuple(int(c) for c in ROVER_COLORS[idx % 2])
        frame = self.rover_cam_frames[idx] if idx < len(self.rover_cam_frames) else None
        if frame is None:
            img = np.full((h, w, 3), (26, 30, 28), np.uint8)
            cv2.putText(img, 'rover camera starting...', (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 180), 1, cv2.LINE_AA)
        else:
            img = cv2.resize(frame, (w, h))
        with self.rescue_lock:
            r = self.rovers[idx]
            state, speed, yaw = r.state, r.v, r.yaw
            dist = r.dist_to_end() if r.route else 0.0
            m = self.coord.missions.get(r.mission_id) if r.mission_id is not None else None
            mtxt = f'M{m.id} {m.name}' if m else 'no mission (idle)'
        cv2.rectangle(img, (0, 0), (w, 24), (16, 22, 20), -1)
        cv2.putText(img, f'ROVER {idx + 1} CAM [{state.upper()}]  SPD:{speed:3.1f}m/s  HDG:{(math.degrees(yaw) + 360) % 360:03.0f}  GOAL:{dist:4.1f}m',
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 22), (w, h), (16, 22, 20), -1)
        cv2.putText(img, f'{mtxt[:34]}  | SIMULATED view from the voxel map', (8, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 240, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 2)
        return img

    def survivor_labels(self) -> list[str]:
        return [f'{m.name} (M{m.id})' for m in self.coord.missions.values()]

    @staticmethod
    def _ned_to_px(north: float, east: float, cx: int, cy: int, scale: float) -> tuple[int, int]:
        px = int(cx + east * scale)
        py = int(cy - north * scale)
        return px, py

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER INDIVIDUAL DRONE CAMERA VIEW (REAL STREAM OR SYNTHETIC HUD)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_drone_cam(self, idx: int, w: int, h: int) -> np.ndarray:
        """Render camera feed for drone `idx` with top badge and bottom task banner."""
        w = max(140, int(w))
        h = max(100, int(h))
        col = DRONE_COLORS_BGR[idx % 4]

        with self.swarm_lock:
            d = self.swarm_drones[idx] if idx < len(self.swarm_drones) else None
            d = dict(d) if d else {
                'name': f'UAV-{idx+1}', 'x': 0.0, 'y': 0.0, 'z': -10.0,
                'yaw': 0.0, 'vx': 0.0, 'vy': 0.0, 'pitch': 0.0,
                'task_label': 'Patrol', 'alloc_mode': 'GNN'
            }

        # Check for live camera feed (prioritize YOLO annotated feed for every drone)
        raw_frame = None
        if self.yolo_annotated_frames.get(idx) is not None and (time.time() - self.last_yolo_img_times.get(idx, 0.0)) < 3.0:
            raw_frame = self.yolo_annotated_frames[idx]
        elif self.drone_cameras[idx] is not None:
            raw_frame = self.drone_cameras[idx]

        if raw_frame is not None:
            cam_img = cv2.resize(raw_frame, (w, h))
        else:
            # High-fidelity synthetic camera view matching this drone's pose and heading
            cam_img = np.zeros((h, w, 3), dtype=np.uint8)
            pitch = float(d.get('pitch', 0.0))
            horizon_y = int(h * 0.48 + math.degrees(pitch) * 2.2)
            horizon_y = max(15, min(h - 15, horizon_y))
            # Sky & Terrain
            cam_img[:horizon_y, :] = (175, 125, 60)
            cam_img[horizon_y:, :] = (50, 78, 52)
            cv2.line(cam_img, (0, horizon_y), (w, horizon_y), (200, 240, 200), 1)

            # Perspective ground grid
            cx = w // 2
            for i in range(-4, 5):
                x_bot = cx + i * int(w * 0.28)
                x_top = cx + i * int(w * 0.06)
                cv2.line(cam_img, (x_top, horizon_y), (x_bot, h), (70, 105, 75), 1, cv2.LINE_AA)

            # Center target reticle
            dyaw = float(d.get('yaw', 0.0))
            dn = float(d.get('target_x', 0.0)) - float(d['x'])
            de = float(d.get('target_y', 0.0)) - float(d['y'])
            bearing = math.atan2(de, dn)
            rel_ang = (bearing - dyaw + math.pi) % (2.0 * math.pi) - math.pi
            tx_px = max(20, min(w - 20, int(cx + (rel_ang / (math.pi * 0.5)) * (w * 0.4))))
            cv2.drawMarker(cam_img, (tx_px, horizon_y), col, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
            cv2.circle(cam_img, (cx, h // 2), 16, (0, 255, 200), 1, cv2.LINE_AA)

        # ── Sleek Top Badge Overlay ───────────────────────────────────────
        alt_agl = max(0.0, -float(d['z']))
        yaw_deg = (math.degrees(float(d['yaw'])) + 360.0) % 360.0
        spd = math.hypot(float(d.get('vx', 0.0)), float(d.get('vy', 0.0)))
        role_tag = 'LEAD YOLO' if idx == 0 else 'SCOUT'

        cv2.rectangle(cam_img, (0, 0), (w, 24), (16, 22, 20), -1)
        badge_txt = f"{d['name']} [{role_tag}]  ALT:{alt_agl:4.1f}m  HDG:{yaw_deg:03.0f}°  SPD:{spd:3.1f}m/s"
        cv2.putText(cam_img, badge_txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # ── Bottom Task Banner ────────────────────────────────────────────
        cv2.rectangle(cam_img, (0, h - 22), (w, h), (16, 22, 20), -1)
        mode_sh = 'GNN' if 'GNN' in d.get('alloc_mode', '') else 'HEU'
        task_txt = f"TASK: {d.get('task_label', '')[:20]} [{mode_sh}]"
        cv2.putText(cam_img, task_txt, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 240, 255), 1, cv2.LINE_AA)

        # Outer drone colored border
        cv2.rectangle(cam_img, (0, 0), (w - 1, h - 1), col, 2)
        return cam_img

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: 4-DRONE SWARM 2D TACTICAL MAP + 2x2 QUAD-CAMERA MULTI-VIEW
    # ══════════════════════════════════════════════════════════════════════════
    def render_tactical_page(self, pane_w: int = 760, pane_h: int = 720) -> np.ndarray:
        W, H = max(480, pane_w), max(480, pane_h)
        cx, cy = W // 2, H // 2

        with self.swarm_lock:
            drones = [dict(d) for d in self.swarm_drones]
            trails = [list(tr) for tr in self.swarm_trails]
            comm_adj = [list(row) for row in self.comm_adj]
            alloc_status = self.allocator_status

        max_dist = 52.0
        for d in drones:
            max_dist = max(
                max_dist,
                abs(float(d['x'])) + 14.0,
                abs(float(d['y'])) + 14.0,
                abs(float(d['target_x'])) + 14.0,
                abs(float(d['target_y'])) + 14.0,
            )
        scale = (min(W, H) * 0.44) / max_dist

        if self.sat_map is not None:
            sh, sw = self.sat_map.shape[:2]
            scx, scy = sw // 2, sh // 2
            half_sat = int(min(sw // 2 - 2, max(140, max_dist * 2.5)))
            crop = self.sat_map[scy - half_sat:scy + half_sat, scx - half_sat:scx + half_sat]
            map_img = cv2.resize(crop, (W, H))
        else:
            map_img = np.full((H, W, 3), (48, 68, 52), dtype=np.uint8)

        grid_step_m = 10 if max_dist <= 75 else 25
        for gm in range(-300, 301, grid_step_m):
            gx, gy = self._ned_to_px(gm, gm, cx, cy, scale)
            if 0 <= gx < W:
                cv2.line(map_img, (gx, 0), (gx, H), (65, 85, 65), 1)
            if 0 <= gy < H:
                cv2.line(map_img, (0, gy), (W, gy), (65, 85, 65), 1)

        for r_m in (20, 40, 60, 100):
            r_px = int(r_m * scale)
            if 15 < r_px < int(min(W, H) * 0.49):
                cv2.circle(map_img, (cx, cy), r_px, (90, 160, 90), 1, cv2.LINE_AA)
                cv2.putText(
                    map_img, f'{r_m}m', (cx + 6, cy - r_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 245, 180), 1, cv2.LINE_AA
                )

        cv2.line(map_img, (cx, 0), (cx, H), (100, 180, 100), 1)
        cv2.line(map_img, (0, cy), (W, cy), (100, 180, 100), 1)
        cv2.putText(map_img, 'NORTH (+X)', (cx + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(map_img, 'SOUTH (-X)', (cx + 10, H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 230, 180), 1, cv2.LINE_AA)
        cv2.putText(map_img, 'EAST (+Y)', (W - 115, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(map_img, 'WEST (-Y)', (12, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 230, 180), 1, cv2.LINE_AA)

        # Helipad at (0,0)
        cv2.circle(map_img, (cx, cy), 6, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(map_img, (cx, cy), 8, (40, 40, 40), 2, cv2.LINE_AA)

        # Outer Ring Collapsed Buildings
        for bn, be, blbl in [
            (26.0, 32.0, 'Collapsed Fire Stn NE'),
            (26.0, -32.0, 'Collapsed Police Stn NW'),
            (-24.0, 30.0, 'Collapsed House SE'),
            (-24.0, -30.0, 'Collapsed House SW'),
            (36.0, 0.0, 'Collapsed House N'),
            (-36.0, 0.0, 'Collapsed Police Stn S'),
        ]:
            px_b, py_b = self._ned_to_px(bn, be, cx, cy, scale)
            cv2.rectangle(map_img, (px_b - 13, py_b - 9), (px_b + 13, py_b + 9), (70, 80, 120), -1)
            cv2.rectangle(map_img, (px_b - 13, py_b - 9), (px_b + 13, py_b + 9), (160, 180, 230), 2)
            cv2.putText(map_img, blbl, (px_b - 42, py_b - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 215, 250), 1, cv2.LINE_AA)

        # Walking & Standing Survivors
        for sn, se, slbl in [
            (8.0, 8.0, 'Walk Survivor #1'),
            (-8.0, -8.0, 'Walk Survivor #2'),
            (10.0, 4.0, 'Standing Male #1'),
            (6.0, -8.0, 'Standing Female #1'),
            (-5.0, 9.0, 'Standing Male #2'),
        ]:
            px_p, py_p = self._ned_to_px(sn, se, cx, cy, scale)
            cv2.circle(map_img, (px_p, py_p), 6, (0, 255, 110), -1, cv2.LINE_AA)
            cv2.circle(map_img, (px_p, py_p), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(map_img, slbl, (px_p + 9, py_p + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 255, 180), 1, cv2.LINE_AA)

        # Lying-Down Injured Victims
        for ln, le, llbl in [
            (8.15, 2.36, 'Lying Victim #1'),
            (4.60, -6.78, 'Lying Victim #2'),
            (-8.54, -3.21, 'Lying Victim #3'),
            (-7.77, 6.58, 'Lying Victim #4'),
        ]:
            px_l, py_l = self._ned_to_px(ln, le, cx, cy, scale)
            cv2.circle(map_img, (px_l, py_l), 7, (0, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(map_img, (px_l, py_l), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(map_img, llbl, (px_l + 9, py_l + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 190, 255), 1, cv2.LINE_AA)

        # Vehicles
        for cn, ce, clbl in [
            (-8.0, 11.0, 'Ambulance'),
            (3.0, 12.0, 'Stranded Prius'),
            (-3.0, -11.0, 'Damaged SUV'),
        ]:
            px_c, py_c = self._ned_to_px(cn, ce, cx, cy, scale)
            cv2.rectangle(map_img, (px_c - 7, py_c - 4), (px_c + 7, py_c + 4), (255, 175, 40), -1)
            cv2.rectangle(map_img, (px_c - 7, py_c - 4), (px_c + 7, py_c + 4), (0, 0, 0), 1)
            cv2.putText(map_img, clbl, (px_c + 9, py_c + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 205, 100), 1, cv2.LINE_AA)

        # Peer-to-Peer GNN Communication Links
        n_d = len(drones)
        for i in range(n_d):
            for j in range(i + 1, n_d):
                if i < len(comm_adj) and j < len(comm_adj[i]) and comm_adj[i][j] > 0.5:
                    p1 = self._ned_to_px(float(drones[i]['x']), float(drones[i]['y']), cx, cy, scale)
                    p2 = self._ned_to_px(float(drones[j]['x']), float(drones[j]['y']), cx, cy, scale)
                    cv2.line(map_img, p1, p2, (210, 240, 110), 1, cv2.LINE_AA)

        # Drone Trails, Targets & Icons
        for idx, d in enumerate(drones[:4]):
            col = DRONE_COLORS_BGR[idx % 4]
            tr = trails[idx]
            if len(tr) > 1:
                pts = np.array([self._ned_to_px(nx, ny, cx, cy, scale) for nx, ny in tr], dtype=np.int32)
                cv2.polylines(map_img, [pts], False, col, 2, cv2.LINE_AA)

            dx_px, dy_px = self._ned_to_px(float(d['x']), float(d['y']), cx, cy, scale)
            tx_px, ty_px = self._ned_to_px(float(d['target_x']), float(d['target_y']), cx, cy, scale)

            cv2.line(map_img, (dx_px, dy_px), (tx_px, ty_px), col, 1, cv2.LINE_AA)
            cv2.drawMarker(map_img, (tx_px, ty_px), col, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

            mode_short = 'GNN' if 'GNN' in d.get('alloc_mode', '') else 'HEU'
            cv2.putText(
                map_img, f"T{idx + 1}[{mode_short}]",
                (tx_px + 8, ty_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA
            )

            dyaw = float(d['yaw'])
            hx = int(dx_px + 22 * math.sin(dyaw))
            hy = int(dy_px - 22 * math.cos(dyaw))
            cv2.arrowedLine(map_img, (dx_px, dy_px), (hx, hy), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(map_img, (dx_px, dy_px), 9, col, -1, cv2.LINE_AA)
            cv2.circle(map_img, (dx_px, dy_px), 11, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.putText(
                map_img, f"{d['name']} ({-float(d['z']):.0f}m)",
                (dx_px + 11, dy_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 2, cv2.LINE_AA
            )

        # Telemetry Card Overlay
        cv2.rectangle(map_img, (10, 10), (495, 192), (16, 24, 20), -1)
        cv2.rectangle(map_img, (10, 10), (495, 192), (0, 220, 160), 1)
        cv2.putText(
            map_img, 'DECENTRALIZED 4-DRONE SWARM (P2P GNN + HEURISTIC)',
            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 2, cv2.LINE_AA
        )
        cv2.putText(
            map_img, f'ALLOCATOR : {alloc_status[:46]}',
            (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 240, 255), 1, cv2.LINE_AA
        )

        for idx, d in enumerate(drones[:4]):
            col = DRONE_COLORS_BGR[idx % 4]
            y_row = 78 + idx * 24
            m_tag = 'GNN' if d.get('alloc_mode') == 'GNN (GraphSAGE)' else (
                'GNN+H' if 'GNN' in d.get('alloc_mode', '') else 'HEUR'
            )
            line_txt = (
                f"{d['name']} ({float(d['x']):+4.0f}N,{float(d['y']):+4.0f}E,{-float(d['z']):.0f}m) "
                f"[{m_tag}] -> {d.get('task_label', '')[:24]}"
            )
            cv2.putText(map_img, line_txt, (18, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1, cv2.LINE_AA)

        det_names = [d.get('class', '?') for d in self.detections]
        cv2.putText(
            map_img, f'YOLO SAR  : {det_names[:4] if det_names else "4 drones scanning disaster zone..."}',
            (18, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (120, 255, 190), 1, cv2.LINE_AA
        )

        cv2.rectangle(map_img, (0, H - 38), (W, H), (22, 30, 24), -1)
        cv2.putText(
            map_img, f'SWARM MISSION: "{self.last_cmd[:64]}"', (14, H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 235, 255), 1, cv2.LINE_AA
        )

        # ── Right Pane: 2x2 QUAD-CAMERA SWARM GRID (or Single Drone Focus) ──
        if self.cam_layout == 'quad':
            hw = W // 2
            hh = H // 2
            c0 = self._render_drone_cam(0, hw, hh)
            c1 = self._render_drone_cam(1, hw, hh)
            c2 = self._render_drone_cam(2, hw, hh)
            c3 = self._render_drone_cam(3, hw, hh)
            top_half = np.hstack([c0, c1])
            bot_half = np.hstack([c2, c3])
            cam_pane = np.vstack([top_half, bot_half])
            cv2.rectangle(cam_pane, (0, 0), (W, 26), (18, 24, 28), -1)
            cv2.putText(
                cam_pane,
                '4-DRONE LIVE CAMERA MULTI-VIEW (Click any drone cam to enlarge/focus)',
                (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 220), 2, cv2.LINE_AA
            )
        elif self.cam_layout == 'rovers':
            cam_pane = np.vstack([self._render_rover_cam(0, W, H // 2), self._render_rover_cam(1, W, H - H // 2)])
            cv2.putText(cam_pane, 'ROVER CAMERAS (simulated first-person view from the voxel map)', (12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 220), 1, cv2.LINE_AA)
        elif isinstance(self.cam_layout, str) and self.cam_layout.startswith('r'):
            ridx = int(self.cam_layout[1:])
            cam_pane = self._render_rover_cam(ridx, W, H)
            cv2.putText(cam_pane, f'FOCUSED CAMERA: ROVER {ridx + 1} (Click a camera button to switch)', (12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            focus_idx = int(self.cam_layout)
            cam_pane = self._render_drone_cam(focus_idx, W, H)
            cv2.rectangle(cam_pane, (0, 0), (W, 28), (18, 24, 28), -1)
            cv2.putText(
                cam_pane,
                f'FOCUSED CAMERA: UAV-{focus_idx + 1} (Click to return to 4-Cam Quad Grid)',
                (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2, cv2.LINE_AA
            )

        map_img = cv2.resize(map_img, (W, H))
        cam_pane = cv2.resize(cam_pane, (W, H))
        return np.hstack([map_img, cam_pane])

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: 4-DRONE COLLABORATIVE WHOLE SCANNED 3D OCTOMAP + 4-CAM PIP
    # ══════════════════════════════════════════════════════════════════════════
    def render_scanned_map_page(self, pane_w: int = 760, pane_h: int = 720) -> np.ndarray:
        W, H = max(480, pane_w), max(480, pane_h)
        left_2d = cv2.resize(self._render_2d_occupancy_scan(W, H), (W, H))
        right_3d = cv2.resize(self._render_3d_octomap_viewer(W, H), (W, H))
        return np.hstack([left_2d, right_3d])

    def _render_2d_occupancy_scan(self, W: int, H: int) -> np.ndarray:
        with self.scan_lock:
            occ = self.occ_grid.copy()
            hgrid = self.height_grid.copy()
            drone_beams = {k: list(v) for k, v in self.drone_scan_endpoints.items()}
            num_voxels = len(self.voxels_3d)

        with self.swarm_lock:
            drones = [dict(d) for d in self.swarm_drones]
            trails = [list(tr) for tr in self.swarm_trails]
            comm_adj = [list(row) for row in self.comm_adj]

        grid_rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        grid_rgb[occ == 0] = (42, 36, 32)
        grid_rgb[occ == 1] = (75, 118, 68)

        occ_mask = (occ == 2)
        if np.any(occ_mask):
            norm_h = np.clip(hgrid / 32.0, 0.0, 1.0)
            b = np.clip((1.0 - norm_h) * 220, 30, 255).astype(np.uint8)
            g = np.clip(np.sin(norm_h * math.pi) * 255, 40, 255).astype(np.uint8)
            r = np.clip(norm_h * 255 + 60, 80, 255).astype(np.uint8)
            grid_rgb[occ_mask, 0] = b[occ_mask]
            grid_rgb[occ_mask, 1] = g[occ_mask]
            grid_rgb[occ_mask, 2] = r[occ_mask]

        scan_img = cv2.resize(grid_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        cx, cy = W // 2, H // 2
        scale = W / (self.grid_half_m * 2.0)

        for gm in range(-45, 46, 15):
            px, py = self._ned_to_px(gm, gm, cx, cy, scale)
            cv2.line(scan_img, (px, 0), (px, H), (65, 85, 70), 1)
            cv2.line(scan_img, (0, py), (W, py), (65, 85, 70), 1)
            if gm != 0:
                cv2.putText(scan_img, f'{gm:+d}m', (px + 3, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 190, 160), 1, cv2.LINE_AA)

        n_d = len(drones)
        for i in range(n_d):
            for j in range(i + 1, n_d):
                if i < len(comm_adj) and j < len(comm_adj[i]) and comm_adj[i][j] > 0.5:
                    p1 = self._ned_to_px(float(drones[i]['x']), float(drones[i]['y']), cx, cy, scale)
                    p2 = self._ned_to_px(float(drones[j]['x']), float(drones[j]['y']), cx, cy, scale)
                    cv2.line(scan_img, p1, p2, (200, 230, 120), 1, cv2.LINE_AA)

        for idx, d in enumerate(drones[:4]):
            col = DRONE_COLORS_BGR[idx % 4]
            dx_px, dy_px = self._ned_to_px(float(d['x']), float(d['y']), cx, cy, scale)

            for hn, he, _hz in drone_beams.get(idx, [])[::2]:
                hx_px, hy_px = self._ned_to_px(hn, he, cx, cy, scale)
                cv2.line(scan_img, (dx_px, dy_px), (hx_px, hy_px), col, 1, cv2.LINE_AA)
                cv2.circle(scan_img, (hx_px, hy_px), 2, (0, 255, 255), -1)

            tr = trails[idx]
            if len(tr) > 1:
                pts = np.array([self._ned_to_px(nx, ny, cx, cy, scale) for nx, ny in tr], dtype=np.int32)
                cv2.polylines(scan_img, [pts], False, col, 2, cv2.LINE_AA)

            tx_px, ty_px = self._ned_to_px(float(d['target_x']), float(d['target_y']), cx, cy, scale)
            cv2.drawMarker(scan_img, (tx_px, ty_px), col, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

            cv2.circle(scan_img, (dx_px, dy_px), 8, col, -1, cv2.LINE_AA)
            cv2.circle(scan_img, (dx_px, dy_px), 10, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(scan_img, d['name'], (dx_px + 10, dy_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 2, cv2.LINE_AA)

        landmarks = [
            (26, 32, 'Collapsed Fire Stn NE'),
            (26, -32, 'Collapsed Police Stn NW'),
            (-24, 30, 'Collapsed House SE'),
            (-24, -30, 'Collapsed House SW'),
            (36, 0, 'Collapsed House N'),
            (-36, 0, 'Collapsed Police Stn S'),
            (8, 8, 'Survivors NE'),
            (-8, -8, 'Survivors SW'),
        ]
        for ln, le, ltxt in landmarks:
            lx, ly = self._ned_to_px(ln, le, cx, cy, scale)
            cv2.putText(scan_img, ltxt, (lx - 42, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 245, 255), 1, cv2.LINE_AA)

        scanned_cells = int(np.count_nonzero(occ > 0))
        occ_cells = int(np.count_nonzero(occ == 2))
        cov_pct = 100.0 * scanned_cells / float(occ.size)

        cv2.rectangle(scan_img, (0, 0), (W, 42), (18, 24, 28), -1)
        cv2.putText(
            scan_img, '4-DRONE COLLABORATIVE WHOLE SCANNED 2D OCCUPANCY & HEIGHT MAP',
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 200), 2, cv2.LINE_AA
        )

        cv2.rectangle(scan_img, (12, 52), (435, 185), (20, 28, 32), -1)
        cv2.rectangle(scan_img, (12, 52), (435, 185), (0, 200, 160), 1)
        cv2.putText(scan_img, f'SWARM COVERAGE : {cov_pct:.1f}% of 120m x 120m (4 Drones Active)', (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 160), 1, cv2.LINE_AA)
        cv2.putText(scan_img, f'3D OCTOMAP     : {num_voxels} occupied 3D voxels ({occ_cells} cells)', (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 235, 255), 1, cv2.LINE_AA)
        for idx, d in enumerate(drones[:4]):
            col = DRONE_COLORS_BGR[idx % 4]
            m_tag = 'GNN' if 'GNN' in d.get('alloc_mode', '') else 'HEU'
            cv2.putText(
                scan_img,
                f"{d['name']} [{m_tag}]: {d.get('task_label', '')[:28]}",
                (20, 118 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1, cv2.LINE_AA
            )

        cv2.rectangle(scan_img, (0, H - 38), (W, H), (18, 24, 28), -1)
        cv2.putText(
            scan_img,
            'LEGEND: [Dark]=Unexplored  [Green]=Free Space  [Cyan->Red]=3D Obstacles  [Colored Cross]=GNN/Heuristic Drone Task',
            (12, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 240, 220), 1, cv2.LINE_AA
        )
        return scan_img

    def _render_3d_octomap_viewer(self, W: int, H: int) -> np.ndarray:
        if self.use_legacy_viewer:
            return self._render_3d_octomap_viewer_legacy(W, H)
        canvas = self.viewer.render(W, H, self._build_scene())
        if W > 380 and H > 360:
            pip_w, pip_h = 340, 240
            qw, qh = pip_w // 2, pip_h // 2
            cams = [self._render_drone_cam(i, qw, qh) for i in range(4)]
            pip_grid = np.vstack([np.hstack([cams[0], cams[1]]), np.hstack([cams[2], cams[3]])])
            px1, py1 = W - pip_w - 12, 44
            cv2.rectangle(canvas, (px1 - 2, py1 - 2), (px1 + pip_w + 2, py1 + pip_h + 2), (0, 255, 200), 2)
            canvas[py1:py1 + pip_h, px1:px1 + pip_w] = pip_grid
            cv2.putText(canvas, 'SWARM 4-CAM PiP', (px1 + 8, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)
            ry = py1 + pip_h + 8
            rw, rh = pip_w // 2, 120
            if ry + rh < H - 6:
                for k in range(2):
                    rc = self._render_rover_cam(k, rw, rh)
                    canvas[ry:ry + rh, px1 + k * rw:px1 + (k + 1) * rw] = rc
        cv2.putText(canvas, 'Left-drag: orbit | Right-drag: pan | Wheel: zoom | Double-click: reset',
                    (10, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 225, 255), 1, cv2.LINE_AA)
        return canvas

    def _render_3d_octomap_viewer_legacy(self, W: int, H: int) -> np.ndarray:
        canvas = np.full((H, W, 3), (26, 32, 38), dtype=np.uint8)

        if self.auto_rotate_3d:
            self.view_azim = (time.time() * 14.0) % 360.0

        az = math.radians(self.view_azim)
        el = math.radians(self.view_elev)
        ca, sa = math.cos(az), math.sin(az)
        ce, se = math.sin(el), math.cos(el)

        cx, cy = W // 2, int(H * 0.60)
        scale = (min(W, H) * 0.0082) * self.view_zoom

        def project_3d(nx: float, ey: float, hz: float) -> tuple[int, int, float]:
            rx = nx * ca - ey * sa
            ry = nx * sa + ey * ca
            px = int(cx + rx * scale)
            py = int(cy - (ry * ce + hz * se * 1.35) * scale)
            depth = ry * se - hz * ce
            return px, py, depth

        for gm in range(-50, 51, 10):
            p1 = project_3d(gm, -50, 0.0)
            p2 = project_3d(gm, 50, 0.0)
            col = (75, 95, 85) if gm == 0 else (45, 58, 65)
            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), col, 1, cv2.LINE_AA)
            p3 = project_3d(-50, gm, 0.0)
            p4 = project_3d(50, gm, 0.0)
            cv2.line(canvas, (p3[0], p3[1]), (p4[0], p4[1]), col, 1, cv2.LINE_AA)

        for lbl, ln, le in [('NORTH (+X)', 54, 0), ('SOUTH (-X)', -54, 0), ('EAST (+Y)', 0, 54), ('WEST (-Y)', 0, -54)]:
            lp = project_3d(ln, le, 0.0)
            cv2.putText(canvas, lbl, (lp[0] - 35, lp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 230, 170), 1, cv2.LINE_AA)

        with self.scan_lock:
            voxel_items = list(self.voxels_3d.items())

        with self.swarm_lock:
            drones = [dict(d) for d in self.swarm_drones]
            trails = [list(tr) for tr in self.swarm_trails]
            comm_adj = [list(row) for row in self.comm_adj]

        projected_voxels = []
        for (vx, vy, vz), h_val in voxel_items:
            px, py, depth = project_3d(float(vx), float(vy), float(vz))
            if 10 <= px < W - 10 and 45 <= py < H - 40:
                projected_voxels.append((depth, px, py, h_val))

        projected_voxels.sort(key=lambda item: item[0], reverse=True)

        box_r = max(3, int(3.8 * self.view_zoom))
        for _depth, px, py, h_val in projected_voxels:
            norm_h = min(1.0, max(0.0, h_val / 32.0))
            b = int((1.0 - norm_h) * 220 + 25)
            g = int(math.sin(norm_h * math.pi) * 230 + 25)
            r = int(norm_h * 235 + 20)
            cv2.rectangle(canvas, (px - box_r, py - box_r), (px + box_r, py + box_r), (b, g, r), -1)
            cv2.rectangle(canvas, (px - box_r, py - box_r), (px + box_r, py + box_r), (20, 25, 30), 1)

        # 3D Peer-to-Peer GNN Communication Links
        n_d = len(drones)
        for i in range(n_d):
            for j in range(i + 1, n_d):
                if i < len(comm_adj) and j < len(comm_adj[i]) and comm_adj[i][j] > 0.5:
                    di, dj = drones[i], drones[j]
                    pi = project_3d(float(di['x']), float(di['y']), max(0.0, -float(di['z'])))
                    pj = project_3d(float(dj['x']), float(dj['y']), max(0.0, -float(dj['z'])))
                    cv2.line(canvas, (pi[0], pi[1]), (pj[0], pj[1]), (180, 240, 120), 1, cv2.LINE_AA)

        # 4 Drones & 3D Flight Trajectories
        for idx, d in enumerate(drones[:4]):
            col = DRONE_COLORS_BGR[idx % 4]
            dx, dy, dz = float(d['x']), float(d['y']), float(d['z'])
            d_alt = max(0.0, -dz)
            tr = trails[idx]
            if len(tr) > 1:
                trail_3d = []
                for tidx, (tn, te) in enumerate(tr):
                    t_alt = d_alt * (tidx / max(1, len(tr) - 1))
                    tp = project_3d(tn, te, t_alt)
                    trail_3d.append((tp[0], tp[1]))
                pts_3d = np.array(trail_3d, dtype=np.int32)
                cv2.polylines(canvas, [pts_3d], False, col, 2, cv2.LINE_AA)

            d_ground = project_3d(dx, dy, 0.0)
            d_air = project_3d(dx, dy, d_alt)
            cv2.line(canvas, (d_ground[0], d_ground[1]), (d_air[0], d_air[1]), col, 1, cv2.LINE_AA)
            cv2.circle(canvas, (d_ground[0], d_ground[1]), 4, (80, 120, 140), -1, cv2.LINE_AA)
            cv2.circle(canvas, (d_air[0], d_air[1]), 8, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (d_air[0], d_air[1]), 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"{d['name']} ({d_alt:.0f}m)", (d_air[0] + 12, d_air[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 2, cv2.LINE_AA
            )

        # ── 4-Drone PiP Quad-Camera Multi-View on Page 2 as well! ─────────
        if W > 380 and H > 360:
            pip_w, pip_h = 340, 240
            qw, qh = pip_w // 2, pip_h // 2
            p0 = self._render_drone_cam(0, qw, qh)
            p1 = self._render_drone_cam(1, qw, qh)
            p2 = self._render_drone_cam(2, qw, qh)
            p3 = self._render_drone_cam(3, qw, qh)
            pip_grid = np.vstack([np.hstack([p0, p1]), np.hstack([p2, p3])])

            px1, py1 = W - pip_w - 12, 52
            cv2.rectangle(canvas, (px1 - 2, py1 - 2), (px1 + pip_w + 2, py1 + pip_h + 2), (0, 255, 200), 2)
            canvas[py1:py1 + pip_h, px1:px1 + pip_w] = pip_grid
            cv2.putText(canvas, 'SWARM 4-CAM PiP', (px1 + 8, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (W, 42), (18, 24, 28), -1)
        cv2.putText(
            canvas, 'LIVE 4-DRONE 3D OCTOMAP VOXEL VIEWER + P2P GNN LINKS + 4-CAM PiP',
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 235, 255), 2, cv2.LINE_AA
        )

        cv2.rectangle(canvas, (0, H - 38), (W, H), (18, 24, 28), -1)
        rot_mode = 'AUTO-ORBITING 3D CAMERA' if self.auto_rotate_3d else f'MANUAL 3D VIEW (Az={self.view_azim:.0f} El={self.view_elev:.0f})'
        cv2.putText(
            canvas, f'{rot_mode}  |  Drag mouse on 3D view to rotate, Scroll to zoom',
            (14, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 225, 255), 1, cv2.LINE_AA
        )
        return canvas


class DroneControlApp:
    """Multi-Page Tkinter GUI for the 4-Drone Decentralized Swarm with 4-Camera Multi-View."""

    def __init__(self, node: MapGuiNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('4-Drone Decentralized Swarm (GNN + Heuristic) — Live 2D Map, 4-Camera Multi-View & 3D OctoMap')
        self.root.geometry('1580x930')
        self.root.minsize(1200, 740)
        self.root.configure(bg='#121816')

        self.active_page = 'tactical'
        self._drag_last = None

        # ── Top Navigation Page & Camera Switcher Bar ──────────────────────
        nav_frame = tk.Frame(self.root, bg='#1a2420', bd=1, relief=tk.RAISED)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(
            nav_frame, text='VIEW:', font=('DejaVu Sans', 10, 'bold'),
            fg='#90e0c0', bg='#1a2420'
        ).pack(side=tk.LEFT, padx=(8, 4), pady=6)

        self.btn_page_tactical = tk.Button(
            nav_frame, text='[Page 1] 2D Map & 4-Cam Grid',
            font=('DejaVu Sans', 10, 'bold'), padx=10, pady=3,
            command=lambda: self._switch_page('tactical')
        )
        self.btn_page_tactical.pack(side=tk.LEFT, padx=3, pady=4)

        self.btn_page_scanned = tk.Button(
            nav_frame, text='[Page 2] Whole Scanned 3D OctoMap',
            font=('DejaVu Sans', 10, 'bold'), padx=10, pady=3,
            command=lambda: self._switch_page('scanned_map')
        )
        self.btn_page_scanned.pack(side=tk.LEFT, padx=3, pady=4)

        # Camera View Switcher Buttons
        tk.Label(
            nav_frame, text='CAMERAS:', font=('DejaVu Sans', 9, 'bold'),
            fg='#ffdb4d', bg='#1a2420'
        ).pack(side=tk.LEFT, padx=(14, 4), pady=6)

        self.btn_cam_quad = tk.Button(
            nav_frame, text='4-CAM GRID', font=('DejaVu Sans', 9, 'bold'), padx=8, pady=3,
            command=lambda: self._set_cam_layout('quad')
        )
        self.btn_cam_quad.pack(side=tk.LEFT, padx=2)

        self.btn_cam_drones = []
        for i in range(4):
            b = tk.Button(
                nav_frame, text=f'UAV-{i + 1} CAM', font=('DejaVu Sans', 9, 'bold'), padx=8, pady=3,
                command=lambda did=i: self._set_cam_layout(did)
            )
            b.pack(side=tk.LEFT, padx=2)
            self.btn_cam_drones.append(b)

        self.btn_cam_rovers = []
        for text, lay in (('ROVER CAMS', 'rovers'), ('ROVER 1 CAM', 'r0'), ('ROVER 2 CAM', 'r1')):
            b = tk.Button(nav_frame, text=text, font=('DejaVu Sans', 9, 'bold'), padx=8, pady=3,
                          command=lambda l=lay: self._set_cam_layout(l))
            b.pack(side=tk.LEFT, padx=2)
            self.btn_cam_rovers.append((b, lay))

        tk.Button(
            nav_frame, text='Clear Scanned Map', font=('DejaVu Sans', 9, 'bold'),
            bg='#4a2525', fg='#ffdede', padx=10, pady=3,
            command=self.node.clear_scanned_map
        ).pack(side=tk.RIGHT, padx=6, pady=4)

        self.btn_autorot = tk.Button(
            nav_frame, text='3D Auto-Orbit: OFF', font=('DejaVu Sans', 9, 'bold'),
            bg='#244438', fg='#b0ffdf', padx=10, pady=3,
            command=self._toggle_autorot
        )
        self.btn_autorot.pack(side=tk.RIGHT, padx=4, pady=4)

        self._update_nav_buttons()

        # ── Main Video Display Canvas ─────────────────────────────────────
        self.video_label = tk.Label(self.root, bg='#121816', bd=0)
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.video_label.bind('<ButtonPress-1>', self._on_mouse_down)
        self.video_label.bind('<B1-Motion>', self._on_mouse_drag)
        self.video_label.bind('<Button-4>', lambda _e: self._on_zoom(1.1))
        self.video_label.bind('<Button-5>', lambda _e: self._on_zoom(0.9))
        self.video_label.bind('<MouseWheel>', self._on_wheel)
        self.video_label.bind('<ButtonPress-3>', self._on_pan_down)
        self.video_label.bind('<B3-Motion>', self._on_pan_drag)
        self.video_label.bind('<ButtonPress-2>', self._on_pan_down)
        self.video_label.bind('<B2-Motion>', self._on_pan_drag)
        self.video_label.bind('<Double-Button-1>', self._on_double_click)

        # ── Bottom Interactive Command & Preset Panel ─────────────────────
        ctrl_frame = tk.Frame(self.root, bg='#1d2822', bd=1, relief=tk.RIDGE)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(2, 8))

        input_row = tk.Frame(ctrl_frame, bg='#1d2822')
        input_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(
            input_row, text='SWARM COMMAND:', font=('DejaVu Sans', 11, 'bold'),
            fg='#7fffd4', bg='#1d2822'
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.cmd_entry = tk.Entry(
            input_row, font=('DejaVu Sans', 12), bg='#0f1713', fg='#ffffff',
            insertbackground='#00ffcc', relief=tk.FLAT, bd=6
        )
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.cmd_entry.insert(0, 'Map and scan the whole surrounding')
        self.cmd_entry.bind('<Return>', self._on_send)

        tk.Button(
            input_row, text='SEND TO 4 DRONES', font=('DejaVu Sans', 11, 'bold'),
            bg='#00b87c', fg='#000000', activebackground='#00e699',
            padx=16, pady=4, command=self._on_send
        ).pack(side=tk.LEFT)

        preset_row = tk.Frame(ctrl_frame, bg='#1d2822')
        preset_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(4, 8))

        tk.Label(
            preset_row, text='SWARM PRESETS:', font=('DejaVu Sans', 9, 'bold'),
            fg='#b0d0c0', bg='#1d2822'
        ).pack(side=tk.LEFT, padx=(0, 6))

        presets = [
            ('4-Drone GNN Whole-Map Scan', 'Map and scan the whole surrounding'),
            ('Toggle GNN / Heuristic Only', 'Toggle GNN / Heuristic mode'),
            ('Swarm Find Survivors', 'Search for the person'),
            ('4-Drone Formation Orbit', 'Move in circles at 12m height'),
            ('4-Drone Square Sweep', 'Fly a 26 metre square pattern at 11m'),
            ('Hold All 4 Drones', 'Hover here'),
            ('RTL & Land Swarm', 'Return to home and land'),
        ]
        for btn_label, cmd_str in presets:
            b = tk.Button(
                preset_row, text=btn_label, font=('DejaVu Sans', 9, 'bold'),
                bg='#006655' if 'GNN' in btn_label else '#2b3d34',
                fg='#ffffff' if 'GNN' in btn_label else '#e0fff4',
                activebackground='#3e584b', activeforeground='#ffffff',
                padx=8, pady=3,
                command=lambda c=cmd_str: self._send_preset(c)
            )
            b.pack(side=tk.LEFT, padx=3)

        self._build_viewer_controls()
        self._last_snap = 0.0
        if os.environ.get('SWARM_GUI_PAGE') == 'scanned_map':      # start directly on the 3D map page
            self._switch_page('scanned_map')
        self._photo = None
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._update_gui()

    def _switch_page(self, page_name: str):
        self.active_page = page_name
        self._update_nav_buttons()
        if page_name == 'scanned_map':
            self.viewer_ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2, before=self.video_label)
        else:
            self.viewer_ctrl.pack_forget()

    # ── 3-D viewer controls: layers, thresholds, views, focus, rover operations ──
    def _build_viewer_controls(self):
        node = self.node
        o = node.viewer.opts
        bg = '#1d2822'
        f = tk.Frame(self.root, bg=bg, bd=1, relief=tk.RIDGE)
        self.viewer_ctrl = f
        lab = dict(bg=bg, fg='#90e0c0', font=('DejaVu Sans', 9, 'bold'))
        chk = dict(bg=bg, fg='#e0fff4', selectcolor='#0f1713', activebackground=bg, activeforeground='#ffffff',
                   font=('DejaVu Sans', 9))

        row1 = tk.Frame(f, bg=bg)
        row1.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(row1, text='LAYERS:', **lab).pack(side=tk.LEFT, padx=(0, 4))
        layers = [('Occupied', 'show_occupied'), ('Ground vox', 'show_ground_voxels'), ('Obstacle vox', 'show_obstacle_voxels'),
                  ('Free space', 'show_free'), ('Unknown', 'show_unknown'), ('Recent', 'show_recent'),
                  ('New obstacles', 'show_new_obstacles'), ('Coverage', 'show_coverage'), ('Shadows', 'show_shadows'),
                  ('AO', 'show_ao'), ('Edges', 'show_edges'), ('Trails', 'show_trails'), ('Routes', 'show_routes'),
                  ('Survivors', 'show_survivors')]
        self._layer_vars = {}
        for label, attr in layers:
            var = tk.BooleanVar(value=getattr(o, attr))
            self._layer_vars[attr] = var
            tk.Checkbutton(row1, text=label, variable=var, command=lambda a=attr, v=var: setattr(o, a, bool(v.get())),
                           **chk).pack(side=tk.LEFT, padx=2)

        row2 = tk.Frame(f, bg=bg)
        row2.pack(fill=tk.X, padx=6, pady=(0, 0))

        def slider(text, attr, lo, hi, step, width=95):
            tk.Label(row2, text=text, **lab).pack(side=tk.LEFT, padx=(6, 1))
            sc = tk.Scale(row2, from_=lo, to=hi, resolution=step, orient=tk.HORIZONTAL, length=width, showvalue=True,
                          bg=bg, fg='#e0fff4', troughcolor='#0f1713', highlightthickness=0, font=('DejaVu Sans', 8),
                          command=lambda v, a=attr: setattr(o, a, float(v)))
            sc.set(getattr(o, attr))
            sc.pack(side=tk.LEFT)

        slider('Occ. thr', 'occ_threshold', 0.5, 0.97, 0.01)
        slider('Z min', 'z_min', 0.0, 20.0, 0.5)
        slider('Z max', 'z_max', 1.0, 60.0, 1.0)
        slider('Opacity', 'opacity', 0.2, 1.0, 0.05)
        slider('Voxel size', 'voxel_scale', 0.4, 1.0, 0.05)
        tk.Label(row2, text='Colour', **lab).pack(side=tk.LEFT, padx=(8, 2))
        self._color_mode = tk.StringVar(value=o.color_mode)
        cb = ttk.Combobox(row2, textvariable=self._color_mode, values=['height', 'occupancy', 'age', 'class'],
                          width=9, state='readonly')
        cb.pack(side=tk.LEFT)
        cb.bind('<<ComboboxSelected>>', lambda _e: setattr(o, 'color_mode', self._color_mode.get()))

        row3 = tk.Frame(f, bg=bg)
        row3.pack(fill=tk.X, padx=6, pady=(0, 4))
        btn = dict(font=('DejaVu Sans', 9, 'bold'), bg='#2b3d34', fg='#e0fff4', activebackground='#3e584b', padx=6, pady=1)
        tk.Label(row3, text='VIEW:', **lab).pack(side=tk.LEFT, padx=(0, 3))
        for text, name in (('Top', 'top'), ('Side', 'side'), ('Isometric', 'iso'), ('Orbit', 'orbit')):
            tk.Button(row3, text=text, command=lambda n=name: node.viewer.cam.preset(n), **btn).pack(side=tk.LEFT, padx=1)
        tk.Button(row3, text='Reset', command=node.viewer.cam.reset, **btn).pack(side=tk.LEFT, padx=1)

        tk.Label(row3, text='FOCUS:', **lab).pack(side=tk.LEFT, padx=(10, 3))
        self._focus_var = tk.StringVar(value='UAV-1')
        self._focus_box = ttk.Combobox(row3, textvariable=self._focus_var, width=18, state='readonly',
                                       values=self._focus_values())
        self._focus_box.pack(side=tk.LEFT)
        self._focus_box.bind('<Button-1>', lambda _e: self._focus_box.configure(values=self._focus_values()))
        tk.Button(row3, text='Go', command=self._do_focus, **btn).pack(side=tk.LEFT, padx=2)

        tk.Label(row3, text='ROVERS:', **lab).pack(side=tk.LEFT, padx=(12, 3))
        tk.Label(row3, text='detections', bg=bg, fg='#c0e0d0', font=('DejaVu Sans', 8)).pack(side=tk.LEFT)
        self._det_var = tk.StringVar(value='Simulated detector')
        self._det_names = {'Simulated detector': 'sim', 'Real YOLO registry': 'swarm', 'Both': 'both', 'Off': 'off'}
        dcb = ttk.Combobox(row3, textvariable=self._det_var, values=list(self._det_names), width=17, state='readonly')
        dcb.pack(side=tk.LEFT, padx=2)
        dcb.bind('<<ComboboxSelected>>', lambda _e: node.set_det_source(self._det_names[self._det_var.get()]))
        self._unknown_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row3, text='Allow unknown ground', variable=self._unknown_var,
                       command=lambda: node.set_allow_unknown(self._unknown_var.get()), **chk).pack(side=tk.LEFT, padx=4)
        tk.Button(row3, text='Drop rubble on route', command=node.drop_obstacle_on_route, **btn).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text='Reset rovers', command=lambda: node.reset_rescue(False), **btn).pack(side=tk.LEFT, padx=2)

    def _focus_values(self):
        vals = [f'UAV-{i + 1}' for i in range(4)] + ['Rover 1', 'Rover 2']
        vals += [f'Survivor: {lbl}' for lbl in self.node.survivor_labels()]
        return vals

    def _do_focus(self):
        v = self._focus_var.get()
        if v.startswith('UAV-'):
            self.node.focus_entity('drone', int(v[4:]) - 1)
        elif v.startswith('Rover'):
            self.node.focus_entity('rover', int(v[6:]) - 1)
        elif v.startswith('Survivor'):
            self.node.focus_entity('survivor', max(0, self._focus_values().index(v) - 6))

    def _set_cam_layout(self, layout):
        self.node.cam_layout = layout
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        if self.active_page == 'tactical':
            self.btn_page_tactical.configure(bg='#00c888', fg='#000000')
            self.btn_page_scanned.configure(bg='#283830', fg='#c0e0d0')
        else:
            self.btn_page_tactical.configure(bg='#283830', fg='#c0e0d0')
            self.btn_page_scanned.configure(bg='#00d8ff', fg='#000000')

        for b, lay in self.btn_cam_rovers:
            on = self.node.cam_layout == lay
            b.configure(bg='#ffb020' if on else '#283830', fg='#000000' if on else '#c0e0d0')
        if self.node.cam_layout == 'quad':
            self.btn_cam_quad.configure(bg='#00e699', fg='#000000')
            for b in self.btn_cam_drones:
                b.configure(bg='#283830', fg='#c0e0d0')
        elif isinstance(self.node.cam_layout, str):
            self.btn_cam_quad.configure(bg='#283830', fg='#c0e0d0')
            for b in self.btn_cam_drones:
                b.configure(bg='#283830', fg='#c0e0d0')
        else:
            self.btn_cam_quad.configure(bg='#283830', fg='#c0e0d0')
            for i, b in enumerate(self.btn_cam_drones):
                if self.node.cam_layout == i:
                    b.configure(bg='#00d8ff', fg='#000000')
                else:
                    b.configure(bg='#283830', fg='#c0e0d0')

    def _toggle_autorot(self):
        cam = self.node.viewer.cam
        cam.auto_orbit = not cam.auto_orbit
        self.node.auto_rotate_3d = cam.auto_orbit
        state = 'ON' if cam.auto_orbit else 'OFF'
        self.btn_autorot.configure(text=f'3D Auto-Orbit: {state}')

    def _on_mouse_down(self, event):
        self._drag_last = (event.x, event.y)
        if self.active_page == 'tactical':
            win_w = max(1100, self.video_label.winfo_width())
            win_h = max(560, self.video_label.winfo_height())
            half_w = win_w // 2
            # If user clicked on the right-half camera pane
            if event.x >= half_w:
                if self.node.cam_layout == 'quad':
                    rx = event.x - half_w
                    ry = event.y
                    is_right = (rx >= (half_w // 2))
                    is_bot = (ry >= (win_h // 2))
                    quad_idx = (2 if is_bot else 0) + (1 if is_right else 0)
                    self._set_cam_layout(quad_idx)
                else:
                    self._set_cam_layout('quad')

    def _in_3d_pane(self, x: int) -> bool:
        return self.active_page == 'scanned_map' and x >= max(1100, self.video_label.winfo_width()) // 2

    def _on_mouse_drag(self, event):
        if self._drag_last is not None and self._in_3d_pane(event.x):
            dx = event.x - self._drag_last[0]
            dy = event.y - self._drag_last[1]
            self.node.viewer.cam.orbit(-dx * 0.45, dy * 0.35)
            self.btn_autorot.configure(text='3D Auto-Orbit: OFF')
        self._drag_last = (event.x, event.y)

    def _on_pan_down(self, event):
        self._pan_last = (event.x, event.y)

    def _on_pan_drag(self, event):
        last = getattr(self, '_pan_last', None)
        if last is not None and self._in_3d_pane(event.x):
            self.node.viewer.cam.pan(event.x - last[0], event.y - last[1], max(1, self.video_label.winfo_height()))
        self._pan_last = (event.x, event.y)

    def _on_zoom(self, factor: float):
        self.node.viewer.cam.zoom(factor)

    def _on_wheel(self, event):
        if self._in_3d_pane(event.x):
            self._on_zoom(1.1 if event.delta > 0 else 0.9)

    def _on_double_click(self, event):
        if self._in_3d_pane(event.x):
            self.node.viewer.cam.reset()

    def _on_send(self, _event=None):
        text = self.cmd_entry.get().strip()
        if text:
            self.node.send_user_command(text)

    def _send_preset(self, cmd_str: str):
        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, cmd_str)
        if 'Map and scan' in cmd_str:
            self._switch_page('scanned_map')
        elif 'Search' in cmd_str or 'Fly' in cmd_str or 'circles' in cmd_str:
            self._switch_page('tactical')
        self.node.send_user_command(cmd_str)

    def _update_gui(self):
        try:
            win_w = max(1100, self.video_label.winfo_width())
            win_h = max(560, self.video_label.winfo_height())
            pane_w = max(520, win_w // 2)
            pane_h = max(520, win_h)

            if self.active_page == 'tactical':
                bgr = self.node.render_tactical_page(pane_w=pane_w, pane_h=pane_h)
            else:
                bgr = self.node.render_scanned_map_page(pane_w=pane_w, pane_h=pane_h)

            snap_dir = os.environ.get('SWARM_GUI_SNAPSHOT')            # optional: save what the GUI shows (for checks)
            if snap_dir and time.time() - self._last_snap > 6.0:
                self._last_snap = time.time()
                os.makedirs(snap_dir, exist_ok=True)
                cv2.imwrite(os.path.join(snap_dir, f'{self.active_page}.png'), bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = PilImage.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=self._photo)
        except Exception as e:
            pass
        finally:
            self.root.after(50, self._update_gui)  # 20 FPS

    def _on_close(self):
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = MapGuiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        app = DroneControlApp(node)
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Decentralized 4-Drone Swarm Brain Node — GNN + Heuristic Task Allocation & PX4 Control.

Coordinates 4 autonomous drones (`UAV-1`, `UAV-2`, `UAV-3`, `UAV-4`) in a decentralized swarm:
1. Subscribes to PX4 telemetry for all 4 drones:
   - UAV-1 (drone 0): `/fmu/out/...` & `/fmu/in/...`
   - UAV-2 (drone 1): `/uav_1/fmu/out/...` & `/uav_1/fmu/in/...`
   - UAV-3 (drone 2): `/uav_2/fmu/out/...` & `/uav_2/fmu/in/...`
   - UAV-4 (drone 3): `/uav_3/fmu/out/...` & `/uav_3/fmu/in/...`
2. Broadcasts decentralized peer-to-peer state on `/swarm/peer_state` and aggregated
   telemetry on `/swarm/state` at 10 Hz.
3. Runs the GraphSAGE GNN Task Allocator (`SwarmGNNAllocator`) with automatic Heuristic
   Frontier Auction fallback so all 4 drones collaboratively scan 100% of the 120m x 120m
   disaster environment without overlapping or colliding.
4. Supports natural language LLM commands and swarm modes (Whole-Map GNN Scan, Heuristic-Only
   Scan, Multi-Drone Survivor Search, 4-Drone Synchronized Orbit, Square Sweep, Hold, Land).
"""

import json
import math
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from px4_msgs.msg import (
    BatteryStatus,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleGlobalPosition,
    VehicleOdometry,
    VehicleStatus,
)

from drone_agent.command_translator import CommandTranslator
from drone_agent.llm_client import LLMClient
from drone_agent.swarm_gnn_allocator import DroneAgentState, SwarmGNNAllocator


NUM_DRONES = 4

# Spawn positions in World NED (North, East) and staggered safe cruise altitudes (m AGL)
# Gazebo ENU spawn poses:
#   UAV-1: ENU (0, 3, 0)   -> NED N=+3.0, E=0.0,  Alt=9.0m
#   UAV-2: ENU (3, 0, 0)   -> NED N=0.0,  E=+3.0, Alt=10.5m
#   UAV-3: ENU (0, -3, 0)  -> NED N=-3.0, E=0.0,  Alt=12.0m
#   UAV-4: ENU (-3, 0, 0)  -> NED N=0.0,  E=-3.0, Alt=13.5m
SWARM_SPECS = [
    {'id': 0, 'name': 'UAV-1', 'ns': '',        'spawn_ned': (3.0, 0.0),  'alt_agl': 9.0},
    {'id': 1, 'name': 'UAV-2', 'ns': '/px4_1',  'spawn_ned': (0.0, 3.0),  'alt_agl': 10.5},
    {'id': 2, 'name': 'UAV-3', 'ns': '/px4_2',  'spawn_ned': (-3.0, 0.0), 'alt_agl': 12.0},
    {'id': 3, 'name': 'UAV-4', 'ns': '/px4_3',  'spawn_ned': (0.0, -3.0), 'alt_agl': 13.5},
]


class DroneUnit:
    """Encapsulates per-drone telemetry, PX4 publishers, translator, and decentralized state."""

    def __init__(self, node: Node, spec: dict, px4_qos: QoSProfile):
        self.node = node
        self.drone_id: int = spec['id']
        self.name: str = spec['name']
        self.ns: str = spec['ns']
        self.spawn_n, self.spawn_e = spec['spawn_ned']
        self.cruise_alt_agl: float = spec['alt_agl']

        self.translator = CommandTranslator(
            drone_id=self.drone_id,
            default_alt_agl=self.cruise_alt_agl,
        )
        # Initialize drone at its spawn position in World NED
        self.state = DroneAgentState(
            drone_id=self.drone_id,
            name=self.name,
            x=self.spawn_n,
            y=self.spawn_e,
            z=0.0,
            target_x=self.spawn_n,
            target_y=self.spawn_e,
            target_z=-self.cruise_alt_agl,
            task_label='Standby @ Helipad',
            alloc_mode='GNN (GraphSAGE)',
        )
        self.translator.update_position(self.spawn_n, self.spawn_e, 0.0)
        self.translator.target_x = self.spawn_n
        self.translator.target_y = self.spawn_e
        self.translator.target_z = -self.cruise_alt_agl

        self.odom_received = False
        self.last_odom_time = 0.0
        self.frame_calibrated = False
        self.landing_active = False
        self.task_assigned_time = 0.0

        # Topic prefix ('' for drone 0, '/uav_1' for drone 1, etc.)
        p = self.ns

        # Subscribers
        node.create_subscription(
            VehicleOdometry, f'{p}/fmu/out/vehicle_odometry', self._odom_cb, px4_qos
        )
        node.create_subscription(
            VehicleGlobalPosition, f'{p}/fmu/out/vehicle_global_position', self._gps_cb, px4_qos
        )
        node.create_subscription(
            VehicleStatus, f'{p}/fmu/out/vehicle_status', self._status_cb, px4_qos
        )
        node.create_subscription(
            VehicleStatus, f'{p}/fmu/out/vehicle_status_v1', self._status_cb, px4_qos
        )
        node.create_subscription(
            BatteryStatus, f'{p}/fmu/out/battery_status', self._battery_cb, px4_qos
        )

        # Publishers
        self.offboard_pub = node.create_publisher(
            OffboardControlMode, f'{p}/fmu/in/offboard_control_mode', 10
        )
        self.setpoint_pub = node.create_publisher(
            TrajectorySetpoint, f'{p}/fmu/in/trajectory_setpoint', 10
        )
        self.command_pub = node.create_publisher(
            VehicleCommand, f'{p}/fmu/in/vehicle_command', 10
        )
        self.command_pub_v1 = node.create_publisher(
            VehicleCommand, f'{p}/fmu/in/vehicle_command_v1', 10
        )

    def _odom_cb(self, msg: VehicleOdometry):
        raw_n = float(msg.position[0])
        raw_e = float(msg.position[1])
        raw_d = float(msg.position[2])

        # Auto-calibrate whether PX4 EKF2 local origin is at spawn pose or world (0,0)
        if not self.frame_calibrated:
            self.frame_calibrated = True
            if math.hypot(raw_n - self.spawn_n, raw_e - self.spawn_e) < 1.2:
                self.translator.frame_offset_x = 0.0
                self.translator.frame_offset_y = 0.0
            else:
                self.translator.frame_offset_x = self.spawn_n
                self.translator.frame_offset_y = self.spawn_e

        self.odom_received = True
        self.last_odom_time = time.time()

        world_n = raw_n + self.translator.frame_offset_x
        world_e = raw_e + self.translator.frame_offset_y
        world_d = raw_d

        self.state.x = world_n
        self.state.y = world_e
        self.state.z = world_d
        self.state.vx = float(msg.velocity[0])
        self.state.vy = float(msg.velocity[1])
        self.state.vz = float(msg.velocity[2])

        q = msg.q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.state.yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        if abs(world_d) > 2.0:
            self.state.armed = True

        self.translator.update_position(world_n, world_e, world_d)

    def _gps_cb(self, msg: VehicleGlobalPosition):
        if not self.translator.home_set and not math.isnan(msg.lat) and not math.isnan(msg.lon) and (abs(msg.lat) > 0.0001 or abs(msg.lon) > 0.0001):
            self.translator.set_home(msg.lat, msg.lon, msg.alt)

    def _status_cb(self, msg: VehicleStatus):
        self.state.armed = (msg.arming_state == 2)

    def _battery_cb(self, msg: BatteryStatus):
        self.state.battery = float(msg.remaining) if msg.remaining > 0.0 else 0.95

    def publish_extra_commands(self, extra_msgs: list):
        now_us = int(time.time() * 1e6)
        for _, msg in extra_msgs:
            msg.timestamp = now_us
            self.command_pub.publish(msg)
            self.command_pub_v1.publish(msg)

    def tick_10hz(self, dt: float = 0.1):
        """Publish 10Hz PX4 offboard setpoints and update kinematic state if PX4 SITL is warming up."""
        now_us = int(time.time() * 1e6)
        offboard_msg = self.translator.get_offboard_control_mode()
        offboard_msg.timestamp = now_us
        self.offboard_pub.publish(offboard_msg)

        sp_msg = self.translator.get_setpoint_tick()
        sp_msg.timestamp = now_us
        self.setpoint_pub.publish(sp_msg)

        # Keep DroneAgentState target synchronized with translator
        self.state.target_x = self.translator.target_x
        self.state.target_y = self.translator.target_y
        self.state.target_z = self.translator.target_z

        # If this drone's PX4 SITL instance hasn't published fresh odometry in the last 1.2s,
        # smoothly integrate 3D quadcopter kinematics toward the setpoint so all 4 drones
        # ALWAYS fly and scan seamlessly even during PX4 EKF2 warmup or low-RAM throttling.
        if (time.time() - self.last_odom_time) > 1.2:
            if self.landing_active:
                goal_n, goal_e, goal_d = self.spawn_n, self.spawn_e, 0.0
            else:
                goal_n = float(sp_msg.position[0]) + self.translator.frame_offset_x
                goal_e = float(sp_msg.position[1]) + self.translator.frame_offset_y
                goal_d = float(sp_msg.position[2])

            dn = goal_n - self.state.x
            de = goal_e - self.state.y
            dd = goal_d - self.state.z
            dist_xy = math.hypot(dn, de)
            max_spd_xy = 4.2
            max_spd_z = 2.8

            # Climb vertically first before horizontal translation when below 4m AGL
            if abs(self.state.z) < 4.0 and not self.landing_active:
                vx, vy = 0.0, 0.0
            elif dist_xy > 0.15:
                step_xy = min(max_spd_xy, dist_xy * 1.4)
                vx = (dn / dist_xy) * step_xy
                vy = (de / dist_xy) * step_xy
                self.state.yaw = math.atan2(vy, vx)
            else:
                vx, vy = 0.0, 0.0

            vz = max(-max_spd_z, min(max_spd_z, dd * 1.5))

            self.state.x += vx * dt
            self.state.y += vy * dt
            self.state.z += vz * dt
            self.state.vx = vx
            self.state.vy = vy
            self.state.vz = vz
            self.state.armed = not (self.landing_active and abs(self.state.z) < 0.3)
            self.state.battery = max(0.15, self.state.battery - 0.00004)
            self.translator.update_position(self.state.x, self.state.y, self.state.z)


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Parameters
        self.declare_parameter('llm_interval_sec', 7.0)
        self.declare_parameter('offboard_rate_hz', 10.0)
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('ollama_model', 'qwen2.5:3b')

        self.llm_interval = self.get_parameter('llm_interval_sec').value
        offboard_rate = self.get_parameter('offboard_rate_hz').value

        ollama_url = self.get_parameter('ollama_url').value
        ollama_model = self.get_parameter('ollama_model').value
        self.llm = LLMClient(ollama_url=ollama_url, model=ollama_model)

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Initialize the 4 decentralized drone units
        self.drones: list[DroneUnit] = [
            DroneUnit(self, spec, px4_qos) for spec in SWARM_SPECS
        ]

        # Initialize the GraphSAGE GNN + Heuristic Fallback Swarm Task Allocator
        self.allocator = SwarmGNNAllocator()
        self.force_heuristic_only = False

        # Shared/merged 240x240 occupancy grid (0=unknown, 1=free, 2=obstacle) for frontier calculation
        self.swarm_occ_grid = np.zeros((240, 240), dtype=np.uint8)
        self._seed_static_obstacles_in_grid()

        # Swarm mission state
        # Modes: 'swarm_scan' (autonomous GNN/Heuristic exploration), 'survivor_search', 'orbit', 'square', 'manual', 'land'
        self.swarm_mode = 'swarm_scan'
        self.current_command = 'Swarm GNN + Heuristic Whole-Area Scan (4 Drones)'
        self.latest_detections = '[]'
        self.last_detection_classes = set()
        self._llm_busy = False
        self._auto_started = False
        self.last_realloc_time = 0.0

        # Subscribers
        self.create_subscription(String, '/yolo/detections', self._yolo_cb, 10)
        self.create_subscription(String, '/user_command', self._user_cmd_cb, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)

        # Decentralized peer-to-peer & swarm telemetry publishers
        self.peer_pub = self.create_publisher(String, '/swarm/peer_state', 20)
        self.swarm_state_pub = self.create_publisher(String, '/swarm/state', 10)

        # Timers
        period = 1.0 / offboard_rate
        self.create_timer(period, self._offboard_swarm_loop)
        self.create_timer(0.25, self._decentralized_peer_broadcast_loop)
        self.create_timer(2.2, self._swarm_task_allocation_loop)
        self.create_timer(2.5, self._auto_start_swarm)

        self.get_logger().info(
            f'4-Drone Decentralized Swarm Brain started! Allocator: {self.allocator.status_reason}'
        )

    def _seed_static_obstacles_in_grid(self):
        """Mark known collapsed building footprints as obstacles (2) in the 240x240 grid."""
        from drone_agent.command_translator import BUILDING_OBSTACLES_NED
        for bn, be, brad, _bh in BUILDING_OBSTACLES_NED:
            gx, gy = self.allocator._ned_to_grid(bn, be)
            r_px = int((brad * 0.65) / self.allocator.grid_res)
            cv2.circle(self.swarm_occ_grid, (gx, gy), r_px, 2, -1)

    def _auto_start_swarm(self):
        if self._auto_started:
            return
        self._auto_started = True
        self.get_logger().info(
            'Auto-starting 4-Drone Decentralized Swarm GNN + Heuristic Whole-Area Scan!'
        )
        self.swarm_mode = 'swarm_scan'
        self._run_gnn_or_heuristic_allocation(force=True)

    def _run_gnn_or_heuristic_allocation(self, force: bool = False):
        """Run decentralized GNN (with Heuristic fallback) to assign tasks to drones needing a new goal."""
        if self.swarm_mode != 'swarm_scan':
            return

        now = time.time()
        # Identify which drones have reached their assigned target (< 4.5m) or timed out (> 14s)
        drones_needing_task = [
            d for d in self.drones
            if force
            or (math.hypot(d.state.x - d.state.target_x, d.state.y - d.state.target_y) < 4.5 and abs(d.state.z) > 4.0)
            or (now - d.task_assigned_time) > 14.0
        ]
        if not drones_needing_task:
            return

        self.last_realloc_time = now
        saved_gnn_flag = self.allocator.gnn_loaded
        if self.force_heuristic_only:
            self.allocator.gnn_loaded = False

        try:
            states = [d.state for d in self.drones]
            assignments = self.allocator.allocate_tasks(
                states, self.swarm_occ_grid, mission_mode='scan'
            )
        finally:
            self.allocator.gnn_loaded = saved_gnn_flag

        need_ids = {d.drone_id for d in drones_needing_task}
        log_parts = []
        for d in self.drones:
            if d.drone_id in need_ids:
                tn, te, lbl, mode, score = assignments[d.drone_id]
                d.state.task_label = lbl
                d.state.alloc_mode = mode
                d.state.gnn_score = score
                d.landing_active = False
                d.task_assigned_time = now

                cmd = {
                    'action': 'position_ned',
                    'x': tn,
                    'y': te,
                    'z': -d.cruise_alt_agl,
                }
                extra = d.translator.process_command(cmd)
                d.publish_extra_commands(extra)
            log_parts.append(
                f'{d.name}->{d.state.task_label} ({d.state.target_x:+.0f}N,{d.state.target_y:+.0f}E) [{d.state.alloc_mode}]'
            )

        self.get_logger().info('Swarm Task Allocation: ' + ' | '.join(log_parts))

    def _enforce_px4_arming(self, drone_id: int):
        px4_bin = Path.home() / 'PX4-Autopilot/build/px4_sitl_default/bin'
        param_bin = px4_bin / 'px4-param'
        cmd_bin = px4_bin / 'px4-commander'
        if not param_bin.exists():
            return
        try:
            subprocess.run([str(param_bin), '--instance', str(drone_id), 'set', 'NAV_DLL_ACT', '0'], capture_output=True, timeout=0.6)
            subprocess.run([str(param_bin), '--instance', str(drone_id), 'set', 'NAV_RCL_ACT', '0'], capture_output=True, timeout=0.6)
            subprocess.run([str(param_bin), '--instance', str(drone_id), 'set', 'COM_RCL_EXCEPT', '4'], capture_output=True, timeout=0.6)
            subprocess.run([str(cmd_bin), '--instance', str(drone_id), 'mode', 'offboard'], capture_output=True, timeout=0.6)
            subprocess.run([str(cmd_bin), '--instance', str(drone_id), 'arm', '-f'], capture_output=True, timeout=0.6)
        except Exception:
            pass

    def _swarm_task_allocation_loop(self):
        """Periodic check to update exploration grid and reallocate tasks across the 4 drones."""
        # Ensure any PX4 instance that finished EKF2 boot after initial startup gets armed & OFFBOARD
        if self.swarm_mode != 'land':
            for d in self.drones:
                if not d.state.armed or abs(d.state.z) < 1.5:
                    self._enforce_px4_arming(d.drone_id)
                    extra = d.translator._arm_and_offboard()
                    d.publish_extra_commands(extra)

        # Update explored free space around all 4 airborne drones (32m LiDAR scan radius)
        scan_r_cells = int(32.0 / self.allocator.grid_res)
        for d in self.drones:
            if abs(d.state.z) > 2.5:
                gx, gy = self.allocator._ned_to_grid(d.state.x, d.state.y)
                # Mark circle of free space where unknown (0 -> 1), preserving obstacles (2)
                mask = np.zeros_like(self.swarm_occ_grid)
                cv2.circle(mask, (gx, gy), scan_r_cells, 1, -1)
                self.swarm_occ_grid[(mask == 1) & (self.swarm_occ_grid == 0)] = 1

        # If coverage > 94%, reset visit counts so the swarm continues patrolling dynamically
        cov = float(np.count_nonzero(self.swarm_occ_grid > 0)) / float(self.swarm_occ_grid.size)
        if cov > 0.94 and len(self.allocator.visit_counts) > 25:
            self.allocator.visit_counts.clear()

        self._run_gnn_or_heuristic_allocation(force=False)

    def _offboard_swarm_loop(self):
        """10Hz control loop for all 4 drones + inter-drone repulsion + `/swarm/state` broadcast."""
        # 1. Update peer positions for each drone's decentralized collision avoidance
        for i, di in enumerate(self.drones):
            peers = [
                (dj.state.x, dj.state.y, dj.state.z)
                for j, dj in enumerate(self.drones)
                if i != j and math.hypot(di.state.x - dj.state.x, di.state.y - dj.state.y) <= self.allocator.comm_range_m
            ]
            di.translator.update_peers(peers)

        # 2. Tick 10Hz offboard control on all 4 drones
        for d in self.drones:
            d.tick_10hz(dt=0.1)

        # 3. Publish combined swarm state on `/swarm/state` for GUI & monitoring
        states = [d.state for d in self.drones]
        comm_adj = self.allocator.compute_comm_adj(states).tolist()
        payload = {
            'timestamp': time.time(),
            'swarm_mode': self.swarm_mode,
            'allocator_status': (
                'Heuristic Only (Forced)' if self.force_heuristic_only else self.allocator.status_reason
            ),
            'comm_range_m': self.allocator.comm_range_m,
            'comm_adj': comm_adj,
            'drones': [
                {
                    'id': d.drone_id,
                    'name': d.name,
                    'x': round(d.state.x, 2),
                    'y': round(d.state.y, 2),
                    'z': round(d.state.z, 2),
                    'vx': round(d.state.vx, 2),
                    'vy': round(d.state.vy, 2),
                    'vz': round(d.state.vz, 2),
                    'yaw': round(d.state.yaw, 3),
                    'armed': bool(d.state.armed),
                    'odom_live': bool(d.odom_received),
                    'battery': round(d.state.battery, 2),
                    'target_x': round(d.state.target_x, 2),
                    'target_y': round(d.state.target_y, 2),
                    'target_z': round(d.state.target_z, 2),
                    'task_label': d.state.task_label,
                    'alloc_mode': d.state.alloc_mode,
                    'gnn_score': round(d.state.gnn_score, 3),
                }
                for d in self.drones
            ],
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.swarm_state_pub.publish(msg)

    def _decentralized_peer_broadcast_loop(self):
        """4Hz decentralized peer-to-peer packet broadcast per drone on `/swarm/peer_state`."""
        now = time.time()
        for d in self.drones:
            pkt = {
                'drone_id': d.drone_id,
                'name': d.name,
                'pos_ned': [round(d.state.x, 2), round(d.state.y, 2), round(d.state.z, 2)],
                'vel_ned': [round(d.state.vx, 2), round(d.state.vy, 2), round(d.state.vz, 2)],
                'target_ned': [round(d.state.target_x, 2), round(d.state.target_y, 2), round(d.state.target_z, 2)],
                'task': d.state.task_label,
                'alloc_mode': d.state.alloc_mode,
                'timestamp': now,
            }
            m = String()
            m.data = json.dumps(pkt)
            self.peer_pub.publish(m)

    def _scan_cb(self, msg: LaserScan):
        """Apply LiDAR obstacle repulsion to UAV-1."""
        d0 = self.drones[0]
        yaw = d0.state.yaw
        rep_n, rep_e, count = 0.0, 0.0, 0
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.4 < r < 5.5 and math.isfinite(r):
                obs_bearing_ned = yaw - angle
                strength = (5.5 - r) / 5.5
                rep_n -= strength * math.cos(obs_bearing_ned)
                rep_e -= strength * math.sin(obs_bearing_ned)
                count += 1
            angle += msg.angle_increment
        if count > 0:
            d0.translator.update_lidar_repulsion(rep_n * 2.5 / count, rep_e * 2.5 / count)
        else:
            d0.translator.update_lidar_repulsion(0.0, 0.0)

    def _yolo_cb(self, msg: String):
        self.latest_detections = msg.data

    def _user_cmd_cb(self, msg: String):
        command = msg.data.strip()
        if not command:
            return
        self.current_command = command
        low = command.lower()
        self.get_logger().info(f'New Swarm Command: "{command}"')

        # 1. Fast local swarm mode commands (instant response)
        if low in ('reset', 'reset memory', 'clear memory', 'new mission'):
            self.llm.reset_memory()
            self.swarm_occ_grid[self.swarm_occ_grid == 1] = 0
            self.allocator.visit_counts.clear()
            self.swarm_mode = 'swarm_scan'
            self._run_gnn_or_heuristic_allocation(force=True)
            return

        if 'toggle gnn' in low or 'heuristic only' in low or 'force heuristic' in low:
            self.force_heuristic_only = not self.force_heuristic_only
            mode_str = 'HEURISTIC ONLY' if self.force_heuristic_only else 'GNN (GraphSAGE) + HEURISTIC FALLBACK'
            self.get_logger().info(f'Switched task allocator mode -> {mode_str}')
            self.swarm_mode = 'swarm_scan'
            self._run_gnn_or_heuristic_allocation(force=True)
            return

        if 'use gnn' in low or 'gnn mode' in low:
            self.force_heuristic_only = False
            self.swarm_mode = 'swarm_scan'
            self._run_gnn_or_heuristic_allocation(force=True)
            return

        if ('map' in low and 'scan' in low) or 'whole surrounding' in low or 'swarm scan' in low:
            self.swarm_mode = 'swarm_scan'
            self._run_gnn_or_heuristic_allocation(force=True)
            return

        if 'search' in low and ('person' in low or 'survivor' in low or 'victim' in low):
            self.swarm_mode = 'survivor_search'
            mode_lbl = 'GNN (GraphSAGE)' if not self.force_heuristic_only else 'Heuristic Fallback'
            # UAV-1 orbits central survivor plaza at r=14m so its YOLOv8 camera frames all 9 survivors;
            # UAV-2, UAV-3, UAV-4 sweep NW, SW, SE survivor & rescue vehicle sectors!
            d0 = self.drones[0]
            d0.state.task_label = 'YOLOv8 Survivor Lock-On Orbit (r=14m)'
            d0.state.alloc_mode = mode_lbl
            d0.landing_active = False
            d0.publish_extra_commands(d0.translator.process_command({
                'action': 'orbit', 'cx': 0.0, 'cy': 0.0, 'alt_z': -9.0, 'radius': 14.0, 'speed': 3.2
            }))

            wing_tasks = [
                (10.0, -10.0, 'Inspect NW Standing & Lying Survivors'),
                (-10.0, -10.0, 'Inspect SW Walking & Lying Victims'),
                (-10.0, 11.0, 'Inspect SE Ambulance & Survivors'),
            ]
            for idx, d in enumerate(self.drones[1:]):
                cn, ce, lbl = wing_tasks[idx % 3]
                d.state.task_label = lbl
                d.state.alloc_mode = mode_lbl
                d.landing_active = False
                d.publish_extra_commands(d.translator.process_command({
                    'action': 'orbit', 'cx': cn, 'cy': ce, 'alt_z': -d.cruise_alt_agl, 'radius': 9.0, 'speed': 3.0,
                    'phase_rad': (idx + 1) * (math.pi * 0.5),
                }))
            return

        if 'hover' in low or 'hold' in low or 'stop' in low:
            self.swarm_mode = 'manual'
            for d in self.drones:
                d.state.task_label = 'Hold Position'
                d.translator.process_command({'action': 'hold'})
                d.translator.target_x = d.state.x
                d.translator.target_y = d.state.y
                d.translator.target_z = d.state.z
            return

        if 'land' in low or 'return to home' in low or 'rtl' in low:
            self.swarm_mode = 'land'
            for d in self.drones:
                d.state.task_label = f'RTL & Land @ ({d.spawn_n:+.0f},{d.spawn_e:+.0f})'
                d.landing_active = True
                extra = d.translator.process_command({'action': 'land'})
                d.publish_extra_commands(extra)
            return

        # Otherwise call LLM in background thread and distribute the command across all 4 drones
        self._call_llm()

    def _call_llm(self):
        if self._llm_busy:
            return
        self._llm_busy = True
        threading.Thread(target=self._call_llm_worker, daemon=True).start()

    def _call_llm_worker(self):
        try:
            drone_state = self._format_swarm_state()
            self.get_logger().info(f'Calling {self.llm.backend} LLM ({self.llm.model}) for 4-drone swarm...')
            cmd = self.llm.ask(drone_state, self.current_command, self.latest_detections)
            action = cmd.get('action', 'orbit')
            self.get_logger().info(f'LLM Swarm Action: {cmd}')

            if action == 'orbit':
                self.swarm_mode = 'orbit'
                cx = float(cmd.get('cx', cmd.get('x', 0.0)))
                cy = float(cmd.get('cy', cmd.get('y', 0.0)))
                radius = float(cmd.get('radius', 15.0))
                speed = float(cmd.get('speed', 3.4))
                for i, d in enumerate(self.drones):
                    d.landing_active = False
                    d.state.task_label = f'Swarm Orbit Phase {i * 90} deg (r={radius:.0f}m)'
                    extra = d.translator.process_command({
                        'action': 'orbit',
                        'cx': cx,
                        'cy': cy,
                        'alt_z': -d.cruise_alt_agl,
                        'radius': radius,
                        'speed': speed,
                        'phase_rad': i * (math.pi * 0.5),
                    })
                    d.publish_extra_commands(extra)

            elif action in ('square_survey', 'square'):
                self.swarm_mode = 'square'
                side = float(cmd.get('side', 24.0))
                speed = float(cmd.get('speed', 3.0))
                for i, d in enumerate(self.drones):
                    d.landing_active = False
                    d.state.task_label = f'Swarm Square Sweep Corner #{i + 1}'
                    extra = d.translator.process_command({
                        'action': 'square_survey',
                        'side': side,
                        'speed': speed,
                        'alt_z': -d.cruise_alt_agl,
                    })
                    d.publish_extra_commands(extra)

            elif action in ('position_ned', 'goto'):
                self.swarm_mode = 'manual'
                base_x = float(cmd.get('x', 0.0))
                base_y = float(cmd.get('y', 0.0))
                # Formation offsets around target so 4 drones never stack on the same point
                offsets = [(4.0, 4.0), (4.0, -4.0), (-4.0, -4.0), (-4.0, 4.0)]
                for i, d in enumerate(self.drones):
                    ox, oy = offsets[i % 4]
                    tx, ty = base_x + ox, base_y + oy
                    d.landing_active = False
                    d.state.task_label = f'Formation Goto ({tx:+.0f}N,{ty:+.0f}E)'
                    extra = d.translator.process_command({
                        'action': 'position_ned',
                        'x': tx,
                        'y': ty,
                        'z': -d.cruise_alt_agl,
                    })
                    d.publish_extra_commands(extra)

            else:
                # Default to GNN + Heuristic whole-area scan
                self.swarm_mode = 'swarm_scan'
                self._run_gnn_or_heuristic_allocation(force=True)

        except Exception as e:
            self.get_logger().error(f'LLM call failed ({e}) -> falling back to GNN/Heuristic Swarm Scan')
            self.swarm_mode = 'swarm_scan'
            self._run_gnn_or_heuristic_allocation(force=True)
        finally:
            self._llm_busy = False

    def _format_swarm_state(self) -> str:
        lines = [f'Swarm Mode: {self.swarm_mode} (4 Drones)']
        for d in self.drones:
            lines.append(
                f'{d.name}: pos=({d.state.x:.1f}N, {d.state.y:.1f}E, {-d.state.z:.1f}m AGL) '
                f'task="{d.state.task_label}" [{d.state.alloc_mode}]'
            )
        return '\n'.join(lines)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

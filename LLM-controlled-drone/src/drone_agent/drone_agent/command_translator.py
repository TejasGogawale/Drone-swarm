#!/usr/bin/env python3
"""Translates LLM & Swarm GNN/Heuristic commands into PX4 messages with 3D Obstacle & Swarm Avoidance.

Converts high-level drone commands (takeoff, goto, orbit, land, etc.)
into PX4 OffboardControlMode, TrajectorySetpoint, and VehicleCommand messages,
while enforcing a 3D safe corridor around all collapsed buildings, obstacles,
and neighboring swarm drones.
"""

import math
import time

import numpy as np

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
)


# PX4 MAV_CMD constants
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_SET_ROI_LOCATION = 195
MAV_CMD_DO_CHANGE_SPEED = 178

# PX4 navigation modes
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0

# 3D Building Obstacle Cylinders in NED frame: (north_m, east_m, safe_radius_m, height_m)
BUILDING_OBSTACLES_NED = [
    (26.0, 32.0, 15.0, 12.5),    # Collapsed Fire Station NE
    (26.0, -32.0, 14.0, 10.0),   # Collapsed Police Station NW
    (-24.0, 30.0, 12.0, 7.5),    # Collapsed House SE
    (-24.0, -30.0, 12.0, 7.5),   # Collapsed House SW
    (36.0, 0.0, 12.0, 7.5),      # Collapsed House North
    (-36.0, 0.0, 14.0, 10.0),    # Collapsed Police Station South
]


from drone_agent.flight_smoothing import JerkLimitedCarrot, YawSmoother

class CommandTranslator:
    """Translates LLM & Swarm commands to PX4 ROS2 messages with 3D wall + inter-drone avoidance."""

    def __init__(
        self,
        drone_id: int = 0,
        default_alt_agl: float = 9.5,
        home_lat: float = 0.0,
        home_lon: float = 0.0,
        home_alt: float = 0.0,
    ):
        self.drone_id = drone_id
        self.default_alt_agl = default_alt_agl
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_alt = home_alt
        self.home_set = False

        # Live drone position in World NED
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

        # Offset between PX4 local EKF2 origin and World NED origin (auto-calibrated at startup)
        self.frame_offset_x = 0.0
        self.frame_offset_y = 0.0

        # Live LiDAR repulsive vector (North, East)
        self.lidar_repulse_n = 0.0
        self.lidar_repulse_e = 0.0

        # Peer drone positions in World NED for decentralized inter-drone collision avoidance
        self.peer_positions: list[tuple[float, float, float]] = []

        # Current target setpoint in World NED (updated by commands)
        self.target_x = 0.0  # North (meters, NED)
        self.target_y = 0.0  # East (meters, NED)
        self.target_z = -abs(default_alt_agl)  # Down (meters, NED — negative = up)
        self.target_yaw = float('nan')  # NaN = auto-face velocity/target
        self.target_speed = 4.0
        self._tick_dt = 0.04

        # Orbit state
        self.orbiting = False
        self.orbit_center_x = 0.0
        self.orbit_center_y = 0.0
        self.orbit_radius = 15.0
        self.orbit_speed = 3.5
        self.orbit_alt_z = -abs(default_alt_agl)
        self.orbit_angle = (drone_id * math.pi * 0.5)  # Stagger 4 drones by 90 deg

        # Square pattern state
        self.square_active = False
        self.square_waypoints = []  # List of (x, y) NED waypoints
        self.square_wp_index = 0
        self.square_alt_z = -abs(default_alt_agl)
        self.square_speed = 2.5
        self.square_threshold = 2.0  # meters — how close before moving to next wp

        # Setpoint smoothing: the drone follows a jerk-limited carrot, not the raw target (steady camera)
        self.carrot = JerkLimitedCarrot(vmax_xy=2.5, amax_xy=0.8, jmax=1.2)    # gentle cruise: a steady camera beats a fast one
        self.yaw_smoother = YawSmoother(rate_max_deg=6.0, acc_max_deg=10.0)   # slow, so the camera never whips
        self._hdg_vec = np.zeros(2)          # low-passed direction of travel (unit-ish vector)
        self.current_yaw = float('nan')
        self._last_tick_t = None
        # Gazebo often runs slower than real time. The smoother must advance in SIMULATION time, otherwise a
        # carrot limited to 0.8 m/s^2 in wall time is 3x harsher in sim time when the real-time factor is 0.58.
        self.rtf = 1.0
        self._clk_sim = None
        self._clk_wall = None
        self._facing_target = False

    def update_peers(self, peers_ned: list[tuple[float, float, float]]):
        """Update positions of other swarm drones for inter-drone repulsion."""
        self.peer_positions = list(peers_ned)

    def apply_obstacle_avoidance(self, n: float, e: float, d: float) -> tuple[float, float, float]:
        """Ensure (n, e, d) never enters a building wall or collides with another swarm drone."""
        alt_agl = abs(d)
        safe_n, safe_e, safe_d = float(n), float(e), float(d)

        # Keep minimum flight altitude of 5.0m AGL during horizontal flight
        if math.hypot(safe_n, safe_e) > 2.5 and alt_agl < 5.0:
            safe_d = -5.0
            alt_agl = 5.0

        for obs_n, obs_e, obs_rad, obs_h in BUILDING_OBSTACLES_NED:
            dn = safe_n - obs_n
            de = safe_e - obs_e
            dist = math.hypot(dn, de)
            # If below building roof + 4.5m safety buffer and inside horizontal safety radius:
            if alt_agl < (obs_h + 4.5) and dist < obs_rad:
                if dist < 1e-3:
                    ang = math.atan2(-obs_e, -obs_n)
                else:
                    ang = math.atan2(de, dn)
                safe_n = obs_n + (obs_rad + 1.5) * math.cos(ang)
                safe_e = obs_e + (obs_rad + 1.5) * math.sin(ang)

            # Also check if straight line from current drone position to target crosses building
            seg_dn = safe_n - self.current_x
            seg_de = safe_e - self.current_y
            seg_len2 = seg_dn * seg_dn + seg_de * seg_de
            if seg_len2 > 1.0 and alt_agl < (obs_h + 4.0):
                t = max(
                    0.0,
                    min(
                        1.0,
                        ((obs_n - self.current_x) * seg_dn + (obs_e - self.current_y) * seg_de) / seg_len2,
                    ),
                )
                closest_n = self.current_x + t * seg_dn
                closest_e = self.current_y + t * seg_de
                if math.hypot(closest_n - obs_n, closest_e - obs_e) < (obs_rad - 1.0):
                    safe_d = min(safe_d, -(obs_h + 4.5))
                    alt_agl = abs(safe_d)

        # Apply decentralized inter-drone repulsion if any peer drone is within 6.5m horizontally
        min_drone_sep = 6.5
        for pn, pe, _pd in self.peer_positions:
            ddn = self.current_x - pn
            dde = self.current_y - pe
            pdist = math.hypot(ddn, dde)
            if 0.05 < pdist < min_drone_sep:
                push = ((min_drone_sep - pdist) / min_drone_sep) * 3.0
                safe_n += (ddn / pdist) * push
                safe_e += (dde / pdist) * push

        # Apply live 3D LiDAR close-proximity repulsion if any wall/object is near
        safe_n += self.lidar_repulse_n
        safe_e += self.lidar_repulse_e
        return safe_n, safe_e, safe_d

    def update_lidar_repulsion(self, repulse_n: float, repulse_e: float):
        """Update real-time repulsive vector from 3D LiDAR point cloud."""
        self.lidar_repulse_n = max(-4.0, min(4.0, float(repulse_n)))
        self.lidar_repulse_e = max(-4.0, min(4.0, float(repulse_e)))

    def update_sim_time(self, sim_s: float):
        """Feed PX4's odometry timestamp (seconds, simulation clock) to estimate the real-time factor."""
        wall = time.monotonic()
        if self._clk_sim is None:
            self._clk_sim, self._clk_wall = float(sim_s), wall
            return
        if wall - self._clk_wall >= 0.5:
            inst = (float(sim_s) - self._clk_sim) / (wall - self._clk_wall)
            if math.isfinite(inst) and inst > 0.0:
                self.rtf = float(np.clip(0.7 * self.rtf + 0.3 * inst, 0.05, 1.5))
            self._clk_sim, self._clk_wall = float(sim_s), wall

    def update_yaw(self, yaw: float):
        self.current_yaw = float(yaw)

    def update_position(self, x: float, y: float, z: float):
        """Update the drone's current World NED position for waypoint tracking."""
        self.current_x = float(x)
        self.current_y = float(y)
        self.current_z = float(z)
        if self.square_active and self.square_waypoints:
            wp = self.square_waypoints[self.square_wp_index]
            dist = math.sqrt((x - wp[0]) ** 2 + (y - wp[1]) ** 2)
            if dist < self.square_threshold:
                self.square_wp_index = (self.square_wp_index + 1) % len(self.square_waypoints)
                wp = self.square_waypoints[self.square_wp_index]
                self.target_x = wp[0]
                self.target_y = wp[1]

    def set_home(self, lat: float, lon: float, alt: float):
        """Set the home/reference position for GPS-to-local conversions."""
        self.home_lat = lat
        self.home_lon = lon
        self.home_alt = alt
        self.home_set = True

    def gps_to_local(self, lat: float, lon: float, alt: float):
        """Convert GPS coordinates to local NED frame relative to home."""
        if not self.home_set:
            return 0.0, 0.0, -(alt - self.home_alt)

        dlat = lat - self.home_lat
        dlon = lon - self.home_lon

        north_m = dlat * 111_139.0
        east_m = dlon * 111_139.0 * math.cos(math.radians(self.home_lat))
        down_m = -(alt - self.home_alt)

        return north_m, east_m, down_m

    def process_command(self, cmd: dict) -> list:
        """Translate an LLM or Swarm command dict into a list of PX4 messages to publish."""
        action = cmd.get('action', 'hold')
        messages = []

        # ── Arm + OFFBOARD mode ──────────────────────────────────────────
        if action in ('arm_offboard', 'takeoff', 'take_off'):
            self.orbiting = False
            self.square_active = False
            if 'alt' in cmd or 'altitude' in cmd or 'z' in cmd:
                val = cmd.get('alt', cmd.get('altitude', cmd.get('z', self.default_alt_agl)))
                self.target_z = -abs(float(val))
            messages.extend(self._arm_and_offboard())

        # ── Direct NED position (primary navigation primitive) ───────────
        elif action == 'position_ned':
            raw_x = float(cmd.get('x', self.target_x))
            raw_y = float(cmd.get('y', self.target_y))
            if 'alt' in cmd or 'altitude' in cmd:
                val = cmd.get('alt', cmd.get('altitude'))
                raw_z = -abs(float(val))
            else:
                z_val = float(cmd.get('z', self.target_z))
                raw_z = -abs(z_val) if z_val != 0.0 else -abs(self.default_alt_agl)
            self.target_x, self.target_y, self.target_z = self.apply_obstacle_avoidance(
                raw_x, raw_y, raw_z
            )
            yaw = cmd.get('yaw_rad')
            if yaw is not None:
                self.target_yaw = float(yaw)
            else:
                # No commanded heading: let the setpoint smoother choose it from the direction of travel.
                # (This used to set yaw to the bearing of every NEW target, so each re-allocation by the
                # swarm brain, about every 8 s, spun the camera round to face the new goal.)
                self.target_yaw = float('nan')
            self.orbiting = False
            self.square_active = False
            messages.extend(self._arm_and_offboard())

        # ── GPS-based goto (legacy alias) ────────────────────────────────
        elif action == 'goto':
            lat = cmd.get('lat', self.home_lat)
            lon = cmd.get('lon', self.home_lon)
            alt = cmd.get('alt', abs(self.target_z))
            n, e, d = self.gps_to_local(lat, lon, alt)
            self.target_x, self.target_y, self.target_z = self.apply_obstacle_avoidance(n, e, d)
            self.orbiting = False
            self.square_active = False
            messages.extend(self._arm_and_offboard())

        # ── Circular orbit ───────────────────────────────────────────────
        elif action == 'orbit':
            req_radius = float(cmd.get('radius', 15.0))
            self.orbit_speed = min(4.0, max(1.5, float(cmd.get('speed', 3.5))))
            self.square_active = False

            if 'cx' in cmd or 'x' in cmd or 'alt_z' in cmd or 'z' in cmd:
                self.orbit_center_x = float(cmd.get('cx', cmd.get('x', 0.0)))
                self.orbit_center_y = float(cmd.get('cy', cmd.get('y', 0.0)))
                raw_z = cmd.get('alt_z', cmd.get('z', cmd.get('alt', cmd.get('altitude', -self.default_alt_agl))))
                self.orbit_alt_z = -abs(float(raw_z)) if float(raw_z) != 0.0 else -abs(self.default_alt_agl)
            else:
                lat = cmd.get('lat', self.home_lat)
                lon = cmd.get('lon', self.home_lon)
                alt = cmd.get('alt', cmd.get('altitude', abs(self.target_z)))
                n, e, _ = self.gps_to_local(lat, lon, alt)
                self.orbit_center_x = n
                self.orbit_center_y = e
                self.orbit_alt_z = -abs(float(alt))

            if abs(self.orbit_alt_z) < 16.0:
                max_center_offset = 6.0
                c_dist = math.hypot(self.orbit_center_x, self.orbit_center_y)
                if c_dist > max_center_offset:
                    scale = max_center_offset / c_dist
                    self.orbit_center_x *= scale
                    self.orbit_center_y *= scale
                self.orbit_radius = max(8.0, min(16.0, req_radius))
                self.orbit_alt_z = min(-8.5, self.orbit_alt_z)
            else:
                self.orbit_radius = max(8.0, min(35.0, req_radius))

            self.target_z = self.orbit_alt_z
            if 'phase_rad' in cmd:
                self.orbit_angle = float(cmd['phase_rad'])
            else:
                self.orbit_angle = math.atan2(
                    self.current_y - self.orbit_center_y,
                    self.current_x - self.orbit_center_x,
                )
            self.orbiting = True
            messages.extend(self._arm_and_offboard())

        # ── Square / rectangular survey ──────────────────────────────────
        elif action in ('square_survey', 'square'):
            side = min(32.0, float(cmd.get('side', 22.0)))
            speed = float(cmd.get('speed', 3.0))
            if 'alt_z' in cmd:
                alt_z = float(cmd['alt_z'])
            else:
                alt_z = -abs(float(cmd.get('alt', abs(self.target_z))))
            alt_z = min(-8.5, alt_z)
            cx = float(cmd.get('cx', 0.0))
            cy = float(cmd.get('cy', 0.0))
            half = side / 2.0
            wps = [
                (cx + half, cy + half),
                (cx + half, cy - half),
                (cx - half, cy - half),
                (cx - half, cy + half),
            ]
            # Stagger starting corner by drone_id so 4 drones sweep all 4 sides in parallel
            rot = self.drone_id % 4
            self.square_waypoints = wps[rot:] + wps[:rot]
            self.square_wp_index = 0
            self.square_alt_z = alt_z
            self.square_speed = speed
            self.square_active = True
            self.orbiting = False
            self.target_x = self.square_waypoints[0][0]
            self.target_y = self.square_waypoints[0][1]
            self.target_z = self.square_alt_z
            messages.extend(self._arm_and_offboard())

        # ── Camera ROI ───────────────────────────────────────────────────
        elif action in ('look_at_gps', 'look_at'):
            lat = cmd.get('lat', self.home_lat)
            lon = cmd.get('lon', self.home_lon)
            alt = cmd.get('alt', 0.0)
            msg = self._make_vehicle_command(
                MAV_CMD_DO_SET_ROI_LOCATION,
                param5=lat,
                param6=lon,
                param7=alt,
            )
            messages.append(('vehicle_command', msg))

        # ── Speed / heading ──────────────────────────────────────────────
        elif action == 'set_speed':
            self.target_speed = float(cmd.get('speed', 4.0))
            self.orbit_speed = min(4.5, self.target_speed)

        elif action == 'set_heading':
            heading_deg = cmd.get('heading_deg', cmd.get('heading', 0.0))
            self.target_yaw = math.radians(float(heading_deg))

        # ── Land / RTL ───────────────────────────────────────────────────
        elif action == 'land':
            self.orbiting = False
            self.square_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_LAND)
            messages.append(('vehicle_command', msg))

        elif action == 'rtl':
            self.orbiting = False
            self.square_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_RETURN_TO_LAUNCH)
            messages.append(('vehicle_command', msg))

        # ── Hold: stop all patterns, brain_node sets position ────────────
        elif action == 'hold':
            self.orbiting = False
            self.square_active = False

        return messages

    def get_setpoint_tick(self) -> TrajectorySetpoint:
        """Generate the current TrajectorySetpoint (in PX4 local frame) for the 10Hz publish loop."""
        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)
        now = time.monotonic()
        wall_dt = 0.04 if self._last_tick_t is None else min(0.5, max(0.005, now - self._last_tick_t))
        self._tick_dt = min(0.25, max(0.002, wall_dt * self.rtf))      # simulation-time step
        self._last_tick_t = now

        # Safe vertical takeoff climb: if drone is currently low (< 4.2m AGL) and needs to fly higher,
        # climb straight up first before translating horizontally so it never clips cars/rubble!
        if abs(self.current_z) < 4.2 and abs(self.target_z) >= 4.5:
            world_n = self.current_x
            world_e = self.current_y
            world_d = min(self.target_z, -7.0)
            yaw = self.target_yaw
        elif self.square_active:
            wp = self.square_waypoints[self.square_wp_index]
            world_n, world_e, world_d = self.apply_obstacle_avoidance(wp[0], wp[1], self.square_alt_z)
            yaw = math.atan2(world_e - self.current_y, world_n - self.current_x)
        elif self.orbiting:
            dt = self._tick_dt
            angular_vel = self.orbit_speed / max(self.orbit_radius, 1.0)
            self.orbit_angle += angular_vel * dt

            raw_n = self.orbit_center_x + self.orbit_radius * math.cos(self.orbit_angle)
            raw_e = self.orbit_center_y + self.orbit_radius * math.sin(self.orbit_angle)
            world_n, world_e, world_d = self.apply_obstacle_avoidance(raw_n, raw_e, self.orbit_alt_z)
            self.target_x = world_n
            self.target_y = world_e
            self.target_z = world_d

            # Point yaw inward toward orbit center so camera scans the disaster plaza
            dx = self.orbit_center_x - world_n
            dy = self.orbit_center_y - world_e
            yaw = math.atan2(dy, dx)
        else:
            # Smooth velocity-capped carrot step toward (target_x, target_y) so flight is smooth
            world_n, world_e, world_d = self.apply_obstacle_avoidance(
                self.target_x, self.target_y, self.target_z
            )
            dn = world_n - self.current_x
            de = world_e - self.current_y
            dist = math.hypot(dn, de)
            # Heading policy for the default waypoint flight: an explicit commanded yaw wins; otherwise
            # turn toward the direction of travel only as fast as a slow low-pass of the velocity allows.
            # (The old code snapped yaw to the bearing of the target; the allocator flips targets every ~8 s,
            # so the camera was whipped through up to 180 degrees each time.)
            yaw = self.target_yaw

        # Smooth: the setpoint is the carrot (limited speed / acceleration / jerk), with velocity and
        # acceleration feed-forward, instead of the raw target that PX4 would chase at full authority.
        cur = np.array([self.current_x, self.current_y, self.current_z])
        if not self.carrot.ready or float(np.linalg.norm(self.carrot.p - cur)) > 6.0:
            self.carrot.reset(cur)                    # first tick, or the drone was moved/landed: re-sync
        p, v, a = self.carrot.step([world_n, world_e, world_d], self._tick_dt)
        if not self.yaw_smoother.ready:
            self.yaw_smoother.reset(self.current_yaw if math.isfinite(self.current_yaw) else 0.0)
        speed_xy = math.hypot(float(v[0]), float(v[1]))
        if speed_xy > 1.2:
            alpha = min(1.0, self._tick_dt / 6.0)
            self._hdg_vec = (1 - alpha) * self._hdg_vec + alpha * np.array([v[0], v[1]]) / speed_xy
        if yaw is not None and math.isfinite(yaw):
            want_yaw = yaw
        elif float(np.linalg.norm(self._hdg_vec)) > 0.8:        # only a CONSISTENT direction of travel turns the camera
            want_yaw = math.atan2(float(self._hdg_vec[1]), float(self._hdg_vec[0]))
        else:
            want_yaw = float('nan')                    # hold the current heading
        yaw_s, yaw_rate = self.yaw_smoother.step(want_yaw, self._tick_dt)

        # Convert World NED setpoint to this drone's PX4 local EKF2 frame
        msg.timestamp = int(time.time() * 1e6)
        msg.position[0] = float(p[0] - self.frame_offset_x)
        msg.position[1] = float(p[1] - self.frame_offset_y)
        msg.position[2] = float(p[2])
        nan = float('nan')
        msg.velocity = [float(v[0]), float(v[1]), float(v[2])]
        msg.acceleration = [float(a[0]), float(a[1]), float(a[2])]
        msg.jerk = [nan, nan, nan]
        msg.yaw = float(yaw_s)
        msg.yawspeed = float(yaw_rate)
        return msg

    def get_offboard_control_mode(self) -> OffboardControlMode:
        """Generate an OffboardControlMode message (position control)."""
        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        return msg

    def _arm_and_offboard(self) -> list:
        """Generate messages to arm the drone and switch to OFFBOARD mode."""
        messages = []

        arm_cmd = self._make_vehicle_command(
            MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        messages.append(('vehicle_command', arm_cmd))

        mode_cmd = self._make_vehicle_command(
            MAV_CMD_DO_SET_MODE,
            param1=1.0,
            param2=PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        )
        messages.append(('vehicle_command', mode_cmd))

        return messages

    def _make_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> VehicleCommand:
        """Create a VehicleCommand message addressed to this PX4 instance."""
        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param3 = param3
        msg.param4 = param4
        msg.param5 = param5
        msg.param6 = param6
        msg.param7 = param7
        msg.target_system = 0  # 0 = accepted by any MAV_SYS_ID on this drone's namespace
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        return msg

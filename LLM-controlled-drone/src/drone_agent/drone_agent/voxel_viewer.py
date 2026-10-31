"""3-D voxel viewer for the swarm's occupancy map (software renderer on numpy + OpenCV, no new dependency).

It replaces the old page-2 painter that stamped a fixed 7-pixel square for every voxel. Differences:
  * every occupied voxel is a real cube whose edge is the map resolution (x voxel scale), in perspective;
  * only EXPOSED faces are generated (faces touching another occupied voxel are dropped), then back-face
    culled, depth-sorted and drawn, so adjacent voxels merge into walls with crisp per-voxel edges;
  * directional + ambient lighting, per-face ambient occlusion from neighbouring voxels, distance fog,
    and ground-plane shadows / AO / unknown-vs-free shading rendered as a texture and projected onto the
    ground with a per-pixel ray cast (so any camera angle works);
  * colour by height / occupancy probability / age / class, with highlights for recently updated voxels
    and newly detected obstacles;
  * rovers, drones, survivors, routes, trails and detection-transmission animations drawn in the SAME
    world frame as the voxels (north, east, height above ground);
  * an orbit/pan/zoom camera with smooth motion, preset views and focus-on-entity.

Viewer settings never write to the VoxelMap: the viewer only reads snapshots.

Geometry (the big arrays) is rebuilt on a background thread when the map or the filters change, so the
GUI keeps its frame rate while exploration adds thousands of voxels.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

# --------------------------------------------------------------------------- palette (BGR)
BG_TOP = np.array([34, 24, 18], np.float32)          # command-centre dark blue/grey
BG_BOT = np.array([22, 16, 12], np.float32)
UNKNOWN_GROUND = np.array([28, 22, 18], np.float32)
FREE_GROUND = np.array([66, 78, 52], np.float32)
DRONE_COLORS = [(255, 200, 40), (60, 220, 90), (60, 90, 255), (240, 90, 220)]     # BGR, UAV-1..4
ROVER_COLORS = [(0, 215, 255), (255, 150, 30)]                                    # Rover 1 amber, Rover 2 azure
SURVIVOR_COLOR = (60, 80, 255)
LIGHT_DIR = np.array([-0.45, 0.35, 0.82], np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _rgb(r, g, b):
    return np.array([b, g, r], np.float32)


# --------------------------------------------------------------------------- options
@dataclass
class ViewOptions:
    show_occupied: bool = True
    show_ground_voxels: bool = True        # voxels below `ground_band`
    show_obstacle_voxels: bool = True      # voxels above it
    show_free: bool = False                # free-space voxels seen along sensor rays
    show_unknown: bool = True              # darken never-observed ground
    show_recent: bool = True               # highlight recently updated voxels + recently explored ground
    show_new_obstacles: bool = True
    show_coverage: bool = False            # per-agent exploration coverage tint
    show_shadows: bool = True
    show_ao: bool = True
    show_edges: bool = True
    show_trails: bool = True
    show_routes: bool = True
    show_survivors: bool = True
    show_links: bool = True
    occ_threshold: float = 0.5             # occupancy probability needed to draw a voxel
    z_min: float = 0.0                     # vertical height filter (m above ground)
    z_max: float = 60.0
    opacity: float = 1.0
    voxel_scale: float = 1.0               # drawn cube edge = resolution * scale
    color_mode: str = "height"             # height | occupancy | age | class
    ground_band: float = 0.5
    recent_window: float = 6.0
    new_window: float = 12.0
    render_scale: float = 0.7              # solid voxels + ground are drawn at this fraction of the pane, then upscaled
    face_budget: int = 10000               # max faces drawn per frame (farthest dropped first)
    model_scale: float = 2.2               # rovers/drones are tiny at map scale: draw them larger

    def geometry_key(self):
        return (self.show_occupied, self.show_ground_voxels, self.show_obstacle_voxels, self.occ_threshold,
                round(self.z_min, 2), round(self.z_max, 2), round(self.voxel_scale, 2), self.color_mode,
                self.show_ao, self.ground_band, self.show_free)


# --------------------------------------------------------------------------- camera
class OrbitCamera:
    """Orbit camera around a target. Azimuth is measured from north toward east, elevation up from the ground."""

    def __init__(self, target=(0.0, 0.0, 4.0), az=215.0, el=32.0, dist=95.0, fov=52.0):
        self.home = dict(target=np.array(target, float), az=az, el=el, dist=dist)
        self.target = np.array(target, float)
        self.az, self.el, self.dist, self.fov = az, el, dist, fov
        self.goal = dict(target=self.target.copy(), az=az, el=el, dist=dist)
        self.smooth = 9.0                # 1/seconds: how fast the camera glides to its goal
        self._pose_cache = None
        self.auto_orbit = False

    # user input -----------------------------------------------------------------
    def orbit(self, d_az: float, d_el: float):
        self.auto_orbit = False
        self.goal["az"] = (self.goal["az"] + d_az) % 360.0
        self.goal["el"] = float(np.clip(self.goal["el"] + d_el, 4.0, 89.0))

    def pan(self, dx_px: float, dy_px: float, height_px: float):
        self.auto_orbit = False
        scale = 2.0 * self.goal["dist"] * math.tan(math.radians(self.fov) / 2) / max(1.0, height_px)
        right, up, fwd = self._basis(self.goal["az"], self.goal["el"])
        ground_fwd = np.array([fwd[0], fwd[1], 0.0])
        norm = np.linalg.norm(ground_fwd)
        ground_fwd = ground_fwd / norm if norm > 1e-6 else np.array([1.0, 0, 0])
        self.goal["target"] = self.goal["target"] - right * dx_px * scale + ground_fwd * dy_px * scale
        self.goal["target"][2] = max(0.0, self.goal["target"][2])

    def zoom(self, factor: float):
        self.goal["dist"] = float(np.clip(self.goal["dist"] / factor, 8.0, 260.0))

    def reset(self):
        self.auto_orbit = False
        self.goal.update(target=self.home["target"].copy(), az=self.home["az"], el=self.home["el"],
                         dist=self.home["dist"])

    def preset(self, name: str):
        self.auto_orbit = False
        if name == "top":
            self.goal.update(el=89.0, az=180.0, dist=max(self.goal["dist"], 110.0))
        elif name == "side":
            self.goal.update(el=6.0, az=180.0, dist=max(self.goal["dist"], 105.0))
        elif name == "iso":
            self.goal.update(el=35.0, az=225.0)
        elif name == "orbit":
            self.auto_orbit = True

    def focus(self, point, dist: float | None = None):
        self.auto_orbit = False
        self.goal["target"] = np.array([point[0], point[1], max(0.0, point[2] if len(point) > 2 else 0.0)], float)
        if dist is not None:
            self.goal["dist"] = dist

    def update(self, dt: float):
        self._pose_cache = None
        if self.auto_orbit:
            self.goal["az"] = (self.goal["az"] + 14.0 * dt) % 360.0
        k = 1.0 - math.exp(-self.smooth * dt)
        self.target = self.target + (self.goal["target"] - self.target) * k
        d_az = (self.goal["az"] - self.az + 180.0) % 360.0 - 180.0
        self.az = (self.az + d_az * k) % 360.0
        self.el += (self.goal["el"] - self.el) * k
        self.dist += (self.goal["dist"] - self.dist) * k

    # projection -----------------------------------------------------------------
    @staticmethod
    def _basis(az_deg: float, el_deg: float):
        az, el = math.radians(az_deg), math.radians(el_deg)
        cam_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        fwd = -cam_dir
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(up_w, fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        return right, up, fwd

    def pose(self):
        if self._pose_cache is None:
            right, up, fwd = self._basis(self.az, self.el)
            self._pose_cache = (self.target - fwd * self.dist, right, up, fwd)
        return self._pose_cache

    def project(self, pts: np.ndarray, W: int, H: int):
        """pts (...,3) world (north, east, up) -> (u, v, depth) arrays."""
        pos, right, up, fwd = self.pose()
        f = (H / 2.0) / math.tan(math.radians(self.fov) / 2)
        d = pts - pos
        xc, yc, zc = d @ right, d @ up, d @ fwd
        z = np.maximum(zc, 1e-3)
        return W / 2.0 + f * xc / z, H / 2.0 - f * yc / z, zc


class FirstPersonCamera(OrbitCamera):
    """Camera fixed to a vehicle: eye position + heading (0 = north, toward east) + pitch."""

    def __init__(self, fov: float = 78.0):
        super().__init__(fov=fov)
        self.eye = np.array([0.0, 0.0, 0.95])
        self.yaw = 0.0
        self.pitch = math.radians(-5.0)

    def set_pose(self, n: float, e: float, yaw: float, height: float = 0.95, pitch_deg: float = -5.0):
        self.eye = np.array([n, e, height], float)
        self.yaw = float(yaw)
        self.pitch = math.radians(pitch_deg)
        self._pose_cache = None

    def update(self, dt: float):                       # no smoothing: the camera is bolted to the rover
        self._pose_cache = None

    def pose(self):
        if self._pose_cache is None:
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            fwd = np.array([math.cos(self.yaw) * cp, math.sin(self.yaw) * cp, sp])
            right = np.cross([0.0, 0.0, 1.0], fwd)
            right /= np.linalg.norm(right)
            up = np.cross(fwd, right)
            self._pose_cache = (self.eye.copy(), right, up, fwd)
        return self._pose_cache


# --------------------------------------------------------------------------- scene description
@dataclass
class Scene:
    drones: list = field(default_factory=list)       # {id, name, n, e, alt, yaw, trail:[(n,e,alt)]}
    rovers: list = field(default_factory=list)       # {id, name, n, e, yaw, state, wheel, trail:[(n,e)], route:[(n,e)], mission}
    survivors: list = field(default_factory=list)    # {id, name, n, e, conf, drone_id, state, t_detect, radius}
    transmissions: list = field(default_factory=list)  # {kind, t0, drone_n, drone_e, drone_alt, rover_n, rover_e, target}
    links: list = field(default_factory=list)        # [(i, j)] drone comm links
    hub: tuple = (0.0, 0.0)                          # coordinator position
    stats: dict = field(default_factory=dict)
    now: float = 0.0
    source_label: str = ""


# --------------------------------------------------------------------------- geometry cache
@dataclass
class Geometry:
    version: int = -1
    key: tuple = ()
    verts: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3), np.float32))
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))   # shaded base colour (BGR 0-255)
    vox: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))              # voxel index per face
    t_occ: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_last: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_vox: int = 0
    free_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    ground_tex: np.ndarray | None = None
    ground_meta: dict = field(default_factory=dict)
    build_ms: float = 0.0


def _ramp(x, stops):
    """Piecewise-linear colour ramp. x in [0,1], stops = [(pos, rgb), ...] -> BGR float32 (N,3)."""
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([[s[1][2], s[1][1], s[1][0]] for s in stops], np.float32)
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, pos, cols[:, c])
    return out


def voxel_colors(mode, z_mid, prob, age, res, opts: ViewOptions):
    if mode == "occupancy":
        return _ramp((prob - 0.5) / 0.5, [(0.0, (172, 176, 182)), (0.45, (96, 150, 205)), (1.0, (28, 84, 232))])
    if mode == "age":
        return _ramp(1.0 - np.clip(age / 120.0, 0, 1), [(0.0, (82, 88, 100)), (0.6, (110, 160, 200)), (1.0, (120, 235, 255))])
    if mode == "class":
        cols = np.empty((len(z_mid), 3), np.float32)
        cols[:] = _rgb(120, 130, 140)
        cols[z_mid < opts.ground_band] = _rgb(150, 125, 95)           # ground-level clutter
        cols[(z_mid >= opts.ground_band) & (z_mid < 2.5)] = _rgb(196, 170, 120)   # vehicles, rubble
        cols[z_mid >= 2.5] = _rgb(112, 150, 205)                       # building mass
        return cols
    # height (default): slate -> concrete -> sky blue
    return _ramp(z_mid / 14.0, [(0.0, (92, 100, 112)), (0.35, (128, 142, 160)), (0.7, (150, 188, 220)), (1.0, (196, 226, 246))])


def build_geometry(vm, opts: ViewOptions) -> Geometry:
    """Exposed-face mesh of the occupied voxels (pure function of the map snapshot and the filters)."""
    t0 = time.perf_counter()
    res = vm.res
    g = Geometry(version=vm.version, key=opts.geometry_key())
    ijk, prob, t_occ, t_last, _ = vm.occupied_arrays(opts.occ_threshold)
    keep = np.ones(len(ijk), bool)
    if len(ijk):
        zc = (ijk[:, 2] + 0.5) * res
        keep &= (zc >= opts.z_min) & (zc <= opts.z_max)
        if not opts.show_ground_voxels:
            keep &= zc >= opts.ground_band
        if not opts.show_obstacle_voxels:
            keep &= zc < opts.ground_band
    if not opts.show_occupied:
        keep[:] = False
    ijk, prob, t_occ, t_last = ijk[keep], prob[keep], t_occ[keep], t_last[keep]
    g.n_vox = len(ijk)
    if opts.show_free:
        fk, _ft = vm.free3d_array()
        if len(fk):
            g.free_pts = ((fk.astype(np.float32) + 0.5) * (2 * res)).astype(np.float32)
    g.ground_tex, g.ground_meta = _ground_texture(vm, opts, ijk, t_last)
    if g.n_vox == 0:
        return g

    # dense occupancy grid with a 1-voxel pad for O(1) neighbour lookups
    lo = ijk.min(axis=0) - 2
    hi = ijk.max(axis=0) + 3
    dims = (hi - lo).astype(int)
    grid = np.zeros(dims, bool)
    loc = (ijk - lo).astype(int)
    grid[loc[:, 0], loc[:, 1], loc[:, 2]] = True

    half = 0.5 * res * opts.voxel_scale
    z_mid = (ijk[:, 2] + 0.5) * res
    centers = (ijk.astype(np.float32) + 0.5) * res
    base = voxel_colors(opts.color_mode, z_mid, prob, np.zeros(len(ijk)), res, opts)

    verts_l, cent_l, norm_l, col_l, vox_l = [], [], [], [], []
    for d_idx, d in enumerate(AXES):
        d = np.array(d)
        nb = loc + d
        exposed = ~grid[nb[:, 0], nb[:, 1], nb[:, 2]]
        idx = np.nonzero(exposed)[0]
        if not len(idx):
            continue
        a = int(np.nonzero(d)[0][0])
        b, c = [x for x in range(3) if x != a]
        corner = np.zeros((4, 3), np.float32)
        for q, (sb, sc) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            corner[q, a] = d[a]
            corner[q, b] = sb
            corner[q, c] = sc
        cen = centers[idx]
        verts_l.append(cen[:, None, :] + corner[None, :, :] * half)
        cent_l.append(cen + d.astype(np.float32) * half)
        norm_l.append(np.broadcast_to(d.astype(np.float32), (len(idx), 3)))
        vox_l.append(idx.astype(np.int32))
        shade = 0.42 + 0.58 * max(0.0, float(np.dot(d, LIGHT_DIR)))
        if opts.show_ao:
            ao = np.zeros(len(idx), np.float32)
            n0 = loc[idx] + d
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if db == 0 and dc == 0:
                        continue
                    off = np.zeros(3, int)
                    off[b], off[c] = db, dc
                    q = n0 + off
                    ao += grid[q[:, 0], q[:, 1], q[:, 2]]
            f = (1.0 - 0.085 * ao)[:, None]
        else:
            f = 1.0
        col_l.append(np.clip(base[idx] * shade * f, 0, 255))
    g.verts = np.concatenate(verts_l).astype(np.float32)
    g.centers = np.concatenate(cent_l).astype(np.float32)
    g.normals = np.concatenate(norm_l).astype(np.float32)
    g.colors = np.concatenate(col_l).astype(np.float32)
    g.vox = np.concatenate(vox_l)
    g.t_occ, g.t_last = t_occ, t_last
    g.build_ms = (time.perf_counter() - t0) * 1000.0
    return g


def _ground_texture(vm, opts: ViewOptions, ijk, t_last):
    """Ground-plane texture in map coordinates: unknown/free, explored-by tint, shadows, AO, grid."""
    n = vm.n
    gl, seen, gt = vm.ground_snapshot()
    tex = np.empty((n, n, 3), np.float32)
    known = gl > 0.0
    tex[:] = UNKNOWN_GROUND if opts.show_unknown else FREE_GROUND * 0.55
    tex[known] = FREE_GROUND * (0.78 + 0.22 * np.clip(gl[known] / 3.0, 0, 1))[:, None]
    meta = {"known_frac": float(known.mean())}
    if opts.show_coverage:
        tint = np.zeros((n, n, 3), np.float32)
        cnt = np.zeros((n, n), np.float32)
        for bit in range(6):
            m = ((seen >> bit) & 1).astype(np.float32)
            col = np.array(DRONE_COLORS[bit] if bit < 4 else ROVER_COLORS[bit - 4], np.float32)
            tint += m[..., None] * col
            cnt += m
        mask = cnt > 0
        tint[mask] /= cnt[mask][:, None]
        tex[mask] = tex[mask] * 0.45 + tint[mask] * 0.55
    # column heights -> shadow and ambient occlusion
    if len(ijk):
        off = int(round(vm.half / vm.res))
        rows, cols = ijk[:, 0] + off, ijk[:, 1] + off
        ok = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        hmap = np.zeros((n, n), np.float32)
        np.maximum.at(hmap, (rows[ok], cols[ok]), (ijk[ok, 2] + 1) * vm.res)
        occ = hmap > 0
        if opts.show_ao:
            blur = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 1.6)
            tex *= (1.0 - 0.55 * np.clip(blur * 1.8, 0, 1) * (~occ))[..., None]
        if opts.show_shadows:
            shadow = np.zeros((n, n), bool)
            ldir = -LIGHT_DIR[:2] / max(1e-6, np.linalg.norm(LIGHT_DIR[:2]))   # shadow points away from the light
            tan_e = math.tan(math.asin(float(LIGHT_DIR[2])))
            for s in range(1, 34):
                dr, dc = int(round(ldir[0] * s)), int(round(ldir[1] * s))
                lim = s * vm.res * 0.707 * tan_e
                src = hmap > lim
                r0, r1 = max(0, dr), min(n, n + dr)
                c0, c1 = max(0, dc), min(n, n + dc)
                shadow[r0:r1, c0:c1] |= src[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            shadow &= ~occ
            tex[shadow] *= 0.58
    if opts.show_recent:
        rec = (vm.now() - gt) < opts.recent_window
        tex[rec] = tex[rec] * 0.78 + np.array([120, 90, 30], np.float32) * 0.22
    # 10 m grid lines
    step = int(round(10.0 / vm.res))
    tex[::step, :] = tex[::step, :] * 0.8 + 18
    tex[:, ::step] = tex[:, ::step] * 0.8 + 18
    return np.clip(tex, 0, 255).astype(np.uint8), meta


# --------------------------------------------------------------------------- procedural models
def _make_templates():
    corners = np.zeros((6, 4, 3), np.float32)
    for k, d in enumerate(AXES):
        d = np.array(d)
        a = int(np.nonzero(d)[0][0])
        b, c = [x for x in range(3) if x != a]
        for q, (sb, sc) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            corners[k, q, a], corners[k, q, b], corners[k, q, c] = d[a], sb, sc
    return corners, np.array(AXES, np.float32)


_BOX_CORNERS, _BOX_NORMALS = _make_templates()


def box_faces(center, size, yaw, color, up=0.0):
    """Faces of an oriented box (yaw about the vertical). Returns verts (6,4,3), centers, normals, colors(BGR float)."""
    cn, ce, cz = center
    half = np.array(size, np.float32) / 2.0
    cy, sn = math.cos(yaw), math.sin(yaw)
    rot_t = np.array([[cy, sn, 0], [-sn, cy, 0], [0, 0, 1]], np.float32)       # transpose of the yaw rotation
    origin = np.array([cn, ce, cz + up], np.float32)
    verts = (_BOX_CORNERS * half) @ rot_t + origin
    cents = (_BOX_NORMALS * half) @ rot_t + origin
    norms = _BOX_NORMALS @ rot_t
    shade = 0.42 + 0.58 * np.clip(norms @ LIGHT_DIR, 0, 1)
    cols = np.clip(np.array(color, np.float32)[None, :] * shade[:, None], 0, 255)
    return verts, cents, norms, cols


def rover_model(n, e, yaw, color, scale, wheel_phase=0.0):
    """Chassis + 4 wheels + camera mast/sensor + front light bar + colour ID stripe."""
    s = scale
    parts = []

    def part(off, size, col, lift=0.0):
        # offset in the rover frame (forward, left, up)
        cy, sn = math.cos(yaw), math.sin(yaw)
        wn = n + (off[0] * cy - off[1] * sn) * s
        we = e + (off[0] * sn + off[1] * cy) * s
        parts.append(box_faces((wn, we, off[2] * s), [x * s for x in size], yaw, col))

    dark = (52, 54, 58)
    part((0, 0, 0.30), (0.95, 0.52, 0.20), (84, 88, 94))                  # chassis
    part((-0.05, 0, 0.43), (0.55, 0.40, 0.08), color)                      # ID colour deck
    for fx in (-0.34, 0.34):
        for fy in (-0.33, 0.33):
            part((fx, fy, 0.16), (0.26, 0.12, 0.26), dark)                # wheels (blocky)
    part((0.30, 0, 0.60), (0.06, 0.06, 0.36), (140, 140, 146))            # mast
    part((0.30, 0, 0.82), (0.16, 0.12, 0.10), (30, 30, 34))               # camera/sensor module
    part((0.30, 0.0, 0.82), (0.17, 0.05, 0.05), (255, 230, 120))          # camera lens glow
    part((0.50, 0, 0.30), (0.06, 0.34, 0.07), (60, 230, 255))             # front light bar => heading
    part((-0.40, 0, 0.30), (0.04, 0.30, 0.05), (40, 40, 220))             # rear lights
    return parts


def drone_model(n, e, alt, yaw, color, scale):
    s = scale
    parts = []

    def part(off, size, col):
        cy, sn = math.cos(yaw), math.sin(yaw)
        parts.append(box_faces((n + (off[0] * cy - off[1] * sn) * s, e + (off[0] * sn + off[1] * cy) * s,
                                alt + off[2] * s), [x * s for x in size], yaw, col))

    part((0, 0, 0), (0.34, 0.34, 0.10), (70, 74, 80))
    part((0, 0, 0.07), (0.22, 0.22, 0.04), color)
    part((0.0, 0.0, 0.0), (0.95, 0.05, 0.04), (150, 150, 155))
    part((0.0, 0.0, 0.0), (0.05, 0.95, 0.04), (150, 150, 155))
    for fx, fy in ((0.46, 0.0), (-0.46, 0.0), (0.0, 0.46), (0.0, -0.46)):
        part((fx, fy, 0.05), (0.26, 0.26, 0.015), (210, 210, 215))   # rotor discs
    return parts


def person_model(n, e, color, lying, scale=1.0):
    parts = []
    if lying:
        parts.append(box_faces((n, e, 0.2 * scale), [1.7 * scale, 0.45 * scale, 0.22 * scale], 0.6, color))
        parts.append(box_faces((n + 0.9 * scale * 0.82, e + 0.9 * scale * 0.56, 0.22 * scale),
                               [0.26 * scale, 0.26 * scale, 0.26 * scale], 0.6, (170, 200, 235)))
    else:
        parts.append(box_faces((n, e, 0.45 * scale), [0.3 * scale, 0.22 * scale, 0.9 * scale], 0.0, (60, 60, 70)))
        parts.append(box_faces((n, e, 1.2 * scale), [0.40 * scale, 0.28 * scale, 0.55 * scale], 0.0, color))
        parts.append(box_faces((n, e, 1.65 * scale), [0.26 * scale, 0.26 * scale, 0.26 * scale], 0.0, (170, 200, 235)))
    return parts


# --------------------------------------------------------------------------- viewer
class VoxelViewer:
    def __init__(self, vm, options: ViewOptions | None = None):
        self.vm = vm
        self.opts = options or ViewOptions()
        self.cam = OrbitCamera()
        self.geo = Geometry()
        self._pending: Geometry | None = None
        self._building = False
        self._lock = threading.Lock()
        self._t_last_frame = time.perf_counter()
        self._last_req = 0.0
        self.fps = 0.0
        self.frame_ms = 0.0
        self.draw_faces = 0
        self.focus_label = ""
        self._bg_cache = None
        self._ground_cache = None
        self.external_geo = False        # first-person viewers reuse the main viewer's geometry

    @classmethod
    def first_person(cls, main: 'VoxelViewer', **opt_overrides) -> 'VoxelViewer':
        """A viewer that shares `main`'s map and geometry but looks from a vehicle-mounted camera."""
        o = replace(main.opts, render_scale=0.85, face_budget=7000, **opt_overrides)
        v = cls(main.vm, o)
        v.cam = FirstPersonCamera()
        v.external_geo = True
        v.geo = main.geo
        return v

    def render_pov(self, main: 'VoxelViewer', W: int, H: int, scene: Scene, rover: dict) -> np.ndarray:
        """First-person image from `rover`'s camera mast (rendered from the voxel map: a SIMULATED camera)."""
        self.geo = main.geo
        self.opts = replace(main.opts, render_scale=self.opts.render_scale, face_budget=self.opts.face_budget)
        self.cam.set_pose(rover['n'], rover['e'], rover['yaw'])
        rv = [dict(r, hide_model=(r['id'] == rover['id'])) for r in scene.rovers]
        sc = Scene(drones=scene.drones, rovers=rv, survivors=scene.survivors, transmissions=[], links=[],
                   hub=scene.hub, stats={}, now=scene.now)
        return self.render(W, H, sc, hud='none')

    # ------------------------------------------------------------------ geometry thread
    def _needs_build(self) -> bool:
        return self.geo.version != self.vm.version or self.geo.key != self.opts.geometry_key() \
            or self.geo.ground_tex is None

    def request_build(self, force: bool = False):
        now = time.perf_counter()
        if self._building or not self._needs_build():
            return
        if not force and now - self._last_req < 1.0:
            return
        self._last_req = now
        self._building = True
        opts = replace(self.opts)

        def work():
            try:
                g = build_geometry(self.vm, opts)
                with self._lock:
                    self._pending = g
            finally:
                self._building = False

        threading.Thread(target=work, daemon=True).start()

    def build_now(self):
        self.geo = build_geometry(self.vm, replace(self.opts))

    def _swap(self):
        with self._lock:
            if self._pending is not None:
                self.geo, self._pending = self._pending, None

    # ------------------------------------------------------------------ rendering
    def render(self, W: int, H: int, scene: Scene | None = None, async_geometry: bool = True,
               hud: str = 'full') -> np.ndarray:
        t0 = time.perf_counter()
        scene = scene or Scene()
        dt = max(1e-3, min(0.2, t0 - self._t_last_frame))
        self._t_last_frame = t0
        before = (self.cam.az, self.cam.el, self.cam.dist, tuple(self.cam.target))
        self.cam.update(dt)
        self._cam_moving = before != (self.cam.az, self.cam.el, self.cam.dist, tuple(self.cam.target))
        if self.external_geo:
            pass
        elif async_geometry:
            self.request_build()
            self._swap()
        elif self._needs_build():
            self.build_now()
        o = self.opts
        sc = float(np.clip(o.render_scale, 0.35, 1.0))
        sw, sh = max(64, int(W * sc)), max(64, int(H * sc))
        img = self._background_and_ground(sw, sh)
        faces = self._gather_faces(scene)
        self._draw_free(img, sw, sh)
        self._draw_faces(img, sw, sh, faces, scene)
        if (sw, sh) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)     # lines, labels and HUD stay sharp
        self._draw_overlays(img, W, H, scene)
        if hud == 'full':
            self._draw_hud(img, W, H, scene)
        self.frame_ms = (time.perf_counter() - t0) * 1000.0
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        return img

    def _background(self, W, H):
        if self._bg_cache is None or self._bg_cache.shape[:2] != (H, W):
            t = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
            col = BG_TOP[None, None, :] * (1 - t) + BG_BOT[None, None, :] * t
            self._bg_cache = np.ascontiguousarray(np.broadcast_to(col, (H, W, 3)).astype(np.uint8))
        return self._bg_cache.copy()

    def _background_and_ground(self, W, H):
        """Sky gradient + projected ground texture as one frame, cached while the camera and map are idle."""
        tex = self.geo.ground_tex
        key = (id(tex), W, H, round(float(self.cam.az), 2), round(float(self.cam.el), 2),
               round(float(self.cam.dist), 2), tuple(np.round(self.cam.target, 2)))
        if self._ground_cache is not None and self._ground_cache[0] == key:
            return self._ground_cache[1].copy()
        img = self._background(W, H)
        moving = self._ground_cache is None or self._ground_cache[0][3:] != key[3:]
        self._draw_ground(img, W, H, coarse=moving)
        self._ground_cache = (key, img.copy())
        return img

    def _draw_ground(self, img, W, H, coarse=False):
        tex = self.geo.ground_tex
        if tex is None:
            return
        pos, right, up, fwd = self.cam.pose()
        f = (H / 2.0) / math.tan(math.radians(self.cam.fov) / 2)
        div = 3 if coarse else 2                              # coarser while the camera moves, sharp when idle
        sw, sh = max(8, W // div), max(8, H // div)
        us = ((np.arange(sw, dtype=np.float32) + 0.5) * (W / sw) - W / 2) / f
        vs = ((np.arange(sh, dtype=np.float32) + 0.5) * (H / sh) - H / 2) / f
        a = us[None, :, None] * right[None, None, :].astype(np.float32)
        b = vs[:, None, None] * up[None, None, :].astype(np.float32)
        dirs = fwd.astype(np.float32)[None, None, :] + a - b
        dz = dirs[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-pos[2] / dz).astype(np.float32)
        valid = (dz < -1e-3) & (t > 0)
        half, res = self.vm.half, self.vm.res
        map_r = (pos[0] + t * dirs[..., 0] + half) / res
        map_c = (pos[1] + t * dirs[..., 1] + half) / res
        inside = valid & (map_r >= 0) & (map_r < tex.shape[0] - 1) & (map_c >= 0) & (map_c < tex.shape[1] - 1)
        map_r = np.where(inside, map_r, -1).astype(np.float32)
        map_c = np.where(inside, map_c, -1).astype(np.float32)
        warped = cv2.remap(tex, map_c, map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        dist = np.where(inside, t * np.sqrt((dirs ** 2).sum(axis=2)), 0).astype(np.float32)
        fog = np.clip((dist - 70.0) / 160.0, 0, 0.75)[..., None]
        bg = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = (warped.astype(np.float32) * (1 - fog) + bg * fog).astype(np.uint8)
        big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(inside.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[mask] = big[mask]

    def _gather_faces(self, scene: Scene):
        """Static voxel faces (with time-based highlights) + dynamic model faces."""
        g = self.geo
        o = self.opts
        verts, cents, norms, cols = g.verts, g.centers, g.normals, g.colors
        if len(verts) and (o.show_recent or o.show_new_obstacles):
            now = self.vm.now()
            cols = cols.copy()
            tl, to = g.t_last[g.vox], g.t_occ[g.vox]
            if o.show_recent:
                a = np.clip(1.0 - (now - tl) / o.recent_window, 0.0, 1.0)[:, None] * 0.55
                cols = cols * (1 - a) + np.array([255, 235, 90], np.float32) * a          # cyan-ish flash
            if o.show_new_obstacles:
                pulse = 0.65 + 0.35 * math.sin(time.perf_counter() * 7.0)
                a = (np.clip(1.0 - (now - to) / o.new_window, 0.0, 1.0) * (to > -1e8))[:, None] * 0.75 * pulse
                cols = cols * (1 - a) + np.array([40, 120, 255], np.float32) * a          # orange: new obstacle
        dv, dc, dn, dcol = [], [], [], []
        s = o.model_scale
        models = []
        for r in scene.rovers:
            if r.get('hide_model'):
                continue
            models += rover_model(r["n"], r["e"], r["yaw"], r["color"], s, r.get("wheel", 0.0))
        for d in scene.drones:
            models += drone_model(d["n"], d["e"], d["alt"], d.get("yaw", 0.0), DRONE_COLORS[d["id"] % 4], s)
        if o.show_survivors:
            for sv in scene.survivors:
                lying = "Lying" in sv.get("name", "")
                col = (70, 215, 90) if sv.get("state") == "rescued" else SURVIVOR_COLOR
                models += person_model(sv["n"], sv["e"], col, lying, 1.0)
        for v, c, n, k in models:
            dv.append(v), dc.append(c), dn.append(n), dcol.append(k)
        if dv:
            verts = np.concatenate([verts] + dv) if len(verts) else np.concatenate(dv)
            cents = np.concatenate([cents] + dc) if len(cents) else np.concatenate(dc)
            norms = np.concatenate([norms] + dn) if len(norms) else np.concatenate(dn)
            cols = np.concatenate([cols] + dcol) if len(cols) else np.concatenate(dcol)
        return verts, cents, norms, cols

    def _draw_faces(self, img, W, H, faces, scene):
        verts, cents, norms, cols = faces
        n = len(verts)
        self.draw_faces = 0
        if n == 0:
            return
        o = self.opts
        pos, right, up, fwd = self.cam.pose()
        to_cam = pos[None, :] - cents
        facing = np.einsum("ij,ij->i", norms, to_cam) > 0
        depth = (cents - pos) @ fwd
        ok = facing & (depth > 1.0)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return
        u, v, zc = self.cam.project(verts[idx].reshape(-1, 3), W, H)
        u = u.reshape(-1, 4)
        v = v.reshape(-1, 4)
        zc = zc.reshape(-1, 4)
        onscreen = (zc.min(axis=1) > 0.5) & (u.max(axis=1) > -20) & (u.min(axis=1) < W + 20) & \
                   (v.max(axis=1) > -20) & (v.min(axis=1) < H + 20)
        idx, u, v = idx[onscreen], u[onscreen], v[onscreen]
        dep = depth[idx]
        budget = o.face_budget if not getattr(self, '_cam_moving', False) else int(o.face_budget * 0.6)
        if len(idx) > budget:                            # LOD: drop the farthest faces (fewer while the camera moves)
            keep = np.argpartition(dep, budget)[:budget]
            idx, u, v, dep = idx[keep], u[keep], v[keep], dep[keep]
        order = np.argsort(-dep)                         # painter's algorithm: far to near
        idx, u, v, dep = idx[order], u[order], v[order], dep[order]
        pts = np.stack([u, v], axis=2).astype(np.int32)
        fog = np.clip((dep - 60.0) / 190.0, 0.0, 0.7)[:, None]
        shaded = cols[idx] * (1.0 - fog) + BG_TOP[None, :] * 1.4 * fog
        shaded = np.clip(shaded, 0, 255).astype(np.uint8)
        bw = u.max(axis=1) - u.min(axis=1)
        bh = v.max(axis=1) - v.min(axis=1)
        small = np.maximum(bw, bh) <= 6.0
        layer = img if o.opacity >= 0.99 else img.copy()
        # --- far / small faces: vectorised block fill, painter order kept by face-major flattening
        si = np.nonzero(small)[0]
        order_all = np.arange(len(idx))
        if len(si):
            u0 = np.floor(u[si].min(axis=1)).astype(np.int32)
            v0 = np.floor(v[si].min(axis=1)).astype(np.int32)
            w_ = np.maximum(1, np.ceil(bw[si]).astype(np.int32))
            h_ = np.maximum(1, np.ceil(bh[si]).astype(np.int32))
            mw, mh = int(w_.max()), int(h_.max())
            dx, dy = np.meshgrid(np.arange(mw), np.arange(mh), indexing="ij")
            dx, dy = dx.ravel(), dy.ravel()
            valid = (dx[None, :] < w_[:, None]) & (dy[None, :] < h_[:, None])
            fi, qi = np.nonzero(valid)                            # face-major => far-to-near order preserved
            xs = np.clip(u0[fi] + dx[qi], 0, W - 1)
            ys = np.clip(v0[fi] + dy[qi], 0, H - 1)
            layer[ys, xs] = shaded[si][fi]
        # --- big faces: polygon fill with crisp voxel edges
        edge = (shaded.astype(np.float32) * 0.55).astype(np.uint8)
        fill, poly = cv2.fillConvexPoly, cv2.polylines
        big = np.nonzero(~small)[0]
        for i in big:
            c = shaded[i]
            p = pts[i]
            fill(layer, p, (int(c[0]), int(c[1]), int(c[2])))
            if o.show_edges and bw[i] > 10.0:
                e = edge[i]
                poly(layer, [p], True, (int(e[0]), int(e[1]), int(e[2])), 1)
        self.draw_faces = len(idx)
        if layer is not img:
            cv2.addWeighted(layer, o.opacity, img, 1.0 - o.opacity, 0, dst=img)

    def _draw_free(self, img, W, H):
        """Free-space voxels (observed along sensor rays) as faint points, behind the solid voxels."""
        pts = self.geo.free_pts
        if not self.opts.show_free or not len(pts):
            return
        u, v, z = self.cam.project(pts, W, H)
        ok = (z > 1.0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
        col = np.array([200, 170, 90], np.float32)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            img[vi + dy, ui + dx] = (img[vi + dy, ui + dx] * 0.62 + col * 0.38).astype(np.uint8)

    # ------------------------------------------------------------------ overlays
    def _p(self, pts, W, H):
        pts = np.asarray(pts, np.float32).reshape(-1, 3)
        u, v, z = self.cam.project(pts, W, H)
        return np.stack([u, v], axis=1), z

    def _line3d(self, img, pts, color, thick=2, dashed=False, W=0, H=0):
        if len(pts) < 2:
            return
        uv, z = self._p(pts, W, H)
        for k in range(len(uv) - 1):
            if z[k] < 0.5 or z[k + 1] < 0.5:
                continue
            if dashed and (k // 2) % 2:
                continue
            cv2.line(img, tuple(np.int32(uv[k])), tuple(np.int32(uv[k + 1])), color, thick, cv2.LINE_AA)

    def _ring(self, img, n, e, radius, color, W, H, z=0.15, thick=2):
        ang = np.linspace(0, 2 * math.pi, 40)
        pts = np.stack([n + radius * np.cos(ang), e + radius * np.sin(ang), np.full_like(ang, z)], axis=1)
        self._line3d(img, pts, color, thick, W=W, H=H)

    def _label(self, img, text, xy, color, scale=0.45, bg=True):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = int(xy[0]), int(xy[1])
        if bg:
            cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 3), (14, 16, 18), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def _draw_overlays(self, img, W, H, scene: Scene):
        o = self.opts
        t = time.perf_counter()
        # drone comm links + trails
        if o.show_links:
            for i, j in scene.links:
                a, b = scene.drones[i], scene.drones[j]
                self._line3d(img, [(a["n"], a["e"], a["alt"]), (b["n"], b["e"], b["alt"])], (110, 150, 90), 1, W=W, H=H)
        for d in scene.drones:
            col = DRONE_COLORS[d["id"] % 4]
            if o.show_trails and len(d.get("trail", [])) > 1:
                self._line3d(img, [(a, b, c) for a, b, c in d["trail"]], col, 1, W=W, H=H)
            uv, z = self._p([(d["n"], d["e"], d["alt"])], W, H)
            if z[0] > 1:
                self._line3d(img, [(d["n"], d["e"], 0.0), (d["n"], d["e"], d["alt"])], (90, 100, 110), 1, W=W, H=H)
                self._label(img, f"{d['name']} {d['alt']:.0f}m", (uv[0, 0] + 14, uv[0, 1] - 10), col, 0.42)
        # rover routes (the exact polyline the rover follows) and trails
        if o.show_routes:
            for r in scene.rovers:
                col = tuple(int(c) for c in r["color"])
                if len(r.get("trail", [])) > 1:
                    self._line3d(img, [(a, b, 0.12) for a, b in r["trail"]], tuple(int(c * 0.6) for c in col), 2, W=W, H=H)
                route = r.get("route") or []
                if len(route) > 1:
                    pts = [(a, b, 0.2) for a, b in route]
                    self._line3d(img, pts, (0, 0, 0), 5, W=W, H=H)
                    self._line3d(img, pts, col, 3, W=W, H=H)
                    for a, b in route[1:]:
                        uv, z = self._p([(a, b, 0.25)], W, H)
                        if z[0] > 1:
                            cv2.circle(img, tuple(np.int32(uv[0])), 4, col, -1, cv2.LINE_AA)
                            cv2.circle(img, tuple(np.int32(uv[0])), 5, (20, 20, 20), 1, cv2.LINE_AA)
        # rover name tags
        for r in scene.rovers:
            if r.get('hide_model'):
                continue
            col = tuple(int(c) for c in r["color"])
            uv, z = self._p([(r["n"], r["e"], 2.3)], W, H)
            if z[0] > 1:
                self._label(img, f"{r['name']} [{r.get('state', '')}]", (uv[0, 0] + 8, uv[0, 1] - 6), col, 0.45)
        # survivors: marker, detection radius, confidence, detecting drone
        if o.show_survivors:
            for sv in scene.survivors:
                rescued = sv.get("state") == "rescued"
                col = (80, 220, 100) if rescued else SURVIVOR_COLOR
                pulse = 0.5 + 0.5 * math.sin(t * 5.0 + sv["id"])
                rad = sv.get("radius", 3.0)
                self._ring(img, sv["n"], sv["e"], rad * (1.0 + (0.0 if rescued else 0.08 * pulse)), col, W, H, thick=2)
                self._ring(img, sv["n"], sv["e"], 1.0, (255, 255, 255), W, H, thick=1)
                self._line3d(img, [(sv["n"], sv["e"], 0.2), (sv["n"], sv["e"], 9.0)], col, 2, W=W, H=H)
                uv, z = self._p([(sv["n"], sv["e"], 9.4)], W, H)
                if z[0] > 1:
                    tag = f"{sv.get('name', 'Survivor')}  {sv.get('conf', 0) * 100:.0f}%  UAV-{sv.get('drone_id', 0) + 1}"
                    if rescued:
                        tag += "  RESCUED"
                    self._label(img, tag, (uv[0, 0] - 40, uv[0, 1]), col, 0.42)
        # transmissions: drone -> coordinator -> rover, animated pulses along arcs
        for tr in scene.transmissions:
            age = scene.now - tr["t0"]
            hub = (scene.hub[0], scene.hub[1], 22.0)
            if tr["kind"] == "to_coordinator" and tr.get("from"):
                a = (tr["from"][0], tr["from"][1], tr["from"][2])
                b = hub
                col = (60, 200, 255)
            elif tr["kind"] == "to_rover":
                a = hub
                b = (tr["rover_n"], tr["rover_e"], 1.2)
                col = (120, 255, 160)
            else:
                continue
            frac = min(1.0, age / 1.4)
            arc = []
            for q in np.linspace(0, 1, 28):
                p = np.array(a) * (1 - q) + np.array(b) * q
                p[2] += 9.0 * math.sin(math.pi * q)
                arc.append(tuple(p))
            self._line3d(img, arc, tuple(int(c * 0.45) for c in col), 1, dashed=True, W=W, H=H)
            k = int(frac * 27)
            self._line3d(img, arc[max(0, k - 5):k + 1], col, 3, W=W, H=H)
            uv, z = self._p([arc[k]], W, H)
            if z[0] > 1:
                cv2.circle(img, tuple(np.int32(uv[0])), 6, col, -1, cv2.LINE_AA)
        # coordinator hub marker
        uv, z = self._p([(scene.hub[0], scene.hub[1], 0.0), (scene.hub[0], scene.hub[1], 22.0)], W, H)
        if z.min() > 1:
            cv2.line(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), (150, 160, 120), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.int32(uv[1])), 7, (60, 200, 255), 1, cv2.LINE_AA)
            self._label(img, "COORDINATOR", (uv[1, 0] + 10, uv[1, 1] + 4), (60, 200, 255), 0.4)

    # ------------------------------------------------------------------ HUD + legend
    def _draw_hud(self, img, W, H, scene: Scene):
        o = self.opts
        cv2.rectangle(img, (0, 0), (W, 34), (22, 28, 24), -1)
        head = f"3D OCCUPANCY MAP  res {self.vm.res:.2f} m   voxels {self.geo.n_vox}   known ground {self.geo.ground_meta.get('known_frac', 0) * 100:.0f}%"
        cv2.putText(img, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 215, 120), 1, cv2.LINE_AA)
        info = f"{self.fps:4.0f} fps  {self.frame_ms:4.0f} ms  faces {self.draw_faces}"
        cv2.putText(img, info, (W - 235, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 190, 160), 1, cv2.LINE_AA)
        if scene.source_label:
            cv2.putText(img, scene.source_label, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 210, 255), 1, cv2.LINE_AA)
        # mission panel
        if scene.stats:
            lines = [f"MISSIONS  detected {scene.stats.get('detected', 0)}  queued {scene.stats.get('queued', 0)}  "
                     f"en-route {scene.stats.get('en_route', 0)}  rescued {scene.stats.get('rescued', 0)}  "
                     f"replans {scene.stats.get('replans', 0)}"]
            lines += scene.stats.get("lines", [])[:5]
            h = 14 * len(lines) + 10
            cv2.rectangle(img, (6, 40), (min(W - 6, 520), 40 + h), (24, 30, 28), -1)
            cv2.rectangle(img, (6, 40), (min(W - 6, 520), 40 + h), (70, 90, 80), 1)
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (12, 54 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 235, 215), 1, cv2.LINE_AA)
        self._legend(img, W, H)

    def _legend(self, img, W, H):
        o = self.opts
        items = []
        if o.color_mode == "occupancy":
            items.append(("occupied  p 0.5 -> 1.0", None))
        else:
            items.append((f"occupied ({o.color_mode})", None))
        items += [("free ground (observed)", tuple(int(c) for c in FREE_GROUND)),
                  ("unknown (never observed)", tuple(int(c) for c in UNKNOWN_GROUND + 25)),
                  ("recently updated", (255, 235, 90)), ("newly detected obstacle", (40, 120, 255)),
                  ("survivor", SURVIVOR_COLOR), ("rescued", (80, 220, 100)),
                  ("Rover 1 route", ROVER_COLORS[0]), ("Rover 2 route", ROVER_COLORS[1])]
        w, h = 214, 16 * len(items) + 12
        x0, y0 = W - w - 8, H - h - 30
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (24, 28, 26), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 90, 80), 1)
        for i, (label, col) in enumerate(items):
            y = y0 + 16 + 16 * i
            if col is None:
                xs = np.linspace(0, 1, 40)
                ramp = voxel_colors(o.color_mode, xs * 14.0, 0.5 + xs * 0.5, (1 - xs) * 120, self.vm.res, o)
                for k in range(40):
                    c = ramp[k]
                    cv2.line(img, (x0 + 8 + k // 2 * 1, y - 8), (x0 + 8 + k // 2, y + 2), (int(c[0]), int(c[1]), int(c[2])), 1)
                for k in range(40):
                    c = ramp[k]
                    cv2.rectangle(img, (x0 + 8 + k, y - 9), (x0 + 9 + k, y + 3), (int(c[0]), int(c[1]), int(c[2])), -1)
                cv2.putText(img, label, (x0 + 54, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 225, 215), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x0 + 8, y - 9), (x0 + 44, y + 3), col, -1)
                cv2.putText(img, label, (x0 + 54, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 225, 215), 1, cv2.LINE_AA)

"""ROS-free geometry + appearance helpers that turn REAL detector boxes into Re-ID observations.

Camera model = the one in yolo_detector.py (mono_cam 1280x960, HFOV 1.74 rad, mounted 25 deg
below the horizon; pitch handled as `mount - body_pitch`). The forward projection here repeats
that math so a round-trip test can prove the back-projection is its exact inverse.

Ground-plane assumption: the foot of a detected person lies on NED down = `ground_d` (flat
ground). That is wrong on rubble or stairs; the upgrade path is ray-casting the box foot
against the fused ESDF/mesh (SAR_SWARM_SPEC.md section 4.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """Drone pose in its NED frame (angles in radians)."""
    n: float = 0.0
    e: float = 0.0
    d: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class Camera:
    width: int = 1280
    height: int = 960
    hfov: float = 1.74
    mount_pitch: float = 0.436332

    @property
    def fx(self) -> float:
        return (self.width * 0.5) / math.tan(self.hfov * 0.5)


def project_ned_to_pixel(cam: Camera, pose: Pose, tn: float, te: float, td: float):
    """Forward model (same as YoloDetector._project_ned_to_pixel). Returns (u, v, depth) or None."""
    dn, de, dd = tn - pose.n, te - pose.e, td - pose.d
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    x1, y1, z1 = dn * cy + de * sy, -dn * sy + de * cy, dd
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    x2, y2, z2 = x1, y1 * cr + z1 * sr, -y1 * sr + z1 * cr
    th = cam.mount_pitch - pose.pitch
    ct, st = math.cos(th), math.sin(th)
    x_cam, y_cam, z_cam = x2 * ct + z2 * st, y2, -x2 * st + z2 * ct
    if x_cam <= 1.0:
        return None
    return (cam.width * 0.5 + y_cam / x_cam * cam.fx,
            cam.height * 0.5 + z_cam / x_cam * cam.fx, x_cam)


def pixel_to_ray_ned(cam: Camera, pose: Pose, u: float, v: float) -> np.ndarray:
    """Direction (not normalised, unit depth along the optical axis) of pixel (u, v) in NED."""
    a = (u - cam.width * 0.5) / cam.fx          # right
    b = (v - cam.height * 0.5) / cam.fx         # down
    th = cam.mount_pitch - pose.pitch
    ct, st = math.cos(th), math.sin(th)
    x2, y2, z2 = ct - b * st, a, st + b * ct    # undo the camera pitch
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    x1, y1, z1 = x2, y2 * cr - z2 * sr, y2 * sr + z2 * cr   # undo roll
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.array([x1 * cy - y1 * sy, x1 * sy + y1 * cy, z1])   # undo yaw -> (n, e, d)


def pixel_to_ground(cam: Camera, pose: Pose, u: float, v: float, ground_d: float = 0.0,
                    max_range: float = 80.0):
    """Intersect the pixel ray with the plane NED-down = ground_d. Returns np.array([n, e]) or None
    (ray points at/above the horizon, hits behind the camera, or is farther than max_range)."""
    ray = pixel_to_ray_ned(cam, pose, u, v)
    if ray[2] <= 1e-3:
        return None
    s = (ground_d - pose.d) / ray[2]
    if s <= 0 or s * float(np.linalg.norm(ray)) > max_range:
        return None
    return np.array([pose.n + s * ray[0], pose.e + s * ray[1]])


def ground_covariance(cam: Camera, pose: Pose, u: float, v: float, px_sigma: float,
                      att_sigma: float = math.radians(1.5), alt_sigma: float = 0.3,
                      ground_d: float = 0.0, eps: float = 1e-4):
    """2x2 covariance of the ground point from pixel, attitude and altitude noise (numerical
    Jacobian). Horizontal pose drift is NOT included: it is correlated across a robot's
    observations and enters the registry separately as `pose_var`."""
    names = ("u", "v", "roll", "pitch", "yaw", "d")
    base = [u, v, pose.roll, pose.pitch, pose.yaw, pose.d]

    def f(x):
        q = Pose(pose.n, pose.e, x[5], x[2], x[3], x[4])
        g = pixel_to_ground(cam, q, x[0], x[1], ground_d, max_range=1e9)
        return None if g is None else g - np.array([pose.n, pose.e])

    f0 = f(base)
    if f0 is None:
        return None
    J = np.zeros((2, len(base)))
    for i in range(len(base)):
        hi = list(base)
        hi[i] += eps
        fi = f(hi)
        if fi is None:
            return None
        J[:, i] = (fi - f0) / eps
    sig = np.diag([px_sigma ** 2, px_sigma ** 2, att_sigma ** 2, att_sigma ** 2, att_sigma ** 2, alt_sigma ** 2])
    return J @ sig @ J.T


# --------------------------------------------------------------------------- appearance
def color_histogram(rgb_crop: np.ndarray, bins: int = 4) -> np.ndarray:
    """Placeholder appearance feature: Hellinger-normalised joint RGB histogram of the upper
    and lower half of the crop (shirt vs trousers). Length 2 * bins**3. NOT a trained Re-ID
    network; replace with MobileNet-ReID/OSNet and recalibrate the cosine thresholds."""
    crop = np.asarray(rgb_crop)
    if crop.ndim != 3 or crop.shape[0] < 4 or crop.shape[1] < 2:
        return np.zeros(2 * bins ** 3)
    half = crop.shape[0] // 2
    out = []
    for part in (crop[:half], crop[half:]):
        q = (part.reshape(-1, 3).astype(np.int64) * bins) // 256
        idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
        h = np.bincount(idx, minlength=bins ** 3).astype(float)
        out.append(np.sqrt(h / max(1.0, h.sum())))
    return np.concatenate(out)


class RunningCentering:
    """Colour histograms are non-negative, so every pair has a high cosine. Subtracting the
    running mean over all crops seen makes unrelated people roughly orthogonal, which is the
    regime the registry thresholds assume."""

    def __init__(self):
        self.n = 0
        self.mean = None

    def __call__(self, raw: np.ndarray):
        raw = np.asarray(raw, float)
        if self.mean is None:
            self.mean = np.zeros_like(raw)
        self.n += 1
        self.mean += (raw - self.mean) / self.n
        c = raw - self.mean
        norm = float(np.linalg.norm(c))
        return c / norm if norm > 1e-9 and self.n >= 5 else None   # too few crops to trust: no appearance yet

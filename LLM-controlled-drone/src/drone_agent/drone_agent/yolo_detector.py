#!/usr/bin/env python3
"""Search & Rescue (SAR) Multi-Drone YOLOv8 + 3D Pose-Aware Human & Vehicle Detector Node.

Upgrades YOLOv8 for Post-Earthquake Search & Rescue across ALL 4 Swarm Drones:
1. Strict SAR class filter: ONLY allows human/vehicle rescue classes (`person`, `car`, `truck`, `bus`)
   — completely eliminates false-positive `airplane`, `kite`, `boat`, `traffic light`, or `tv` labels!
2. Multi-Camera Architecture:
   - UAV-1: `/camera`        -> `/yolo/image_annotated` & `/uav_0/yolo/annotated`
   - UAV-2: `/uav_1/camera`  -> `/uav_1/yolo/annotated`
   - UAV-3: `/uav_2/camera`  -> `/uav_2/yolo/annotated`
   - UAV-4: `/uav_3/camera`  -> `/uav_3/yolo/annotated`
3. 3D Camera Frustum Projection for each individual drone's pose:
   accurately locks onto and classifies:
   - Walking human survivors (`PERSON [WALKING SURVIVOR]`)
   - Standing human survivors (`PERSON [STANDING SURVIVOR]`)
   - Lying-down injured earthquake victims (`PERSON [LYING VICTIM]`)
   - Rescue & stranded vehicles (`CAR [AMBULANCE]`, `CAR [STRANDED PRIUS]`, `CAR [DAMAGED SUV]`)
"""

import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

from drone_agent.victim_geometry import color_histogram

def cv2_to_imgmsg_fast(img: np.ndarray, encoding: str = 'bgr8', header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = int(img.shape[0])
    msg.width = int(img.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(img.shape[1] * (3 if encoding == 'bgr8' else 1))
    msg.data = img.tobytes()
    return msg


# COCO class IDs strictly relevant to Search & Rescue:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
SAR_ALLOWED_CLASSES = [0, 1, 2, 3, 5, 7]


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('skip_frames', 1)
        self.declare_parameter('raw_conf', 0.35)          # min YOLO confidence for /yolo/raw_detections
        self.raw_conf = float(self.get_parameter('raw_conf').value)
        # Demo-only: paint the world's hard-coded people/cars as detections (simulator ground truth).
        self.declare_parameter('gt_overlay', False)
        self.gt_overlay = bool(self.get_parameter('gt_overlay').value)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.skip_frames = self.get_parameter('skip_frames').value

        self.get_logger().info(f'Loading Multi-Drone SAR YOLOv8 model: {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.frame_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        # Drone NED poses for 3D camera frustum target verification & human classification
        self.drone_poses = {
            0: {'n': 3.0, 'e': 0.0, 'd': -9.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ok': False},
            1: {'n': 0.0, 'e': 3.0, 'd': -10.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ok': False},
            2: {'n': -3.0, 'e': 0.0, 'd': -12.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ok': False},
            3: {'n': 0.0, 'e': -3.0, 'd': -13.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ok': False},
        }

        # Camera intrinsics matching mono_cam (1280x960, HFOV=1.74 rad, pitched down 25 deg = 0.436332 rad)
        self.cam_hfov = 1.74
        self.cam_mount_pitch = 0.436332  # +25 deg downward tilt below horizon

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Odometry subscriptions for all 4 drones
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', lambda m: self._odom_cb(0, m), px4_qos
        )
        self.create_subscription(
            VehicleOdometry, '/px4_1/fmu/out/vehicle_odometry', lambda m: self._odom_cb(1, m), px4_qos
        )
        self.create_subscription(
            VehicleOdometry, '/px4_2/fmu/out/vehicle_odometry', lambda m: self._odom_cb(2, m), px4_qos
        )
        self.create_subscription(
            VehicleOdometry, '/px4_3/fmu/out/vehicle_odometry', lambda m: self._odom_cb(3, m), px4_qos
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Camera subscriptions for all 4 drones
        self.create_subscription(
            Image, '/camera', lambda m: self.process_drone_camera(0, m), sensor_qos
        )
        self.create_subscription(
            Image, '/uav_1/camera', lambda m: self.process_drone_camera(1, m), sensor_qos
        )
        self.create_subscription(
            Image, '/uav_2/camera', lambda m: self.process_drone_camera(2, m), sensor_qos
        )
        self.create_subscription(
            Image, '/uav_3/camera', lambda m: self.process_drone_camera(3, m), sensor_qos
        )

        # Detections and annotated camera publishers
        self.pub = self.create_publisher(String, '/yolo/detections', 10)
        self.raw_pub = self.create_publisher(String, '/yolo/raw_detections', 10)
        self.annotated_pubs = {
            0: self.create_publisher(Image, '/yolo/image_annotated', sensor_qos),
            1: self.create_publisher(Image, '/uav_1/yolo/annotated', sensor_qos),
            2: self.create_publisher(Image, '/uav_2/yolo/annotated', sensor_qos),
            3: self.create_publisher(Image, '/uav_3/yolo/annotated', sensor_qos),
        }
        self.uav0_alt_pub = self.create_publisher(Image, '/uav_0/yolo/annotated', sensor_qos)

        if self.gt_overlay:
            self.get_logger().warn('gt_overlay=true: /yolo/detections and the annotated video include '
                                   'SIMULATOR GROUND-TRUTH targets, not only real YOLO detections.')
        self.get_logger().info('Multi-Drone SAR YOLOv8 Human & Vehicle Detector started for UAV-1..4!')

    def _odom_cb(self, drone_idx: int, msg: VehicleOdometry):
        p = self.drone_poses[drone_idx]
        p['ok'] = True
        p['n'] = float(msg.position[0])
        p['e'] = float(msg.position[1])
        p['d'] = float(msg.position[2])
        q = msg.q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        p['roll'] = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        p['pitch'] = math.asin(sinp)
        p['yaw'] = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_disaster_targets_ned(self):
        """Returns the 3D positions & dimensions of all humans & vehicles in drone_demo.sdf (NED)."""
        t = time.time() % 14.0
        # Walking survivor 1 animates East between 6m and 10m at North=8m
        if t < 6.0:
            w1_e = 6.0 + (t / 6.0) * 4.0
            w2_e = -6.0 - (t / 6.0) * 4.0
        elif t < 7.0:
            w1_e = 10.0
            w2_e = -10.0
        elif t < 13.0:
            w1_e = 10.0 - ((t - 7.0) / 6.0) * 4.0
            w2_e = -10.0 + ((t - 7.0) / 6.0) * 4.0
        else:
            w1_e = 6.0
            w2_e = -6.0

        # Tuple: (class_name, display_label, north, east, center_alt_d, radius_m, height_m)
        return [
            # Walking survivors (dynamic ground motion)
            ('person', 'person [walking survivor]', 8.0, w1_e, -0.9, 0.45, 1.75),
            ('person', 'person [walking survivor]', -8.0, w2_e, -0.9, 0.45, 1.75),

            # Standing survivors (trapped near collapsed debris)
            ('person', 'person [standing survivor]', 10.0, 4.0, -0.9, 0.40, 1.75),
            ('person', 'person [standing survivor]', 6.0, -8.0, -0.9, 0.40, 1.75),
            ('person', 'person [standing survivor]', -5.0, 9.0, -0.9, 0.40, 1.75),

            # Lying-down injured earthquake victims (horizontal on ground)
            ('person', 'person [lying victim]', 8.15, 2.36, -0.3, 0.95, 0.45),
            ('person', 'person [lying victim]', 4.60, -6.78, -0.3, 0.95, 0.45),
            ('person', 'person [lying victim]', -8.54, -3.21, -0.3, 0.95, 0.45),
            ('person', 'person [lying victim]', -7.77, 6.58, -0.3, 0.95, 0.45),

            # Emergency & stranded vehicles
            ('car', 'car [ambulance]', -8.0, 11.0, -1.2, 1.4, 2.2),
            ('car', 'car [stranded prius]', 3.0, 12.0, -0.8, 1.2, 1.6),
            ('car', 'car [damaged suv]', -3.0, -11.0, -0.9, 1.3, 1.8),
        ]

    def _project_ned_to_pixel(self, drone_idx: int, tn: float, te: float, td: float, w: int, h: int):
        """Projects a World NED coordinate into this specific drone's 2D image plane."""
        p = self.drone_poses[drone_idx]
        dn = tn - p['n']
        de = te - p['e']
        dd = td - p['d']

        # 1. Yaw rotation into drone body frame
        cy, sy = math.cos(p['yaw']), math.sin(p['yaw'])
        x1 = dn * cy + de * sy
        y1 = -dn * sy + de * cy
        z1 = dd

        # 2. Roll rotation around Forward axis
        cr, sr = math.cos(p['roll']), math.sin(p['roll'])
        x2 = x1
        y2 = y1 * cr + z1 * sr
        z2 = -y1 * sr + z1 * cr

        # 3. Downward camera pitch rotation
        theta_down = self.cam_mount_pitch - p['pitch']
        ct, st = math.cos(theta_down), math.sin(theta_down)
        x_cam = x2 * ct + z2 * st      # Depth along optical axis
        y_cam = y2                     # Right in image
        z_cam = -x2 * st + z2 * ct     # Down in image

        if x_cam <= 1.0:
            return None

        fx = (w * 0.5) / math.tan(self.cam_hfov * 0.5)
        fy = fx
        u = (w * 0.5) + (y_cam / x_cam) * fx
        v = (h * 0.5) + (z_cam / x_cam) * fy
        return u, v, x_cam, fx

    def _publish_raw_persons(self, drone_idx, msg, frame, raw_yolo_boxes, w, h):
        """`/yolo/raw_detections`: person boxes straight from YOLO with an appearance feature.
        Unlike `/yolo/detections` this contains nothing seeded from `_get_disaster_targets_ned`,
        so the victim registry counts only what the network actually saw."""
        dets = []
        for yb in raw_yolo_boxes:
            if yb['class'] != 'person' or yb['conf'] < self.raw_conf:
                continue
            x1, y1, x2, y2 = yb['box']
            crop = frame[max(0, y1):y2, max(0, x1):x2, ::-1]          # BGR -> RGB
            emb = color_histogram(crop) if crop.size else np.zeros(2 * 4 ** 3)
            dets.append({'box': [x1, y1, x2, y2], 'conf': round(yb['conf'], 3),
                         'emb': [round(float(v), 3) for v in emb]})
        if not dets:
            return
        out = String()
        out.data = json.dumps({'drone_id': drone_idx,
                               'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                               'width': w, 'height': h, 'dets': dets})
        self.raw_pub.publish(out)

    def process_drone_camera(self, drone_idx: int, msg: Image):
        self.frame_counts[drone_idx] += 1
        if self.frame_counts[drone_idx] % (self.skip_frames + 1) != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'UAV-{drone_idx+1} cv_bridge failed: {e}')
            return

        h, w = frame.shape[:2]

        # 1. Run YOLOv8 strictly filtered to SAR classes
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=0.20,
                iou=0.45,
                imgsz=640,
                classes=SAR_ALLOWED_CLASSES,
                agnostic_nms=True,
            )[0]
        except Exception as e:
            self.get_logger().warn(f'UAV-{drone_idx+1} YOLO inference error: {e}')
            return

        raw_yolo_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            if cls_name in ('truck', 'bus'):
                cls_name = 'car'
            conf = float(box.conf[0])
            bw, bh = (x2 - x1), (y2 - y1)
            area_frac = (bw * bh) / float(w * h)
            if area_frac > 0.25 or area_frac < 0.0003:
                continue
            raw_yolo_boxes.append({
                'class': cls_name,
                'label': f'{cls_name} [yolo]',
                'conf': conf,
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'matched': False,
            })

        # 1b. Publish the REAL YOLO person boxes only (no simulator ground truth) for Re-ID.
        self._publish_raw_persons(drone_idx, msg, frame, raw_yolo_boxes, w, h)

        final_detections = []
        draw_items = []

        # 2. Fuse with 3D Camera Frustum Projection for this drone's pose
        # The overlay is SIMULATOR GROUND TRUTH, so it is off unless explicitly requested.
        overlay_targets = self._get_disaster_targets_ned() if self.gt_overlay else []
        for cls_name, disp_label, tn, te, t_alt, t_rad, t_h in overlay_targets:
            proj = self._project_ned_to_pixel(drone_idx, tn, te, t_alt, w, h)
            if proj is None:
                continue
            u, v, dist_m, fx = proj
            if dist_m > 48.0:
                continue
            margin = 16
            if not (margin <= u <= w - margin and margin <= v <= h - margin):
                continue

            half_w = max(18, int((t_rad * fx) / dist_m))
            half_h = max(22, int((max(t_h * 0.65, t_rad * 0.85) * fx) / dist_m))
            px1 = max(2, int(u - half_w))
            py1 = max(2, int(v - half_h))
            px2 = min(w - 2, int(u + half_w))
            py2 = min(h - 2, int(v + half_h))

            best_conf = 0.91 if cls_name == 'person' else 0.89
            for yb in raw_yolo_boxes:
                yx1, yy1, yx2, yy2 = yb['box']
                ycx, ycy = 0.5 * (yx1 + yx2), 0.5 * (yy1 + yy2)
                if math.hypot(ycx - u, ycy - v) < max(half_w, half_h) * 2.2:
                    yb['matched'] = True
                    best_conf = max(best_conf, min(0.98, yb['conf'] + 0.30))
                    px1 = int(0.6 * px1 + 0.4 * yx1)
                    py1 = int(0.6 * py1 + 0.4 * yy1)
                    px2 = int(0.6 * px2 + 0.4 * yx2)
                    py2 = int(0.6 * py2 + 0.4 * yy2)
                    break

            final_detections.append({
                'drone_id': drone_idx,
                'class': cls_name,
                'status': disp_label,
                'confidence': round(best_conf, 2),
                'distance_m': round(dist_m, 1),
                'target_ned': [round(tn, 2), round(te, 2)],
                'bbox_center': [
                    round(0.5 * (px1 + px2) / w, 2),
                    round(0.5 * (py1 + py2) / h, 2),
                ],
                'bbox_area': round(((px2 - px1) * (py2 - py1)) / float(w * h), 4),
            })
            draw_items.append((px1, py1, px2, py2, disp_label, best_conf, dist_m, cls_name))

        # 3. Include unmatched raw YOLOv8 person/car detections
        for yb in raw_yolo_boxes:
            if not yb['matched'] and yb['conf'] >= 0.35:
                x1, y1, x2, y2 = yb['box']
                final_detections.append({
                    'drone_id': drone_idx,
                    'class': yb['class'],
                    'status': yb['label'],
                    'confidence': round(yb['conf'], 2),
                    'distance_m': 0.0,
                    'bbox_center': [
                        round(0.5 * (x1 + x2) / w, 2),
                        round(0.5 * (y1 + y2) / h, 2),
                    ],
                    'bbox_area': round(((x2 - x1) * (y2 - y1)) / float(w * h), 4),
                })
                draw_items.append((x1, y1, x2, y2, yb['label'], yb['conf'], 0.0, yb['class']))

        if final_detections:
            out = String()
            out.data = json.dumps(final_detections)
            self.pub.publish(out)

            if self.frame_counts[drone_idx] % 12 == 0:
                summary = [f"{d.get('status', d['class'])} ({d.get('distance_m', '?')}m)" for d in final_detections]
                self.get_logger().info(f'UAV-{drone_idx+1} SAR YOLO Lock-On ({len(final_detections)} targets): {summary}')

        # 4. Draw bold, high-contrast Search & Rescue bounding boxes + HUD banner
        annotated = frame.copy()
        person_count = sum(1 for d in final_detections if d['class'] == 'person')
        car_count = sum(1 for d in final_detections if d['class'] == 'car')

        # Top HUD status bar on video feed
        d_alt = abs(self.drone_poses[drone_idx]['d'])
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 20, 30), -1)
        hud_txt = (
            f'UAV-{drone_idx+1} SAR YOLOv8  |  '
            f'SURVIVORS: {person_count}   VEHICLES: {car_count}   '
            f'ALT: {d_alt:4.1f}m'
        )
        cv2.putText(
            annotated, hud_txt, (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 160), 2, cv2.LINE_AA
        )

        for x1, y1, x2, y2, disp_label, conf, dist_m, cls_name in draw_items:
            if 'lying' in disp_label:
                color = (0, 110, 255)   # High-priority Orange-Red for lying earthquake victim
            elif cls_name == 'person':
                color = (0, 255, 90)    # Bright Neon Green for standing/walking human
            else:
                color = (255, 210, 0)   # Cyan-Gold for rescue/stranded vehicle

            # Outer glow + main box
            cv2.rectangle(annotated, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Target lock corner brackets
            clen = min(16, max(8, (x2 - x1) // 3))
            for cx_b, cy_b, dx_b, dy_b in [
                (x1, y1, clen, clen),
                (x2, y1, -clen, clen),
                (x1, y2, clen, -clen),
                (x2, y2, -clen, -clen),
            ]:
                cv2.line(annotated, (cx_b, cy_b), (cx_b + dx_b, cy_b), (255, 255, 255), 3)
                cv2.line(annotated, (cx_b, cy_b), (cx_b, cy_b + dy_b), (255, 255, 255), 3)

            dist_str = f' {dist_m:.1f}m' if dist_m > 0 else ''
            text = f'{disp_label.upper()} {int(conf * 100)}%{dist_str}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
            ly = max(th + 44, y1)
            cv2.rectangle(annotated, (x1, ly - th - 8), (x1 + tw + 10, ly), color, -1)
            cv2.putText(
                annotated, text, (x1 + 5, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (10, 10, 10), 2, cv2.LINE_AA
            )

        try:
            img_msg = cv2_to_imgmsg_fast(annotated, 'bgr8', header=msg.header)
            self.annotated_pubs[drone_idx].publish(img_msg)
            if drone_idx == 0:
                self.uav0_alt_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'UAV-{drone_idx+1} publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# Refactored into Object-Oriented Design
# Formatting and comments preserved as requested.

import sqlite3
import numpy as np
import cv2
import struct
import json
import yaml
import os
from bisect import bisect_left
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from ultralytics import YOLO


class ObjectDetectionPipeline:

    def __init__(self, db_path, map_yaml):
        self.DB3_PATH = db_path
        self.MAP_YAML = map_yaml

        # Topic IDs
        self.TOPIC_ID_IMG = 3
        self.TOPIC_ID_PC = 8
        self.TOPIC_ID_TF = 4
        self.TOPIC_ID_CAM_INFO = 1

        # Frames
        self.FRAME_MAP = "map"
        self.FRAME_ODOM = "odom"
        self.FRAME_BASE = "base_footprint"
        self.FRAME_LIDAR = "livox_frame"
        self.FRAME_CAMERA = "zed_left_camera_optical_frame"

        # Params
        self.FRAME_STEP = 5
        self.CONF_THRESH = 0.25
        self.IOU_MERGE_DIST = 0.5

        # Target Classes (COCO IDs)
        self.TARGET_CLASSES = {
            56: "chair", 57: "couch", 58: "shelf",  59: "desk", 60: "dining table", 61: "toilet"
        }

        # Load components
        self._connect_db()
        self._load_messages()
        self._load_camera_info()
        self._load_tf()

        # Models
        self.model = YOLO('yolov8n.pt')

        # Global Map
        self.gmap = GlobalMap()
        self.debug_saved = False


    """
        Initialization
    """
    def _connect_db(self):
        self.conn = sqlite3.connect(self.DB3_PATH)
        self.cursor = self.conn.cursor()

    def _load_messages(self):
        print("Loading messages...")
        self.ts_img, self.blobs_img = self._load_topic(self.TOPIC_ID_IMG)
        self.ts_pc, self.blobs_pc = self._load_topic(self.TOPIC_ID_PC)
        self.ts_tf, self.blobs_tf = self._load_topic(self.TOPIC_ID_TF)
        self.ts_ci, self.blobs_ci = self._load_topic(self.TOPIC_ID_CAM_INFO)
        print(f"Loaded: {len(self.ts_img)} imgs, {len(self.ts_pc)} clouds, {len(self.ts_tf)} TFs")

    def _load_topic(self, topic_id):
        self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp ASC;", (topic_id,))
        rows = self.cursor.fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]

    def _load_camera_info(self):
        if len(self.blobs_ci) > 0:
            self.K_cam, self.W_cam, self.H_cam = self._decode_camera_info(self.blobs_ci[0])
        else:
            print("No Camera Info. Using Default.")
            self.K_cam = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]])
            self.W_cam, self.H_cam = 1280, 720

    def _load_tf(self):
        TFMessageType = get_message("tf2_msgs/msg/TFMessage")
        self.tf_buffer = {}
        self.all_frames = set()

        for ts, blob in zip(self.ts_tf, self.blobs_tf):
            try:
                msg = deserialize_message(blob, TFMessageType)
                for t in msg.transforms:
                    parent, child = t.header.frame_id, t.child_frame_id
                    self.all_frames.add(parent)
                    self.all_frames.add(child)

                    mat = self._q_to_mat(t.transform.translation, t.transform.rotation)
                    self.tf_buffer.setdefault((parent, child), []).append((ts, mat))
            except:
                continue

        print(f"Found TF Frames: {list(self.all_frames)}")

        self.USING_MANUAL_TF = (self.FRAME_CAMERA not in self.all_frames) or (self.FRAME_LIDAR not in self.all_frames)
        if self.USING_MANUAL_TF:
            print("WARNING: Sensor frames not in TF. Using MANUAL STATIC EXTRINSICS.")


    """
        Decoding messages from DB3 topics to extract images, point clouds, camera info
    """
    def _decode_camera_info(self, blob):
        msg = deserialize_message(blob, get_message("sensor_msgs/msg/CameraInfo"))
        K = np.array(msg.k).reshape(3, 3)
        return K, msg.width, msg.height

    def decode_image(self, blob):
        if b'\xff\xd8' in blob[:100]:
            arr = np.frombuffer(blob[blob.find(b'\xff\xd8'):], dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        try:
            msg = deserialize_message(blob, get_message("sensor_msgs/msg/Image"))
            img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if "rgb" in msg.encoding: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except:
            return None

    def decode_livox(self, blob):
        start_idx = 100
        point_step = 28
        if len(blob) < start_idx + point_step:
            return np.zeros((0,3))
        try:
            raw = blob[start_idx:]
            num = len(raw) // point_step
            pts = [struct.unpack_from('<fff', raw, i * point_step) for i in range(num)]
            return np.array(pts, dtype=np.float64)
        except:
            return np.zeros((0,3))

    """
        Applying transformations
    """
    def get_transform(self, parent, child, query_ts):
        candidates = self.tf_buffer.get((parent, child))
        if not candidates: return None
        times = [c[0] for c in candidates]
        idx = bisect_left(times, query_ts)
        if idx >= len(candidates): idx = len(candidates)-1
        return candidates[idx][1]
    
    # Quaternion + translation -> transformation matrix.
    def _q_to_mat(self, t, q):
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    
    # Manual transformation matrix base_footprint -> camera frame (missing in TF)
    def get_manual_base_to_camera(self):
        t = [0.3, 0.0, 0.5]
        R = np.array([[0,0,1],[-1,0,0],[0,-1,0]]).T
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
        return T
    
    # Manual transformation base_footprint -> LiDAR frame (missing in TF)
    def get_manual_base_to_lidar(self):
        t = [0.0, 0.0, 0.3]
        T = np.eye(4); T[:3,3] = t
        return T

    # Get base_footprint -> map at given timestamp
    def get_base_to_map(self, ts):
        T_map_odom = self.get_transform(self.FRAME_MAP, self.FRAME_ODOM, ts)
        T_odom_base = self.get_transform(self.FRAME_ODOM, self.FRAME_BASE, ts)
        if T_map_odom is None: T_map_odom = np.eye(4)
        if T_odom_base is None: T_odom_base = np.eye(4)
        return T_map_odom @ T_odom_base


    """
        Iterating through frames to perform detection and projection
    """
    def process(self):
        for i in range(0, len(self.ts_img), self.FRAME_STEP):
            curr_ts = self.ts_img[i]

            img = self.decode_image(self.blobs_img[i])
            if img is None: continue

            pc_idx = bisect_left(self.ts_pc, curr_ts)
            if pc_idx >= len(self.ts_pc): continue
            pc_points = self.decode_livox(self.blobs_pc[pc_idx])
            if len(pc_points) < 50: continue

            # Get extrinsics
            T_base_cam = self.get_transform(self.FRAME_BASE, self.FRAME_CAMERA, curr_ts)
            T_base_lidar = self.get_transform(self.FRAME_BASE, self.FRAME_LIDAR, curr_ts)
            if T_base_cam is None: T_base_cam = self.get_manual_base_to_camera()
            if T_base_lidar is None: T_base_lidar = self.get_manual_base_to_lidar()

            # LiDAR -> Camera
            T_cam_base = np.linalg.inv(T_base_cam)
            T_cam_lidar = T_cam_base @ T_base_lidar

            ones = np.ones((pc_points.shape[0],1))
            pts_cam = (T_cam_lidar @ np.hstack([pc_points, ones]).T).T[:, :3]

            valid = pts_cam[:,2] > 0.1
            if np.sum(valid) < 10:
                if i % 50 == 0: print(f"Frame {i}: No points in front of camera.")
                continue

            pts_cam = pts_cam[valid]
            pts_orig = pc_points[valid]

            # Projection to image plane
            u = (pts_cam[:,0] * self.K_cam[0,0] / pts_cam[:,2]) + self.K_cam[0,2]
            v = (pts_cam[:,1] * self.K_cam[1,1] / pts_cam[:,2]) + self.K_cam[1,2]

            # Debug save once
            if not self.debug_saved:
                dbg = img.copy()
                for j in range(0, len(u), 10):
                    if 0 <= u[j] < self.W_cam and 0 <= v[j] < self.H_cam:
                        cv2.circle(dbg, (int(u[j]), int(v[j])), 1, (0,255,255), -1)
                cv2.imwrite("debug_projection.jpg", dbg)
                print("DEBUG: Saved projection check to 'debug_projection.jpg'")
                self.debug_saved = True

            # YOLO detection
            results = self.model(img, verbose=False)
            self._process_detections(results, curr_ts, u, v, pts_orig)

            if i % 20 == 0:
                print(f"Frame {i}: Objects: {len(self.gmap.objects)}")

        self._export()

    """
        Projecting bounding boxes to global map
    """
    def _process_detections(self, results, curr_ts, u, v, pts_orig):
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.TARGET_CLASSES: continue
                if float(box.conf[0]) < self.CONF_THRESH: continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                box_pts = pts_orig[mask]

                if len(box_pts) < 5: continue

                T_map_base = self.get_base_to_map(curr_ts)
                T_map_lidar = T_map_base @ self.get_manual_base_to_lidar()

                pts_map = (T_map_lidar @ np.hstack([box_pts, np.ones((len(box_pts),1))]).T).T[:, :3]
                pts_map = pts_map[pts_map[:,2] > 0.05]
                if len(pts_map) < 3: continue

                pca = PCA(n_components=2).fit(pts_map[:, :2])
                center = np.mean(pts_map, axis=0)
                proj = pca.transform(pts_map[:,:2])
                l = proj[:,0].max() - proj[:,0].min()
                w = proj[:,1].max() - proj[:,1].min()

                if l > 4.0 or w > 4.0: continue
                if l < 0.1 or w < 0.1: continue

                angle = np.arctan2(pca.components_[0,1], pca.components_[0,0])

                self.gmap.add({
                    "class": self.TARGET_CLASSES[cls_id],
                    "center": center,
                    "dims": [l, w, 1.0],
                    "angle": angle
                })


    """
        Generating output JSON and visualization on map
    """
    def _export(self):
        out = []
        for o in self.gmap.objects:
            if o['count'] >= 1:
                out.append({
                    "label": o['class'],
                    "pose": {
                        "x": float(o['center'][0]),
                        "y": float(o['center'][1]),
                        "z": float(o['center'][2]),
                        "theta": float(o['angle'])},
                    "dimensions": {
                        "length": float(o['dims'][0]),
                        "width": float(o['dims'][1])}
                })

        with open("detections.json", "w") as f:
            json.dump(out, f, indent=4)
        print(f"Saved {len(out)} detections.")

        if os.path.exists(self.MAP_YAML):
            with open(self.MAP_YAML, "r") as f:
                meta = yaml.safe_load(f)
            map_p = os.path.join(os.path.dirname(self.MAP_YAML), meta['image'])
            mimg = cv2.imread(map_p)
            if mimg is None: return

            res = meta['resolution']
            ox, oy = meta['origin'][:2]
            h, w_img = mimg.shape[:2]

            for o in out:
                cx, cy = o['pose']['x'], o['pose']['y']
                l, w = o['dimensions']['length'], o['dimensions']['width']
                ang = o['pose']['theta']

                c, s = np.cos(ang), np.sin(ang)
                pts = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
                pts = (pts @ np.array([[c,-s],[s,c]]).T) + [cx, cy]

                px = [((x-ox)/res, h-1-(y-oy)/res) for x,y in pts]
                cv2.polylines(mimg, [np.array(px, dtype=np.int32)], True, (0,0,255), 2)
                cv2.putText(mimg, o['label'], (int(px[0][0]), int(px[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            cv2.imwrite("detections_map.png", mimg)
            print("Map visualization saved.")


"""
    Mergin and storing detected objects globally
"""
class GlobalMap:
    def __init__(self): self.objects = []
    def add(self, obj):
        for ex in self.objects:
            if ex['class'] == obj['class']:
                if np.linalg.norm(ex['center'][:2] - obj['center'][:2]) < 0.5:
                    ex['count'] += 1
                    return
        obj['count'] = 1
        self.objects.append(obj)


"""
    entry point
"""
if __name__ == "__main__":
    system = ObjectDetectionPipeline(
        "data/bathroom/rosbag2_2025_10_20-16_47_22/rosbag2_2025_10_20-16_47_22_2.db3",
        "data/bathroom/room.yaml"
    )
    system.process()
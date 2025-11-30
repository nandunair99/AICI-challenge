import sqlite3
import numpy as np
import cv2
import struct
from bisect import bisect_left

# -----------------------------------------------------------
# 1. Open DB3 (ROS2 bag)
# -----------------------------------------------------------
conn = sqlite3.connect("data/office/rosbag2_2025_10_20-16_09_39/rosbag2_2025_10_20-16_09_39_1.db3")
cursor = conn.cursor()

# Topics (replace with yours)
IMG_TOPIC_ID = 3       # /zed/... image topic
PC_TOPIC_ID  = 8       # /livox/lidar

# -----------------------------------------------------------
# 2. Utility: Read (timestamp, data) messages for a topic
# -----------------------------------------------------------
def load_topic_messages(topic_id):
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp ASC;",
        (topic_id,)
    )
    rows = cursor.fetchall()
    timestamps = [r[0] for r in rows]
    blobs = [r[1] for r in rows]
    return timestamps, blobs


img_timestamps, img_blobs = load_topic_messages(IMG_TOPIC_ID)
pc_timestamps,  pc_blobs  = load_topic_messages(PC_TOPIC_ID)

print("Images:", len(img_timestamps))
print("Point clouds:", len(pc_timestamps))

# -----------------------------------------------------------
# 3. Function: find nearest PC timestamp for image timestamp
# -----------------------------------------------------------
def find_nearest_pc(img_ts):
    """Return index of point cloud with closest timestamp to img_ts."""
    idx = bisect_left(pc_timestamps, img_ts)

    if idx == 0:
        return 0
    if idx == len(pc_timestamps):
        return len(pc_timestamps) - 1

    before = pc_timestamps[idx - 1]
    after = pc_timestamps[idx]

    return idx if abs(after - img_ts) < abs(img_ts - before) else idx - 1

# -----------------------------------------------------------
# 4. Image decoding helper (extract JPEG inside blob)
# -----------------------------------------------------------
def decode_image(blob):
    jpeg_start = blob.find(b'\xff\xd8')
    if jpeg_start == -1:
        return None
    arr = np.frombuffer(blob[jpeg_start:], dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# -----------------------------------------------------------
# 5. Lidar decoding helper (Livox)
# -----------------------------------------------------------
def decode_livox_pointcloud(blob, header_offset=500):
    data = blob[header_offset:]
    point_step = 28  # x,y,z,i (4*4) + timestamp(8) + tag(4)
    num_points = len(data) // point_step
    points = np.zeros((num_points, 6), dtype=np.float64)

    for i in range(num_points):
        offset = i * point_step
        x, y, z, intensity = struct.unpack_from('<ffff', data, offset)
        timestamp = struct.unpack_from('<Q', data, offset + 16)[0]
        tag = struct.unpack_from('<I', data, offset + 24)[0]
        points[i] = [x, y, z, intensity, timestamp, tag]

    return points

# -----------------------------------------------------------
# 6. Build synchronized image + point cloud pairs
# -----------------------------------------------------------
paired_frames = []

for i, (img_ts, img_blob) in enumerate(zip(img_timestamps, img_blobs)):
    pc_idx = find_nearest_pc(img_ts)

    paired_frames.append({
        "img_timestamp": img_ts,
        "pc_timestamp": pc_timestamps[pc_idx],
        "img_blob": img_blob,
        "pc_blob": pc_blobs[pc_idx]
    })

print(f"Total synchronized frames: {len(paired_frames)}")


# -----------------------------------------------------------
# 7. Example: decode and show frame #200
# -----------------------------------------------------------
frame_id = 200
frame = paired_frames[frame_id]

img = decode_image(frame["img_blob"])
pc  = decode_livox_pointcloud(frame["pc_blob"])

print("\nFrame:", frame_id)
print("Image TS:", frame["img_timestamp"])
print("PC TS   :", frame["pc_timestamp"])
print("PC shape:", pc.shape)
print("First points:\n", pc[:5])

cv2.imshow("Synchronized Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

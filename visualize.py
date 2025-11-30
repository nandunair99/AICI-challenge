import sqlite3, json, yaml, struct, numpy as np, cv2
from ultralytics import YOLO
from sklearn.decomposition import PCA

# --- Config ---
bag_file = "data/office/rosbag2_2025_10_20-16_09_39/rosbag2_2025_10_20-16_09_39_1.db3"
map_file = "data/office/room.pgm"
map_yaml = "data/office/room.yaml"
output_map_file = "result_map.png"
classes_to_detect = ["chair","couch","table","shelf","bathtub","wc"]

# --- Load map ---
map_img = cv2.imread(map_file, cv2.IMREAD_COLOR)
with open(map_yaml, 'r') as f:
    map_meta = yaml.safe_load(f)
resolution = map_meta['resolution']
origin = map_meta['origin']  # [x0, y0, theta]

def world_to_map(x,y):
    mx = int((x - origin[0])/resolution)
    my = map_img.shape[0] - int((y - origin[1])/resolution)
    return mx,my

# --- SQLite connection ---
conn = sqlite3.connect(bag_file)
cursor = conn.cursor()


# --- Pick image + point cloud by topic IDs ---
img_topic_id = 4   # '/zed/zed_node/rgb/image_rect_color/compressed'
pc_topic_id  = 8   # '/livox/lidar'

def get_messages_by_id(topic_id):
    cursor.execute("SELECT data FROM messages WHERE topic_id=?;", (topic_id,))
    return [row[0] for row in cursor.fetchall()]

# Load all messages
img_msgs = get_messages_by_id(img_topic_id)
print("depth of image msgs:", img_msgs[:10])

#  Projecting object detections into an occupancy grid map:

## Approach
In this approach LiDAR obtained from db3 are transformed into camera frame  and projected onto the image. YOLOv8 detects the objects in the image  and corresponding liDAR points are extracted and transformed to map frame. PCA estimates the objects orientation, dimensions and position. Observations are merged over frames to build a global object map, saved as JSON and optionally visualized on a 2D map.

Steps:
1. Load sensor data from a ROS2 SQLite database (.db3)
2. Parse and store TF messages to maintain a transform tree
3. Point Cloud Transformations like LiDAR frame -> camera frame
4. Project the transformed LiDAR points into the 2D camera image
5. Run YOLOv8 on the camera image to detect objects
6. Associate LiDAR points with detected bounding boxes in the image using the projected 2D points
7. Transform the associated 3D points from LiDAR frame to global map frame
8. Apply PCA on the 3D points in map frame to estimate object orientation
9. Merge multiple observations of the same object over time using IOU
10. Generate a JSON file with object class, pose, and dimensions.

Challenges faced:
This approach requires us to have base_footprint -> camera_frame and base_footprint to livox_frame mapping in the /tf topic which was not present. So I have made assumptions for those matrices based on the assumed mounting positions of the sensors which breaks projection and mapping. 

Another alternative I thought of was performing 3D object detection using clustering, but since the point clouds were sparse, which made localization unreliable.

Note: Code starting point- obj_detection_pipeline.py
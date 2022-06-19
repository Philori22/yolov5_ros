# yolov5_ros

This is a ROS interface/node for using YOLOv5 for real time object detection on a ROS image topic received via laptop camera. It is a crude construct making use of the official YOLOv5 repository, to allow for offline detection when using `torch.hub.load()` function.

#! /usr/bin/env python3

from calendar import c
import rospy
from std_msgs.msg import String

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import os

# from objectdetection import *
device = (torch.device("cuda") if torch.cuda.is_available() else ("cpu"))

class obj_det:
    def __init__(self) -> None:
        # Load Yolov5s model
        self.model = torch.hub.load( '/home/cabbage/catkin_ws/src/rosbot_demo/yolov5' , 'custom', '/home/cabbage/catkin_ws/src/yolov5_ros/scripts/yolov5s.pt', source='local')

        # Initialise ROS-based packages
        self.bridge = CvBridge()
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.img_pub = rospy.Publisher("/output_img", Image)  
        rospy.spin()
    
    def image_callback(self, msg):
        
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        except CvBridgeError as e:
            print(e)

        results = self.model(cv2_img)
        class_list = results.names
        img_bbox = self.draw_bboxes_for_yolov5(results=results, img=cv2_img, class_list=class_list)
        print("done processing shit")

        rosmsg = self.bridge.cv2_to_imgmsg(img_bbox, "bgr8")
        self.img_pub.publish(rosmsg)
        rospy.loginfo("got an image")

    def draw_bboxes_for_yolov5(self, results=None, img=None, class_list=None):
        '''
        NB! not used, since bboxes are scaled to im0, and not sure where im0 comes from:
        https://github.com/ultralytics/yolov5/blob/master/detect.py#L151
        '''
        img_bbox = img.copy()
        for det in results.xywhn[0]:
            print("det.cpu().numpy().tolist() = ", det.cpu().numpy().tolist())
        for det in results.xywh[0]:
            print("det = ", det)
            print("det.cpu().numpy().tolist() = ", det.cpu().numpy().tolist())
            x,y,w,h,conf,class_idx = det.cpu().numpy().tolist()
            x,y,w,h = int(x),int(y),int(w),int(h)
            class_idx = int(class_idx)
            class_name = class_list[class_idx]
            print("x,y,w,h = ", x,y,w,h, "conf = ", conf, "class_name = ", class_name)
            cv2.rectangle(img_bbox,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img_bbox, class_name,(x,y-30),0,0.7,(0,255,0))
            cv2.putText(img_bbox, format(conf, ".3f") ,(x+w-30,y-10),0,0.7,(0,255,0))
        return img_bbox

def main():
    rospy.init_node('yolov5_detector')    
    od = obj_det()

if __name__ == '__main__':
    main()
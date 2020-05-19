#!/usr/bin/env python
import os
import time
import numpy as np
import imutils

from bspline_path import  approximate_b_spline_path
from cubic_spline_planner import  Spline2D

from scipy import ndimage
import scipy.special
from skimage.measure import  label
from sklearn import linear_model, datasets



from bspline_path import  approximate_b_spline_path

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


from std_msgs.msg import Float32MultiArray,Int32MultiArray
from sensor_msgs.msg import Image

import rospy


show_animation = True

        
def adjust_gamma(image, gamma=0.8):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    image = np.power(image/255.0, gamma)
    return (image * 255.0).astype(np.uint8)



class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.lane_pred_sub= rospy.Subscriber("/Lane/pred", Image, self.lane_pred_callback)
        self.lane_mask_sub= rospy.Subscriber("Drive/pred_main", Image, self.lane_main_callback)
        self.lane_exist_sub= rospy.Subscriber("/Lane/exist", Float32MultiArray, self.lane_exist_callback)

        self.lane_pred_ok = False
        self.lane_main_ok = False
        self.lane_exist_ok = False

    def lane_pred_callback(self, msg):
        self.lane_pred = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.lane_pred_ok = True
    def lane_main_callback(self,msg):
        self.lane_main = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.lane_main_ok = True
    def lane_exist_callback(self,msg):
        self.lane_exist = msg.data
        self.lane_exist_ok = True

def main(curve_pub,curve_stutus_pub,curve_poly_pub):
    img_sub = Img_Sub()
    rate = rospy.Rate(10)
    while not (img_sub.lane_pred_ok and img_sub.lane_main_ok and img_sub.lane_exist_ok):
        time.sleep(0.2)
        continue
    while not rospy.is_shutdown():
        bridge = CvBridge()
        lane_exist = img_sub.lane_exist
        # Loading Image 
        pred_lane = img_sub.lane_pred.copy()
        # pred_lane = adjust_gamma(pred_lane)

        pred_main = img_sub.lane_main.copy()
        pred_lane_gray = cv2.cvtColor(pred_lane,cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((30,30),np.uint8)  
        height ,width ,channel=  pred_lane.shape
        # predict class
        start = time.time()


        # Otsu's thresholding after Gaussian filtering ------------
        _,main_mask = cv2.threshold(pred_main[:,:,1],0,255,cv2.THRESH_OTSU)
        # Boundary caculate ---------------------------------------
        # Find line point from boundary of drive area
        main_mask_dilate = cv2.dilate(main_mask.copy(),kernel)
        _,pred_lane_bin = cv2.threshold(pred_lane_gray,0,255,cv2.THRESH_OTSU)
        boundary = main_mask_dilate-main_mask 
        filter_mask = np.logical_and(boundary,pred_lane_bin)




        # # Linear Binary classifier -------------------------
        # c = np.polyfit([dst_corner_y,center_main_y],[dst_corner_x,center_main_x],1)
        # poly = np.poly1d(c)


       # Resample method -----------------------------
       
        index = np.array(np.arange(pred_lane_gray.shape[0]*pred_lane_gray.shape[1]))
        sample = np.zeros(pred_lane_gray.shape).flatten()
        p_left = (pred_lane[:,:,1].flatten().astype(float)/pred_lane[:,:,1].astype(float).sum())
        p_right = (pred_lane[:,:,2].flatten().astype(float)/pred_lane[:,:,2].astype(float).sum())
        # print(pred_lane[:,:,0].astype(float).sum())
        ss_left = []
        ss_right =  []
        lane_status = [1,1]
        try:
            ss_left = np.random.choice(index,(576//16,1024//16),replace=False,p=p_left)
        except:
            lane_status[0] = 0
            rospy.WARN("no enough point Left")
        try:
            ss_right = np.random.choice(index,(576//16,1024//16),replace=False,p=p_right)
        except:
            lane_status[1] = 0
            rospy.WARN("no enough point Right")
            
        
        
        sample[ss_left] = 1
        sample[ss_right] = 2
        sample = sample.reshape(pred_lane_gray.shape)
        boundary[boundary>1]=1
        filter_sample = sample * boundary
        


        y = np.arange(int(300),height-10,30)

        left_y,left_x = np.where(filter_sample==1)
        right_y,right_x = np.where(filter_sample==2)
        num_fit_left_point = len(left_y)
        num_fit_right_point = len(right_y)
        
        print('Fit point {}'.format(num_fit_left_point))
        tmp = np.zeros(pred_lane.shape,dtype=np.uint8)
        if lane_status[0] and lane_exist[1]>0.9 and num_fit_left_point>50:
            
            c_left = np.polyfit(left_y,left_x,3)
            poly_left = np.poly1d(c_left)
            left_x = poly_left(y)
            left_x = np.clip(left_x,0,width)


            pts = np.zeros((len(y),1,2),dtype=int)
            pts[:,0,0] = left_x.astype(int)
            pts[:,0,1] = y
        
            cv2.polylines(tmp,[pts],False,(255,0,0),8)
        if lane_status[1] and lane_exist[2]>0.9 and num_fit_right_point>50:

            
            c_right = np.polyfit(right_y,right_x,3)
            poly_right = np.poly1d(c_right)
            right_x = poly_right(y)
            right_x = np.clip(right_x,0,width-10)
        

            pts = np.zeros((len(y),1,2),dtype=int)
            pts[:,0,0] = right_x.astype(int)
            pts[:,0,1] = y
        
            cv2.polylines(tmp,[pts],False,(0,255,0),8)






        # Draw vis plot
        lane_vis = np.zeros(pred_lane.shape,dtype=np.uint8)
        lane_vis[filter_sample==1,1]=255
        lane_vis[filter_sample==2,2]=255

        status_msg = Int32MultiArray()
        status_msg.data = lane_status
        curve_stutus_pub.publish(status_msg)
        curve_pub.publish(bridge.cv2_to_imgmsg(lane_vis,'bgr8'))
        curve_poly_pub.publish(bridge.cv2_to_imgmsg(tmp,'bgr8'))
        stop = time.time()
        print("Time: {}".format(stop-start))
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('LaneFitting', anonymous=True)
    curve_pub = rospy.Publisher("Curve/lane", Image)
    curve_poly_pub = rospy.Publisher("Curve/line_poly", Image)
    curve_stutus_pub = rospy.Publisher("Curve/status", Int32MultiArray)
    main(curve_pub,curve_stutus_pub,curve_poly_pub)




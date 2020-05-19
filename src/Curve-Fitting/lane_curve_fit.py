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
from skimage.feature import peak_local_max
from bezier_path import calc_4points_bezier_path
        
def adjust_gamma(image, gamma=0.8):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    image = np.power(image/255.0, gamma)
    return (image * 255.0).astype(np.uint8)

def g_h_filter(data, x0, dx, g, h, dt=1.):
    x_est = x0
    results = []
    for z in data:
        # prediction step
        x_pred = x_est + (dx*dt)
        dx = dx

        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)

class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.lane_pred_sub= rospy.Subscriber("/Lane/pred", Image, self.lane_pred_callback)
        self.lane_mask_sub= rospy.Subscriber("Drive/pred_main", Image, self.lane_main_callback)
        self.lane_exist_sub= rospy.Subscriber("/Lane/exist", Float32MultiArray, self.lane_exist_callback)
        self.lane_mask_sub= rospy.Subscriber("/Lane/mask", Image, self.lane_mask_callback)

        self.lane_pred_ok = False
        self.lane_main_ok = False
        self.lane_exist_ok = False
        self.lane_mask_ok = False

    def lane_pred_callback(self, msg):
        self.lane_pred = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.lane_pred_ok = True
    def lane_main_callback(self,msg):
        self.lane_main = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.lane_main_ok = True
    def lane_exist_callback(self,msg):
        self.lane_exist = msg.data
        self.lane_exist_ok = True
    def lane_mask_callback(self,msg):
        self.lane_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.lane_mask_ok = True

def main(curve_pub,curve_stutus_pub,curve_poly_pub,curve_dist_pub,line_points_pub):
    img_sub = Img_Sub()
    rate = rospy.Rate(10)
    while not (img_sub.lane_pred_ok and img_sub.lane_main_ok and img_sub.lane_exist_ok and img_sub.lane_mask_ok):
        time.sleep(0.2)
        continue
    
    while not rospy.is_shutdown():
        bridge = CvBridge()
        lane_exist = img_sub.lane_exist
        # Loading Image 
        pred_lane = img_sub.lane_pred.copy()
        pred_lane = adjust_gamma(pred_lane)

        pred_main = img_sub.lane_main.copy()
        pred_main = adjust_gamma(pred_main)
        pred_lane_gray = cv2.cvtColor(pred_lane,cv2.COLOR_BGR2GRAY)

        lane_mask_line = img_sub.lane_mask.copy()
        


        pred_main[lane_mask_line>0,1] = 0

        kernel = np.ones((30,30),np.uint8)  
        height ,width ,channel=  pred_lane.shape
        # predict class
        start = time.time()


        # Otsu's thresholding after Gaussian filtering ------------
        _,main_mask = cv2.threshold(pred_main[:,:,1],0,255,cv2.THRESH_OTSU)
        dist_main_mask = cv2.distanceTransform(main_mask*pred_main[:,:,1],cv2.DIST_L1, 5)
        # Boundary caculate ---------------------------------------
        # Find line point from boundary of drive area
        main_mask_dilate = cv2.dilate(main_mask.copy(),kernel)
        _,pred_lane_bin = cv2.threshold(pred_lane_gray,0,255,cv2.THRESH_OTSU)
        boundary = main_mask_dilate-main_mask 
        filter_mask = np.logical_and(boundary,pred_lane_bin)
        # main_mask = main_mask - pred_lane_bin

        '''
            Labels is a matrix the size of the input image where each element has a value equal to its label.
            Stats is a matrix of the stats that the function calculates.
            stats[label, COLUMN]
            cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
            cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
            cv2.CC_STAT_WIDTH The horizontal size of the bounding box
            cv2.CC_STAT_HEIGHT The vertical size of the bounding box
            cv2.CC_STAT_AREA The total area (in pixels) of the connected component
        '''
        connectivity = 4  
        # Perform the operation
        output = cv2.connectedComponentsWithStats(main_mask.astype(np.uint8), connectivity)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        largest_label = np.argsort(stats[:,cv2.CC_STAT_AREA])[-2]
        main_mask = (labels==largest_label).astype(np.uint8)
        
        mask_height = stats[largest_label,cv2.CC_STAT_HEIGHT]
        main_mask_top_slice = main_mask.copy()
        main_mask_top_slice[height-mask_height+20:] = 0
        top_center = np.array(ndimage.measurements.center_of_mass(main_mask_top_slice)).astype(int)
        # plt.imshow(main_mask_top_slice)
        # plt.show()
        print('-------------')





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
            ss_left = np.random.choice(index,(576//8,1024//8),replace=False,p=p_left)
        except:
            lane_status[0] = 0
            rospy.logwarn("no enough point Left")
        try:
            ss_right = np.random.choice(index,(576//8,1024//8),replace=False,p=p_right)
        except:
            lane_status[1] = 0
            rospy.logwarn("no enough point Right")
            
        
        
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
        # print("Number_left {}".format(num_fit_left_point))
        # print("num_fit_right_point {}".format(num_fit_right_point))
        
        
        tmp = np.zeros(pred_lane.shape,dtype=np.uint8)
        if lane_status[0] and lane_exist[1]>0.8 and num_fit_left_point>80:
            
            c_left = np.polyfit(left_y,left_x,2)
            poly_left = np.poly1d(c_left)
            left_x = poly_left(y)
            left_x = np.clip(left_x,0,width)


            pts = np.zeros((len(y),1,2),dtype=int)
            pts[:,0,0] = left_x.astype(int)
            pts[:,0,1] = y
        
            cv2.polylines(tmp,[pts],False,(255,0,0),8)
        if lane_status[1] and lane_exist[2]>0.8 and num_fit_right_point>80:

            
            c_right = np.polyfit(right_y,right_x,2)
            poly_right = np.poly1d(c_right)
            right_x = poly_right(y)
            right_x = np.clip(right_x,0,width-10)
        

            pts = np.zeros((len(y),1,2),dtype=int)
            pts[:,0,0] = right_x.astype(int)
            pts[:,0,1] = y
        
            cv2.polylines(tmp,[pts],False,(0,255,0),8)

        # Distance Transform  ----------------------------------
        dist_main_mask = dist_main_mask.astype(np.uint8)


        ret, sure_fg = cv2.threshold(dist_main_mask,0.9*dist_main_mask.max(),255,0)

        

        ret, course_fg = cv2.threshold(dist_main_mask,0.3*dist_main_mask.max(),255,0)
        center_mass = np.array(ndimage.measurements.center_of_mass(course_fg)).astype(int)


        # Draw vis plot
        lane_vis = np.zeros(pred_lane.shape,dtype=np.uint8)
        lane_vis[filter_sample==1,1]=255
        lane_vis[filter_sample==2,2]=255

        # Pure center of mass method
        mask_slice = []

        # -----------------




        # Linear Binary classifier -------------------------
        
    
        start_x = float(1024/2.0 )  # [m]
        start_y = float(570.)  # [m]

        
        diff = (center_mass[1]-start_x,center_mass[0]-start_y)
        angle = np.angle([diff[0]+diff[1]*1j])[0]
        start_yaw = angle  # [rad]

        
        
        end_x = float(top_center[1])  # [m]
        end_y = float(top_center[0])  # [m]
        
        diff = (end_x - center_mass[1],end_y - center_mass[0])
        angle_ = np.angle([diff[0]+diff[1]*1j])[0]
        end_yaw = angle_  # [rad]
        offset = 3.0
        
    

        # Plot start and end control point ---------------------------------------
        cv2.circle(dist_main_mask,(int(center_mass[1]),int(center_mass[0])),5,255,3)
        cv2.circle(dist_main_mask,(int(end_x),int(end_y)),5,255,3)
        cv2.circle(dist_main_mask,(int(start_x),int(start_y)),5,255,3)

        path, control_points = calc_4points_bezier_path(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)

        mid_point = path.flatten().tolist()
        msg_midpoint = Float32MultiArray()
        msg_midpoint.data = mid_point
        line_points_pub.publish(msg_midpoint)






        status_msg = Int32MultiArray()
        status_msg.data = lane_status

        
        curve_dist_pub.publish(bridge.cv2_to_imgmsg(dist_main_mask.astype(np.uint8),'mono8'))
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
    curve_dist_pub = rospy.Publisher("Curve/dist", Image)
    line_points_pub = rospy.Publisher("Drive/main_point", Float32MultiArray)
    main(curve_pub,curve_stutus_pub,curve_poly_pub,curve_dist_pub,line_points_pub)




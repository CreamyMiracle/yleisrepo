#!/usr/bin/env python
import rospy
import sys
import os
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from marker_msgs.msg import MarkerDetection
from tf.transformations import euler_from_quaternion
sys.path.insert(1, '/home/ros/catkin_ws/src/moro_g12/src/')
from ekf import EKF
from csv_writer import csv_writer

previous_timestamp = None
ekf = EKF(3, 2, 2)
beacons = [(7.3, 3), (1, 1), (9, 9), (1, 8), (5.8, 8)] # xy coordinates

def get_rotation(quaternion):
    global roll, pitch, yaw
    orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return wrap_to_pi(yaw)


def odom_callback(msg):
    timestamp = msg.header.stamp
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    
    theta = get_rotation(msg.pose.pose.orientation) # radians

    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z

    # Calculate elapsed time
    global previous_timestamp
    if (previous_timestamp == None):
        dt = 1
    else:
        dt = timestamp.to_sec() - previous_timestamp.to_sec()

    previous_timestamp = timestamp;        
    
    # Predict robot state based on it's velocities
    ekf.set_params(v, w, dt)   
    ekf.predict()
    
    # Write data to CSV, note that room center is at 5,5
    csv_writer('odom', 5. + x, 5. + y, theta)
    csv_writer('ekf', (ekf.state_vector[0])[0], (ekf.state_vector[1])[0], (ekf.state_vector[2])[0])
 

def ground_truth_callback(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    
    theta = get_rotation(msg.pose.pose.orientation) # radians
    
    # Write data to CSV, note that room center is at 5,5
    csv_writer('ground_truth', 5. + x, 5. + y, theta)
   

def marker_callback(msg):
    timestamp = msg.header.stamp
    markersList = []
    # get beacons info
    for marker in msg.markers:
        markersList.append((
            marker.ids[0],
            (beacons[marker.ids[0] - 1])[0],
            (beacons[marker.ids[0] - 1])[1],
            marker.pose.position.x, 
            marker.pose.position.y, 
            get_rotation(marker.pose.orientation), # theta in radians
            timestamp))
            
    # Propagate robot state using Kalman filter
    ekf.set_beacons(markersList)
    ekf.propagate_state()
    

def ekf_loc():
    rospy.init_node('ekf_localization', anonymous=True)
    rospy.Subscriber("odom", Odometry, odom_callback)
    rospy.Subscriber("base_pose_ground_truth", Odometry, ground_truth_callback)
    rospy.Subscriber("base_marker_detection", MarkerDetection, marker_callback)
    rospy.spin()


def wrap_to_pi(angle):
    """
    Wraps the input angle between [-pi,pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
    

if __name__ == '__main__':
    # Delete old CSVs
    try:
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'odom.csv'))
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf.csv') )
    except:
        pass
    
    ekf_loc()

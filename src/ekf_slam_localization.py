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
from ekf_slam import EKF_SLAM
from csv_writer import csv_writer

previous_timestamp = None
ekf = EKF_SLAM(5, 3, 2, 2)


def get_rotation(quaternion):
    global roll, pitch, yaw
    orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return wrap_to_pi(yaw)


def odom_callback(msg):
    timestamp = msg.header.stamp
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = get_rotation(msg.pose.pose.orientation)
    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z

    # Calculate elapsed time
    global previous_timestamp
    if (previous_timestamp == None):
        dt = 1
    else:
        dt = timestamp.to_sec() - previous_timestamp.to_sec()

    previous_timestamp = timestamp

    # Prediction step
    ekf.set_params(v, w, dt)
    # ekf.predict()

    # Write results to csv
    csv_writer('odom', 5. + x, 5. + y, theta)


def ground_truth_callback(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    theta = get_rotation(msg.pose.pose.orientation)  # radians

    csv_writer('ground_truth', x, y, theta)


def marker_callback(msg):
    """
    In SLAM only the range and heading are known, not beacon location
    """
    markersList = []
    # get beacons info
    for marker in msg.markers:
        markersList.append((
            np.sqrt(np.square(marker.pose.position.x) + np.square(marker.pose.position.y)),  # range
            get_rotation(marker.pose.orientation),  # theta in radians
            marker.ids[0]))

        # Posteriori step
    ekf.set_observed_features(markersList)
    ekf.predict()
    ekf.update()
    csv_writer('ekf_slam', (ekf.mean[0])[0], (ekf.mean[1])[0], (ekf.mean[2])[0])


def ekf_loc():
    rospy.init_node('ekf_slam_localization', anonymous=True)
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
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf_slam.csv'))
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ground_truth.csv'))
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'odom.csv'))
    except:
        pass

    ekf_loc()
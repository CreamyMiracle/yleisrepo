"""
Run this to visualize data from CSV.
Assumes that the data to be drawn is found in the same folder as this file
"""
import matplotlib.pyplot as plt
import os
import csv

odom_x = []
odom_y = []
ekf_x = []
ekf_y = []
ekf_slam_x = []
ekf_slam_y = []
ground_truth_x = []
ground_truth_y = []


odom_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'odom.csv')
ekf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf.csv')
ekf_slam_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf_slam.csv')
ground_truth_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ground_truth.csv')

try:
    with open(odom_path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            odom_x.append(row[0])
            odom_y.append(row[1])
except:
    pass

try:
    with open(ekf_path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            ekf_x.append(row[0])
            ekf_y.append(row[1])
except:
    pass

try:
    with open(ekf_slam_path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            fixed_row = [0., 0.]
            i = 0
            for val in row:
                val = float(val.replace("[", "", 2).replace("]", "", 2).strip())
                print(val, "   ", i)
                fixed_row[i] = val

                if i == 1:
                    break
                i += 1
            ekf_slam_x.append(fixed_row[0])
            ekf_slam_y.append(fixed_row[1])
except:
    pass

try:
    with open(ground_truth_path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            ground_truth_x.append(row[0])
            ground_truth_y.append(row[1])
except:
    pass

plt.plot(odom_x, odom_y, label='Odometry')
plt.plot(ekf_x, ekf_y, label='EKF')
plt.plot(ekf_slam_x, ekf_slam_y, label='EKF SLAM')
plt.plot(ground_truth_x, ground_truth_y, label='Ground truth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

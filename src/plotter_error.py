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
pf_x = []
pf_y = []
ekf_slam_x = []
ekf_slam_y = []
ground_truth_x = []
ground_truth_y = []


odom_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'odom.csv')
ekf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf.csv')
pf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pf.csv')
ekf_slam_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ekf_slam.csv')
ground_truth_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ground_truth.csv')

with open(odom_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        odom_x.append(row[0])
        odom_y.append(row[1])

with open(ekf_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        ekf_x.append(row[0])
        ekf_y.append(row[1])

with open(pf_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        pf_x.append(float(row[0]))
        pf_y.append(float(row[1]))

with open(ekf_slam_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        ekf_slam_x.append(row[0])
        ekf_slam_y.append(row[1])

with open(ground_truth_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        ground_truth_x.append(float(row[0]))
        ground_truth_y.append(float(row[1]))

import numpy as np
max_len = len(pf_x) # GT has more samples of filters, need to trim it
x = np.abs(np.subtract(pf_x, ground_truth_x[0:max_len]))
y = np.abs(np.subtract(pf_y, ground_truth_y[0:max_len]))

plt.plot(x, label='Error X')
plt.plot(y, label='Error Y')
plt.xlabel('Sample')
plt.ylabel('Error')
plt.legend()
plt.show()

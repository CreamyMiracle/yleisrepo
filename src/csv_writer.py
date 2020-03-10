"""
Write robot coordinates to CSV files in current the same folder as this file
is in
"""
import csv

def csv_writer(filename, x, y, theta):
    import os
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename + '.csv')
    
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow([str(x), str(y), str(theta)])
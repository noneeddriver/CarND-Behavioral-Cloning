import csv
import cv2
import numpy as np
#original data
lines =[]
with open('./data/driving_log1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print('loading udacity-data(clock direction)')    
del lines[0]

with open('./data_clock_1/data4/driving_log.csv') as csvfile1:
    reader1 = csv.reader(csvfile1)
    for line in reader1:
        lines.append(line)
    print('loading data4(clock direction)')
    
with open('./data_clock_1/data5/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)
    print('loading data5(encouter clock direction)')

with open('./data_track1/data6/driving_log.csv') as csvfile3:
    reader3 = csv.reader(csvfile3)
    for line in reader3:
        lines.append(line)
    print('loading data6(recovery)')

with open('./data_track2/data8/driving_log1.csv') as csvfile4:
    reader4 = csv.reader(csvfile4)
    for line in reader4:
        lines.append(line)
    print('loading data8(track 2)')

    angles = []
for line in lines:
    angle = float(line[3])
    angles.append(angle)
print('Original Data Number: ', len(lines))
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# the histogram of the data
fig = plt.figure(figsize=(15,15))
plt.subplot(121)
n, bins, patches = plt.hist(angles, 25, facecolor='green', rwidth =0.8)

plt.xlabel('Angles')
plt.ylabel('data number')
plt.title(str(len(lines))+' Original Data')
plt.axis([-1, 1, 0, 17000])
plt.grid(True)

# print(bins)

#adapt the data to get a balanced distrubution
lines_new = []
angles_new = []
for line in lines:
    if (float(line[3])<0.05) and (float(line[3])>-0.05):
        if np.random.rand()< 0.4:
            lines_new.append(line)
    elif (float(line[3])<-0.05) and (float(line[3])>-0.2):
        if np.random.rand()< 0.75:
            lines_new.append(line)
    else:
        lines_new.append(line)
        
for line in lines_new:
    angle_new = float(line[3])
    angles_new.append(angle_new)
plt.subplot(122)
plt.hist(angles_new, 25, facecolor='green', rwidth =0.8)
plt.xlabel('Angles')
plt.ylabel('data number')
plt.title(str(len(lines_new))+' Adapted Data')
plt.axis([-1, 1, 0, 7000])
plt.grid(True)
fig.savefig('./angle_distribution.png', dpi = 300)
print('Adapted Data Number: ', len(lines_new))
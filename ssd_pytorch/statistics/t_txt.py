import os
import numpy as np


ay = np.loadtxt("result_cam_phone/te.txt", delimiter=",")
a = ay.tolist()

b = []
for i in range(len(a)):
    if i % 3 != 2:
       b.append(a[i])


c = np.array(b).reshape(int(len(b)/2), 2)

c = np.mat(c)
np.savetxt("result_cam_phone/te2.txt", c, fmt="%d %.4f")
#!/usr/bin/env python
# coding: utf-8
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd



net = build_ssd('test', 300, 2)    # initialize SSD
# net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
net.load_weights('../weights/VOC.pth')



image = cv2.imread('../data/VOCdevkit/VOC2007/JPEGImages/000010.jpg', cv2.IMREAD_COLOR)
# uncomment if dataset not downloaded
# images = cv2.imread('../data/VOCdevkit/VOC2007/JPEGImages', cv2.IMREAD_COLOR)
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
# testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
# img_id = 4
# image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
plt.figure(figsize=(10,10))


x = cv2.resize(image, (300, 300)).astype(np.float32)
# kernel = np.array([[-1, -1, -1],
#                       [-1, 9, -1],
#                       [-1, -1, -1]])
# x = cv2.filter2D(x, -1, kernel)



plt.imshow(x)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)



xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)



from data import VOC_CLASSES as labels
top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    # j = 0
    # while detections[0,i,j,0] >= 0.6:
        i=1
        j=0
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        # cv2.rectangle(image,(int(pt[0]),int(pt[1])),(int(pt[2]), int(pt[3])),(5,200,100),40)
        x1 = int(pt[0])
        y1 = int(pt[1])
        x2 = int(pt[2])
        y2 = int(pt[3])
        roi = image[y1:y2, x1:x2]
        # cv2.namedWindow('a',0)
        # cv2.imshow('a',roi)
        # cv2.waitKey(0)
        outpath = "../data/VOCdevkit/VOC2007/bounderbox/000001.jpg"
        cv2.imwrite(outpath, roi)
        plt.show()
        # j+=1


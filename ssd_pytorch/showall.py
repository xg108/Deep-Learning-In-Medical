import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd
from data import VOC_CLASSES as labels
import xml.etree.cElementTree as ET

from PIL import Image, ImageDraw, ImageFont



# with open('data/VOCdevkit/VOC2007/results/new_test_plaque.txt','r') as f1:
#         list1=f1.readlines()
#         sum="0"
#         for i in range(len(list1)):
#            if i >= 2 :
#                break;
#            str=list1[i].split('\n')
#            str=str[0]
#            list2=str.split(' ')
#
#            id=list2[0]
#
#            if  sum==id:
#                continue
#            else:
#             image = cv2.imread('data/VOCdevkit/VOC2007/JPEGImages/%s.jpg'%(id), cv2.IMREAD_COLOR)
#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            # x1=float(list2[2])
            # y1=float(list2[3])
            # x2=float(list[4])
            # y2=float(list[5])
            # cv2.rectangle(rgb_image, (int(x1),int(y1)), (int(x2), int(y2)), (55, 255, 155), 5)
            # plt.figure(figsize=(10,10))
            # plt.imshow(rgb_image)
            # plt.show()
            # sum=id

def getpoint(path):
    pointlist = []
    sum = ""
    with open(path,'r') as f:
        strlist = f.readlines()
        for item in strlist:
            templist = item[0:-1].split(' ')
            if(templist[0] == sum):
                continue
            pointlist.append(templist)
            sum = templist[0]
    return pointlist

def getxml(path):
        tree = ET.parse(path)
        root = tree.getroot()
        for ob in root.iter('object'):
            bbox=ob.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
        return bndbox



# def draw(imgpath,x1,y1,x2,y2):
#     img = cv2.imread(imgpath)
#
#     cv2.rectangle(img,(x1,y1),(x2,y2),(80,180,180),20)
#     # plt.figure(figsize=(10,10))
#     # plt.imshow(img)
#     # plt.show()
#     return img

def draw2(imgpath,str,x1,y1,x2,y2,x3,y3,x4,y4):
    img = cv2.imread(imgpath)

    cv2.rectangle(img,(x1,y1),(x2,y2),(80,180,180),20)
    cv2.putText(img, str, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 25)
    cv2.rectangle(img,(x3,y3),(x4,y4),(5,200,100),40)
    return img

if __name__ == "__main__":  #eval.py
    path = 'data/VOCdevkit/VOC2007/results/vocroc_test_OSCC.txt'
    inputdir = "data/VOCdevkit/VOC2007/JPEGImages/"
    xmlinputdir="data/VOCdevkit/VOC2007/Annotations/"
    outputdir = "data/VOCdevkit/VOC2007/bug80/"
    pointlist = getpoint(path)

    #out = draw(inputdir + "000001.jpg",500,550,2800,2800)
    #cv2.imwrite(outputdir +  "000001.jpg",out)
    for item in pointlist:
        imgpath = inputdir + item[0] + '.jpg'
        outpath = outputdir + item[0] + '.jpg'
        x1 = int(item[2].split('.')[0])
        y1 = int(item[3].split('.')[0])
        x2 = int(item[4].split('.')[0])
        y2 = int(item[5].split('.')[0])
        str=item[1]
        xmlpath = xmlinputdir + item[0] + '.xml'
        boxlist=getxml(xmlpath)
        x3=int(boxlist[0])
        y3 = int(boxlist[1])
        x4 = int(boxlist[2])
        y4 = int(boxlist[3])

        if(x1>=0&y1>=0&x2>=0&y2>=0):
            outimg = draw2(imgpath,str,x1,y1,x2,y2,x3,y3,x4,y4)
            cv2.imwrite(outpath,outimg)
            # height=y4-y3
            # width=x4-x3
            # roi = outimg[y3:y4, x3:x4]
            # cv2.imshow('roi', roi)  # region of interesting
            # cv2.waitKey(0)
            # cv2.imwrite(outpath, roi)




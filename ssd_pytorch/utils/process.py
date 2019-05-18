import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import cv2
import xml.etree.cElementTree as ET
from numpy import random
import numpy as np
import math


def getpoint(path):
    pointlist = []
    files = os.listdir(path)
    for item in files:
            templist = (item.split('.'))[0]
            pointlist.append(templist)
    return pointlist

def getxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    for ob in root.iter('object'):
        bbox = ob.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        return bndbox

def RandomMirror(image,boxes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[0::2] = width - boxes[ 2::-2]
        return image, boxes

def Expand(image,box):
        mean = (104, 117, 123)

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = box.copy()
        # boxes[:, :2] += (int(left), int(top))
        # boxes[:, 2:] += (int(left), int(top))
        boxes[0]=boxes[0]+left
        boxes[1]=boxes[1]+top
        boxes[2]=boxes[2]+left
        boxes[3]=boxes[3]+top

        return image, boxes

def Shift(img,bbox):
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0

        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        # shift_bboxes = list()
        #
        # shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])
        # boxes = bbox.copy()
        # boxes[:, :2] += (int(x), int(y))
        # boxes[:, 2:] += (int(x), int(y))
        bbox[0]=bbox[0]+x
        bbox[1]=bbox[1]+y
        bbox[2]=bbox[2]+x
        bbox[3]=bbox[3]+y
        return shift_img, bbox

def rotation(img,box):
        # cv2.namedWindow('image', 0)
        # cv2.imshow('image', img)
        # cv2.waitKey(30)
        # cv2.destroyAllWindows()
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        list=[90, 180, 270]
        angle = random.choice(list, 1)
        scale = random.uniform(0.7, 0.8)
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # cv2.namedWindow('image', 0)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        # rot_bboxes = list()
        bbox = box.copy()
        # for bbox in boxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
        concat = concat.astype(np.int32)
            # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
            # 加入list中
            # rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        bbox[0]=rx_min
        bbox[1]=ry_min
        bbox[2]=rx_max
        bbox[3]=ry_max


        return rot_img, bbox

if __name__ == "__main__":  #eval.py
    inputdir = "../data/VOCdevkit/VOC2007/JPEGImages/"
    xmlinputdir="../data/VOCdevkit/VOC2007/Annotations/"
    outputdir = "../data/VOCdevkit/VOC2007/bounderbox/"
    pointlist = getpoint(inputdir)

    for item in pointlist:
        imgpath = inputdir + '000001' + '.jpg'
        outpath = outputdir + '000001'+ '.jpg'
        xmlpath = xmlinputdir + '000001' + '.xml'
        boxlist = getxml(xmlpath)
        img = cv2.imread(imgpath)
        # x1 = int(boxlist[0])
        # y1 = int(boxlist[1])
        # x2 = int(boxlist[2])
        # y2 = int(boxlist[3])
        # roi = img[y1:y2, x1:x2]
        # cv2.imwrite(outpath, roi)
        # image, box = Shift(img, boxlist)
        # image,box=rotation(img,box)
        image,box=RandomMirror(img,boxlist)

        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (80, 180, 180), 20)
        cv2.namedWindow('a', 0)
        cv2.imshow('a', image)
        cv2.waitKey(0)
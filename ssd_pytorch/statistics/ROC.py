# ! -*- coding=utf-8 -*-
import pylab as pl
from math import log, exp, sqrt
import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.metrics import auc

figsize = 5, 5
figure, ax = plt.subplots(figsize=figsize)
evaluate_result = "F:/2019.1.19项目拷贝/5.17/val.txt"
ys = []
scores = []
with open(evaluate_result, 'r') as fs:
    for line in fs:
        label, score = line.strip().split(' ')
        label = int(label)
        score = float(score)
        ys.append(label)
        scores.append(score)

fpr, tpr, thresholds = metrics.roc_curve(ys, scores, pos_label=1)
AUC = auc(fpr, tpr)
print(AUC)

# evaluate_result_phone = "F:/2019.1.19项目拷贝/result_cam_phone/5.13phone.txt"
# ys_phone = []
# scores_phone = []
# with open(evaluate_result_phone, 'r') as fs:
# for line in fs:
# label, score = line.strip().split(' ')
# label = int(label)
# score = float(score)
# ys_phone.append(label)
# scores_phone.append(score)
# fpr_phone, tpr_phone, thresholds_phone = metrics.roc_curve(ys_phone, scores_phone, pos_label=1)
# AUC_phone = auc(fpr_phone, tpr_phone)
# print(AUC_phone)
font1 = {'family': 'Times New Roman',
         'size': 10,
         'weight': 'bold'
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10
         }
pl.title("The validation set", font1, loc='left')
pl.xlabel("Specificity", font2)
pl.ylabel("Sensitivy", font2)
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(fpr, tpr, color='#000000', linestyle='-', linewidth='1')
# plt.plot(fpr, tpr, color='#d6483e', linestyle='-', linewidth='1', label='DSLRs')
# plt.plot(fpr_phone, tpr_phone, color='#225396', linestyle='-', linewidth='1', label='Phone')
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# plt.legend()
plt.show()
# plt.savefig('Phone.jpg', dpi=300)

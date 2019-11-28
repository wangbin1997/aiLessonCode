# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("dot.csv",header=0)
x1=df["x1"]
x2=df["x2"]
y_c=df["y_c"]
yc=[["red" if y else "blue"]for y in y_c]
yc=np.squeeze(yc)

xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
# #把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
probs = pd.read_csv('probs.csv')#在probs.csv中存每一点的高度
probs=probs.values

#用plt.scatter画出数据集X各行中第1列元素和第2列元素的点即各行的（x1，x2）
plt.scatter(x1, x2, color=np.squeeze(yc))
plt.contour(xx, yy, probs, levels=[0.5])


plt.show()

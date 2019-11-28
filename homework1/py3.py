# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

from PIL import Image
import numpy as np

f = open('mnist.txt', 'r')
for line in f.readlines():
    str=line.split()
    tempath="./mnist_data_jpg/"+str[0]
    print(tempath)
    img = Image.open(tempath)  # 打开0.png
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = np.array(img.convert('L'))
    img_arr = img

    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255-img_arr[i][j]
            if img_arr[i][j] > 25:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0
    im = Image.fromarray(img_arr)  # 将array转成Image
    im.save(tempath)  # 保存图片

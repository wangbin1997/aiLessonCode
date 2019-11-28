# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("SH_600519_high_low.csv",header=0)
y1=df["high"]
y2=df["low"]
x=np.arange(0,47)

plt.xlabel("day")
plt.ylabel("price")
plt.title("Kweichow Moutai")
plt.subplot(2,1,1)
plt.plot(x,y1,"b",label="high")
plt.legend()
plt.subplot(2,1,2)
plt.plot(x,y2,"y",label="low")
plt.legend()
plt.show()
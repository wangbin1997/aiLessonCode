# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

iris_data=pd.read_csv('iris.txt',header=None)
iris_data .columns=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类别']
# print(iris_data)
x_data = iris_data[['花萼长度','花萼宽度','花瓣长度','花瓣宽度']]
y_data = iris_data['类别']

y_data.replace('Iris-setosa', 0,inplace=True)
y_data.replace('Iris-versicolor', 1,inplace=True)
y_data.replace('Iris-virginica', 2,inplace=True)
# print(x_data.values)
# print(y_data.values)
x_data=x_data.values
y_data=y_data.values

np.random.seed(50)
np.random.shuffle(x_data)
np.random.seed(50)
np.random.shuffle(y_data)



x_train=x_data[:-30]
y_train=y_data[:-30]
x_test=x_data[-30:]
y_test=y_data[-30:]

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = tf.cast(x_train, tf.float32)
x_test=tf.cast(x_test,tf.float32)

y_train = tf.cast(y_train, tf.int32)
y_test=tf.cast(y_test,tf.int32)


train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)

w1 = tf.Variable(tf.random.truncated_normal([4, 32], stddev=0.1,seed=1))
b1=tf.Variable(tf.random.truncated_normal([32], stddev=0.1,seed=1))
w2 = tf.Variable(tf.random.truncated_normal([32, 32], stddev=0.1,seed=2))
b2=tf.Variable(tf.random.truncated_normal([32], stddev=0.1,seed=2))
w3 = tf.Variable(tf.random.truncated_normal([32, 3], stddev=0.1,seed=3))
b3=tf.Variable(tf.random.truncated_normal([3], stddev=0.1,seed=3))

lr = 0.1
train_loss_results = []
epoch = 500
loss_all=0
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        # print(step)

        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h2 = tf.matmul(h1,w2) + b2
            y = tf.matmul(h2, w3) + b3

            y_onehot = tf.one_hot(y_train, depth=3)

            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - y))
            loss_all+=loss.numpy()

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step%100==0:
             print(epoch, step, 'loss:', float(loss))
    train_loss_results.append(loss_all/4)
    loss_all=0

    # test(做测试）
    total_correct, total_number = 0, 0
    for step,(x_test, y_test) in enumerate(test_db):

        h1 = tf.matmul(x_test, w1) + b1
        h2 = tf.matmul(h1,w2) + b2
        y = tf.matmul(h2, w3) + b3

        pred=tf.argmax(y, axis=1)


        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        correct=tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc=total_correct/total_number
    print("test_acc:",acc)



# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results)
plt.show()
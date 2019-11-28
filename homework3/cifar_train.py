# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_save_path = './checkpoint/cifar10.tf'

train_path="./cifar_10_image/train/"
test_path="./cifar_10_image/test/"

train_txt="./cifar_10_image/train_label.txt"
test_txt="./cifar_10_image/test_label.txt"

# def generateds(path,txt):
#     f = open(txt, 'r')
#     contents=f.readlines()#按行读取
#     f.close()
#     labels1,labels2,labels3,labels4,labels5 = [],[],[],[],[]
#     images1,images2,images3,images4,images5=[],[],[],[],[]
#     nums=0
#     for content in contents:
#         nums=nums+1
#
#         value=content.split()  #以空格分开，存入数组
#         img_path=path+value[0]
#         img = Image.open(img_path)
#         img = np.array(img)
#         # print("img=", img)
#         img=img/255.
#
#
#         if nums <= 1000:
#             images1.append(img)
#             labels1.append(value[1])
#         elif nums <= 2000:
#             images2.append(img)
#             labels2.append(value[1])
#         elif nums <= 3000:
#             images3.append(img)
#             labels3.append(value[1])
#         elif nums <= 4000:
#             images4.append(img)
#             labels4.append(value[1])
#         elif nums <= 5000:
#             images5.append(img)
#             labels5.append(value[1])
#
#         print('loading : '+content)
#
#
#     x_1 = np.array(images1)
#     x_2 = np.array(images2)
#     x_3 = np.array(images3)
#     x_4 = np.array(images4)
#     x_5 = np.array(images5)
#
#     y_1 = np.array(labels1)
#     y_2 = np.array(labels2)
#     y_3 = np.array(labels3)
#     y_4 = np.array(labels4)
#     y_5 = np.array(labels5)
#
#     y_1 = y_1.astype(np.int64)
#     y_2 = y_2.astype(np.int64)
#     y_3 = y_3.astype(np.int64)
#     y_4 = y_4.astype(np.int64)
#     y_5 = y_5.astype(np.int64)
#     return x_1,x_2,x_3,x_4,x_5,y_1,y_2,y_3,y_4,y_5

def generateds(path,txt):
    f = open(txt, 'r')
    contents=f.readlines()#按行读取
    f.close()
    images, labels = [], []
    for content in contents:
        value=content.split()  #以空格分开，存入数组
        img_path=path+value[0]
        img = Image.open(img_path)
        img = np.array(img)
        img=img/255.
        images.append(img)
        labels.append(value[1])
        print('loading : '+content)

    x_=np.array(images)
    y_=np.array(labels)
    y_ = y_.astype(np.int64)
    return x_,y_

print('-------------load the data-----------------')
# x_train1,x_train2,x_train3,x_train4,x_train5,y_train1,y_train2,y_train3,y_train4,y_train5=generateds(train_path,train_txt)
x_train,y_train=generateds(train_path,train_txt)
x_test,y_test=generateds(test_path,test_txt)

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)

print(x_train.shape)
x_train=x_train.reshape(5,10000,32,32,3)
x_train1=x_train[0,]
x_train2=x_train[1,]
x_train3=x_train[2,]
x_train4=x_train[3,]
x_train5=x_train[4,]

print(x_train1.shape)


# x_test,null,null,null,null,y_test,null,null,null,null=generateds(test_path,test_txt)

x_train1 = x_train1.reshape(x_train1.shape[0], 32, 32, 3)
x_train2 = x_train2.reshape(x_train1.shape[0], 32, 32, 3)
x_train3 = x_train3.reshape(x_train1.shape[0], 32, 32, 3)
x_train4 = x_train4.reshape(x_train1.shape[0], 32, 32, 3)
x_train5 = x_train5.reshape(x_train1.shape[0], 32, 32, 3)

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
# x_train1,x_train2,x_train3,x_train4,x_train5, x_test = x_train1 / 255.0,x_train2/ 255.0,x_train3 / 255.0,x_train4 / 255.0,x_train5 / 255.0, x_test / 255.0



# 随机打乱数据
# np.random.seed(116)
# np.random.shuffle(x_train1)
# np.random.seed(116)
# np.random.shuffle(y_train1)
#
# np.random.seed(116)
# np.random.shuffle(x_train2)
# np.random.seed(116)
# np.random.shuffle(y_train2)
#
# np.random.seed(116)
# np.random.shuffle(x_train3)
# np.random.seed(116)
# np.random.shuffle(y_train3)
#
# np.random.seed(116)
# np.random.shuffle(x_train4)
# np.random.seed(116)
# np.random.shuffle(y_train4)
#
# np.random.seed(116)
# np.random.shuffle(x_train5)
# np.random.seed(116)
# np.random.shuffle(y_train5)

np.random.seed(116)
np.random.shuffle(x_test)
np.random.seed(116)
np.random.shuffle(y_test)

print(x_train2)
print(x_test)

x_train1=tf.convert_to_tensor(x_train1)
x_train2=tf.convert_to_tensor(x_train2)
x_train3=tf.convert_to_tensor(x_train3)
x_train4=tf.convert_to_tensor(x_train4)
x_train5=tf.convert_to_tensor(x_train5)

x_test=tf.convert_to_tensor(x_test)

# y_train=tf.squeeze(y_train, axis=1)
# y_test=tf.squeeze(y_test, axis=1)


image_gen_train = ImageDataGenerator(
                                     rescale=1./255,#归至0～1
                                     rotation_range=45,#随机45度旋转
                                     width_shift_range=.15,#宽度偏移
                                     height_shift_range=.15,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=0.5#将图像随机缩放到50％
                                     )
image_gen_train.fit(x_train1)
image_gen_train.fit(x_train2)
image_gen_train.fit(x_train3)
image_gen_train.fit(x_train4)
image_gen_train.fit(x_train5)


model = tf.keras.models.Sequential([
    Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=(32,32,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    Dropout(0.2),

    Conv2D(64, kernel_size=(5,5), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


if os.path.exists(model_save_path+'.index'):
    print('-------------load the model-----------------')
    # model.load_weights(model_save_path)
for i in range(5):
    # history = model.fit(x_train, y_train, epochs=1,batch_size=32, validation_data=(x_test, y_test), validation_freq=2)
    # history = model.fit(image_gen_train.flow(x_train1, y_train1), epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(image_gen_train.flow(x_train2, y_train2), epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(image_gen_train.flow(x_train3, y_train3), epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(image_gen_train.flow(x_train4, y_train4), epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(image_gen_train.flow(x_train5, y_train5), epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    history = model.fit(x_test, y_test, epochs=4,
                        validation_split=0.2, validation_freq=2)
    model.save_weights(model_save_path, save_format='tf')

model.summary()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


file=open('./weights.txt','w')
for line in model.trainable_variables:
    file.write(str(line.name)+'\n')
    file.write(str(line.shape) + '\n')
    file.write(str(line.numpy()) + '\n')
file.close()
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


x_train,y_train=generateds(train_path,train_txt)
x_test,y_test=generateds(test_path,test_txt)

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)


x_train=x_train.reshape(5,10000,32,32,3)
x_train1=x_train[0]
x_train2=x_train[1]
x_train3=x_train[2]
x_train4=x_train[3]
x_train5=x_train[4]

y_train=y_train.reshape(5,10000)
y_train1=y_train[0]
y_train2=y_train[1]
y_train3=y_train[2]
y_train4=y_train[3]
y_train5=y_train[4]

print(x_train1)
print(y_train1)



x_train1 = x_train1.reshape(x_train1.shape[0], 32, 32, 3)
x_train2 = x_train2.reshape(x_train1.shape[0], 32, 32, 3)
x_train3 = x_train3.reshape(x_train1.shape[0], 32, 32, 3)
x_train4 = x_train4.reshape(x_train1.shape[0], 32, 32, 3)
x_train5 = x_train5.reshape(x_train1.shape[0], 32, 32, 3)

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

np.random.seed(116)
np.random.shuffle(x_test)
np.random.seed(116)
np.random.shuffle(y_test)


x_train1=tf.convert_to_tensor(x_train1)
x_train2=tf.convert_to_tensor(x_train2)
x_train3=tf.convert_to_tensor(x_train3)
x_train4=tf.convert_to_tensor(x_train4)
x_train5=tf.convert_to_tensor(x_train5)

x_test=tf.convert_to_tensor(x_test)




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
    model.load_weights(model_save_path)

acc=[]
loss=[]
tacc=[]
tloss=[]
for i in range(1):

    history = model.fit(image_gen_train.flow(x_train1, y_train1), epochs=1,
                        validation_data=(x_test, y_test), validation_freq=1)

    history = model.fit(image_gen_train.flow(x_train2, y_train2), epochs=1,
                        validation_data=(x_test, y_test), validation_freq=1)

    history = model.fit(image_gen_train.flow(x_train3, y_train3), epochs=1,
                        validation_data=(x_test, y_test), validation_freq=1)

    history = model.fit(image_gen_train.flow(x_train4, y_train4), epochs=1,
                        validation_data=(x_test, y_test), validation_freq=1)

    history = model.fit(image_gen_train.flow(x_train5, y_train5), epochs=1,
                        validation_data=(x_test, y_test), validation_freq=1)

    # history = model.fit(x_train1, y_train1, epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(x_train2, y_train2, epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    #
    # history = model.fit(x_train3, y_train3, epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(x_train4, y_train4, epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)
    #
    # history = model.fit(x_train5, y_train5, epochs=1,
    #                     validation_data=(x_test, y_test), validation_freq=1)

    acc.extend(history.history['val_sparse_categorical_accuracy'])
    loss.extend(history.history['val_loss'])
    tacc.extend(history.history['sparse_categorical_accuracy'])
    tloss.extend(history.history['loss'])

    model.save_weights(model_save_path, save_format='tf')

model.summary()




plt.plot(acc,label='val acc')
plt.plot(tacc,label='acc')
plt.legend( )
plt.show()

plt.plot(loss,label='val loss')
plt.plot(tloss,label='loss')
plt.legend( )
plt.show()


np.set_printoptions(threshold=np.inf)

file=open('./weights.txt','w')
for line in model.trainable_variables:
    file.write(str(line.name)+'\n')
    file.write(str(line.shape) + '\n')
    file.write(str(line.numpy()) + '\n')
file.close()
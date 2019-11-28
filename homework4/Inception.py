# !/usr/bin/env python
# coding:utf-8
# Author: Huangyu
# !/usr/bin/env python
# coding:utf-8
# Author: Huangyu
# 利用class结构训练并测试fashion_mnist
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
print(x_train.shape[0])
x_train, x_test = x_train / 255.0, x_test / 255.0

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

image_gen_train = ImageDataGenerator(
    rescale=1,  # 归至0～1
    rotation_range=0,  # 随机0度旋转
    width_shift_range=0.1,  # 宽度偏移
    height_shift_range=0.1,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=1  # 将图像随机缩放到100％
)
image_gen_train.fit(x_train)


class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1): #ch是channels
        super(InceptionBlk, self).__init__()
        # self.ch = ch
        # self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides,padding='same')
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides,padding='same')
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1,padding='same')
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides,padding='same')
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1,padding='same')
        self.c4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides,padding='same')

    def call(self, x, training=None):
        x1 = self.c1(x, training=training)
        x2_1 = self.c2_1(x, training=training)
        x2_2 = self.c2_2(x2_1, training=training)  # 可能这里错误
        x3_1 = self.c3_1(x, training=training)
        x3_2 = self.c3_2(x3_1, training=training)
        x4_1 = self.c4_1(x)
        x4_2 = self.c4_2(x4_1, training=training)
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class fashion_class_inception(Model):
    def __init__(self, num_layers, num_classes, init_ch=16):
        super(fashion_class_inception, self).__init__()
        # self.in_channels = init_ch  #channel是卷积核 此处是输入卷积核个数
        self.out_channels = init_ch
        # self.num_layers = num_layers
        # self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        # self.blocks = tf.keras.models.Sequential([
        #     InceptionBlk(self.out_channels, strides=2),
        #     InceptionBlk(self.out_channels, strides=2),
        #     InceptionBlk(self.out_channels, strides=2),
        #     InceptionBlk(self.out_channels, strides=2)
        # ])
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x, training=None):
        x = self.c1(x, training=training)
        x = self.blocks(x, training=training)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = fashion_class_inception(2, 10)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/cifar10.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 save_best_only=True,
                                                 verbose=2)
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=128), epochs=2000000,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

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

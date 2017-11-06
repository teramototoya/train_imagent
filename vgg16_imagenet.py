import numpy as np
import os
import sys
import math
import time
import random
import datetime
import matplotlib.pyplot as plt

from progressbar import *
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model,Sequential
from keras.datasets import mnist,cifar10
from keras.utils import np_utils, plot_model
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist,cifar10
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

config =tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",allow_growth=True))
config = tf.ConfigProto()
config.gpu_options.allocator_type ='BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.90
tf.Session(config=config)

def get_train_batch():
    index = 1

    global current_index

    B = np.zeros(shape=(batch_size, 256, 256, 3))
    L = np.zeros(shape=(batch_size))

    while index < batch_size:
        try:
            img = load_img("/data/dataset/ILSVRC2012_256/"+ training_images[current_index])
            B[index] = img_to_array(img) - ilsvrc_2012_mean
            B[index] /= 255
            del(img)

            L[index] = training_labels[current_index]

            index = index + 1
            current_index = current_index + 1
        except:
            print("Ignore image {}".format(training_images[current_index]))
            current_index = current_index + 1

    return B, keras.utils.to_categorical(L, num_classes)

def get_test_batch():
    index = 1

    global current_index_test

    B = np.zeros(shape=(batch_size, 256, 256, 3))
    L = np.zeros(shape=(batch_size))

    while index < batch_size:
        try:
            img = load_img("/data/dataset/ILSVRC2012_256/"+ test_images[current_index_test])
            B[index] = img_to_array(img) - ilsvrc_2012_mean
            B[index] /= 255
            del(img)

            L[index] = test_labels[current_index_test]

            index = index + 1
            current_index_test = current_index_test + 1
        except:
            print("Ignore image {}".format(test_images[current_index_test]))
            current_index_test = current_index_test + 1

    return B, keras.utils.to_categorical(L, num_classes)

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

if __name__ == '__main__':
    path_directory = "/ilsvrc2012/train.txt"

    # load image
    dataReader = np.genfromtxt(path_directory, delimiter = ",", dtype = 'str')
    training_images = dataReader[:,0]
    training_labels = dataReader[:,1]

    # split data
    training_images, test_images, training_labels, test_labels = train_test_split(training_images, training_labels, test_size=0.2, random_state=0)

    # load ilsvrc_2012_mean
    ilsvrc_2012_mean = np.load("ilsvrc_2012_mean.npy")
    ilsvrc_2012_mean = ilsvrc_2012_mean.transpose(1,2,0)

    # create model
    input_tensor = Input(shape=(256, 256, 3))
    vgg16 = VGG16(include_top=True, weights=None, input_tensor=input_tensor)
    vgg16.compile(loss      = 'categorical_crossentropy',
                  optimizer = SGD(lr=0.01, momentum=0.9, decay = 0.0005),
                  metrics   = ['accuracy'])

    batch_size = 32
    epochs = 500
    num_classes = 1000
    nice_n = math.floor(len(training_images) / batch_size) * batch_size

    # create log directory
    TIME = datetime.datetime.today().strftime("%Y%m%d_%H:%M")
    log_path = './log_2/' + TIME + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    callback = TensorBoard(log_path)
    callback.set_model(vgg16)
    train_names = ['train_loss', 'train_acc']
    val_names = ['val_loss', 'val_acc']

    nice_n = np.ceil(len(training_images) / batch_size) * batch_size
    widgets = [FormatLabel(''), Percentage(), ' ', Bar(),' ', ETA(), ' ']
    perm = list(range(len(training_images)))

    for i in range(0, epochs):
        print('epoch {}/{}'.format(i+1, epochs))
        p = ProgressBar(widgets=widgets,maxval=int(nice_n / batch_size)).start()
        current_index = 0
        current_index_test = 0
        val_loss = 0
        val_accuracy = 0
        random.shuffle(perm)
        training_images = [training_images[index] for index in perm]
        training_labels = [training_labels[index] for index in perm]
        batch_no = 0
        while batch_no < int(nice_n / batch_size):
            b, l = get_train_batch()
            logs = vgg16.train_on_batch(b, l)
            loss = logs[0]
            accuracy = logs[1]
            widgets[0] = FormatLabel('batch {}/{} loss {:f} acc {:f} val_loss {:f} val_acc {:f}'.format(int(current_index / batch_size), int(nice_n / batch_size),loss,accuracy,val_loss,val_accuracy))
            write_log(callback, train_names, logs, batch_no)

            if batch_no % 10 == 0:
                b_t, l_t = get_test_batch()
                logs = vgg16.test_on_batch(b_t, l_t)
                val_loss = logs[0]
                val_accuracy = logs[1]
                write_log(callback, val_names, logs, batch_no//10)

            p.update(batch_no)
            batch_no += 1

        if i % 10 == 0:
            vgg16.save("./save/vgg16_model_2.e{epoch:02d}-l{loss:.2f}-a{acc:.2f}.hdf5".format(**{"epoch":i,"loss":loss, "acc":accuracy}))
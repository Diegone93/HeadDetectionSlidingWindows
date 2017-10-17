import sys
import os
import random


# SETTAGGIO SCHEDA GRAFICA
if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
else:
    gpu_id = "gpu"
    cnmem = "0.70"
print("Argument: gpu={}".format(gpu_id))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from models import faceNet2
from batch_generators import load_names, load_images, load_names_val
from keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

random.seed(1769)

from matplotlib import pyplot as plt
import keras as k
from keras.callbacks import EarlyStopping, ModelCheckpoint

class LossHistory(k.callbacks.Callback):

    def __init__(self):
        plt.ion()
        fig = plt.figure()
        self.plot_loss = fig.add_subplot(211)
        self.plot_val_loss = fig.add_subplot(212)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.val_loss = 0
        self.loss = 0

    def on_epoch_end(self, epoch, logs={}):

        self.val_loss = logs.get('val_loss')
        self.loss = logs.get('loss')
        self.losses.append(self.loss)
        self.val_losses.append(self.val_loss)

        self.plot_loss.plot(self.val_losses, 'r')
        self.plot_val_loss.plot(self.losses, 'r')

        plt.draw()


if __name__ == '__main__':

    # image dimension
    rows = 64 #256
    cols = 64 #192

    # deep parameters
    patience = 100
    batch_size = 1000
    n_epoch = 500

    # training parameters
    data_augmentation = True
    b_crop = False
    b_rescale = False
    b_scale = False
    b_normcv2 = True
    b_tanh = True
    limit_train = 1
    #limit_train = -1
    limit_test = 10
    #limit_test = -1
    b_debug = False
    fulldepth = False
    removeBackground = True
    equalize = False

    model = faceNet2(rows, cols)
    print("Load weights ...")
    model.load_weights('weights_5_16bit\weights.013-0.03280.hdf5')
    print("Done.")

    opt = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['acc'])
    model.summary()

    # loading train name
    train_data_names = load_names()
    # loading validation name
    val_data_names = load_names_val()

    tmp = load_names(augm=5)
    train_data_names = train_data_names + tmp


    # cutting train data
    random.shuffle(train_data_names)
    if limit_train == -1:
        limit_train = len(train_data_names)
    train_data_names = train_data_names[:limit_train]

    random.shuffle(val_data_names)

    # cutting validation data
    if limit_test == -1:
        limit_test = len(val_data_names)
    val_data_names = val_data_names[:limit_test]



    def generator():
        random.shuffle(train_data_names)
        while True:
            for it in range(0, len(train_data_names), batch_size):
                X, Y = load_images(train_data_names[it:it + batch_size],b_debug=b_debug,crop=b_crop, rescale=b_rescale, scale=b_scale, normcv2=b_normcv2, fulldepth=fulldepth, rows=rows, cols=cols,removeBackground=removeBackground,equalize=equalize)
                yield X, Y
    def generator_val():
        while True:
            for it in range(0, len(val_data_names), batch_size):
                val_data_X, val_data_Y = load_images(val_data_names[it:it + batch_size], crop=b_crop, rescale=b_rescale, scale=b_scale, b_debug=b_debug, normcv2=b_normcv2, rows=rows,fulldepth=fulldepth, cols=cols,removeBackground=removeBackground,equalize=equalize)
                yield val_data_X,val_data_Y


    his = LossHistory()
    model.fit_generator(generator(),
                        nb_epoch=n_epoch,
                        validation_data=generator_val(),
                        nb_val_samples=len(val_data_names),
                        samples_per_epoch=len(train_data_names),
                        callbacks=[his,EarlyStopping(patience=patience),
                        ModelCheckpoint("weights_5_16bit/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)]
                        )
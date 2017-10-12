
from keras.layers import Input, Convolution2D, Dense, merge, Flatten, Dropout, ZeroPadding2D, LSTM, BatchNormalization
from keras.layers import MaxPooling2D
from keras.models import Model



def faceNet2(rows, cols):
    # canale
    ch = 1

    # dimensioni
    rows = rows
    cols = cols

    input = Input(shape=(ch, rows, cols))
    x = Convolution2D(32, 5, 5, activation='tanh')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 5, 5, activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 4, 4, activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 3, 3, activation='tanh')(x)

    x = Convolution2D(128, 3, 3, activation='tanh')(x)

    x = Flatten()(x)

    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(84, activation='tanh')(x)
    x = Dropout(0.5)(x)

    output = Dense(2, activation='softmax')(x)

    return Model(input=input, output=output)


def faceNet3(rows, cols):
    # canale
    ch = 1

    # dimensioni
    rows = rows
    cols = cols

    input = Input(shape=(ch, rows, cols)) #64

    # 1 block
    x = Convolution2D(96, 3, 3, border_mode='same', activation='tanh')(input)
    x = Convolution2D(96, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='tanh')(x)

    # 2 block
    x = Convolution2D(96, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(96, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='tanh')(x)

    # 3 block
    x = Convolution2D(128, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2), activation='tanh')(x)

    # 4 block
    x = Convolution2D(256, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', activation='tanh')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2), activation='tanh')(x)


    x = Flatten()(x)

    x = Dense(256, activation='tanh')(x)

    x = Dense(512, activation='tanh')(x)

    output = Dense(2, activation='softmax')(x)

    return Model(input=input, output=output)


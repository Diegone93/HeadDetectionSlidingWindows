from keras.layers import Input, Conv2D, Dense, merge, Flatten, Dropout, ZeroPadding2D, LSTM, BatchNormalization
from keras.layers import MaxPooling2D
from keras.models import Model

def faceNet2(rows, cols):
    # canale
    ch = 1

    # dimensioni
    rows = rows
    cols = cols
    input = Input(shape=(ch, rows, cols))


    x = Conv2D(32, (5, 5), activation='tanh')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (5, 5), activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (4, 4), activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='tanh')(x)

    x = Conv2D(128, (3, 3), activation='tanh')(x)

    x = Flatten()(x)

    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(84, activation='tanh')(x)
    x = Dropout(0.5)(x)

    output = Dense(2, activation='softmax')(x)

    return Model(inputs=input, outputs=output)

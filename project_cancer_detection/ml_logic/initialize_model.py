# INIT MODEL and COMPILE
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam


def init_model(l_rate=0.001):
    model = models.Sequential()

    model.add(Rescaling(scale=1./255,input_shape=(96,96,3)))

    # Lets add convolution layers,
    model.add(layers.Conv2D(32, kernel_size=2, activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(32, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(32, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))


    model.add(layers.Flatten())

    model.add(layers.Dense(30, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

     ### Model compilation
    optim = Adam(learning_rate=l_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    return model


def init_model_2(l_rate=0.001):

    model = models.Sequential()

    model.add(Rescaling(scale=1./255,input_shape=(96,96,3)))

    # Lets add convolution layers,
    model.add(layers.Conv2D(8, kernel_size=2, activation='relu'))

    model.add(layers.Conv2D(16, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(16, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(32, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(32, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(16, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())

    model.add(layers.Dense(40, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

     ### Model compilation
    optim = Adam(learning_rate=l_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def load_VGG():
    model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape =(96,96,3)
    )
    return model

def set_nontrainable_layers(model):
    model.trainable = False
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(10, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')
    dropout_layer = layers.Dropout(rate=0.20)

    model = models.Sequential([
        base_model,
        dropout_layer,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])
    return model

def init_VGG(l_rate=0.001):
    model = load_VGG()
    model = add_last_layers(model)

    optim = Adam(learning_rate=l_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model

# INIT MODEL and COMPILE
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def init_model(l_rate, decay_rate, decay_steps):
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
    lr_schedule = ExponentialDecay(l_rate,
                               decay_steps = decay_steps,    # every 2000 iterations
                               decay_rate = decay_rate)      # we multiply the learning rate by the decay_rate
                                                      # PS: we have appox 404 x 70% /16 = 18 iterations per epoch

    optim = Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,  # get's Adam with learning rate
                  metrics=['accuracy'])

    return model


def init_model_2(l_rate,decay_rate,decay_steps):

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
    lr_schedule = ExponentialDecay(l_rate,
                            decay_steps = decay_steps,    # every 2000 iterations
                            decay_rate = decay_rate)      # we multiply the learning rate by the decay_rate
                                                   # PS: we have appox 404 x 70% /16 = 18 iterations per epoch


    optim = Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model


def load_VGG():
    model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape =(96,96,3)
    )
    return model

def load_ResNet50():
    model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape =(96,96,3)
    )
    return model

def load_MobileNetV2():
    model = MobileNetV2(
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
    pooling_layer = layers.GlobalAveragePooling2D()
    flatten_layer = layers.Flatten()
    dense_layer_100 = layers.Dense(100, activation='relu')
    dense_layer_40 = layers.Dense(40, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')
    dropout_layer = layers.Dropout(rate=0.20)

    model = models.Sequential([
        base_model,
        pooling_layer,
        #dropout_layer,
        flatten_layer,
        dense_layer_100,
        dropout_layer,
        dense_layer_40,
        prediction_layer
    ])
    return model

def init_VGG(l_rate,decay_rate,decay_steps):
    model = load_VGG()
    model = add_last_layers(model)

    lr_schedule = ExponentialDecay(l_rate,
                            decay_steps = decay_steps,    # every 2000 iterations
                            decay_rate = decay_rate)      # we multiply the learning rate by the decay_rate
                                                   # PS: we have appox 404 x 70% /16 = 18 iterations per epoch


    optim = Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model

def init_ResNet50(l_rate,decay_rate,decay_steps):
    model = load_ResNet50()
    model = add_last_layers(model)

    lr_schedule = ExponentialDecay(l_rate,
                            decay_steps = decay_steps,    # every 2000 iterations
                            decay_rate = decay_rate)      # we multiply the learning rate by the decay_rate
                                                   # PS: we have appox 404 x 70% /16 = 18 iterations per epoch


    optim = Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model

def init_MobileNetV2(l_rate, decay_rate, decay_steps):
    model = load_MobileNetV2()
    model = add_last_layers(model)

    # lr_schedule = ExponentialDecay(l_rate,
    #                         decay_steps = decay_steps,    # every 2000 iterations
    #                         decay_rate = decay_rate)      # we multiply the learning rate by the decay_rate
    #                                                # PS: we have appox 404 x 70% /16 = 18 iterations per epoch

    optim = Adam(learning_rate=l_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model

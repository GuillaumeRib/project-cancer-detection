# INIT MODEL and COMPILE
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam

def init_model():
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
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def init_model_2():
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
    optim = Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model

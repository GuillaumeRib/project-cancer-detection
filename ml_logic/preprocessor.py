
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocessed(local_train_path, local_test_path):

    # ImageGenerator preprocess images / minimum params / to read .tif images
    train_datagen = ImageDataGenerator(validation_split=0.2)
    test_datagen = ImageDataGenerator()

    # Load from directory to flow passsed into ImageGenerator
    train_generator = train_datagen.flow_from_directory(local_train_path,
                                                        subset='training',
                                                        target_size=(96,96),
                                                        batch_size=16,
                                                        class_mode='binary')

    val_generator = train_datagen.flow_from_directory(local_train_path,
                                                    subset='validation',
                                                    target_size=(96,96),
                                                    batch_size=16,
                                                    class_mode='binary')


    test_generator = test_datagen.flow_from_directory(local_test_path,
                                                    target_size=(96,96),
                                                    class_mode='binary')

    print("⭐️Train data was generated")
    print("⭐️Validation data was generated")
    print("⭐️Test data was generated")

    return train_generator, val_generator, test_generator

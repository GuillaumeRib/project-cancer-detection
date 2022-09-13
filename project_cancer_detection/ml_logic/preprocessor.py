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
    return train_generator, val_generator, test_generator


def preprocessed_VGG(local_train_path, local_test_path):
    from keras.applications.vgg16 import preprocess_input
    # ImageGenerator preprocess images / minimum params / to read .tif images
    train_datagen = ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    return train_generator, val_generator, test_generator

def preprocessed_ResNet50(local_train_path, local_test_path):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    # ImageGenerator preprocess images / minimum params / to read .tif images
    train_datagen = ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    return train_generator, val_generator, test_generator

from webbrowser import get
from tensorflow.keras.callbacks import EarlyStopping
from project_cancer_detection.ml_logic.initialize_model import init_model, init_model_2, init_VGG, init_ResNet50
from project_cancer_detection.ml_logic.preprocessor import preprocessed, preprocessed_ResNet50, preprocessed_VGG
import os
import mlflow

####################################
# Select the model
#model_name='Baseline_CNN'
#model_name='V2_CNN'
#model_name='VGG16_transfer'
model_name='ResNet50'

# Sample size
sample_size = '_5k'

# Init model params
l_rate = 0.0005

# Fit Model Parameters (from get_history function)
epochs = 50
batch_size = 32
verbose_model = 1

# EarlyStopping
patience=3
verbose=1
####################################


def get_paths():
    DATA_SOURCE = os.environ.get("DATA_SOURCE")

    if DATA_SOURCE == 'local':
        train_path = os.environ.get('LOCAL_TRAIN_PATH')+sample_size
        test_path = os.environ.get('LOCAL_TEST_PATH')+sample_size

    if DATA_SOURCE == 'cloud':
        train_path = os.environ.get('CLOUD_TRAIN_PATH')
        test_path = os.environ.get('CLOUD_TEST_PATH')

    print(f'### Sourcing data from {DATA_SOURCE} ... ###\n')
    return train_path,test_path


def get_history(train_generator, val_generator):

    if model_name == 'Baseline_CNN':
        model = init_model(l_rate)
    elif model_name == 'V2_CNN':
        model = init_model_2(l_rate)
    elif model_name == 'VGG16_transfer':
        model = init_VGG(l_rate)
    elif model_name == 'ResNet50':
        model = init_ResNet50(l_rate)

    es = EarlyStopping(patience=patience, restore_best_weights=True,verbose=verbose)


    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data=val_generator,
                        batch_size = batch_size,
                        verbose = verbose_model,
                        callbacks = [es])
    return history, model


def evaluate(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return results

def save_model(model, model_outputs, batch_size, epochs, model_name, l_rate, sample_size):
    mlflow.set_tracking_uri("https://mlflow.lewagon.ai")
    mlflow.set_experiment(experiment_name="project-cancer-detection")

    with mlflow.start_run():

        params = dict(batch_size=batch_size, epochs=epochs, l_rate=l_rate, model_name=model_name, sample_size=sample_size)
        metrics = dict(loss=model_outputs[0], accuracy=model_outputs[1])

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.keras.log_model(keras_model=model,
                            artifact_path="model",
                            keras_module="tensorflow.keras",
                            registered_model_name="cancer_detection_model")


def load_model():
    mlflow.set_tracking_uri("https://mlflow.lewagon.ai")
    model_uri = "MLFLOW_MODEL_NAME"                            # 1) if you write "latest" intead of "2" it'll load the latest model;
    model = mlflow.keras.load_model(model_uri=model_uri)       # 2) you can change "2" to any number of the version you want to load
    return model


if __name__ == '__main__':
    # Store your train & test paths from .env file
    train_path, test_path = get_paths()

    print('### Preprocessing & generators starting ... ###')
    if model_name == 'VGG16_transfer':
        train_generator, val_generator, test_generator = preprocessed_VGG(train_path, test_path)
        print('### Preprocessing & generators done! ###\n')
    if model_name == 'ResNet50':
        train_generator, val_generator, test_generator = preprocessed_ResNet50(train_path, test_path)
        print('### Preprocessing & generators done! ###\n')
    else:
        train_generator, val_generator, test_generator = preprocessed(train_path, test_path)
        print('### Preprocessing & generators done! ###\n')

    print('### Model fit starting ... ###')
    history, model = get_history(train_generator, val_generator)
    print('### Model fit done ! Starting evaluation ... ###\n')

    print('### Evaluation starting ... ###')
    #evaluate_model(model, test_generator)
    model_outputs = evaluate(model, test_generator)

    print('### Evaluation done ! Saving params & model to MLFlow ... ###\n')
    save_model(model, model_outputs, batch_size, epochs, model_name, l_rate, sample_size)

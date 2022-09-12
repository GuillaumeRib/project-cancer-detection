from webbrowser import get
from tensorflow.keras.callbacks import EarlyStopping
from project_cancer_detection.ml_logic.initialize_model import init_model
from project_cancer_detection.ml_logic.preprocessor import preprocessed
import os
import mlflow

# Model Parameters (from get_history function)
epochs = 1
batch_size = 32
verbose_model = 1

# EarlyStopping
patience=3
verbose=1

def get_paths():
    DATA_SOURCE = os.environ.get("DATA_SOURCE")

    if DATA_SOURCE == 'local':
        train_path = os.environ.get('LOCAL_TRAIN_PATH')
        test_path = os.environ.get('LOCAL_TEST_PATH')

    if DATA_SOURCE == 'cloud':
        train_path = os.environ.get('CLOUD_TRAIN_PATH')
        test_path = os.environ.get('CLOUD_TEST_PATH')

    print(f'### Sourcing data from {DATA_SOURCE} ... ###\n')
    return train_path,test_path


def get_history(train_generator, val_generator):
    model = init_model()
    es = EarlyStopping(patience=patience, restore_best_weights=True,verbose=verbose)
    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data=val_generator,
                        batch_size = batch_size,
                        verbose = verbose_model,
                        callbacks = [es])
    return history, model

def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return print(f"The accuracy on the test set is of {results[1]*100:.2f} %")

def evaluate(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return results


def save_model(model, model_outputs, batch_size, epochs):
    mlflow.set_tracking_uri("https://mlflow.lewagon.ai")
    mlflow.set_experiment(experiment_name="project-cancer-detection")

    with mlflow.start_run():

        params = dict(batch_size=batch_size, epochs=epochs)
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
    # Link to your sample train_path (manually selected for now)
    '''
    local_train_path = os.environ['LOCAL_TRAIN_PATH']
    local_test_path = os.environ['LOCAL_TEST_PATH']
    cloud_train_path = os.environ['CLOUD_TRAIN_PATH']
    cloud_test_path = os.environ['CLOUD_TEST_PATH']
    '''

    train_path, test_path = get_paths()

    print('### Preprocessing & generators starting ... ###')
    train_generator, val_generator, test_generator = preprocessed(train_path, test_path)
    print('### Preprocessing & generators done! ###\n')

    print('### Model fit starting ... ###')
    history, model = get_history(train_generator, val_generator)
    print('### Model fit done ! Starting evaluation ... ###\n')

    evaluate_model(model, test_generator)
    model_outputs = evaluate(model, test_generator)
    save_model(model, model_outputs, batch_size, epochs)

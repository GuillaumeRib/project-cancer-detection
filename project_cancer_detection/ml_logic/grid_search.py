from webbrowser import get
from tensorflow.keras.callbacks import EarlyStopping
from project_cancer_detection.ml_logic.initialize_model import init_model, init_model_2, init_VGG
from project_cancer_detection.ml_logic.preprocessor import preprocessed
import os
import mlflow
#from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# Select the model
model_name='Baseline_CNN'
#model_name='V2_CNN'
#model_name='VGG16_transfer'

patience=3
verbose=1



def get_paths():
    DATA_SOURCE = os.environ.get("DATA_SOURCE")

    if DATA_SOURCE == 'local':
        train_path = os.environ.get('LOCAL_TRAIN_PATH')#+sample_size
        test_path = os.environ.get('LOCAL_TEST_PATH')#+sample_size

    if DATA_SOURCE == 'cloud':
        train_path = os.environ.get('CLOUD_TRAIN_PATH')
        test_path = os.environ.get('CLOUD_TEST_PATH')

    print(f'### Sourcing data from {DATA_SOURCE} ... ###\n')
    return train_path,test_path


def save_model(best_estimator):
    mlflow.set_tracking_uri("https://mlflow.lewagon.ai")
    mlflow.set_experiment(experiment_name="project-cancer-detection")

    with mlflow.start_run():

        params = dict(batch_size=best_estimator['batch_size'], epochs=best_estimator['epochs'], l_rate=best_estimator['alpha'], scoring=best_estimator['scoring'])
        metrics = dict(accuracy=best_estimator['accuracys'])

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
        train_generator, val_generator, test_generator = preprocessed(train_path, test_path)
        print('### Preprocessing & generators done! ###\n')
    else:
        train_generator, val_generator, test_generator = preprocessed(train_path, test_path)
        print('### Preprocessing & generators done! ###\n')

    #####################################################################################
    #                                GRID SEARCH                                        #
    # Instanciate model                                                                 #

    es = EarlyStopping(patience=patience, restore_best_weights=True,verbose=verbose)

    model = init_model()

    # Hyperparameter Grid
    grid = {'alpha': [0.001, 0.01, 0.1]}
    scoring = ['accuracy']

    batch_size = 32
    epochs = [10, 50]

    def getModel(optimizer):
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

        return model


    optimizer = ['Adam']
    epochs = [10, 50]

    param_grid = dict(epochs=epochs, optimizer=optimizer)

    Kmodel = KerasClassifier(build_fn=getModel, verbose=1)
    grid = GridSearchCV(estimator=Kmodel, param_grid=param_grid, scoring=scoring, n_jobs=1, refit='accuracy', cv=3)
    grid_result = grid.fit(train_generator, train_generator,
                            epochs = epochs,
                            validation_data=val_generator,
                            batch_size = batch_size,
                            verbose = 1,
                            callbacks = [es])

    # Fit data to Grid Search
    best_estimator = grid.best_estimator_
    print(f"The best estimator is:", best_estimator)
    #                                                                                     #
    ######################################################################################
    print('### Evaluation done ! Saving params & model to MLFlow ... ###\n')
    save_model(best_estimator)

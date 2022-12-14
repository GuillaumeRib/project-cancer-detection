from tensorflow.keras.callbacks import EarlyStopping
from project_cancer_detection.ml_logic.initialize_model import init_model
from project_cancer_detection.ml_logic.preprocessor import preprocessed
import os

def get_history(train_generator, val_generator):
    model = init_model()
    epochs = 1
    batch_size = 32
    es = EarlyStopping(patience=3, restore_best_weights=True,verbose=1)
    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data=val_generator,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks = [es])
    return history, model

def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return print(f"The accuracy on the test set is of {results[1]*100:.2f} %")

def evaluate_model_04():
    return print("The function works")

if __name__ == '__main__':
    # Link to your sample train_path (manually selected for now)
    local_train_path = os.environ['LOCAL_TRAIN_PATH']
    local_test_path = os.environ['LOCAL_TEST_PATH']
    cloud_train_path = os.environ['CLOUD_TRAIN_PATH']
    cloud_test_path = os.environ['CLOUD_TEST_PATH']

    print("1")
    train_generator, val_generator, test_generator = preprocessed(cloud_train_path, cloud_test_path)
    history, model = get_history(train_generator, val_generator)
    evaluate_model(model, test_generator)
    #evaluate_model_04()

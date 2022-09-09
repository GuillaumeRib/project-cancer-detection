import sys
sys.path.insert(1, '/home/naz/code/GuillaumeRib/project-cancer-detection/ml_logic')   # DON'T FORGET TO CHANGE ABSOLUTE PATH

from tensorflow.keras.callbacks import EarlyStopping
from initialize_model import init_model
#from initialize_model import init_model
from preprocessor import preprocessed


def get_history(train_generator, val_generator):
    model = init_model()
    epochs = 15
    batch_size = 32
    es = EarlyStopping(patience=3, restore_best_weights=True,verbose=1)

    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data=val_generator,
                        batch_size = batch_size,
                        verbose = 0,
                        callbacks = [es])
    return history, model

def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return print(f'✅The accuracy on the test set is of {results[1]*100:.2f} %')


def dum_model():
    return print("Function works")


if __name__ == '__main__':

    # Link to your sample train_path (manually selected for now)
    local_train_path = '../train_small/'
    local_test_path = '../test_small/'

    train_generator, val_generator, test_generator = preprocessed(local_train_path, local_test_path)
    history, model = get_history(train_generator, val_generator)
    evaluate_model(model, test_generator)
    #dum_model()

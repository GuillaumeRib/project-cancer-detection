import sys
sys.path.insert(1,“/Users/17i/code/GuillaumeRib/project-cancer-detection/project-cancer-detection/ml_logic”)

from tensorflow.keras.callbacks import EarlyStopping
from ml_logic.initialize_model import init_model
from ml_logic.preprocessor import preprocessed

# Link to your sample train_path (manually selected for now)
local_train_path = '../raw_data/SAMPLES/TRAIN_5K'
local_test_path = '../raw_data/SAMPLES/TEST_5K'

train_generator, val_generator, test_generator = preprocessed(local_train_path, local_test_path)
model = init_model()
epochs = 15
batch_size = 32
es = EarlyStopping(patience=3, restore_best_weights=True,verbose=1)


def get_history(model, train_generator, val_generator):
    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data=val_generator,
                        batch_size = batch_size,
                        verbose = 0,
                        callbacks = [es])
    return history
def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator, verbose = 1 )
    return print(f’The accuracy on the test set is of {results[1]*100:.2f} %‘)

def evaluate_model_04():
    return print(f’The function works’)

evaluate_model()

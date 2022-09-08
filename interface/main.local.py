from tensorflow.keras.callbacks import EarlyStopping
from ml_logic import initialize_model
from ml_logic import preprocessor


train_generator, val_generator, test_generator = preprocessor()
model = initialize_model()
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
    return print(f'The accuracy on the test set is of {results[1]*100:.2f} %')

def evaluate_model_04():
    return print(f'The function works')


evaluate_model_04()

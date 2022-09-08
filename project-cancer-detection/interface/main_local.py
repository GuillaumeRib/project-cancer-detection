from ml_logic.data import  clean_data

def preprocess_and_train():

    print("***START preprocess_and_train process")
    clean_data()
    return print("END preprocess_and_train process***")

preprocess_and_train()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.interface.main import preprocess_features
from project_cancer_detection.interface.main_local import load_model
from project_cancer_detection.ml_logic.preprocessor import preprocessed_MobileNetV2_img
# just for test commit

app = FastAPI()
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?
@app.get("/predict")
def predict(img_path):
    """
    Need to pass a path to a single image path
    """
    img_preproc = preprocessed_MobileNetV2_img(img_path)
    y_pred = app.state.model.predict(img_preproc)

    return {'Probability of tumor in image': float(y_pred)}


@app.get("/")
def root():
    return {'greeting': 'Hello'}

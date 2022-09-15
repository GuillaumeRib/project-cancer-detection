from fastapi import FastAPI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from project_cancer_detection.interface.main_local import load_model
from PIL import Image
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
    im = Image.open(img_path)
    im_np = np.array(im)
    im_np = np.expand_dims(im_np, 0)

    # preprocessing with MobileNetV2
    im_prep = preprocess_input(im_np)

    # prediction on model
    model_preds = model.predict(im_prep)
    model_preds = model_preds.tolist()
    return {'Probability of tumor in image': round(model_preds[0][0], 2)}


@app.get("/")
def root():
    return {'greeting': 'Hello'}

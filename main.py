from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import warnings
import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

def load_model():
    global xgb_model_carregado
    #with open("xgb_model.pkl", "rb") as f:
    #    xgb_model_carregado = pickle.load(f)

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8,8))
    img_array = np.array(img)
    img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    img_array = img_array.reshape(1, -1)
    prediction = xgb_model_carregado.predict(img_array)
    return {"prediction": prediction}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

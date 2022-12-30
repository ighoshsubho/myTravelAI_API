from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version1__ as model_version1
from app.model.model import __version2__ as model_version2
import csv


app = FastAPI()


class data(BaseModel):
    airport_dist: float
    railway_dist: float
    Elevation: float
    rating: float

class PredictionOut(BaseModel):
    Name: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version1": model_version1, "model_version": model_version2}


@app.post("/predict")
def predict(payload: data):
    Name = predict_pipeline(dict(payload))
    return {"Result": Name}
#/workspace/myTravelAI_API/app/model/tourism.csv
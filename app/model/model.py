import pickle
import re
from pathlib import Path
import csv
import pandas as pd
import numpy as np

__version1__ = "0.1.0"
__version2__ = "1.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version1__}.pkl", "rb") as f1:
    model1 = pickle.load(f1)

with open(f"{BASE_DIR}/trained_pipeline-{__version2__}.pkl", "rb") as f2:
    model2 = pickle.load(f2)


classes1 = [
    "East",
    "North",
    "North East",
    "North West",
    "South",
    "South East",
    "South West",
    "West"
]

classes2 = [
    "Beach",
    "Hill Station",
    "Temple",
    "Zoological Park"
]

def predict_pipeline(data):
    # CURRENT_DIR = os.path.dirname(__file__)
    # file_path = os.path.join(CURRENT_DIR, 'tourism.csv')
    #  with open("/workspace/myTravelAI_API/app/model/tourism.csv","r") as f:
    #      tourism = csv.reader(f)
    # tourism = '../app/model/tourism.csv'
    # with open(tourism) as f:
    #     reader = csv.reader(f)
    tourism = open("/workspace/myTravelAI_API/app/model/tourism.csv","r+")
    dataset = pd.read_csv(tourism)
    print(dataset)
    data = pd.DataFrame(data,index = [0],columns=['airport_dist(km)','railway_dist(km)','Elevation(m)','rating'])
    pred1 = model1.predict(data)
    pred2 = model2.predict(data)
    Region = classes1[pred1[0]]
    Type = classes2[pred2[0]]
    data_final = dataset.loc[data['Region']==Region]
    data_final = data_final.loc[data_final['Type']==Type]
    if len(data_final['Name'].index) > 1:
        data_test = data_final[abs(data_final['airport_dist(km)']-data['airport_dist(km)'])<60]
        data_test = data_test[abs(data_final['railway_dist(km)']-data['railway_dist(km)'])<60]
        data_test = data_test[abs(data_final['Elevation(m)']-data['Elevation(m)'])<200]
        data_test = data_test[abs(data_final['rating']-data['rating'])<3]
        return data_test['Name'][data_test.index[0]]
    elif len(data_final['Name'].index) == 1:
        return data_final['Name'][data_final.index[0]]
    else:
        return None
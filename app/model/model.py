import pickle
import re
from pathlib import Path
import csv
import pandas as pd

__version1__ = "0.1.0"
__version2__ = "1.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version1__}.pkl", "rb") as f1:
    model1 = pickle.load(f1)

with open(f"{BASE_DIR}/trained_pipeline-{__version2__}.pkl", "rb") as f2:
    model2 = pickle.load(f2)

def predict_pipeline(data):
    print(data)
    tourism = open("/workspace/myTravelAI_API/app/model/tourism.csv","r+")
    dataset = pd.read_csv(tourism)
    data = pd.DataFrame(data,index = [0],columns=['airport_dist(km)','railway_dist(km)','Elevation(m)','rating'])
    pred1 = model1.predict(data)
    pred2 = model2.predict(data)
    Region = pred1[0]
    Type = pred2[0]
    data_final = dataset[dataset['Region']==Region]
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

data = {"airport_dist(km)":20.0	, "railway_dist(km)":100.0, "Elevation(m)":2210.0, "rating":7.0}

print(predict_pipeline(data))
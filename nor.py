import pandas as pd

tourism = open("/workspace/myTravelAI_API/app/model/tourism.csv","r+")
dataset = pd.read_csv(tourism)
dataset.dropna(inplace=True)

print(dataset)
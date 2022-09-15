import pickle
import pandas as pd
with open("D:\study_data\emotional/merged_training.pkl","rb") as fr:
    data = pickle.load(fr)
    
print(data)

data.to_csv("D:\study_data\emotional/twitter_emotional.csv")




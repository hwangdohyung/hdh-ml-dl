#hugging face twitter 
import numpy as np 
import pickle
import pandas as pd
with open("D:\study_data\emotional/merged_training.pkl","rb") as fr:
    data1 = pickle.load(fr)
    
print(data1.shape)       # (416809, 2)
print(data1.describe())  # unique 6

data1.to_csv("D:\study_data\emotional/data1_twitter.csv")


# text emotional
# CrowdFlower가 2016년에 만든 The Emotion in Text 데이터세트는 감정으로 레이블이 지정된 트윗입니다. 
data2 = pd.read_csv('D:\study_data\emotional/text_emotional.csv')
print('====================================')
print(data2.describe())  # unique 6

# twitt emotions
# kaggle 13개의 서로 다른 감정 40000 레코드 


# kaggle emotional



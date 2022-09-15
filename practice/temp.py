import pickle
import pandas as pd


#1. hugging face twitter ####################
with open("D:\study_data\emotional/merged_training.pkl","rb") as fr:
    data1 = pickle.load(fr)
data1= pd.DataFrame(data1)
data1.columns = ['sentence','label']


#sentence

#sentence_id

#label
#dysphoria,euphoria,neutral

print(data1.shape)       # (416809, 2)
print(data1.describe())  # unique 6
data1.to_csv("D:\study_data\emotional/data1_twitter.csv")

#2. text emotional #########################
# CrowdFlower가 2016년에 만든 The Emotion in Text 데이터세트는 감정으로 레이블이 지정된 트윗입니다. 


data2 = pd.read_csv('D:\study_data\emotional/data2_text_emotional.csv')
print(data2.shape) # (416809, 3)
print(data2.value_counts('emotions')) # 6


#joy         141067
# sadness     121187
# anger        57317
# fear         47712
# love         34554
# surprise     14972
# dtype: int64


#3. twitt emotions #########################
# kaggle 13개의 서로 다른 감정 40000 레코드 
data3 = pd.read_csv('D:\study_data\emotional/data3_tweet_emotions.csv')
print(data3.shape) #(40000, 3)
print(data3.value_counts('sentiment')) # 13
# sentiment
# neutral       8638
# worry         8459
# happiness     5209
# sadness       5165
# love          3842
# surprise      2187
# fun           1776
# relief        1526
# hate          1323
# empty          827
# enthusiasm     759
# boredom        179
# anger          110
# dtype: int64


#4. all data #########################
# 언론,뉴스,신문 sentence 
data4 = pd.read_csv('D:\study_data\emotional/data4_alldata.csv')
print(data4.shape) #(4845, 2)
data4.columns = ['emotions','sentence']
print(data4.value_counts('emotions')) # 3
#emotions
# neutral     2878
# positive    1363
# negative     604
# dtype: int64


#5. kaggle emotional #########################
data5_1 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/test.csv')
data5_2 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/training.csv')
data5_3 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/validation.csv')
data5 = pd.concat([data5_1,data5_2,data5_3], axis=0) 
data5.to_csv("D:\study_data\emotional/data5_kaggle_emotional.csv")
print(data5.shape) #(20000, 2)
print(data5.value_counts('label')) #5
# label
# 1    6761
# 0    5797
# 3    2709
# 4    2373
# 2    1641
# 5     719
# dtype: int64


#6. hugging emotional #########################
data6_1 = pd.read_csv('D:\study_data\emotional\hugging_emotional/train.csv')
data6_2 = pd.read_csv('D:\study_data\emotional\hugging_emotional/test.csv')
data6 = pd.concat([data6_1,data6_2], axis=0)
data6.to_csv("D:\study_data\emotional/data6_hugging_emotional.csv")
print(data6.value_counts('class'))
# class
# fear       2252
# anger      1701
# joy        1616
# sadness    1533
# dtype: int64



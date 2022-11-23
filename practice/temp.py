import pickle
import pandas as pd

#sentence

#sentence_id

#label
#dysphoria,euphoria,neutral

#1. hugging face twitter ####################
# CrowdFlower가 2016년에 만든 The Emotion in Text 데이터세트는 감정으로 레이블이 지정된 트윗입니다.
with open("D:\study_data\emotional/merged_training.pkl","rb") as fr:
    data1 = pickle.load(fr)
data1= pd.DataFrame(data1)
print(data1.shape)       # (416809, 2)
print(data1.describe())  # unique 6
# data1.to_csv("D:\study_data\emotional/data1_twitter.csv")
data1.columns = ['sentence','labels']

data1_label = data1['labels']
print(data1_label)
newlist = []
    
for i in data1_label :
    if i == 'joy':
        newlist += ['euphoria']
    elif i == 'love':
        newlist += ['euphoria']
    elif i == 'suprise':
        newlist += ['euphoria']
    else:
        newlist += ['dysphoria']    

# print(newlist)
newlist = pd.DataFrame(newlist,columns=['label'])
data1.reset_index(inplace=True,drop=True)
data1 = pd.concat([data1,newlist],axis=1)
data1 = data1.drop(columns='labels',axis=1)

print(data1)
# data1.to_csv("D:\study_data\emotional\last/data1_twitter.csv")



#3. twitt emotions #########################
# kaggle 13개의 서로 다른 감정 40000 레코드 
data3 = pd.read_csv('D:\study_data\emotional/data3_tweet_emotions.csv',index_col=0)
print(data3.shape) #(40000, 2)
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
print(data3.shape)
data3_idx =  data3[data3['sentiment'] == 'empty'].index
data3 = data3.drop(index = data3_idx)
print(data3.value_counts('sentiment'))
data3.columns = ['labels','sentence']
data3_label = data3['labels']
newlist1 = []
for i in data3_label :
    if i == 'neutral':
        newlist1 += ['neutral']
    elif i == 'happiness':
        newlist1 += ['euphoria']
    elif i == 'suprise':
        newlist1 += ['euphoria']
    elif i == 'love':
        newlist1 += ['euphoria']
    elif i == 'fun':
        newlist1 += ['euphoria']
    elif i == 'relief':
        newlist1 += ['euphoria']
    elif i == 'enthusiasm':
        newlist1 += ['euphoria']
    else:
        newlist1 += ['dysphoria']    

newlist1 = pd.DataFrame(newlist1,columns=['label'])
data3.reset_index(inplace=True,drop=True)
data3 = pd.concat([data3,newlist1],axis=1)
data3 = data3.drop(columns='labels',axis=1)
# data3.to_csv("D:\study_data\emotional\last/data3_twitt.csv")

#4. all data #########################
# 언론,뉴스,신문 sentence 
data4 = pd.read_csv('D:\study_data\emotional/data4_alldata.csv')
print(data4.shape) #(4845, 2)
data4.columns = ['label','sentence']
# emotions
# neutral     2878
# positive    1363
# negative     604
# dtype: int64
data4 = data4.replace({'label' : 'positive'}, 'euphoria') 
data4 = data4.replace({'label' : 'negative'}, 'dysphoria') 
data4 = data4[['sentence','label']]
# data4.to_csv("D:\study_data\emotional\last/data4_alldata.csv")


#5. kaggle emotional #########################
data5_1 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/test.csv')
data5_2 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/training.csv')
data5_3 = pd.read_csv('D:\study_data\emotional\kaggle_emotional/validation.csv')
data5 = pd.concat([data5_1,data5_2,data5_3], axis=0) 
# data5.to_csv("D:\study_data\emotional/data5_kaggle_emotional.csv")
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
# 0 (sadness), 1 (joy), 2 (love), 3 (anger), 4 (fear), 5 (surprise)


data5.columns = ['sentence','labels']

data5_label = data5['labels']

newlist5 = []

for i in data5_label :
    if i == 1:
        newlist5 += ['euphoria']
    elif i == 2:
        newlist5 += ['euphoria']
    elif i == 5:
        newlist5 += ['euphoria']
    else:
        newlist5 += ['dysphoria']    

# print(newlist)
newlist5 = pd.DataFrame(newlist5,columns=['label'])
data5.reset_index(inplace=True,drop=True)
data5 = pd.concat([data5,newlist5],axis=1)
data5 = data5.drop(columns='labels',axis=1)

print(data5.value_counts('label'))
# data5.to_csv("D:\study_data\emotional\last/data5_kaggle_emotional.csv")


#6. hugging emotional #########################
data6_1 = pd.read_csv('D:\study_data\emotional\hugging_emotional/train.csv',index_col=0)
data6_2 = pd.read_csv('D:\study_data\emotional\hugging_emotional/test.csv',index_col=0)
data6 = pd.concat([data6_1,data6_2], axis=0)
# data6.to_csv("D:\study_data\emotional/data6_hugging_emotional.csv")
print(data6.value_counts('class'))
# class
# fear       2252
# anger      1701
# joy        1616
# sadness    1533
# dtype: int64
data6 = data6.drop(columns=['sentiment_intensity','class_intensity','labels'])
data6.columns = ['sentence','label']
data6 = data6.replace({'label' : 'joy'}, 'euphoria') 
data6 = data6.replace({'label' : 'fear'}, 'dysphoria') 
data6 = data6.replace({'label' : 'anger'}, 'dysphoria') 
data6 = data6.replace({'label' : 'sadness'}, 'dysphoria') 
print(data6.value_counts('label'))
data6.reset_index(inplace=True,drop=True)
# data6.to_csv("D:\study_data\emotional\last/data6_hugging_emotional.csv")

alldata = pd.concat([data1,data3,data4,data5,data6],ignore_index=True)
alldata.to_csv("D:\study_data\emotional/last/alldata_emotional.csv",encoding='utf-8')

print(alldata.shape)
 
 
 
 
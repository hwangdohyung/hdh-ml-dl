import pandas as pd 
import numpy as np 

path = 'D:\study_data\KERC22Dataset\KERC22Dataset_PublicTest/'
# x = pd.read_csv(path + 'train_data.csv')
x = pd.read_csv(path + "train_data.tsv",delimiter='\t',)

y = pd.read_csv(path + "train_labels.csv",)

print(x.shape,y.shape)

print(y)

df = pd.concat([x, y], axis=1)

print(df)
print(df.shape)

df.drop(['sentence_id'],axis=1)

df.to_csv(path + 'df.csv',index=False)





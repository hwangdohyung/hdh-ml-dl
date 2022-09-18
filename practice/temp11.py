import pandas as pd 
import numpy as np 

path = 'D:\study_data\KERC22Dataset\KERC22Dataset_PublicTest/'
# x = pd.read_csv(path + 'train_data.csv')
x = pd.read_csv(path + "train_data.tsv",delimiter='\t',index_col='sentence_id')

y = pd.read_csv(path + "train_labels.csv",index_col='sentence_id')


df = pd.concat([x, y], axis=1)



df.to_csv(path + 'df.csv')






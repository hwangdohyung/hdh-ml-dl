import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)




import numpy as np 
import pandas as pd
from sklearn import metrics
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf
import datetime


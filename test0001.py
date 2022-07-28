import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

pre_path = "D:\study_data\_data\image\pix2pix"


import os

clr_img_path = []



for img_path in os.listdir(pre_path) :
    clr_img_path.append(os.path.join(pre_path, img_path))
    

clr_img_path.sort()



from PIL import Image
from keras.preprocessing.image import img_to_array

X = []
y = []

for i in range(4) :
    
    img1 = cv2.cvtColor(cv2.imread(clr_img_path[i]), cv2.COLOR_BGR2RGB)
     
    y.append(img_to_array(Image.fromarray(cv2.resize(img1,(128,128)))))

y = np.array(y)

y = (y/127.5) - 1

LAMBDA = 100
BATCH_SIZE = 16
BUFFER_SIZE  = 400

train_dataset = tf.data.Dataset.from_tensor_slices((X))
valid_dataset = tf.data.Dataset.from_tensor_slices((y))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
valid_dataset = valid_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)


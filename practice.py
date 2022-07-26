#pt3장이상 time table 무조건 들어가야함 월~일 까지 무엇을 할것인지
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array

pred = cv2.imread("D:\study_data\_data\image\pix2pix/ad.jpg")
pred = cv2.resize(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB), (256,256))

pred = np.array(pred)

plt.figure(figsize = (50,50))

 
plt.imshow(pred/ 255.0)
plt.axis('off')
plt.title('ColorImage')
plt.show()
print(pred.shape)
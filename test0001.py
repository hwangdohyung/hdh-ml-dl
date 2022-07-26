import tensorflow as tf 
import os 
import time 
from matplotlib import pyplot as plt 
from IPython import display

PATH = 'D:\study_data\_data\image\gan/'


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
  image = tf.io.read_file(image_file)

  w = tf.shape(image)[1]

  w = w // 2
  color_image = image[:, :w, :]
  gray_image = image[:, w:, :]

  color_image = tf.cast(gray_image, tf.float32)
  gray_image = tf.cast(color_image, tf.float32)

  return gray_image, color_image

gray, color = load(PATH +'image0001.jpg')
plt.figure()
plt.imshow(gray/255.0)
plt.figure()
plt.imshow(color/255.0)
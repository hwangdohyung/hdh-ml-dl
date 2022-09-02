import tensorflow as tf

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
        print('gpu 돈다')
else:
        print('gpu 안도라')
        
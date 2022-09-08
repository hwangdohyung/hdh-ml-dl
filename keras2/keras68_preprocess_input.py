from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input,decode_predictions
import numpy as np 

model = ResNet50(weights='imagenet')

img_path = 'D:\study_data\_data\dog/123456.png'
img = image.load_img(img_path, target_size=(224, 224))
print(img)

x = image.img_to_array(img)
print('====================== image.img_to_array(img) =================')
print(x, '\n', x.shape) 
 
x = np.expand_dims(x, axis=0) # 여기 axis: 늘려주고싶은 차원의 위치를 잡아준다. 
print('====================== np.expand_dims(x, axis=0) =================')
print('\n', x.shape) 
print(np.min(x), np.max(x))

x = preprocess_input(x)
print('====================== x = preprocess_input(x) =================')
print(x, '\n', x.shape) 
print(np.min(x), np.max(x))

print('====================== model.predict =================')
preds = model.predict(x)
print(preds, '\n', preds.shape )

print('결과는 : ', decode_predictions(preds, top=5)[0])



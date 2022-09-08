from keras.applications import VGG16, VGG19
from keras.applications import ResNet50,ResNet50V2
from keras.applications import ResNetRS101,ResNet101V2,ResNetRS152,ResNet152V2
from keras.applications import DenseNet121,DenseNet169,DenseNet201
from keras.applications import InceptionV3,InceptionResNetV2
from keras.applications import MobileNet,MobileNetV2
from keras.applications import MobileNetV3Small,MobileNetV3Large
from keras.applications import NASNetLarge,NASNetMobile
from keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB7
from keras.applications import Xception

m_list = [VGG16,VGG19,ResNet50,ResNet50V2,ResNetRS101,ResNet101V2,ResNet152V2,
          DenseNet121,DenseNet169,DenseNet201,ResNetRS152,InceptionV3,InceptionResNetV2,
          MobileNet,MobileNetV2,MobileNetV3Small,MobileNetV3Large,NASNetLarge,NASNetMobile,
          EfficientNetB0,EfficientNetB1,EfficientNetB7,Xception]
last = []
weight = []
train = []


for i in m_list:
    model = i()
    model.trainable = False
    model.summary()
    print

# print('===================================================')
# print('모델명 : ', )
# print('전체 가중치 갯수 : ', )
# print('훈련 가능 가중치 갯수 : ', )



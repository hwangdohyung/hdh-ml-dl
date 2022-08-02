import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os

clr_path = "D:\study_data\_data\image\gan\color"
gry_path = "D:\study_data\_data\image\gan\gray"
last_path = "D:\study_data\_data\image\pix2pix"

clr_img_path = [] 
gry_img_path = []
last_img_path = []

for img_path in os.listdir(clr_path) :  
    clr_img_path.append(os.path.join(clr_path, img_path))  
    
for img_path in os.listdir(gry_path) :
    gry_img_path.append(os.path.join(gry_path, img_path))

for img_path in os.listdir(last_path) :
    last_img_path.append(os.path.join(last_path, img_path))

from PIL import Image
from keras.preprocessing.image import img_to_array

X = []
y = []
z = []

for i in range(5000) :
    
    img1 = cv2.cvtColor(cv2.imread(clr_img_path[i]), cv2.COLOR_BGR2RGB)    
    img2 = cv2.cvtColor(cv2.imread(gry_img_path[i]), cv2.COLOR_BGR2RGB)    
     
    y.append(img_to_array(cv2.resize(img1,(128,128))))   
    X.append(img_to_array(cv2.resize(img2,(128,128))))   

X = np.array(X)
y = np.array(y)

for i in range(5) :
    img3 = cv2.cvtColor(cv2.imread(last_img_path[i]), cv2.COLOR_BGR2RGB)   
    z.append(img_to_array(cv2.resize(img3,(128,128))))  
     
z = np.array(z)

X = (X/127.5) - 1
y = (y/127.5) - 1
z = (z/127.5) - 1

LAMBDA = 100
BATCH_SIZE = 64
BUFFER_SIZE  = 5000
TEST_BATCH = 6

train_dataset = tf.data.Dataset.from_tensor_slices((X, y)) 
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((z))
test_dataset = test_dataset.batch(batch_size=TEST_BATCH)

from tensorflow_addons.layers import SpectralNormalization
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization

############################### 인코더 #################################

init = RandomNormal(mean = 0.0, stddev = 0.02)
def d_block (x_input, filters, strides, padding, batch_norm, inst_norm) :
    
    x = Conv2D(filters, (4, 4),
               strides=strides,
               padding=padding,
               use_bias= False,
               kernel_initializer = init)(x_input)
    
    if batch_norm == True :
        x = BatchNormalization   ()(x)
    if inst_norm  == True :
        x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

############################## 디코더 ##################################

def u_block (x, skip, filters, strides, padding, batch_norm, inst_norm) :
    
    x = Conv2DTranspose(filters, (4, 4),
                        strides=strides,
                        padding=padding,
                        use_bias= False,
                        kernel_initializer = init)(x)
    
    if batch_norm == True :
        x = BatchNormalization   ()(x)
    if inst_norm  == True :
        x = InstanceNormalization()(x)
    x = ReLU()(x)
    conc_x = Concatenate()([x , skip])
    
    return conc_x

def PatchGAN (image_shape) :
    
    genI = Input(shape =  image_shape)
    inpI = Input(shape =  image_shape)
    conc = Concatenate()([genI, inpI])
    
    c064 = d_block(conc, 2**6, 2, 'same', False, False)
    c128 = d_block(c064, 2**7, 2, 'same', False, True )
    c256 = d_block(c128, 2**8, 2, 'same', True , False)
    
    temp = ZeroPadding2D()(c256)
    
    c512 = d_block(temp, 2**9, 1,'valid', True , False)
    
    temp = ZeroPadding2D()(c512)
    
    c001 = Conv2D(2**0, (4,4), strides=1, padding = 'valid', activation = 'sigmoid', kernel_initializer=init)(temp)
    
    model = Model(inputs = [genI, inpI], outputs = c001)
    return model

dis0 = PatchGAN((128,128,3,))


def mod_Unet () :
    
    srcI = Input(shape = (128,128,3,))
    
    # Contracting path
    
    c064 = d_block(srcI, 2**6, 2, 'same', False, False) # _______________________.
    c128 = d_block(c064, 2**7, 2, 'same', True , False) # ____________________.  .
    c256 = d_block(c128, 2**8, 2, 'same', True , False) # _________________.  .  .
    c512 = d_block(c256, 2**9, 2, 'same', True , False) # ______________.  .  .  .
    d512 = d_block(c512, 2**9, 2, 'same', True , False) # ___________.  .  .  .  .
    e512 = d_block(d512, 2**9, 2, 'same', True , False) # ________.  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    # Bottleneck layer                                            .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    f512 = d_block(e512, 2**9, 2, 'same', True , False) #         .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    # Expanding  path                                             .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    u512 = u_block(f512, e512, 2**9, 2, 'same', True, False)# ____.  .  .  .  .  .
    u512 = u_block(u512, d512, 2**9, 2, 'same', True, False)# _______.  .  .  .  .
    u512 = u_block(u512, c512, 2**9, 2, 'same', True, False)# __________.  .  .  .
    u256 = u_block(u512, c256, 2**8, 2, 'same', True, False)# _____________.  .  .
    u128 = u_block(u256, c128, 2**7, 2, 'same', True, False)# ________________.  .
    u064 = u_block(u128, c064, 2**6, 2, 'same', False, True)# ___________________.
    
    
    genI = Conv2DTranspose(3, (4,4), strides = 2, padding = 'same', activation = 'tanh', kernel_initializer = init)(u064)
    
    model = Model(inputs = srcI, outputs = genI)
    return model


gen0 = mod_Unet()

 # (W//1) x (H//1)]


bin_entropy = keras.losses.BinaryCrossentropy(from_logits = True)

def gen_loss (dis_gen_output, target_image, gen_output) :

    ad_loss = bin_entropy(tf.ones_like (dis_gen_output) ,  dis_gen_output)
    l1_loss = tf.reduce_mean(tf.abs(tf.subtract(target_image,gen_output)))
    
    
    total_loss = ad_loss + (LAMBDA*l1_loss)
    
    return total_loss, ad_loss, l1_loss

def dis_loss (dis_gen_output, dis_tar_output) :
    
    gen_loss = bin_entropy(tf.zeros_like(dis_gen_output), dis_gen_output)
    tar_loss = bin_entropy(tf.ones_like (dis_tar_output), dis_tar_output)
    
    total_dis_loss = gen_loss + tar_loss
    return total_dis_loss

g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)
d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)


@tf.function
def train_on_batch (input_image, tar_image) :
    
    with tf.GradientTape(persistent = True) as  g :
        
        gen_image = gen0(input_image, training=True)
        
        dis_tar_output = dis0([input_image, tar_image], training = True)
        dis_gen_output = dis0([input_image, gen_image], training = True)
        
        g_loss = gen_loss(dis_gen_output, tar_image, gen_image)
        d_loss = dis_loss(dis_gen_output, dis_tar_output)
        
    # compute gradients
    g_gradients = g.gradient(g_loss, gen0.trainable_variables) # generatorLoss
    d_gradients = g.gradient(d_loss, dis0.trainable_variables)   # dis loss

    # apply gradient descent
    g_optimizer.apply_gradients(zip(g_gradients, gen0.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, dis0.trainable_variables))
 
 
def fig (input_image, gen_image, tar_image) :
    
    plt.figure(figsize = (20, 20))
    
    plt.subplot(1,3,1)
    plt.imshow((input_image[0] + 1.0) / 2.0)
    plt.title('BandW Image',fontsize = 20)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow((gen_image[0] + 1.0) / 2.0)
    plt.title('GenerateImg',fontsize = 20)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow((tar_image[0] + 1.0) / 2.0)
    plt.title('Colored Img',fontsize = 20)
    plt.axis('off')
    
    plt.show()

def fit (EPOCHS = 200) :
     
    for epoch in range(EPOCHS) :
        
        print(f'Epoch {epoch} out of {EPOCHS}')
        
        for n, (input_image, tar_image) in train_dataset.enumerate() :
         
            train_on_batch(input_image, tar_image)
        
        if epoch  :
            global_gen_image = gen0(input_image,training = True)
            fig(input_image, global_gen_image, tar_image)

fit(EPOCHS = 2)

# for b_w_image,tar_image in train_dataset.take(20) :
#     gen_image = gen0(b_w_image , training = True)
#     fig(b_w_image, gen_image, tar_image)

def fig1 (input_image, gen_image) :
    
    plt.figure(figsize = (10, 10))
    
    plt.subplot(1,2,1)
    plt.imshow((input_image[0] + 1.0) / 2.0)
    plt.title('Input Image',fontsize = 20)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow((gen_image[0] + 1.0) / 2.0)
    plt.title('GenerateImg',fontsize = 20)
    plt.axis('off')
    
    plt.show()

for input_image in test_dataset.take(5) :
    gen_image = gen0(input_image , training = True)
    fig1(input_image, gen_image)


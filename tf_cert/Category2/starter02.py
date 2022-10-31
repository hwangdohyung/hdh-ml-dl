# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Flatten
from sklearn.metrics import accuracy_score
import numpy as np 
def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    
    # YOUR CODE HERE
    model = Sequential()
    model.add(Dense(20,input_shape=(28,28)))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Dense(20))
    model.add(Dense(10, activation='softmax'))
    
    
    model.compile(loss= 'sparse_categorical_crossentropy',optimizer='adam')
    model.fit(x_train,y_train, epochs=100,batch_size=256,verbose=1)
    loss = model.evaluate(x_test,y_test)
    print('loss : ', loss)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict,axis=1)
    acc = accuracy_score(y_test,y_predict)
    print('acc : ', acc)
    
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")





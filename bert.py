
import os 
import re 
import json 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('seaborn-white')
from tqdm import tqdm 
from keras.utils import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
tf.random.set_seed(111)
np.random.seed(111)
print(tf.__version__)
import transformers as tr
from transformers import TFBertModel,BertTokenizer
print(tr.__version__)

BATCH_SIZE = 32
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN = 39 

import urllib.request

train_file = urllib.request.urlopen('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt')
test_file  = urllib.request.urlopen('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt')

train_data = pd.read_table(train_file)
test_data = pd.read_table(test_file)

train_data = train_data.dropna()
test_data = test_data.dropna()

print(train_data.head())
print(test_data.head())

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',cache_dir='bert_ckpt',do_lower_case=False)

def bert_tokenizer(sentence, MAX_LEN):
    
    encoded_dict = tokenizer.encode_plus(
        text = sentence,
        add_special_tokens = True,
        max_length = MAX_LEN,
        pad_to_max_length = True,
        return_attention_mask = True
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    
    return input_id, attention_mask, token_type_id

input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []

for train_sentence, train_label in tqdm(zip(train_data['document'], train_data['label']), total=len(train_data)):
  try:
    input_id,attention_mask, token_type_id = bert_tokenizer(train_sentence, MAX_LEN)
    

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    train_data_labels.append(train_label)
  except Exception as e:
      print(e)
      pass

train_movie_input_ids = np.array(input_ids, dtype=int)
train_movie_attention_masks = np.array(attention_masks, dtype=int)
train_movie_token_type_ids = np.array(token_type_ids, dtype=int)
train_movie_inputs = (train_movie_input_ids,train_movie_attention_masks,train_movie_token_type_ids)
train_data_labels = np.asarray(train_data_labels, dtype= np.int32)
    
print('Sentences: {}\nLabels: {}'.format(len(train_movie_input_ids),len(train_data_labels)))

idx = 5 
input_id = train_movie_input_ids[idx]
attention_mask = train_movie_attention_masks[idx]
token_type_id = train_movie_token_type_ids[idx]

print(input_id)
print(attention_mask)
print(token_type_id)
print(tokenizer.decode(input_id))

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()
        
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifer = tf.keras.layers.Dense(num_class,
                                               kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                               name = 'Classifier')
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training = False):
            outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids = token_type_ids)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output, training=training)
            logits = self.classifier(pooled_output)
            
            return logits


cls_model = TFBertClassifier(model_name='bert-base-multilingual-cased',
                             dir_path = 'bert_ckpt',
                             num_class=2)        

optimizer = tf.keras.optimizers.Adamax(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss = loss, metrics=[metric])

model_name = 'tf2_bert_naver_movie'

earlystop_callback = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0001,patience=2)

checkpoint_path = os.path.join('./',model_name, 'weight.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print('Directory already exists\n'.format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print('Directory create complete\n'.format(checkpoint_dir))
    

cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy',verbose=1,save_bset_only = True,
                              save_weights_only=True)

history= cls_model.fit(train_movie_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                       validation_split= VALID_SPLIT,callbacks= [earlystop_callback,cp_callback])

print(history.history)




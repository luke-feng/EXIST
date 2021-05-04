import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import keras
import tensorflow_text as text 
logging.basicConfig(level=logging.INFO)
from sklearn.model_selection import train_test_split

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import matplotlib.pyplot as plt



# BERT_URL_en = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
nnlm_es_dim128 = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128/1"
nnlm_en_dim128 = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
BERT_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
# bert_layer = hub.KerasLayer(BERT_URL, trainable=True, name='bert_layer')
preprocessor = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
file_path = '~/Documents/vscode/EXIST/EXIST/'

def from_label_to_number(labelList):
    numberList = pd.DataFrame([1 if x  == 'sexist' else 0 for x in labelList])
    return numberList

def data_prepare():
    trainfile = pd.read_csv(file_path+'EXIST2021_training.tsv', sep='\t', header=0)
    X_raw = trainfile['text']
    y_raw = from_label_to_number(trainfile['task1'])
    X_train, X_dev, y_train, y_dev = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    X_train = tf.constant(X_train)
    X_dev = tf.constant(X_dev)
    return X_train, X_dev, y_train, y_dev


X_train, X_dev, y_train, y_dev =  data_prepare()

# t1 = X_train[0:10]




def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocessor, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(BERT_URL, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    # hub_layer = hub.KerasLayer(nnlm_es_dim128, output_shape=[128], input_shape=[], dtype=tf.string)
    net = outputs['pooled_output']
    # net = tf.keras.layers.concatenate()([net, hub_layer])
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(64, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(text_test))
# print(tf.sigmoid(bert_raw_result))
# tf.keras.utils.plot_model(classifier_model)

# loss
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# optimizer
epochs = 10
optimizer = tf.keras.optimizers.Adam(lr=1e-5)

#early stop
callback = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=5)

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)


print(f'Training model with {BERT_URL}')

train_history = classifier_model.fit(
    x = X_train, y = y_train,
    validation_data=(X_dev, y_dev),
    epochs=epochs,
    shuffle=True,
    callbacks=[callback],
    batch_size=16,
    verbose=1)


dataset_name = 'exist'
saved_model_path = file_path + '{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)

history_dict = train_history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

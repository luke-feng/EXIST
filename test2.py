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


# BERT_URL_multi = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
# BERT_URL_en = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

nnlm_es_dim128 = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128/1"
nnlm_en_dim128 = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"


preprocessor_multi = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
preprocessor_en = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

file_path = '/home/chaofeng/Documents/vscode/EXIST/EXIST/'

BERT_URL_multi = file_path + 'bert_multi/'
BERT_URL_en = file_path + 'bert_en/'

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
    test = pd.read_csv(file_path+'EXIST2021_test.tsv', sep='\t', header=0)
    return X_train, X_dev, y_train, y_dev, test


X_train, X_dev, y_train, y_dev, test =  data_prepare()

# t1 = X_train[0:10]




def build_classifier_model():

    text_input_en = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_en')
    text_input_es = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_es')
    text_input_bert_multi = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_bert_multi')
    text_input_bert_en = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_bert_en')

    hub_layer_en = hub.KerasLayer(nnlm_en_dim128,
                           input_shape=[], dtype=tf.string)

    hub_layer_es = hub.KerasLayer(nnlm_es_dim128,
                           input_shape=[], dtype=tf.string)

    preprocessing_multi_layer = hub.KerasLayer(preprocessor_multi, name='preprocessing_multi_layer')
    encoder_inputs_multi = preprocessing_multi_layer(text_input_bert_multi)
    encoder_multi = hub.KerasLayer(BERT_URL_multi, trainable=True)
    outputs_multi = encoder_multi(encoder_inputs_multi)

    net_bert_multi = outputs_multi['pooled_output']
    net_bert_multi = tf.keras.layers.Dropout(0.1)(net_bert_multi)
    net_bert_multi = tf.keras.layers.Dense(32, activation='relu')(net_bert_multi)

    preprocessing_en_layer = hub.KerasLayer(preprocessor_en, name='preprocessing_en_layer')
    encoder_inputs_en = preprocessing_en_layer(text_input_bert_en)
    encoder_en = hub.KerasLayer(BERT_URL_en, trainable=True)
    outputs_en = encoder_en(encoder_inputs_en)

    net_bert_en = outputs_en['pooled_output']
    net_bert_en = tf.keras.layers.Dropout(0.1)(net_bert_en)
    net_bert_en = tf.keras.layers.Dense(32, activation='relu')(net_bert_en)

    net_en = hub_layer_en(text_input_en)
    net_en = tf.keras.layers.Dense(32, activation='relu')(net_en)
    net_en = tf.keras.layers.Dropout(0.2)(net_en)
    net_en = tf.keras.Model(inputs=text_input_en, outputs=net_en)

    net_es = hub_layer_es(text_input_es)
    net_es = tf.keras.layers.Dense(32, activation='relu')(net_es)
    net_es = tf.keras.layers.Dropout(0.2)(net_es)
    net_es = tf.keras.Model(inputs=text_input_es, outputs=net_es)

    combined = tf.keras.layers.concatenate([net_en.output, net_es.output, net_bert_multi, net_bert_en], axis=-1)

    net = net = tf.keras.layers.Dense(32, activation='relu')(combined)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model([text_input_en, text_input_es, text_input_bert_multi, text_input_bert_en], net)

classifier_model = build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(text_test))
# print(tf.sigmoid(bert_raw_result))
# tf.keras.utils.plot_model(classifier_model)

# loss
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# optimizer
epochs = 3
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

#early stop
callback = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=5)

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)


print(f'Training model with {BERT_URL_multi}')

train_history = classifier_model.fit(
    x = [X_train, X_train, X_train, X_train], y = y_train,
    validation_data=([X_dev, X_dev, X_dev, X_dev] , y_dev),
    epochs=epochs,
    shuffle=True,
    callbacks=[callback],
    batch_size=4,
    verbose=1)


dataset_name = 'exist'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

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

x_test = tf.constant(test['text'])
y_test = classifier_model.predict(x = x_test, batch_size=4)
y = pd.DataFrame(['sexist' if x  == 1 else 'non-sexist' for x in y_test])

case = test['test_case']
ids = test['id']

results = pd.DataFrame(case, ids, y)
results.to_csv(file_path + 'output.tsv', sep='\t', header=False, index=False)
"""
Transfer learning with DistilBert: build a text classifier using the DistilBert language model
The Glue/CoLA dataset is used (https://nyu-mll.github.io/CoLA/).
"""

# Install DistilBert
!pip install -q transformers tfds-nightly

import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd

# May only work on the 2nd try so add an except clause
try:
  from transformers import DistilBertTokenizer, TFDistilBertModel
except Exception as err:
  from transformers import DistilBertTokenizer, TFDistilBertModel

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Dropout

# Import dataset with tf.datasets
import tensorflow_datasets as tfds

dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# -------------------------Data Preparation-------------------------------------

def load_data(save_dir="./"):

  dataset = tfds.load('glue/cola', shuffle_files=True)
  train = tfds.as_dataframe(dataset["train"])
  val = tfds.as_dataframe(dataset["validation"])
  test = tfds.as_dataframe(dataset["test"])

  return train, val, test

def prepare_raw_data(df):

  raw_data = df.loc[:, ["idx", "sentence", "label"]]
  raw_data["label"] = raw_data["label"].astype('category')

  return raw_data

train, val, test = load_data()
train = prepare_raw_data(train)
val = prepare_raw_data(val)
test = prepare_raw_data(test)

# Process raw data: drop duplicates, set minimum number of words in a sentence

def clean_data(df):
  # Remove duplicates in 'sentence' column from df
  cleaned_data = df.drop_duplicates(subset='sentence', keep='first')

  # Calculate number of words in sentence and filter out sentences with less than threshold number of words
  threshold = 3
  cleaned_data['num_words'] = cleaned_data['sentence'].apply(lambda x: len(x.split()))
  cleaned_data = cleaned_data[cleaned_data['num_words'] > threshold]

  return cleaned_data

train = clean_data(train)
val = clean_data(val)
test = clean_data(test)

print(train.head())
print(test.head())

# Use decode to manipulate text (sentences are originally encoded as binary strings)
def extract_text_and_y(df):

  text = [x.decode('utf-8') for x in  df.sentence.values]

  # use a single sigmoid output since there are two output classes
  y = np.array([x for x in df.label.values])

  return text, y

# Encode text using dbert_tokenizer
def encode_text(text):

    encoded_text = dbert_tokenizer(text, padding='max_length', truncation=True, return_tensors="tf", max_length=128)

    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    return input_ids, attention_mask

# Prepare the input for DistilBert model
train_text, train_y = extract_text_and_y(clean_data(train))
val_text, val_y = extract_text_and_y(clean_data(val))
test_text, test_y = extract_text_and_y(clean_data(test))

train_input, train_mask = encode_text(train_text)
val_input, val_mask = encode_text(val_text)
test_input, test_mask = encode_text(test_text)

train_model_inputs_and_masks = {
    'inputs' : train_input,
    'masks' : train_mask
}

val_model_inputs_and_masks = {
    'inputs' : val_input,
    'masks' : val_mask
}

test_model_inputs_and_masks = {
    'inputs' : test_input,
    'masks' : test_mask
}

# ---------------------------Modeling-----------------------------------------

# Build and train model

def build_model(base_model, trainable=False, params={}):

    max_seq_len = train_input.shape[1]
    inputs = Input(shape=(max_seq_len,), name='inputs', dtype='int32')
    masks  = Input(shape=(max_seq_len,), name='masks', dtype='int32')

    base_model.trainable = trainable

    dbert_output = base_model(inputs, attention_mask=masks)

    dbert_last_hidden_state = dbert_output.last_hidden_state # gets the output encoding (vector of 768 values) for each token

    cls_token = dbert_last_hidden_state[:, 0, :] # first token passed into the model; cls_token corresponds to first element of DistilBert sequence and is used to build the sentence classifier network

    my_output = keras.layers.BatchNormalization()(cls_token)
    my_output = Dense(256, activation='relu')(my_output)
    my_output = Dropout(rate=params.get('dropout_rate'), seed=42)(my_output)
    my_output = Dense(128, activation='relu')(my_output)
    my_output = Dropout(rate=params.get('dropout_rate'), seed=42)(my_output)
    probs = Dense(1, activation='sigmoid')(my_output)

    model = keras.Model(inputs=[inputs, masks], outputs=probs)
    model.summary()

    return model

dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
params={'dropout_rate': 0.1,
        'learning_rate': 0.0005
        }

model = build_model(dbert_model, params=params)

# Compile model
def compile_model(model):

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate')), loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(curve='PR', name='PR_AUC'), 'Precision', 'Recall'])
    return model

model = compile_model(model)

# Train model
def train_model(model, model_inputs_and_masks_train, model_inputs_and_masks_val,
    y_train, y_val, batch_size, num_epochs):

    history = model.fit(x=model_inputs_and_masks_train,
                        y=y_train,
                        validation_data=(model_inputs_and_masks_val, y_val),
                        batch_size=batch_size,
                        epochs=num_epochs)

    return model, history

model, history = train_model(model, train_model_inputs_and_masks, val_model_inputs_and_masks, train_y, val_y, batch_size=128, num_epochs=5)

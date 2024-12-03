import pickle
from typing import Union

import numpy as np
import streamlit as st
import tensorflow as tf
from fastapi import FastAPI
from tensorflow.keras.models import load_model

app = FastAPI()

best_model = load_model("iris_classification_tf.keras")
with open("iris_preprocessor.pkl", 'rb') as file:
    preprocessor = pickle.load(file)
scaler = preprocessor['scaler']
one_hot = preprocessor['encoder']


def prediction(user_input):
    
    scaled_data = scaler.transform(user_input)
    pred = tf.one_hot(
        np.argmax(best_model.predict(scaled_data), axis =1), depth =3
    ).numpy()
    return one_hot.inverse_transform(pred)[0][0]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def web_prediction(user_input):
    
    array_input = np.array(
        [eval(user_input)]
    )
    output = prediction(array_input)
    
    return {
        "input" : eval(user_input),
        "result" : output
    }


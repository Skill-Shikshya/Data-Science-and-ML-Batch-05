import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model


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


if __name__ == "__main__":
    st.title("Flower Classifier !")
        
    sL = st.number_input("Enter Sepal Length", min_value=0.0, max_value=10.0, step = 0.1)
    sW = st.number_input("Enter Sepal Width", min_value=0.0, max_value=10.0, step = 0.1)
    pL = st.number_input("Enter Petal Length", min_value=0.0, max_value=10.0, step = 0.1)
    pW = st.number_input("Enter Petal Width", min_value=0.0, max_value=10.0, step = 0.1)
    
    user_input = np.array([[sL, sW, pL, pW]])

    if st.button("Predict !", icon = ":material/settings:"):
        st.write(
            f"Given Flower belongs to {prediction(user_input)} family."
        )

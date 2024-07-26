# Streamlit App

import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('supplement_recommendation_model.pkl')

# Define the app
st.title('Fitness Supplement Recommendation System')

# Input fields
age = st.number_input('Age', min_value=0, max_value=60, value=25)
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Weight (kg)', min_value=10, max_value=150, value=70)
bmi = st.number_input('BMI', min_value=10.0, max_value=30.0, value=24.0)

# Predict button
if st.button('Predict'):
    # Make a prediction
    input_data = np.array([[age, height, weight, bmi]])
    prediction = model.predict(input_data)[0]
    st.write(f'Supplement Recommendation: {prediction}')

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import predict_crop

# Load the trained model
model = joblib.load("crop_model_randomforest1.pkl")

# Load the label encoder categories
label_categories = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
    'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

# App Title
st.title("🌾 Crop Recommendation System")
st.markdown("Predict the best crop to grow based on soil and environmental parameters.")

# User Input Form
with st.form("Crop Input Form"):
    st.header("Input Parameters")
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=None)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=None)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=None)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=None)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=None)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=None)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=None)

    # Submit Button
    submitted = st.form_submit_button("Predict Crop")

data = {
    'N': n, 'P': p, 'K': k,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}
# Handle Prediction
if submitted:
    result = predict_crop(data)
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <h2 style="color: #4CAF50;">Recommended Crop</h2>
            <h1 style="color: #333;">{result}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    #crop calendar

    st.subheader("🗓️ Crop Calendar")
    crop_months = {
        "rice": "June to November", "wheat": "November to April", "maize": "June to September",
        "cotton": "April to October", "millet": "July to October", "barley": "November to March",
        "sugarcane": "October to March", "banana": "Year-round", "apple": "August to October",
        "orange": "November to February", "grapes": "February to April", "papaya": "Year-round",
        "mango": "March to June", "pomegranate": "July to October", "chickpea": "October to March",
        "kidneybeans": "June to October", "blackgram": "June to September", "lentil": "November to March",
        "mothbeans": "July to September", "mungbean": "July to October", "muskmelon": "February to May",
        "watermelon": "January to April", "jute": "May to August"
    }
    crop_period = crop_months.get(result, "No calendar available")
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <h2 style="color: #4CAF50;">Best growing season for recommended crop</h2>
            <h1 style="color: #333;">{crop_period}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

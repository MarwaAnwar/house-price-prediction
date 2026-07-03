import streamlit as st
import joblib
import numpy as np
import os

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter the house details below and click Predict.")

MedInc = st.number_input("Median Income", min_value=0.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=1.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=1.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, value=1.0)
Population = st.number_input("Population", min_value=1.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=1.0, value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict House Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ${prediction[0]*100000:,.2f}")
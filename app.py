import streamlit as st
import joblib
import numpy as np
import os
import time

# ✅ Register the class in globals() to fix model loading issue
from q_learning import QLearningSurgePricing
globals()['QLearningSurgePricing'] = QLearningSurgePricing  

# ✅ Ensure model files exist
if not os.path.exists("demand_model.pkl") or not os.path.exists("q_learning_model.pkl"):
    st.error("❌ Model files not found! Ensure 'demand_model.pkl' and 'q_learning_model.pkl' exist.")
    st.stop()

# ✅ Load models with error handling
try:
    demand_model = joblib.load("demand_model.pkl")
    pricing_model = joblib.load("q_learning_model.pkl")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# 🌟 Set Streamlit page config
st.set_page_config(page_title="🚖 AI Surge Pricing", layout="centered")

# 🚀 Title
st.markdown("<h1 style='text-align: center; color: #FF4500;'>🚖 AI-Powered Surge Pricing</h1>", unsafe_allow_html=True)

# 📌 Sidebar
st.sidebar.image("https://media.giphy.com/media/l0HlOvJ7yaacpuSas/giphy.gif", width=250)
st.sidebar.markdown("### 🎛 Configure Ride Details")

# 🚗 Ride Settings
hour = st.sidebar.slider("⏰ Hour of the day", 0, 23, 12)
traffic = st.sidebar.slider("🚦 Traffic Level (1-10)", 1, 10, 5)

# 🌦 Weather & Event Inputs
weather = st.sidebar.radio("🌦 Weather", ["Clear", "Rain", "Storm"], horizontal=True)
events = st.sidebar.radio("🎉 Nearby Event?", ["No", "Yes"], horizontal=True)

# 📏 Distance Input
distance_km = st.slider("📏 Trip Distance (km)", 2.0, 30.0, 10.0)

# 🎯 Predict Button
if st.button("⚡ Predict Surge Price"):
    with st.spinner("Analyzing demand & pricing... ⏳"):
        time.sleep(2)

        # ✅ Encode categorical inputs
        weather_map = {"Clear": 1, "Rain": 2, "Storm": 3}
        events_map = {"No": 0, "Yes": 1}

        # ✅ Model Input
        features = np.array([[hour, traffic, weather_map[weather], events_map[events], distance_km]], dtype=np.float32)

        try:
            # 🎯 Predict
            predicted_demand = demand_model.predict(features)[0]
            surge_multiplier = pricing_model.get_price_multiplier(hour, traffic, weather_map[weather], events_map[events])

            # 🚖 Base fare per km
            base_fare_per_km = 1.5
            final_fare = round(distance_km * base_fare_per_km * surge_multiplier, 2)

            # 🔥 Display Results
            st.success("✅ Prediction Completed!")
            st.metric("💰 Final Fare ($)", final_fare)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

st.caption("🔍 Powered by AI | XGBoost + Reinforcement Learning 🤖")

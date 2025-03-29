import streamlit as st
import joblib
import numpy as np
import os
import time
from q_learning import QLearningSurgePricing  # ✅ Import the custom class

# ✅ Register the class in globals() to fix loading issue
globals()['QLearningSurgePricing'] = QLearningSurgePricing  

# ✅ Check if model files exist
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

# 🚀 Cool Title with Emojis
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4500;'>🚖 AI-Powered Surge Pricing</h1>
    <h4 style='text-align: center; color: #2E8B57;'>Smart fare calculation based on demand, weather, and traffic.</h4>
    """, 
    unsafe_allow_html=True
)

# 📌 Sidebar with animated icon
st.sidebar.image("https://media.giphy.com/media/l0HlOvJ7yaacpuSas/giphy.gif", width=250)
st.sidebar.markdown("### 🎛 Configure Ride Details")

# 🚗 Ride Settings
hour = st.sidebar.slider("⏰ Hour of the day", 0, 23, 12)
traffic = st.sidebar.slider("🚦 Traffic Level (1-10)", 1, 10, 5)

# 🌦 Weather & Event Inputs
col1, col2 = st.columns(2)
with col1:
    weather = st.radio("🌦 Weather", ["Clear", "Rain", "Storm"], horizontal=True)
with col2:
    events = st.radio("🎉 Nearby Event?", ["No", "Yes"], horizontal=True)

# 📏 Distance Input
distance_km = st.slider("📏 Trip Distance (km)", 2.0, 30.0, 10.0)

# 🎨 Styled Button
st.markdown(
    "<style>div.stButton > button {background-color: #FF4500; color: white; border-radius: 10px;}</style>",
    unsafe_allow_html=True
)

# 🎯 Predict Button with Animation
if st.button("⚡ Predict Surge Price"):
    with st.spinner("Analyzing demand & surge pricing... ⏳"):
        time.sleep(2)

        # ✅ Encode categorical inputs
        weather_map = {"Clear": 1, "Rain": 2, "Storm": 3}
        events_map = {"No": 0, "Yes": 1}

        # ✅ Ensure input is in correct shape
        features = np.array([[hour, traffic, weather_map[weather], events_map[events], distance_km]], dtype=np.float32)
        
        # 📌 Debugging - Show input shape
        st.write("🔍 Feature Input Shape:", features.shape)

        try:
            # 🎯 Predict Demand & Surge Pricing
            predicted_demand = demand_model.predict(features)[0]
            surge_multiplier = pricing_model.get_price_multiplier(hour, traffic, weather_map[weather], events_map[events])

            # 🚖 Base fare per km
            base_fare_per_km = 1.5
            final_fare = round(distance_km * base_fare_per_km * surge_multiplier, 2)

            # 🔥 Display Results
            st.success("✅ Prediction Completed!")
            st.markdown("### 📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔮 Predicted Demand", int(predicted_demand))
            col2.metric("🔥 Surge Multiplier", round(surge_multiplier, 2))
            col3.metric("💰 Final Fare ($)", final_fare)

            # 🚦 Surge Alerts
            if surge_multiplier > 1.5:
                st.warning("⚠️ HIGH DEMAND! Prices are surging 🚀")
            elif surge_multiplier > 1.2:
                st.info("🔄 Moderate surge pricing applied.")
            else:
                st.success("✅ Normal pricing. No surge!")

            # 🏁 Show Ride Summary
            st.markdown("---")
            st.markdown(
                f"""
                **📍 Ride Details:**  
                🚗 **Distance:** {distance_km} km  
                ⏰ **Hour:** {hour}  
                🚦 **Traffic Level:** {traffic}  
                🌦 **Weather:** {weather}  
                🎉 **Event Nearby:** {events}  
                """, 
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.write("🔍 Possible causes:")
            st.write("1. Model input shape mismatch (Check training feature count).")
            st.write("2. Corrupt model file (Try retraining).")

st.markdown("---")
st.caption("🔍 Powered by AI | XGBoost + Reinforcement Learning 🤖")

import streamlit as st
import joblib
import numpy as np

# Load models
demand_model = joblib.load("demand_model.pkl")
pricing_model = joblib.load("q_learning_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Dynamic Surge Pricing", layout="centered")

st.title("🚕 Dynamic Surge Pricing Optimization")
st.markdown("### AI-powered dynamic pricing for ride-hailing services.")

# User Inputs
hour = st.slider("⏰ Hour of the day", 0, 23, 12)
traffic = st.slider("🚦 Traffic Level (1-10)", 1, 10, 5)
weather = st.selectbox("🌦 Weather Condition", ["Clear", "Rain", "Storm"])
events = st.radio("🎉 Is there an event nearby?", ["No", "Yes"])

# Encode categorical inputs
weather_map = {"Clear": 1, "Rain": 2, "Storm": 3}
events_map = {"No": 0, "Yes": 1}

# Predict demand & surge pricing
if st.button("⚡ Predict Demand & Price"):
    features = np.array([[hour, traffic, weather_map[weather], events_map[events]]])
    predicted_demand = demand_model.predict(features)[0]
    surge_multiplier = pricing_model.get_price_multiplier(hour, traffic, weather_map[weather], events_map[events])
    final_fare = round(10 * surge_multiplier, 2)  # Base fare = 10

    # Display results
    st.subheader("📊 Prediction Results")
    st.metric("🔮 Predicted Ride Demand", int(predicted_demand))
    st.metric("🔥 Surge Multiplier", round(surge_multiplier, 2))
    st.metric("💰 Final Fare ($)", final_fare)

    # Display surge status
    if surge_multiplier > 1.5:
        st.warning("⚠️ High demand! Prices are surging 🚀")
    elif surge_multiplier > 1.2:
        st.info("🔄 Moderate surge pricing applied.")
    else:
        st.success("✅ Normal pricing. No surge!")

st.markdown("---")
st.caption("Powered by XGBoost & Reinforcement Learning")

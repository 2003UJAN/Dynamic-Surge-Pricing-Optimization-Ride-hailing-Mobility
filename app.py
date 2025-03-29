import streamlit as st
import joblib
import numpy as np

class QLearningSurgePricing:
    def __init__(self):
        self.q_table = np.zeros((24, 10, 4, 2, 5))  # (Hour, Traffic, Weather, Events, Price Levels)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

    def get_price_multiplier(self, hour, traffic, weather, events):
        state = (hour, traffic - 1, weather - 1, events)  # Adjust indexes
        action = np.argmax(self.q_table[state])
        return 1 + (action * 0.2)  # Surge multiplier (1.0 to 1.8)

try:
    pricing_model = joblib.load("q_learning_model.pkl")
except Exception as e:
    st.error(f"Error loading Q-learning model: {e}")
    pricing_model = QLearningSurgePricing()  # Fallback to a new instance

demand_model = joblib.load("demand_model.pkl")

st.title("🚕 Dynamic Surge Pricing Optimization")

hour = st.slider("⏰ Hour of the day", 0, 23, 12)
traffic = st.slider("🚦 Traffic Level (1-10)", 1, 10, 5)
weather = st.selectbox("🌦 Weather", ["Clear", "Rain", "Storm"])
events = st.radio("🎉 Event nearby?", ["No", "Yes"])

weather_map = {"Clear": 1, "Rain": 2, "Storm": 3}
events_map = {"No": 0, "Yes": 1}

if st.button("⚡ Predict Demand & Price"):
    features = np.array([[hour, traffic, weather_map[weather], events_map[events]]])
    predicted_demand = demand_model.predict(features)[0]
    surge_multiplier = pricing_model.get_price_multiplier(hour, traffic, weather_map[weather], events_map[events])
    final_fare = round(10 * surge_multiplier, 2)

    st.metric("🔮 Predicted Demand", int(predicted_demand))
    st.metric("🔥 Surge Multiplier", round(surge_multiplier, 2))
    st.metric("💰 Final Fare ($)", final_fare)

import streamlit as st
import numpy as np
import joblib
from keras.models import Sequential

model = joblib.load(r"D:\Rent_prediction\src\model\house_rent_prediction_model.pkl")
x_scaler = joblib.load(r"D:\Rent_prediction\src\model\x_scaler.pkl")
y_scaler= joblib.load(r"D:\Rent_prediction\src\model\y_scaler.pkl")

area_map = {"Super Area": 1, "Carpet Area": 2, "Built Area": 3}
city_map = {
    "Mumbai": 4000, "Chennai": 6000, "Bangalore": 5600,
    "Hyderabad": 5000, "Delhi": 1100, "Kolkata": 7000
}
furnish_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
tenant_map = {"Bachelors": 1, "Bachelors/Family": 2, "Family": 3}

# Streamlit UI
st.set_page_config(page_title="🏠 House Rent Prediction", layout="centered")
st.title("🏠 House Rent Prediction App")
st.markdown("Fill in the details below to estimate the rent of your house.")

# User inputs
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
size = st.number_input("Size of the House (in sqft)", min_value=100, max_value=10000, step=10)
area = st.selectbox("Area Type", list(area_map.keys()))
city = st.selectbox("City", list(city_map.keys()))
furnish = st.selectbox("Furnishing Status", list(furnish_map.keys()))
tenant = st.selectbox("Tenant Preference", list(tenant_map.keys()))
bath = st.slider("Number of Bathrooms", 1, 5)

if st.button("Predict Rent"):
    try:
        # Convert inputs to model-ready format
        features = np.array([[bhk, size, area_map[area], city_map[city],
                              furnish_map[furnish], tenant_map[tenant], bath]])

        # Scale features
        scaled_features = x_scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        predicted_rent = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

        st.success(f"💰 Predicted Monthly Rent: ₹ {predicted_rent:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

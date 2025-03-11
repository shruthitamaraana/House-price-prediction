import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
@st.cache_data
def load_data():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target, name="MedHouseVal")
    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (simplified for clarity)
@st.cache_data
def train_model():
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Streamlit app
st.title("🏠 California House Price Prediction")

# ---------------------------
# User Input Section (Integer Values for Rooms/Bedrooms)
# ---------------------------
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider(
        "Median Income (in $10,000)",
        min_value=0.0, max_value=15.0, value=3.0, step=0.1
    )
    house_age = st.slider(
        "House Age (years)",
        min_value=0, max_value=100, value=30, step=1
    )
    # Rooms as integer
    ave_rooms = st.slider(
        "Number of Rooms",
        min_value=1, max_value=10, value=5, step=1
    )
    # Bedrooms as integer
    ave_bedrms = st.slider(
        "Number of Bedrooms",
        min_value=1, max_value=5, value=2, step=1
    )

with col2:
    population = st.slider(
        "Population (Block Group)",
        min_value=0, max_value=5000, value=1000, step=1
    )
    ave_occup = st.slider(
        "Household Members",
        min_value=1, max_value=10, value=3, step=1
    )
    latitude = st.slider(
        "Latitude",
        min_value=32.0, max_value=42.0, value=34.0, step=0.1
    )
    longitude = st.slider(
        "Longitude",
        min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1
    )

# Convert integer inputs to float (to match model expectations)
input_data = pd.DataFrame([[
    med_inc,
    house_age,
    float(ave_rooms),  # Convert to float
    float(ave_bedrms),  # Convert to float
    population,
    ave_occup,
    latitude,
    longitude
]], columns=X.columns)

# Predict and display price
predicted_price = model.predict(input_data)[0]
st.success(f"**Predicted Median House Value**: ${predicted_price*100000:,.2f}")

# ---------------------------
# Model Performance Section
# ---------------------------
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
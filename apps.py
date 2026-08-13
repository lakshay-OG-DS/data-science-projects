import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load('house_model.pkl')

st.title(' Washington House Price Predictor')
st.write('Enter house details below to estimate market value:')

# Input fields
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
    sqft_living = st.number_input('Living Area (sqft)', min_value=300, max_value=10000, value=2000)
    sqft_lot = st.number_input('Lot Area (sqft)', min_value=500, max_value=100000, value=5000)
    floors = st.number_input('Floors', min_value=1, max_value=4, value=1)

with col2:
    waterfront = st.selectbox('Waterfront Property?', [0, 1])
    view = st.slider('View Quality (0-4)', 0, 4, 0)
    condition = st.slider('Condition Rating (1-5)', 1, 5, 3)
    sqft_basement = st.number_input('Basement Area (sqft)', min_value=0, max_value=5000, value=0)

# Extract cities from trained model feature names
model_features = getattr(model, "feature_names_in_", [])
city_columns = [col for col in model_features if col.startswith('city_')]
cities = [col.replace('city_', '') for col in city_columns]

selected_city = st.selectbox('City', sorted(cities) if cities else ['Seattle'])

if st.button('Predict House Price'):
    try:
        # Base numerical features
        data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_basement': sqft_basement
        }

        # Build dummy columns matching training set
        for city_col in city_columns:
            data[city_col] = 1 if city_col == f'city_{selected_city}' else 0

        input_df = pd.DataFrame([data])

        # Ensure exact column ordering as trained model
        if len(model_features) > 0:
            input_df = input_df[model_features]

        log_pred = model.predict(input_df)
        predicted_price = np.expm1(log_pred)[0]

        st.success(f'### Estimated House Price: ${predicted_price:,.2f}')
    except Exception as e:
        st.error(f"Prediction Error: {e}")

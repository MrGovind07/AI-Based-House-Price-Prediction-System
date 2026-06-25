import streamlit as st
import numpy as np
import pickle

# 1. Load your saved model and scaler
model = pickle.load(open("house_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 AI-Based House Price Prediction")
st.write("Enter the complete details of the house to estimate its price:")

# 2. Create input fields for all 18 features (Strictly maintaining the dataset order)
bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.25)
sqft_living = st.number_input("Sqft Living (Interior space in sqft)", min_value=100, value=1800, step=50)
sqft_lot = st.number_input("Sqft Lot (Land space in sqft)", min_value=100, value=5000, step=50)
floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)

waterfront = st.selectbox("Waterfront View?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
view = st.slider("View Quality (0 to 4)", min_value=0, max_value=4, value=0)
condition = st.slider("House Condition (1 to 5)", min_value=1, max_value=5, value=3)
grade = st.slider("Construction Grade (1 to 13)", min_value=1, max_value=13, value=7)

sqft_above = st.number_input("Sqft Above (Space above ground)", min_value=0, value=1500, step=50)
sqft_basement = st.number_input("Sqft Basement (Space below ground)", min_value=0, value=0, step=50)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000, step=1)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=0, step=1)

zipcode = st.number_input("Zipcode", min_value=10000, value=98103, step=1)
lat = st.number_input("Latitude (e.g., 47.51)", format="%.5f", value=47.5112)
long = st.number_input("Longitude (e.g., -122.25)", format="%.5f", value=-122.2570)

sqft_living15 = st.number_input("Sqft Living 15 (Nearest 15 neighbors living space)", min_value=100, value=1800, step=50)
sqft_lot15 = st.number_input("Sqft Lot 15 (Nearest 15 neighbors land space)", min_value=100, value=5000, step=50)


# 3. Predict Button
if st.button("Predict Price"):
    # Grouping all 18 features into a single array in the exact order
    input_data = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
        condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, 
        zipcode, lat, long, sqft_living15, sqft_lot15
    ]
    
    # Convert to numpy array and reshape to 2D
    features = np.array(input_data).reshape(1, -1)
    
    # Transform data using your loaded scaler (No more shape error!)
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    # Display the final price cleanly
    st.success(f"🎉 Estimated House Price: ${prediction[0]:,.2f}")
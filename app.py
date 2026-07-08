import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso

# ---------- Page Config ----------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.write("Car ki details daalo, predicted selling price milega.")

# ---------- Load Data & Train Model (cached so it doesn't retrain every time) ----------
@st.cache_data
def load_data():
    df = pd.read_csv('archive/car data.csv')
    return df

@st.cache_resource
def train_model(df):
    data = df.copy()
    data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    data.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    data.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

    X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
    Y = data['Selling_Price']

    model = Lasso()
    model.fit(X, Y)
    return model

car_dataset = load_data()
model = train_model(car_dataset)

# ---------- User Inputs ----------
st.subheader("Car Details Daalo")

year = st.number_input("Purchase Year", min_value=1990, max_value=2026, value=2015, step=1)
present_price = st.number_input("Present Showroom Price (in Lakhs)", min_value=0.0, value=5.0, step=0.1)
kms_driven = st.number_input("Kms Driven", min_value=0, value=30000, step=500)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# ---------- Encode Inputs ----------
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

current_year = 2026
car_age = current_year - year

input_data = pd.DataFrame([{
    'Year': year,
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'Fuel_Type': fuel_map[fuel_type],
    'Seller_Type': seller_map[seller_type],
    'Transmission': trans_map[transmission],
    'Owner': owner
}])

# ---------- Predict ----------
if st.button("Predict Selling Price"):
    prediction = model.predict(input_data)[0]
    if prediction < 0:
        st.warning("Model negative price predict kar raha hai — inputs check karo.")
    else:
        st.success(f"Predicted Selling Price: ₹ {prediction:.2f} Lakhs")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Lasso

# ---------- Page Config ----------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("""
<style>
.big-price {font-size: 42px; font-weight: 700; color: #16a34a;}
.metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("Car Price Prediction App")
st.write("Car ki details daalo — price live update hoga, koi button dabane ki zaroorat nahi.")

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
    return model, X.columns.tolist()

car_dataset = load_data()
model, feature_names = train_model(car_dataset)

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}
current_year = 2026

tab1, tab2, tab3 = st.tabs(["Predict", "Data Insights", "Model Info"])

# ================= TAB 1: PREDICTION =================
with tab1:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Car Details Daalo")

        year = st.slider("Purchase Year", min_value=1990, max_value=2026, value=2015, step=1)
        present_price = st.slider("Present Showroom Price (in Lakhs)", min_value=0.5, max_value=40.0, value=5.0, step=0.1)
        kms_driven = st.slider("Kms Driven", min_value=0, max_value=200000, value=30000, step=500)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

        car_age = current_year - year

        input_data = pd.DataFrame([{
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_map[fuel_type],
            'Seller_Type': seller_map[seller_type],
            'Transmission': trans_map[transmission],
            'Owner': owner
        }])[feature_names]

        prediction = model.predict(input_data)[0]
        prediction = max(prediction, 0)

    with right:
        st.subheader("Predicted Price")
        st.markdown(f'<p class="big-price">Rs {prediction:.2f} Lakhs</p>', unsafe_allow_html=True)

        depreciation = present_price - prediction
        pct = (depreciation / present_price * 100) if present_price > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Car Age", f"{car_age} yrs")
        c2.metric("Value Lost", f"Rs {depreciation:.2f} L", f"-{pct:.1f}%")
        c3.metric("Kms Driven", f"{kms_driven:,}")

        # Gauge-style indicator: predicted vs present price
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted Selling Price (Lakhs)"},
            gauge={
                'axis': {'range': [0, max(present_price, prediction) * 1.2]},
                'bar': {'color': "#16a34a"},
                'steps': [
                    {'range': [0, present_price * 0.5], 'color': "#fee2e2"},
                    {'range': [present_price * 0.5, present_price], 'color': "#fef9c3"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'value': present_price
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption("Black line = current showroom price. Bar = predicted resale price.")

        # Similar cars from dataset
        similar = car_dataset[
            (car_dataset['Fuel_Type'] == fuel_type) &
            (abs(car_dataset['Year'] - year) <= 2)
        ][['Car_Name', 'Year', 'Selling_Price', 'Kms_Driven']].head(5)

        if not similar.empty:
            st.subheader("Similar Cars in Dataset")
            st.dataframe(similar, use_container_width=True, hide_index=True)

# ================= TAB 2: DATA INSIGHTS =================
with tab2:
    st.subheader("Dataset par ek nazar")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(car_dataset, x="Selling_Price", nbins=30,
                             title="Selling Price Distribution", color_discrete_sequence=["#16a34a"])
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(car_dataset, x="Kms_Driven", y="Selling_Price", color="Fuel_Type",
                           title="Kms Driven vs Selling Price", trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(car_dataset, x="Fuel_Type", y="Selling_Price", color="Transmission",
                   title="Price Spread by Fuel Type and Transmission")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Raw Data (searchable)")
    search = st.text_input("Car name se search karo")
    filtered = car_dataset[car_dataset['Car_Name'].str.contains(search, case=False)] if search else car_dataset
    st.dataframe(filtered, use_container_width=True, hide_index=True)

# ================= TAB 3: MODEL INFO =================
with tab3:
    st.subheader("Model kaise decide karta hai price")

    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    fig4 = px.bar(coefs, x='Coefficient', y='Feature', orientation='h',
                   title="Feature Importance (Lasso Coefficients)",
                   color='Coefficient', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    **Model:** Lasso Regression
    **Training rows:** {} cars
    **Features used:** Year, Present Price, Kms Driven, Fuel Type, Seller Type, Transmission, Owner

    Jitna zyada bar right (green) side mein, utna positive asar price par.
    Jitna zyada bar left (red) side mein, utna negative asar (price kam karta hai).
    """.format(len(car_dataset)))

st.divider()
st.caption("Made with Streamlit | Lasso Regression | Data: CampusX Car Dekho dataset")
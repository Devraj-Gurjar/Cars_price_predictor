# Car Price Prediction App

An interactive web app that predicts the resale price of a used car using a Lasso Regression model, built with Streamlit and Plotly.

**Live app:** https://carspricepredictor-k5nbjf9pqscpee4ddptdae.streamlit.app

---

## Overview

This project takes the CarDekho used-car dataset and turns it into a fully interactive prediction tool. Instead of a static notebook, users can move sliders and instantly see a predicted resale price, compare it against the original showroom price, browse similar cars from the dataset, and inspect which features actually drive the model's decisions.

## Features

- **Live prediction** — price updates instantly as inputs change, no button needed
- **Gauge chart** — visual comparison of predicted resale price vs. original showroom price
- **Similar cars lookup** — shows comparable cars from the dataset (same fuel type, nearby year)
- **Data insights tab** — selling price distribution, kms driven vs. price trend (with regression line), price spread by fuel type and transmission
- **Model info tab** — feature importance chart based on Lasso regression coefficients
- **Searchable raw data table**

## Tech Stack

| Layer | Tool |
|---|---|
| UI / App framework | Streamlit |
| Data handling | Pandas, NumPy |
| Model | scikit-learn (Lasso Regression) |
| Visualization | Plotly |

## Dataset

Source: CarDekho used car dataset (via CampusX).
Features used: `Year`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`.
Target: `Selling_Price`.

## Project Structure

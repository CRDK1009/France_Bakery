import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import joblib

# Load the data
weather_data = pd.read_csv('weather_2021.01.01-2022.10.31.csv')
sales_data = pd.read_csv('bakery_sales_2021-2022.csv')

# Preprocess weather data
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data['day'] = weather_data['date'].dt.day
weather_data['month'] = weather_data['date'].dt.month
weather_data['year'] = weather_data['date'].dt.year

# Preprocess sales data
sales_data['date'] = pd.to_datetime(sales_data['date'])

# Load the trained model
loaded_rf_model = joblib.load('random_forest_model.joblib')

# Merge weather data with sales data
merged_data = pd.merge(weather_data, sales_data, on='date', how='inner')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: white;'>Bakery Productions</h1>", unsafe_allow_html=True)

# User inputs for date
st.markdown("<h3 style='color: #f57242;'>Select a date to get production suggestions:</h3>", unsafe_allow_html=True)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
year = st.number_input('Year', min_value=2021, value=2021)

# Predict tavg for the input date
predicted_tavg = loaded_rf_model.predict(np.array([[day, month, year]]))[0]
st.markdown(f"<h3 style='color: #f57242;'>Predicted Average Temperature for {day}/{month}/{year}: {predicted_tavg:.2f}°C</h3>", unsafe_allow_html=True)

# Filter historical data to include only the same month as input
filtered_data = merged_data[merged_data['month'] == month]

# Find the nearest historical tavg within the same month
nearest_tavg = filtered_data.iloc[(filtered_data['tavg'] - predicted_tavg).abs().argsort()[:1]]
nearest_date = nearest_tavg['date'].values[0]

# Get the articles sold on the nearest historical tavg date
articles_sold = merged_data[merged_data['date'] == nearest_date].groupby('Article').agg({'Quantity': 'sum'}).reset_index()

# Display the suggested quantities for each article
st.markdown(f"<h3 style='color: #f57242;'>Suggested Production Quantities for {day}/{month}/{year}:</h3>", unsafe_allow_html=True)

# Plot the articles and their quantities using Plotly
fig = px.bar(articles_sold, x='Article', y='Quantity', title='Suggested Production Quantities',
             labels={'Quantity': 'Quantity'}, template='plotly_white')
fig.update_layout(title_x=0.3, title_font=dict(size=20, color='white'), xaxis_tickangle=-45)

st.plotly_chart(fig)

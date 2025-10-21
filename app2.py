import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# App title
st.title("📈 Stock Trend Prediction")

# User input
user_input = st.text_input("Enter Stock Ticker", "AAPL")

# Date range
start = "2012-01-01"
end = "2022-12-31"

# Download data
df = yf.download(user_input, start=start, end=end)

# Check if data exists
if df.empty:
    st.error("No data found. Please check the stock ticker symbol.")
    st.stop()

# Show stats
st.subheader("Data from 2012 - 2022")
st.write(df.describe())

# Closing Price vs Time
st.subheader("Closing Price vs Time Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price")
ax.legend()
st.pyplot(fig)

# Closing Price with 100MA
st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df["Close"].rolling(100).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, ma100, label="100MA", color="orange")
ax.legend()
st.pyplot(fig)

# Closing Price with 100MA & 200MA
st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma200 = df["Close"].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, ma100, label="100MA", color="orange")
ax.plot(df.index, ma200, label="200MA", color="green")
ax.legend()
st.pyplot(fig)

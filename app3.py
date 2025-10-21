import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import load_model   # ✅ FIX: added this import
import os

# --- App Title ---
st.set_page_config(layout="wide")
st.title('📈 Stock Trend & Forecasting App')

# --- Date Range ---
start = '2012-01-01'
end = '2022-12-31'

# --- Stock Ticker Input ---
user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA, INFY):', 'AAPL')

try:
    # --- Fetch Stock Data ---
    df = yf.download(user_input, start=start, end=end)
    
    if df.empty:
        st.error("No data found. Please enter a valid ticker symbol.")
    else:
        st.subheader('Stock Data (2012 - 2022)')
        st.write(df.describe())

        # --- Closing Price Chart ---
        st.subheader('📊 Closing Price vs Time')
        fig1, ax1 = plt.subplots(figsize=(12,6))
        ax1.plot(df['Close'], label='Close Price')
        ax1.legend()
        st.pyplot(fig1)

        # --- Closing Price + 100MA ---
        st.subheader('📉 Closing Price with 100-Day MA')
        ma100 = df['Close'].rolling(100).mean()
        fig2, ax2 = plt.subplots(figsize=(12,6))
        ax2.plot(df['Close'], label='Close Price')
        ax2.plot(ma100, label='100MA')
        ax2.legend()
        st.pyplot(fig2)

        # --- Closing Price + 100MA + 200MA ---
        st.subheader('📉 Closing Price with 100-Day & 200-Day MA')
        ma200 = df['Close'].rolling(200).mean()
        fig3, ax3 = plt.subplots(figsize=(12,6))
        ax3.plot(df['Close'], label='Close Price')
        ax3.plot(ma100, label='100MA')
        ax3.plot(ma200, label='200MA')
        ax3.legend()
        st.pyplot(fig3)

        # --- Check if model and scaler exist ---
        if not os.path.exists("keras_model.h5") or not os.path.exists("scaler.save"):
            st.error("Model or Scaler file not found. Please make sure 'keras_model.h5' and 'scaler.save' are in the same folder.")
            st.stop()

        # --- Load LSTM Model and Scaler ---
        model = load_model('keras_model.h5')
        scaler = joblib.load('scaler.save')

        # --- Prepare Last 100 Days for Forecasting ---
        past_100 = df['Close'][-100:].values.reshape(-1,1)
        scaled_input = scaler.transform(past_100)
        input_seq = list(scaled_input.flatten())

        # --- Predict Next 30 Days ---
        n_future = 30
        predictions = []

        for i in range(n_future):
            x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0][0])
            input_seq.append(pred[0][0])

        # --- Inverse Transform the Predicted Values ---
        forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # --- Create Future Dates for Plotting ---
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_future)

        # --- Plot Forecast ---
        st.subheader('🔮 Forecasted Stock Price (Next 30 Days)')
        fig4, ax4 = plt.subplots(figsize=(12,6))
        ax4.plot(df.index, df['Close'], label='Historical')
        ax4.plot(future_dates, forecast, label='Forecast (30 Days)', color='red', linestyle='dashed')
        ax4.legend()
        st.pyplot(fig4)

        # --- Download Button ---
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': forecast.flatten()
        })

        csv = forecast_df.to_csv(index=False).encode()
        st.download_button("📥 Download Forecast as CSV", data=csv, file_name='future_forecast.csv', mime='text/csv')

except Exception as e:
    st.error(f"Error: {e}")

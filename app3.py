import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model   # ✅ FIXED IMPORT
import os

# --- Streamlit Config ---
st.set_page_config(layout="wide")
st.title('📈 Stock Trend & Forecasting App')

# --- Date Range ---
start = '2012-01-01'
end = '2022-12-31'

# --- Stock Ticker Input ---
user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA, INFY):', 'AAPL')

if not user_input.strip():
    st.warning("Please enter a valid stock ticker symbol.")
    st.stop()

try:
    # --- Fetch Stock Data ---
    with st.spinner("📡 Fetching stock data..."):
        df = yf.download(user_input, start=start, end=end)

    if df.empty:
        st.error("No data found. Please enter a valid ticker symbol.")
        st.stop()

    # --- Display Basic Info ---
    st.subheader('📊 Stock Data Summary (2012 - 2022)')
    st.write(df.describe())

    # --- Plot Closing Price ---
    st.subheader('💹 Closing Price vs Time')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Close'], label='Close Price', color='blue')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # --- Plot with 100-Day MA ---
    st.subheader('📈 Closing Price with 100-Day Moving Average')
    ma100 = df['Close'].rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df['Close'], label='Close Price', color='blue')
    ax2.plot(ma100, label='100-Day MA', color='orange')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- Plot with 100 & 200-Day MA ---
    st.subheader('📈 Closing Price with 100 & 200-Day Moving Averages')
    ma200 = df['Close'].rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df['Close'], label='Close Price', color='blue')
    ax3.plot(ma100, label='100-Day MA', color='orange')
    ax3.plot(ma200, label='200-Day MA', color='green')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # --- Model Check ---
    if not os.path.exists("keras_model.h5") or not os.path.exists("scaler.save"):
        st.warning("⚠️ Model or Scaler not found. Showing data analysis only.")
        st.stop()

    # --- Load Model and Scaler ---
    @st.cache_resource
    def load_resources():
        model = load_model('keras_model.h5')
        scaler = joblib.load('scaler.save')
        return model, scaler

    model, scaler = load_resources()

    # --- Prepare Last 100 Days ---
    past_100 = df['Close'][-100:].values.reshape(-1, 1)
    scaled_input = scaler.transform(past_100)
    input_seq = list(scaled_input.flatten())

    # --- Predict Next 30 Days ---
    n_future = 30
    predictions = []

    with st.spinner("🔮 Generating 30-day forecast..."):
        for i in range(n_future):
            x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0][0])
            input_seq.append(pred[0][0])

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # --- Create Future Dates ---
    last_date = df.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_future)

    # --- Plot Forecast ---
    st.subheader('📅 Forecasted Stock Price (Next 30 Business Days)')
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(df.index, df['Close'], label='Historical', color='blue')
    ax4.plot(future_dates, forecast, label='Forecast (Next 30 Days)', color='red', linestyle='dashed')
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    # --- Show Latest Prediction ---
    st.metric(label="Predicted Price for Next Trading Day", value=f"${forecast[0][0]:.2f}")

    # --- Download Button ---
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': forecast.flatten()})
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button("📥 Download Forecast as CSV", data=csv, file_name='future_forecast.csv', mime='text/csv')

except Exception as e:
    st.error(f"❌ Error: {str(e)}")

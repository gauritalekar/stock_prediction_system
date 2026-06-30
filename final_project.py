import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Prediction System", layout="wide")
st.title("📈 Intelligent Risk-Aware Stock Market Prediction System")
st.markdown("LSTM-based Stock Prediction with Buy / Sell / Hold Signals")

# ---------------- SIDEBAR ----------------
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# ---------------- RUN BUTTON ----------------
if st.sidebar.button("Run Prediction"):

    # -------- DATA DOWNLOAD --------
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ No data found.")
        st.stop()

    data = data.reset_index()

    st.subheader("📊 Stock Data")
    st.dataframe(data.tail())

    # -------- FEATURE ENGINEERING --------
    data['MA20'] = data['Close'].rolling(20).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + rs))

    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema12 - ema26

    data.dropna(inplace=True)

    # -------- SCALING --------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'MA20', 'RSI', 'MACD']])

    # -------- DATASET CREATION --------
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # -------- TRAIN / TEST SPLIT --------
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -------- MODEL --------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM Model..."):
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # -------- PREDICTION --------
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    st.metric("📉 Model RMSE", f"{rmse:.4f}")

    # -------- NEXT DAY PREDICTION --------
    last_60 = scaled_data[-60:].reshape(1, 60, 4)
    next_scaled = model.predict(last_60)

    dummy = np.zeros((1, 4))
    dummy[0][0] = next_scaled[0][0]
    next_price = scaler.inverse_transform(dummy)[0][0]

    st.metric("📅 Next Day Predicted Price", f"${next_price:.2f}")

    # -------- SIGNAL GENERATION (FIXED) --------
    signals = []
    confidence = []
    base_index = len(data) - len(predictions)

    for i in range(len(predictions)):
        score = 0

    if float(predictions[i][0]) > float(y_test[i]):
        score += 1

    if float(data['RSI'].iloc[base_index + i]) < 70:
        score += 1

    if float(data['Close'].iloc[base_index + i]) > float(data['MA20'].iloc[base_index + i]):
        score += 1

    if score >= 2:
        signals.append("BUY")
    elif score == 1:
        signals.append("HOLD")
    else:
        signals.append("SELL")

    confidence.append(score / 3)

    signal_df = data.iloc[base_index: base_index + len(signals)].copy()
    signal_df['Signal'] = signals
    signal_df['Confidence'] = confidence

    st.subheader("📌 Trading Signals")
    st.dataframe(signal_df[['Date', 'Close', 'Signal', 'Confidence']].tail(10))

    # -------- BACKTEST --------
    capital = 100000
    shares = 0

    for i in range(len(signal_df)):
        price = float(signal_df['Close'].iloc[i])

        if signal_df['Signal'].iat[i] == "BUY" and capital > 0:
            shares = capital / price
            capital = 0

        elif signal_df['Signal'].iat[i] == "SELL" and shares > 0:
            capital = shares * price
            shares = 0

    final_value = capital + shares * float(signal_df['Close'].iloc[-1])
    st.metric("💰 Final Portfolio Value", f"${final_value:,.2f}")

    # -------- VISUALIZATION --------
    st.subheader("📈 Actual vs Predicted Prices")

    fig, ax = plt.subplots()
    ax.plot(y_test, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.legend()

    st.pyplot(fig)

    st.success("✅ Model ran successfully!")
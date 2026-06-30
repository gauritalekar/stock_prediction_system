# 📈 Intelligent Risk-Aware Stock Market Prediction System

An AI-powered web application that predicts stock market prices using a **Long Short-Term Memory (LSTM)** deep learning model. Built with **Python** and **Streamlit**, the application analyzes historical stock data, calculates technical indicators, predicts the next day's closing price, and generates **Buy**, **Hold**, or **Sell** signals.

---

## 🚀 Features

* 📊 Fetches historical stock data using **Yahoo Finance**
* 🧠 Predicts stock prices using an **LSTM** neural network
* 📈 Calculates technical indicators:

  * 20-Day Moving Average (MA20)
  * Relative Strength Index (RSI)
  * Moving Average Convergence Divergence (MACD)
* 📅 Predicts the next day's stock closing price
* 💹 Generates **Buy / Hold / Sell** trading signals
* 🎯 Displays confidence scores for each signal
* 💰 Performs portfolio backtesting
* 📉 Evaluates model performance using **RMSE**
* 📊 Visualizes actual vs. predicted stock prices
* 🌐 Interactive user interface using **Streamlit**

---

## 🛠️ Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* yfinance

---

## 📂 Project Structure

```text
Stock-Prediction-System/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── assets/                 # Images or screenshots (optional)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Stock-Prediction-System.git
cd Stock-Prediction-System
```

### 2. Create a virtual environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install streamlit tensorflow yfinance pandas numpy matplotlib scikit-learn
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

If your file has a different name:

```bash
streamlit run stock_prediction.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📖 How It Works

1. Enter a stock symbol (e.g., AAPL, TSLA, MSFT).
2. Select the start and end dates.
3. Click **Run Prediction**.
4. The application:

   * Downloads historical stock data
   * Calculates technical indicators
   * Trains the LSTM model
   * Predicts stock prices
   * Generates Buy/Hold/Sell signals
   * Performs portfolio backtesting
   * Displays charts and performance metrics

---

## 📊 Output

The application displays:

* Historical stock data
* Next-day predicted stock price
* Model RMSE
* Trading signals (Buy/Hold/Sell)
* Confidence scores
* Portfolio value after backtesting
* Actual vs. Predicted price graph

---

## 🎯 Future Enhancements

* Multiple stock comparison
* Real-time price prediction
* News sentiment analysis
* Transformer-based prediction models
* Email and mobile notifications
* Advanced portfolio optimization
* Candlestick chart visualization

---

## 📸 Screenshots

Add screenshots of your application here.

Example:

```
assets/home.png
assets/prediction.png
assets/signals.png
```

---

## 👩‍💻 Author

**Gauri Talekar**

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**. Stock market predictions are based on historical data and machine learning models and **should not be considered financial or investment advice**.

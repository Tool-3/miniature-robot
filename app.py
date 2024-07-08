import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Helper functions

def scrape_index_data():
    url = 'https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    rows = soup.find_all('tr', class_='BdT')
    
    dates = []
    closes = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 5:
            date = cols[0].text
            close = cols[4].text
            try:
                date = datetime.datetime.strptime(date, '%b %d, %Y')
                close = float(close.replace(',', ''))
                dates.append(date)
                closes.append(close)
            except ValueError:
                continue

    data = pd.DataFrame({'Date': dates, 'Close': closes})
    data.set_index('Date', inplace=True)
    return data

def clean_data(data):
    data = data.dropna()
    return data

def technical_analysis(data):
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['Signal_Line'] = compute_macd(data['Close'])
    data['Signal'] = np.where(data['Close'] > data['SMA'], 1, 0)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    return macd, signal_line

def price_prediction(data):
    # Prepare data for LSTM
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    prediction_days = 60

    X_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Predict future prices
    test_data = scaled_data[-prediction_days:]
    X_test = [test_data]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    data['Predicted_Price'] = np.nan
    data.iloc[-1, data.columns.get_loc('Predicted_Price')] = predicted_price[0][0]
    
    return data

def sentiment_analysis(data):
    analyzer = SentimentIntensityAnalyzer()
    data['Sentiment'] = data['Close'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    return data

def combined_analysis(data):
    data['Combined_Signal'] = data['Signal'] + data['Sentiment']
    data['Combined_Signal'] = np.where(data['Combined_Signal'] > 1, 1, 0)
    return data

def generate_signals(data):
    data['Buy'] = np.where(data['Combined_Signal'] == 1, data['Close'], np.nan)
    data['Sell'] = np.where(data['Combined_Signal'] == 0, data['Close'], np.nan)
    data['Stop_Loss'] = data['Close'] * 0.95
    return data

# Streamlit app

def main():
    st.title("Index Data Analysis and Prediction")

    # Scrape data
    data = scrape_index_data()
    st.write("### Raw Data")
    st.dataframe(data.tail())

    # Clean data
    data = clean_data(data)
    st.write("### Cleaned Data")
    st.dataframe(data.tail())

    # Technical analysis
    data = technical_analysis(data)
    st.write("### Technical Analysis")
    st.dataframe(data.tail())

    # Price prediction
    data = price_prediction(data)
    st.write("### Price Prediction")
    st.dataframe(data.tail())

    # Sentiment analysis
    data = sentiment_analysis(data)
    st.write("### Sentiment Analysis")
    st.dataframe(data.tail())

    # Combined analysis
    data = combined_analysis(data)
    st.write("### Combined Analysis")
    st.dataframe(data.tail())

    # Generate signals
    data = generate_signals(data)
    st.write("### Buy/Sell Signals with Stop-Loss")
    st.dataframe(data.tail())

    # Plot data
    st.line_chart(data[['Close', 'SMA', 'Predicted_Price', 'Buy', 'Sell']])

if __name__ == "__main__":
    main()

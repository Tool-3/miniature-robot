import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import datetime

# Helper functions

def scrape_index_data():
    # Placeholder for actual scraping logic
    # For demonstration, we use Yahoo Finance to get data
    data = yf.download('^GSPC', period='1y')
    return data

def clean_data(data):
    # Placeholder for actual cleaning logic
    data = data.dropna()
    return data

def technical_analysis(data):
    # Simple moving average as an example of technical analysis
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['Signal'] = np.where(data['Close'] > data['SMA'], 1, 0)
    return data

def price_prediction(data):
    # Linear regression for price prediction as a placeholder
    model = LinearRegression()
    data['Prediction'] = data['Close'].shift(-30)
    data.dropna(inplace=True)  # Drop rows with NaN values after shift
    
    X = np.array(data.drop(columns=['Prediction']))
    y = np.array(data['Prediction'])
    
    model.fit(X, y)
    
    future = np.array(data.drop(columns=['Prediction']))[-30:]
    data['Predicted_Price'] = np.nan
    data.iloc[-30:, data.columns.get_loc('Predicted_Price')] = model.predict(future)
    
    return data

def sentiment_analysis(data):
    # Placeholder for sentiment analysis
    data['Sentiment'] = data['Close'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def combined_analysis(data):
    # Combine technical and sentiment analysis for predictions
    data['Combined_Signal'] = data['Signal'] + data['Sentiment']
    data['Combined_Signal'] = np.where(data['Combined_Signal'] > 1, 1, 0)
    return data

def generate_signals(data):
    # Generate buy/sell signals with stop-loss
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

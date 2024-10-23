import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(
    page_title="trAIde - Stock Price Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar menu
st.sidebar.header('Stock Prediction Parameters')
tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')

# Fetching ticker information
tickerData = yf.Ticker(tickerSymbol)
string_name = tickerData.info.get('longName', 'N/A')

st.subheader(f"Stock Price Prediction: {tickerSymbol} - {string_name}")

# Ticker data
st.header('Historical Stock Data')
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2021, 1, 31))
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
st.write(tickerDf)

# Check if 'Close' column exists and there are enough data points
if 'Close' in tickerDf.columns and len(tickerDf) > 1:
    # Stock Price Prediction using LSTM
    st.header('Stock Price Prediction using LSTM')

    # Prepare the data for prediction
    data = tickerDf['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=64)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Display evaluation results
    st.subheader('Model Evaluation')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

    # Plot actual vs predicted prices
    st.header('Actual vs Predicted Prices')
    prediction_df = pd.DataFrame({'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'Predicted': predictions.flatten()})
    st.write(prediction_df)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=np.arange(len(y_test)), y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), mode='lines', name='Actual'))
    fig_pred.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.flatten(), mode='lines', name='Predicted'))
    fig_pred.update_layout(title='Actual vs Predicted Prices')
    st.plotly_chart(fig_pred)

else:
    st.error("Failed to compute returns. Please check if the 'Close' column exists and there are enough data points.")

"""
Data processing functions for stock price prediction
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def load_stock_data(ticker_symbol, start_date, end_date):
    """
    Load stock data using yfinance
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period (str): Time period to fetch data for
        
    Returns:
        pd.DataFrame: Stock data
    """
    ticker = yf.Ticker(ticker_symbol)
    data = pd.DataFrame(ticker.history(start=start_date, end=end_date))
    
    # Drop unnecessary columns
    if 'Dividends' in data.columns:
        data.drop(columns=['Dividends'], inplace=True)
    if 'Stock Splits' in data.columns:
        data.drop(columns=['Stock Splits'], inplace=True)
    
    return data.reset_index(drop=True)

def load_stock_data_by_period(ticker_symbol, period):
    """
    Load stock data using yfinance
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period (str): Time period to fetch data for
        
    Returns:
        pd.DataFrame: Stock data
    """
    ticker = yf.Ticker(ticker_symbol)
    data = pd.DataFrame(ticker.history(period=period))
    
    # Drop unnecessary columns
    if 'Dividends' in data.columns:
        data.drop(columns=['Dividends'], inplace=True)
    if 'Stock Splits' in data.columns:
        data.drop(columns=['Stock Splits'], inplace=True)
    
    return data.reset_index(drop=True)

def normalize_data(data):
    """
    Normalize data using MinMaxScaler
    
    Args:
        data (pd.DataFrame): Stock data
        
    Returns:
        tuple: Normalized data and scaler object
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def create_dataset(data, time_step=1):
    """
    Create time series dataset with sliding window
    
    Args:
        data (np.ndarray): Normalized data
        time_step (int): Size of the sliding window
        
    Returns:
        tuple: Features X and target y
    """
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        window = data[i:(i + time_step)]
        dataX.append(window)
        dataY.append(data[i + time_step, 3])  # Close price
    return np.array(dataX), np.array(dataY)

def prepare_train_test_data(X, y, test_size):
    """
    Split data into training and testing sets
    
    Args:
        X (np.ndarray): Feature data
        y (np.ndarray): Target data
        test_size (int): Number of samples for testing
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    
    return X_train, X_test, y_train, y_test

def inverse_transform_labels(values_to_invert, scaler):
    values_to_invert_squeezed = tf.squeeze(values_to_invert)
    is_single_value = np.isscalar(values_to_invert)
    
    if is_single_value:
        value_reshaped = np.zeros((1, 5))
        value_reshaped[0, 3] = values_to_invert_squeezed
        inverted = scaler.inverse_transform(value_reshaped)[0, 3]
    else:
        values_to_invert_squeezed_reshaped = np.zeros((len(values_to_invert_squeezed), 5))
        values_to_invert_squeezed_reshaped[:, 3] = values_to_invert_squeezed
        inverted = scaler.inverse_transform(values_to_invert_squeezed_reshaped)[:, 3]
    
    return inverted
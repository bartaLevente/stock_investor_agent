"""
Model training script for stock price prediction

Usage:
    python train_model.py --ticker TICKER --period PERIOD --plot

Arguments:
    --ticker: Stock ticker symbol (default: NVDA)
    --period: Time period to fetch data (default: 3y)
    --plot: Flag to enable plotting training history
"""
import os
import argparse
import numpy as np
import tensorflow as tf

# Import modules
from utils import config
from utils.data_processing import (load_stock_data_by_period, normalize_data, create_dataset, 
                            prepare_train_test_data)
from utils.model import create_lstm_model, train_model
from utils.visualization import plot_stock_prices, plot_training_history
from utils.utils import save_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train stock price prediction model')
    parser.add_argument('--ticker', type=str, default=config.TICKER,
                        help=f'Stock ticker symbol (default: {config.TICKER})')
    parser.add_argument('--period', type=str, default=config.TIME_PERIOD,
                        help=f'Time period to fetch data (default: {config.TIME_PERIOD})')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training history')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print(f"Loading {args.ticker} stock data...")
    # Load data
    stock_data = load_stock_data_by_period(args.ticker, args.period)
    
    if args.plot:
        # Plot original data
        plot_stock_prices(stock_data, title=f"{args.ticker.upper()} Stock Prices", 
                         ticker=args.ticker.upper())
    
    print("Preprocessing data...")
    # Normalize data
    normalized_data, scaler = normalize_data(stock_data)
    
    # Create dataset with sliding window
    X, y = create_dataset(normalized_data, time_step=config.TIME_STEP)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        X, y, test_size=config.TRAIN_TEST_SPLIT
    )
    
    print("Creating and training LSTM model...")
    # Create LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(
        input_shape,
        lstm_units_1=config.LSTM_UNITS_1,
        lstm_units_2=config.LSTM_UNITS_2,
        dense_units_1=config.DENSE_UNITS_1,
        dense_units_2=config.DENSE_UNITS_2,
        dropout_rate_1=config.DROPOUT_RATE_1,
        dropout_rate_2=config.DROPOUT_RATE_2,
        dropout_rate_3=config.DROPOUT_RATE_3,
        dropout_rate_4=config.DROPOUT_RATE_4
    )
    
    # Display model summary
    model.summary()
    
    # Train model
    trained_model, history = train_model(
        model, X_train, y_train, X_test, y_test, 
        epochs=config.EPOCHS, patience=config.PATIENCE
    )
    
    if args.plot:
        # Plot training history
        plot_training_history(history)
    
    # Save model
    model_path = save_model(trained_model, args.ticker.upper() + '_model.keras')
    
    # Save scaler - essential for later predictions
    import pickle
    scaler_filename = os.path.basename(args.ticker.upper() + '_scaler.pkl')
    scaler_path = os.path.join(os.path.dirname(model_path), scaler_filename)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"\nTo make predictions, run: python predict.py --ticker {args.ticker.upper()}")

if __name__ == "__main__":
    main()
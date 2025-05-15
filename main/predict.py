"""
Prediction script for stock price prediction

Usage:
    python predict.py --ticker TICKER [--model_folder my_models] [--plot] [--save_results]

Arguments:
    --ticker: A ticker for the stock, i.e. AAPL, MSFT
    --model_folder: The path to a custom folder containing the model. Better to leave on deafult: models
    --plot: Flag to enable plotting predictions
    --save_results: Flag to save prediction results
"""
import os
import argparse
import tensorflow as tf
import pickle

# Import modules
from utils import config
from utils.data_processing import (load_stock_data_by_period, normalize_data, create_dataset, inverse_transform_labels)
from utils.visualization import plot_predictions 
from utils.utils import (calculate_metrics, save_results)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Make stock price predictions')
    parser.add_argument('--ticker', type=str, required=True,
                        help='A ticker for the stock, i.e. AAPL, MSFT')
    parser.add_argument('--model_folder', type=str, required=False, default="models",
                        help='The path to a custom folder containing the model. Better to leave on deafult: models')
    parser.add_argument('--plot', action='store_true',
                        help='Plot predictions')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    model_path = args.model_folder + "/" + args.ticker.upper() + "_model.keras"
    scaler_path = args.model_folder + "/" + args.ticker.upper() + "_scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Loading model from {model_path}...")
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading {args.ticker} stock data...")
    # Load the most recent data for the same ticker
    stock_data = load_stock_data_by_period(args.ticker, config.TIME_PERIOD)
    
    print("Preparing data for prediction...")
    # Normalize data
    normalized_data, _ = normalize_data(stock_data)
    
    # Create dataset with sliding window
    X, y = create_dataset(normalized_data, time_step=config.TIME_STEP)
    
    # Use the last portion for testing (same as training split)
    X_test = X[-config.TRAIN_TEST_SPLIT:]
    y_test = y[-config.TRAIN_TEST_SPLIT:]
    y_train = y[:-config.TRAIN_TEST_SPLIT]
    
    print("Making predictions on test data...")
    # Make predictions on test data
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    pred_inv, y_test_inv, y_train_inv = inverse_transform_labels(predictions, scaler), inverse_transform_labels(y_test, scaler),inverse_transform_labels(y_train, scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, pred_inv)
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if args.plot:
        # Plot test predictions
        plot_predictions(y_train_inv, y_test_inv, pred_inv, 
                        title=f"{args.ticker.upper()} Stock Price Prediction")
    
    # Save results if requested
    if args.save_results:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        results_path = save_results(metrics, pred_inv, y_test_inv, model_path)
        print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
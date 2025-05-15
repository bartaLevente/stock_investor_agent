"""
Utility functions for stock price prediction
"""
import numpy as np
import os
import datetime
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary with error metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape)
    }

def save_model(model, model_name, model_dir='models'):
    """
    Save model to disk
    
    Args:
        model (tf.keras.Model): Trained model
        model_dir (str): Directory to save model
        
    Returns:
        str: Path to saved model
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    
    return model_path

def save_results(metrics, predictions, actual, model_path, results_dir='results'):
    """
    Save results to disk
    
    Args:
        metrics (dict): Performance metrics
        predictions (np.ndarray): Model predictions
        actual (np.ndarray): Actual values
        model_path (str): Path to saved model
        results_dir (str): Directory to save results
        
    Returns:
        str: Path to saved results
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace('.keras', '')
    results_path = os.path.join(results_dir, f'results_{model_name}.json')
    
    results = {
        'metrics': metrics,
        'model_path': model_path,
        'predictions': predictions.tolist(),
        'actual': actual.tolist(),
        'timestamp': timestamp
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results_path
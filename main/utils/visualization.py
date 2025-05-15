"""
Visualization functions for stock price prediction
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_stock_prices(data, title="Stock Prices", ticker="Stock"):
    """
    Plot stock close prices
    
    Args:
        data (pd.DataFrame): Stock price data
        title (str): Plot title
        ticker (str): Stock ticker name
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label=ticker)
    plt.title(f'{title} Over the Last Year')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot model training history
    
    Args:
        history (tf.keras.callbacks.History): Model training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_train, y_true, y_pred, title="Stock Price Prediction", 
                    show_metrics=True, last_n_days=100):
    """
    Plot actual vs predicted stock prices
    
    Args:
        y_true (np.ndarray): Actual stock prices
        y_pred (np.ndarray): Predicted stock prices
        title (str): Plot title
        show_metrics (bool): Whether to display error metrics
    """
    plt.figure(figsize=(12, 6))
    
    last_n_train = y_train[-last_n_days:]

    plt.plot(range(-len(last_n_train), 0), last_n_train, label='Training Data', color='blue', linewidth=2)
    plt.plot(range(len(y_true)), y_true, label='True Price', color='green', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Price', color='red', linewidth=2)
    
    # Add error metrics to title if requested
    if show_metrics:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        plt.title(f'{title} (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
    else:
        plt.title(title)
    
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
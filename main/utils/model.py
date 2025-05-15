"""
Model definition and training for stock price prediction
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_lstm_model(input_shape, lstm_units_1, lstm_units_2,dense_units_1, dense_units_2, 
                      dropout_rate_1, dropout_rate_2, dropout_rate_3, dropout_rate_4):
    """
    Create LSTM model for stock price prediction
    
    Args:
        input_shape (tuple): Shape of input data (time_step, features)
        lstm_units_1 (int): Number of units in first LSTM layer
        lstm_units_2 (int): Number of units in second LSTM layer
        dense_units (int): Number of units in dense layer
        dropout_rate_1 (float): Dropout rate after first LSTM layer
        dropout_rate_2 (float): Dropout rate after second LSTM layer
        dropout_rate_3 (float): Dropout rate after dense layer
        
    Returns:
        tf.keras.Model: LSTM model
    """
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate_1),
        LSTM(lstm_units_2, return_sequences=False),
        Dropout(dropout_rate_2),
        Dense(dense_units_1),
        Dropout(dropout_rate_3),
        Dense(dense_units_2),
        Dropout(dropout_rate_4),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(), 
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, patience):
    """
    Train LSTM model with early stopping
    
    Args:
        model (tf.keras.Model): LSTM model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        
    Returns:
        tuple: Trained model and training history
    """
    early_stopping = EarlyStopping(
        monitor='mean_absolute_error', 
        patience=patience, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=epochs, 
        callbacks=[early_stopping],
        validation_data=(X_test, y_test)
    )
    
    return model, history

def predict(model, X_test):
    return model.predict(X_test)
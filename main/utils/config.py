
"""
Configuration settings for stock price prediction
"""

TICKER = 'NVDA'
TIME_PERIOD = "3y"

# Data processing settings
TIME_STEP = 30
TRAIN_TEST_SPLIT = 30

# Model settings
LSTM_UNITS_1 = 512
LSTM_UNITS_2 = 256
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.3
DROPOUT_RATE_3 = 0.2
DROPOUT_RATE_4 = 0.2
EPOCHS = 100
PATIENCE = 5
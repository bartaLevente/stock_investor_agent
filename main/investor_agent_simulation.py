
import pickle
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
import pandas as pd
from datetime import datetime, timedelta
from utils import config
from utils.data_processing import (
    load_stock_data, 
    normalize_data, 
    create_dataset, 
    prepare_train_test_data, 
    inverse_transform_labels
)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class InvestorAgent:
    def __init__(self, initial_cash, initial_period='2y', time_step=30):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {}
        self.portfolio_value = []
        self.transactions = []
        self.models = {}
        self.scalers = {}
        self.initial_period = initial_period
        self.time_step = time_step

    def load_model(self, ticker, model_folder = "models"):
        """
        Load a prediction model for a specific stock ticker.
        If model_path is None, it will use a default path pattern.
        """
        try:
            # For TensorFlow models
            model_path = f"{model_folder}/{ticker}_model.keras"
            self.models[ticker] = tf.keras.models.load_model(model_path)
            print(f"Model for {ticker} loaded successfully")

            scaler_path = f"{model_folder}/{ticker}_scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scalers[ticker] = pickle.load(f)
            print(f"Scaler for {ticker} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model for {ticker}: {e}")

    def get_prediction(self, ticker, day, X_test_data):
        """
        Get a prediction for a specific ticker on a specific day.
        """
        current_X_test = tf.expand_dims(X_test_data[ticker][day], axis=0)
        preds = tf.squeeze(self.models[ticker].predict(current_X_test, verbose=0)).numpy()
        return preds

    def make_decision(self, ticker, current_price, prediction, threshold=0.05):
        """
        Make a trading decision based on the prediction and current price.
        """
        if ticker not in self.portfolio:
            self.portfolio[ticker] = 0

        # The current_price should be a scalar value for the specific ticker
        actual_current_price = inverse_transform_labels(current_price, self.scalers[ticker])
        
        # The prediction should be the expected future price
        expected_price = inverse_transform_labels(prediction, self.scalers[ticker])

        if expected_price > actual_current_price * (1 + threshold):
            return "BUY"
        elif expected_price < actual_current_price * (1 - threshold) and self.portfolio[ticker] > 0:
            return "SELL"
        else:
            return "HOLD"
        
    def execute_transaction(self, ticker, action, current_price, day, max_rate=0.1):
        """
        Execute a transaction based on the decision.
        """
        # Convert normalized price back to actual price
        actual_price = inverse_transform_labels(current_price, self.scalers[ticker])
        
        if action == "BUY":
            max_cash_to_use = self.cash * max_rate
            shares_to_buy = int(max_cash_to_use / actual_price)

            if shares_to_buy > 0:
                cost = shares_to_buy * actual_price
                self.cash -= cost
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares_to_buy
                self.transactions.append({
                    'day': day,
                    'action': 'BUY',
                    'ticker': ticker,
                    'cost': cost,
                    'price': actual_price,
                    'shares': shares_to_buy 
                })
                print(f"Day {day+1}: Bought {shares_to_buy} shares of {ticker} at ${actual_price:.2f}")

        elif action == "SELL" and ticker in self.portfolio and self.portfolio[ticker] > 0:
            shares_to_sell = self.portfolio[ticker]
            income = shares_to_sell * actual_price
            self.cash += income
            self.portfolio[ticker] = 0
            self.transactions.append({
                'day': day,
                'action': 'SELL',
                'ticker': ticker,
                'income': income,
                'price': actual_price,
                'shares': shares_to_sell 
            })
            print(f"Day {day+1}: Sold {shares_to_sell} shares of {ticker} at ${actual_price:.2f}")
        elif action == "HOLD":
            print(f"Day {day+1}: HOLD {ticker}")

    def calculate_portfolio_value(self, day, current_prices):
        """
        Calculate the total value of the portfolio (cash + stocks).
        """
        stock_value = 0
        for ticker, shares in self.portfolio.items():
            if ticker in current_prices and shares > 0:
                actual_price = inverse_transform_labels(current_prices[ticker], self.scalers.get(ticker))
                stock_value += shares * actual_price
        
        total_value = self.cash + stock_value
        
        # Record portfolio value
        self.portfolio_value.append({
            'day': day,
            'value': total_value
        })
        
        return total_value

    def display_results(self):
        """
        Display the results of the simulation.
        """
        if not self.portfolio_value:
            print("No portfolio data to display.")
            return
        
        # Convert to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(self.portfolio_value)
        
        # Calculate returns
        initial_value = self.initial_cash
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        print("\n--- SIMULATION RESULTS ---")
        print(f"Starting value: ${initial_value:.2f}")
        print(f"Final value: ${final_value:.2f}")
        print(f"Total return: {total_return:.2f}%")
        
        # Current holdings
        print("\nFinal Portfolio:")
        for ticker, shares in self.portfolio.items():
            if shares > 0:
                print(f"{ticker}: {shares} shares")
        print(f"Cash: ${self.cash:.2f}")


def prepare_data(tickers, time_step, end_date):
    """
    Prepare data for simulation.
    
    Args:
        tickers: List of stock tickers
        time_step: Number of days to look back
        end_date: End date for the simulation
    
    Returns:
        Dictionary of prepared data for each ticker
    """
    price_data = {}
    X_test_data = {}
    start_date = end_date - timedelta(time_step * 10)
    
    for ticker in tickers:
        try:
            # Load historical stock data
            stock_data = load_stock_data(ticker, start_date, end_date)
            
            # Normalize the data
            normalized_data, _ = normalize_data(stock_data)

            # Create dataset with time steps
            X, y = create_dataset(normalized_data, time_step)

            # Prepare training and testing data
            _, X_test, _, prices = prepare_train_test_data(X, y, config.TRAIN_TEST_SPLIT)

            # Store the most recent time_step days of data
            price_data[ticker] = prices[-time_step:]
            X_test_data[ticker] = X_test[-time_step:]
            
        except Exception as e:
            print(f"Error preparing data for {ticker}: {e}")
    
    return price_data, X_test_data


def run_simulation(tickers, days=30, initial_cash=10000):
    """
    Run the trading simulation.
    
    Args:
        tickers: List of stock tickers to trade
        days: Number of days to simulate
        initial_cash: Initial cash amount
    
    Returns:
        InvestorAgent instance after simulation
    """
    agent = InvestorAgent(initial_cash)
    
    today = datetime.now().date()
    
    # Prepare historical data for each ticker
    price_data, X_test_data = prepare_data(tickers, days, today)
    
    # Load models for each ticker
    for ticker in tickers:
        agent.load_model(ticker)
    
    # Run simulation for specified number of days
    for day in range(0, days):
        daily_prices = {}
        
        # Get current prices for each ticker
        for ticker in tickers:
            if ticker in price_data and day < len(price_data[ticker]):
                daily_prices[ticker] = price_data[ticker][day]
        
        if not daily_prices:
            continue
        
        # Process each ticker
        for ticker, current_price in daily_prices.items():
            try:
                prediction = agent.get_prediction(ticker, day, X_test_data)
                if prediction is not None:
                    decision = agent.make_decision(ticker, current_price, prediction)
                    agent.execute_transaction(ticker, decision, current_price, day)

            except Exception as e:
                print(f"Error processing {ticker} on day {day}: {e}")
        
        # Calculate portfolio value at end of day
        agent.calculate_portfolio_value(day, daily_prices)
    
    # Display final results
    agent.display_results()
    
    return agent


if __name__ == "__main__":
    # Example usage
    tickers = ["NVDA","AAPL","GOOGL","MSFT","TSLA","AMZN"]
    agent = run_simulation(tickers, days=30, initial_cash=10000)
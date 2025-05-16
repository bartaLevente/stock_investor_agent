import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from investor_agent_simulation import run_simulation, InvestorAgent

st.set_page_config(layout="wide", page_title="Stock Investor Agent")

st.title("Stock Investor Agent")
st.write("Choose from the available tickers and run a simulation")

options = ["NVDA", "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
selected_tickers = []

st.write("### Choose some tickers:")

for ticker in options:
    if st.checkbox(ticker, key=ticker):
        selected_tickers.append(ticker)


col1, col2 = st.columns(2)
with col1:
    initial_cash = st.slider("Initial cash ($)", 1000, 50000, 10000, 1000)
with col2:
    days = st.slider("Days for simulation", 7, 60, 30, 1)

def display_results_in_streamlit(agent):
    """
    Display the simulation results in Streamlit.
    """
    if not agent.portfolio_value:
        st.warning("No portfolio data to display.")
        return
    
    # Convert to DataFrame for easier analysis
    portfolio_df = pd.DataFrame(agent.portfolio_value)
    
    # Calculate returns
    initial_value = agent.initial_cash
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    st.header("Simulation Results")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Starting Value", f"${initial_value:.2f}")
    with metrics_col2:
        st.metric("Final Value", f"${final_value:.2f}")
    with metrics_col3:
        st.metric("Total Return", f"{total_return:.2f}%", 
                  delta=f"{total_return:.2f}%", 
                  delta_color="normal")
    
    # Portfolio value chart
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portfolio_df['day'], portfolio_df['value'])
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True)
    st.pyplot(fig)
    
    # Current holdings
    st.subheader("Final Portfolio")
    
    holdings = []
    for ticker, shares in agent.portfolio.items():
        if shares > 0:
            holdings.append({"Ticker": ticker, "Shares": shares})
    
    if holdings:
        holdings_df = pd.DataFrame(holdings)
        st.dataframe(holdings_df)
    else:
        st.write("No stocks in final portfolio")
    
    st.metric("Cash on Hand", f"${agent.cash:.2f}")
    
    # Transactions
    if agent.transactions:
        st.subheader("Transaction History")
        transactions_df = pd.DataFrame(agent.transactions)
        st.dataframe(transactions_df)
    else:
        st.write("No transactions recorded")

if st.button("Run Simulation", type="primary"):
    if not selected_tickers:
        st.error("Please select at least one ticker to simulate")
    else:
        with st.spinner(f"Running simulation for {', '.join(selected_tickers)} over {days} days..."):
            try:                
                # Run the simulation
                agent = run_simulation(selected_tickers, days=days, initial_cash=initial_cash)

                # Display results
                display_results_in_streamlit(agent)
                
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
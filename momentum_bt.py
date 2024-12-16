import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt import dump
from datetime import timedelta

# Define the parameter space, including stop-loss and take-profit thresholds
space = [
    Integer(70, 80, name='rsi_overbought'),
    Integer(20, 30, name='rsi_oversold'),
    Real(0.01, 0.1, name='trailing_stop_pct'),
    Real(0.01, 0.1, name='stop_loss_pct'),  # New stop-loss parameter
    Real(0.05, 0.2, name='take_profit_pct')  # New take-profit parameter
]

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df['RSI'] = calculate_rsi(df['Close'], period=5)
    df['SMA'] = df['Close'].rolling(window=10).mean()
    df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
    df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
    return df

# Indicator calculation functions
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

# Enhanced signal generation function with additional risk management
def generate_signals(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    signals = pd.DataFrame(index=data.index)
    signals['Position'] = 0
    signals['Trailing_Stop'] = np.nan
    position_open = 0
    entry_price = None  # Track entry price
    buy_signals = []
    sell_signals = []
    trade_durations = []
    trade_start_time = None

    for i in range(1, len(data)):
        close_price = data['Close'].iloc[i]
        
        # Exit based on trailing stop, stop-loss, or take-profit
        if position_open == 1:  # Long position
            if close_price < signals['Trailing_Stop'].iloc[i - 1] or close_price < entry_price * (1 - stop_loss_pct) or close_price > entry_price * (1 + take_profit_pct):
                signals.loc[data.index[i], 'Position'] = 0
                sell_signals.append(data.index[i])
                position_open = 0
                if trade_start_time:
                    trade_durations.append(data.index[i] - trade_start_time)
                    trade_start_time = None
            else:
                signals.loc[data.index[i], 'Position'] = 1
                signals.loc[data.index[i], 'Trailing_Stop'] = max(
                    signals['Trailing_Stop'].iloc[i - 1],
                    close_price * (1 - trailing_stop_pct)
                )

        elif position_open == -1:  # Short position
            if close_price > signals['Trailing_Stop'].iloc[i - 1] or close_price > entry_price * (1 + stop_loss_pct) or close_price < entry_price * (1 - take_profit_pct):
                signals.loc[data.index[i], 'Position'] = 0
                buy_signals.append(data.index[i])
                position_open = 0
                if trade_start_time:
                    trade_durations.append(data.index[i] - trade_start_time)
                    trade_start_time = None
            else:
                signals.loc[data.index[i], 'Position'] = -1
                signals.loc[data.index[i], 'Trailing_Stop'] = min(
                    signals['Trailing_Stop'].iloc[i - 1],
                    close_price * (1 + trailing_stop_pct)
                )

        # Entry conditions
        else:
            if (data['RSI'].iloc[i] < rsi_oversold) and (close_price < data['BB_lower'].iloc[i]):
                signals.loc[data.index[i], 'Position'] = 1
                signals.loc[data.index[i], 'Trailing_Stop'] = close_price * (1 - trailing_stop_pct)
                entry_price = close_price
                buy_signals.append(data.index[i])
                position_open = 1
                trade_start_time = data.index[i]
                
            elif (data['RSI'].iloc[i] > rsi_overbought) and (close_price > data['BB_upper'].iloc[i]):
                signals.loc[data.index[i], 'Position'] = -1
                signals.loc[data.index[i], 'Trailing_Stop'] = close_price * (1 + trailing_stop_pct)
                entry_price = close_price
                sell_signals.append(data.index[i])
                position_open = -1
                trade_start_time = data.index[i]

    avg_trade_duration = sum(trade_durations, timedelta(0)) / len(trade_durations) if trade_durations else timedelta(0)

    return signals, buy_signals, sell_signals, avg_trade_duration

# Walk-forward analysis with updated objective to penalize drawdown
def walk_forward_analysis(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    signals, _, _, avg_trade_duration = generate_signals(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct)
    returns = data['Close'].pct_change()
    signals['Strategy_Returns'] = returns * signals['Position'].shift()
    cumulative_returns = (1 + signals['Strategy_Returns']).cumprod() - 1
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    sharpe_ratio = (signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std()) * np.sqrt(252) if signals['Strategy_Returns'].std() != 0 else 0
    adjusted_sharpe = sharpe_ratio - max_drawdown  # Penalize high drawdown
    
    win_rate = (signals['Strategy_Returns'] > 0).sum() / (signals['Strategy_Returns'] != 0).sum() if (signals['Strategy_Returns'] != 0).sum() != 0 else 0

    return -adjusted_sharpe, cumulative_returns.iloc[-1], max_drawdown, win_rate, avg_trade_duration

# Objective function for optimization
@use_named_args(space)
def objective(rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    data = load_data("MSFT_1hour_data.csv")
    adjusted_sharpe, _, _, _, _ = walk_forward_analysis(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct)
    return adjusted_sharpe

# Run Bayesian optimization
res = gp_minimize(objective, space, n_calls=40, random_state=0)

# Get best parameters and additional metrics
best_params = res.x
data = load_data("MSFT_1hour_data.csv")
_, cumulative_return, max_drawdown, win_rate, avg_trade_duration = walk_forward_analysis(
    data, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4]
)

# Display results
print("Best parameters found:", best_params)
print("Best Sharpe Ratio:", -res.fun)
print("Best Cumulative Return:", cumulative_return)
print("Smallest Maximum Drawdown during active positions:", max_drawdown)
print("Win Rate:", win_rate)
print("Average Trade Duration:", avg_trade_duration)

# Combined final visualization plot using Plotly
signals, buy_signals, sell_signals, _ = generate_signals(data, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4])

fig = go.Figure()

# Plot the closing price
fig.add_trace(go.Scatter(
    x=data.index, y=data['Close'],
    mode='lines', name='Close Price',
    line=dict(color='blue')
))

# Plot buy signals
fig.add_trace(go.Scatter(
    x=buy_signals, y=data.loc[buy_signals]['Close'],
    mode='markers', name='Buy Signal',
    marker=dict(color='green', symbol='triangle-up', size=10)
))

# Plot sell signals
fig.add_trace(go.Scatter(
    x=sell_signals, y=data.loc[sell_signals]['Close'],
    mode='markers', name='Sell Signal',
    marker=dict(color='red', symbol='triangle-down', size=10)
))

# Layout customization for the Plotly figure
fig.update_layout(
    title="Combined Walk-Forward Analysis with Buy/Sell Signals",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

# Show Plotly figure
fig.show()

# Convergence Plot
plt.figure(figsize=(8, 6))
plot_convergence(res)
plt.title("Convergence Plot")
plt.xlabel("Number of Calls")
plt.ylabel("Objective Value (Negative Sharpe Ratio)")
plt.show()

# Partial Dependence Plot for Parameter Effects
plot_objective(res)
plt.suptitle("Partial Dependence Plot")
plt.show()

# Parameter Evaluations Plot
plot_evaluations(res)
plt.suptitle("Parameter Evaluations")
plt.show()

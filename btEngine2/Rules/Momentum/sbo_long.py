import polars as pl
from btEngine2.Indicators import *
from typing import Optional, List
from numba import njit
import numpy as np


schema = {
    "Signal": pl.Int32,
    "TradeEntry": pl.Float64,
    "TradeExit": pl.Float64,
    "InTrade": pl.Int32,
    # Add other necessary columns here
}
def sbo_long(
    df: pl.DataFrame, 
    X: int, 
    N: int, 
    r: float, 
    atr_type: str = 'atr'
) -> pl.DataFrame:
    """
    Original Breakout Simple Long Strategy.

    :param df: Polars DataFrame containing asset data.
    :param X: Lookback period for rolling max and ATR.
    :param N: Holding period after entry.
    :param r: Multiplier for ATR to determine entry threshold.
    :param atr_type: Type of ATR calculation ('atr' or 'sd').
    :return: Polars DataFrame with trading signals appended to the original DataFrame.
    """
    # Step 1: Compute Effective ATR using the helper function
    df = df.with_columns([pl.col('Close').rolling_max(window_size=X).alias('RollingMax')])
    df = compute_effective_atr(df, X, atr_type)
    
    # Step 2: Calculate the Buy Threshold
    df = df.with_columns([
        (pl.col('RollingMax') + r * pl.col('ATR')).shift(1).alias('Buy_Threshold')
    ])
    
    # Step 3: Identify Buy Signals
    df = df.with_columns([
        (pl.col('Close') > pl.col('Buy_Threshold')).alias('Buy_Signal')
    ])
    
    # Step 4: Initialize trading signal columns
    df = df.with_columns([
        pl.lit(0).alias('Signal'),
        pl.lit(np.nan).alias('TradeEntry'),  # Use NaN for float columns
        pl.lit(np.nan).alias('TradeExit'),
        pl.lit(0).alias('InTrade')
    ])
    
    # Step 5: Convert DataFrame to list of dictionaries for iteration
    df_list = df.to_dicts()
    
    position_open = False
    entry_index = None

    for i in range(X, len(df_list) - 1):
        if not position_open:
            # Check for Buy Signal
            if df_list[i]['Buy_Threshold'] is not None and df_list[i]['Close'] > df_list[i]['Buy_Threshold']:
                try:
                    df_list[i + 1]['Signal'] = 1  # Buy signal
                    df_list[i + 1]['TradeEntry'] = df_list[i + 1]['Close']  # Record the entry price
                    position_open = True
                    entry_index = i + 1
                    df_list[i + 1]['InTrade'] = 1  # Mark as in trade
                except:
                    continue
        elif position_open:
            # Continue holding the position
            df_list[i + 1]['InTrade'] = 1  # Mark as in trade
            
            # Check for Sell Signal after holding period N
            try:
                if i + 1 == entry_index + N:
                    df_list[i + 1]['Signal'] = -1  # Sell signal
                    df_list[i + 1]['TradeExit'] = df_list[i + 1]['Close']  # Record the exit price
                    position_open = False  # Close the position
            except:
                continue

    # Step 6: Replace None with np.nan for consistency
    for row in df_list:
        row['TradeEntry'] = row['TradeEntry'] if row['TradeEntry'] is not None else np.nan
        row['TradeExit'] = row['TradeExit'] if row['TradeExit'] is not None else np.nan

    # Step 7: Convert updated data back to Polars DataFrame and append new columns
    df_updated = pl.from_dicts(df_list, schema=schema)

    # Step 8: Append the new columns (Signal, TradeEntry, TradeExit, InTrade) to the original DataFrame
    df = df.with_columns([
        df_updated['Signal'],
        df_updated['TradeEntry'],
        df_updated['TradeExit'],
        df_updated['InTrade']
    ])

    return df

def sbo_long_pb(
    df: pl.DataFrame,
    X: int,
    N: int,
    r: float,
    lmt_days: int = 1,
    lmt_atr_ratio: float = 0.5,
    lmt_epsilon: float = 0.15,
    atr_type: str = 'atr'
) -> pl.DataFrame:
    """
    Breakout Pullback Long Strategy.

    :param df: Polars DataFrame containing asset data.
    :param X: Lookback period for rolling max and ATR.
    :param N: Holding period after entry.
    :param r: Multiplier for ATR to determine entry threshold.
    :param lmt_days: Number of days to keep the limit order active after breakout.
    :param lmt_atr_ratio: Multiplier for ATR to calculate limit price.
    :param atr_type: Type of ATR calculation ('atr' or 'sd').
    :return: Polars DataFrame with trading signals appended to the original DataFrame.
    """
    # Step 1: Compute Effective ATR and Rolling Max
    df = df.with_columns([
        pl.col('Close').rolling_max(window_size=X).alias('RollingMax')
    ])
    df = compute_effective_atr(df, X, atr_type)
    
    # Step 2: Calculate the Buy Threshold
    df = df.with_columns([
        (pl.col('RollingMax') + r * pl.col('ATR')).shift(1).alias('Buy_Threshold')
    ])
    
    # Step 3: Identify Buy Signals
    df = df.with_columns([
        (pl.col('Close') > pl.col('Buy_Threshold')).alias('Buy_Signal')
    ])
    
    # Step 4: Initialize trading signal columns
    df = df.with_columns([
        pl.lit(0).alias('Signal'),
        pl.lit(np.nan).alias('TradeEntry'),  # Use NaN for float columns
        pl.lit(np.nan).alias('TradeExit'),
        pl.lit(0).alias('InTrade'),
        pl.lit(0).alias('Holding_Period')  # New column to track holding period
    ])
    
    # Step 5: Convert DataFrame to list of dictionaries for iteration
    df_list = df.to_dicts()
    
    position_open = False
    entry_index = None

    # Ensure all keys are present in each dictionary and have consistent types
    default_row = {key: np.nan if isinstance(value, float) else 0 for key, value in df_list[0].items()}
    for row in df_list:
        for key in default_row:
            if key not in row or row[key] is None:
                row[key] = default_row[key]

    # Step 6: Iterate over the DataFrame
    for i in range(X, len(df_list)):
        if not position_open:
            # Check for Buy Signal
            if df_list[i]['Buy_Signal']:
                # Calculate limit price
                breakout_day_close = df_list[i]['Close']
                breakout_day_atr = df_list[i]['ATR']
                lmt_price = breakout_day_close - lmt_atr_ratio * breakout_day_atr
                lmt_epsilon = lmt_epsilon * breakout_day_atr  # lmt_epsilon is 0.2 * ATR
                
                # For the next lmt_days, check if limit order is filled
                order_filled = False
                for day_offset in range(1, lmt_days + 1):
                    next_day_index = i + day_offset
                    if next_day_index >= len(df_list):
                        break  # Reached end of data
                    
                    day_data = df_list[next_day_index]
                    
                    day_low = day_data['Low']
                    day_high = day_data['High']
                    
                    # Adjusted low and high with lmt_epsilon
                    adj_low = day_low + lmt_epsilon
                    adj_high = day_high - lmt_epsilon
                    
                    # Check if lmt_price is within the day's range
                    if adj_low <= lmt_price <= adj_high:
                        # Limit order is filled
                        df_list[next_day_index]['Signal'] = 1  # Buy signal
                        df_list[next_day_index]['TradeEntry'] = lmt_price  # Record the entry price
                        df_list[next_day_index]['InTrade'] = 0  # Mark as in trade
                        df_list[next_day_index]['Holding_Period'] = 0  # Start holding period
                        position_open = True
                        entry_index = next_day_index
                        order_filled = True
                        break  # Exit loop after order is filled
                
                if not order_filled:
                    # Limit order was not filled in the specified lmt_days
                    pass  # Do nothing, wait for next buy signal
        else:
            # We are in a position
            df_list[i]['InTrade'] = 1
            current_holding_period = df_list[i - 1]['Holding_Period'] + 1
            df_list[i]['Holding_Period'] = current_holding_period
            
            if current_holding_period >= N:
                # Exit position
                df_list[i]['Signal'] = -1  # Sell signal
                df_list[i]['TradeExit'] = df_list[i]['Close']  # Exit at current close price
                df_list[i]['InTrade'] = 1
                df_list[i]['Holding_Period'] = 0
                position_open = False
                entry_index = None
    
    # Step 7: Replace None with np.nan for consistency and ensure consistent data types
    for row in df_list:
        for key in row:
            if row[key] is None:
                if isinstance(default_row[key], float):
                    row[key] = np.nan
                elif isinstance(default_row[key], int):
                    row[key] = 0
                else:
                    row[key] = default_row[key]
    
    # Step 8: Define the schema explicitly
    schema = {
        'Date': pl.Date,  # Adjust the type based on your 'Date' column
        'Open': pl.Float64,
        'High': pl.Float64,
        'Low': pl.Float64,
        'Close': pl.Float64,
        'Volume': pl.Int64,  # Adjust if necessary
        'RollingMax': pl.Float64,
        'ATR': pl.Float64,
        'Buy_Threshold': pl.Float64,
        'Buy_Signal': pl.Boolean,
        'Signal': pl.Int64,
        'TradeEntry': pl.Float64,
        'TradeExit': pl.Float64,
        'InTrade': pl.Int64,
        'Holding_Period': pl.Int64
    }
    
    # Step 9: Convert updated data back to Polars DataFrame with explicit schema
    df_updated = pl.from_dicts(df_list, schema=schema)
    
    # Step 10: Append the new columns to the original DataFrame
    df = df.with_columns([
        df_updated['Signal'],
        df_updated['TradeEntry'],
        df_updated['TradeExit'],
        df_updated['InTrade'],
        df_updated['Holding_Period']
    ])
    
    return df


import pandas as pd
import numpy as np

def bo_rsi_long(
    df: pd.DataFrame,
    rsi_param: tuple,
    N: int,
    trend_filter: tuple = None,
    exit_trend_rule: tuple = None
) -> pd.DataFrame:
    """
    Optimized RSI-based breakout strategy with optional trend filters and exit rules.
    
    Parameters:
    - df: DataFrame containing 'Date', 'Open', 'High', 'Close' columns.
    - rsi_param: Tuple (rsi_period, rsi_threshold).
    - N: Number of days to hold the trade.
    - trend_filter: Tuple (moving_avg_period, moving_avg_type) or None.
    - exit_trend_rule: Tuple (moving_avg_period, moving_avg_type) or None.
    
    Returns:
    - df: DataFrame with 'Signal', 'TradeEntry', 'TradeExit', 'InTrade' columns added.
    """

    # Convert DataFrame to Pandas if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialize columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0

    # Convert DataFrame columns to numpy arrays for faster access
    close = df['Close'].values
    open_prices = df['Open'].values
    high = df['High'].values

    # Calculate RSI
    rsi_period, rsi_threshold = rsi_param
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    rsi = df['RSI'].values

    # Apply trend filter if provided
    if trend_filter is not None:
        tf_period, tf_type = trend_filter
        if tf_type.lower() == 'ema':
            trend_ma = df['Close'].ewm(span=tf_period, adjust=False).mean().values
        elif tf_type.lower() == 'sma':
            trend_ma = df['Close'].rolling(window=tf_period).mean().values
        else:
            raise ValueError("Invalid moving average type for trend_filter. Use 'ema' or 'sma'.")
        trend_filter_mask = close > trend_ma
    else:
        trend_filter_mask = np.ones(len(df), dtype=bool)  # Always True if no trend filter

    # Apply exit trend rule if provided
    if exit_trend_rule is not None:
        etr_period, etr_type = exit_trend_rule
        if etr_type.lower() == 'ema':
            exit_ma = df['Close'].ewm(span=etr_period, adjust=False).mean().values
        elif etr_type.lower() == 'sma':
            exit_ma = df['Close'].rolling(window=etr_period).mean().values
        else:
            raise ValueError("Invalid moving average type for exit_trend_rule. Use 'ema' or 'sma'.")
    else:
        exit_ma = np.full(len(df), np.nan)  # Not used if no exit trend rule

    # Identify potential entry points
    entry_conditions = (rsi > rsi_threshold) & trend_filter_mask
    entry_indices = np.where(entry_conditions)[0]

    # Remove entries on the last day since we cannot enter on the next open
    entry_indices = entry_indices[entry_indices < len(df) - 1]

    # Initialize variables to store trade information
    in_trade = np.zeros(len(df), dtype=bool)
    signals = np.zeros(len(df))
    trade_entries = np.full(len(df), np.nan)
    trade_exits = np.full(len(df), np.nan)

    i = 0
    while i < len(entry_indices):
        entry_idx = entry_indices[i] + 1  # Enter at next open
        if in_trade[entry_idx]:
            i += 1
            continue  # Already in trade, skip
        signals[entry_idx] = 1  # Entry signal
        trade_entries[entry_idx] = open_prices[entry_idx]
        trade_start = entry_idx

        # Determine exit conditions
        exit_idx = trade_start
        exit_found = False
        while exit_idx < len(df) - 1:
            exit_idx += 1
            days_in_trade = exit_idx - trade_start
            exit_by_N_days = days_in_trade >= N
            exit_by_high = close[exit_idx] > high[exit_idx - 1]

            if exit_trend_rule is not None and not np.isnan(exit_ma[trade_start]):
                exit_trend_in_effect = close[trade_start] >= exit_ma[trade_start]
            else:
                exit_trend_in_effect = False

            if exit_trend_in_effect:
                # Exit when price closes below the exit trend MA
                if close[exit_idx] < exit_ma[exit_idx]:
                    exit_found = True
                    break
            else:
                # Normal exit rules
                if exit_by_N_days or exit_by_high:
                    exit_found = True
                    break

        if not exit_found:
            exit_idx = len(df) - 1  # Exit at last available data

        signals[exit_idx] = -1  # Exit signal
        trade_exits[exit_idx] = close[exit_idx]

        # Mark in_trade period
        in_trade[trade_start:exit_idx + 1] = True

        # Move to next potential entry after exit
        i += 1
        # Skip entries that occur during the current trade
        while i < len(entry_indices) and entry_indices[i] <= exit_idx:
            i += 1

    # Assign computed values back to DataFrame
    df['Signal'] = signals
    df['TradeEntry'] = trade_entries
    df['TradeExit'] = trade_exits
    df['InTrade'] = in_trade.astype(int)

    df = pl.DataFrame(df)

    return df

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
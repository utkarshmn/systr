import polars as pl
from btEngine2.Indicators import *
from typing import Optional, List
from numba import njit
import numpy as np
import pandas as pd

from scipy.stats import skew


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



def bo_rsi_long(
    df: pd.DataFrame,
    rsi_param: tuple,
    N: int,
    N_min: int = 1,
    trend_filter: tuple = None,
    exit_trend_rule: tuple = None,
    exit_quick_rule: bool = False
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
            if exit_quick_rule and days_in_trade > N_min:
                exit_by_high = close[exit_idx] > high[exit_idx - 1]
            else:
                exit_by_high = False

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
    df['rsi'] = rsi
    try:
        df['ema'] = trend_ma
    except:
        pass
    df['cond'] = entry_conditions.astype(int)
    df['Signal'] = signals
    df['TradeEntry'] = trade_entries
    df['TradeExit'] = trade_exits
    df['InTrade'] = in_trade.astype(int)

    df = pl.DataFrame(df)

    return df


import pandas as pd
import numpy as np

def bo_rsi_long_pb(
    df: pd.DataFrame,
    rsi_param: tuple,
    N: int,
    N_min: int = 1,
    lmt_days: int = 1,
    lmt_atr_ratio: float = 0.5,
    lmt_epsilon: float = 0.15,
    atr_period: int = 14,
    trend_filter: tuple = None,
    exit_quick_rule: bool = False,
    exit_trend_rule: tuple = None
) -> pd.DataFrame:
    """
    RSI-based breakout strategy with pullback entry and optional trend filters and exit rules.
    
    Parameters:
    - df: DataFrame containing 'Date', 'Open', 'High', 'Low', 'Close' columns.
    - rsi_param: Tuple (rsi_period, rsi_threshold).
    - N: Number of days to hold the trade after entry.
    - lmt_days: Number of days to keep the limit order active after breakout.
    - lmt_atr_ratio: Multiplier for ATR to calculate limit price.
    - lmt_epsilon: Small adjustment to avoid false triggers (in ATR units).
    - atr_period: Period for calculating ATR.
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
    low = df['Low'].values

    # Calculate RSI
    rsi_period, rsi_threshold = rsi_param
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    rsi = df['RSI'].values

    # Calculate ATR
    df['ATR'] = compute_atr(df, atr_period)
    atr = df['ATR'].values

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

    # Initialize variables to store trade information
    in_trade = np.zeros(len(df), dtype=bool)
    signals = np.zeros(len(df))
    trade_entries = np.full(len(df), np.nan)
    trade_exits = np.full(len(df), np.nan)

    i = 0
    while i < len(entry_indices):
        signal_idx = entry_indices[i]
        if in_trade[signal_idx]:
            i += 1
            continue  # Already in trade, skip

        # Calculate limit price
        breakout_day_close = close[signal_idx]
        breakout_day_atr = atr[signal_idx]
        lmt_price = breakout_day_close - lmt_atr_ratio * breakout_day_atr
        lmt_epsilon_adjusted = lmt_epsilon * breakout_day_atr

        # For the next lmt_days, check if limit order is filled
        order_filled = False
        for day_offset in range(1, lmt_days + 1):
            next_day_idx = signal_idx + day_offset
            if next_day_idx >= len(df):
                break  # Reached end of data

            if in_trade[next_day_idx]:
                continue  # Already in trade

            day_low = low[next_day_idx]
            day_high = high[next_day_idx]

            # Adjusted low and high with lmt_epsilon
            adj_low = day_low + lmt_epsilon_adjusted
            adj_high = day_high - lmt_epsilon_adjusted

            # Check if lmt_price is within the day's range
            if adj_low <= lmt_price <= adj_high:
                # Limit order is filled
                signals[next_day_idx] = 1  # Entry signal
                trade_entries[next_day_idx] = lmt_price  # Entry price
                trade_start = next_day_idx
                in_trade[trade_start] = True

                # Determine exit conditions
                exit_idx = trade_start
                exit_found = False
                while exit_idx < len(df) - 1:
                    exit_idx += 1
                    days_in_trade = exit_idx - trade_start
                    exit_by_N_days = days_in_trade >= N

                    if exit_quick_rule and days_in_trade > N_min:
                        exit_by_high = close[exit_idx] > high[exit_idx - 1]
                    else:
                        exit_by_high = False

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

                    in_trade[exit_idx] = True  # Mark as in trade

                if not exit_found:
                    exit_idx = len(df) - 1  # Exit at last available data

                signals[exit_idx] = -1  # Exit signal
                trade_exits[exit_idx] = close[exit_idx]
                in_trade[exit_idx] = True

                # Mark in_trade period
                in_trade[trade_start:exit_idx + 1] = True

                # Move to next potential entry after exit
                i += 1
                # Skip entries that occur during the current trade
                while i < len(entry_indices) and entry_indices[i] <= exit_idx:
                    i += 1

                order_filled = True
                break  # Exit the limit order loop after order is filled

        if not order_filled:
            # Limit order was not filled in the specified lmt_days
            i += 1  # Move to the next entry signal

    # Assign computed values back to DataFrame
    df['rsi'] = rsi
    try:
        df['ema'] = trend_ma
    except:
        pass
    df['cond'] = entry_conditions.astype(int)
    df['Signal'] = signals
    df['TradeEntry'] = trade_entries
    df['TradeExit'] = trade_exits
    df['InTrade'] = in_trade.astype(int)

    # If you need to convert back to Polars DataFrame
    df = pl.DataFrame(df)

    return df


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def bo_rsi_long_ml(
    df: pd.DataFrame,
    rsi_param: tuple,
    N: int,
    min_dps: int,
    max_dps: int,
    N_min: int = 1,
    probability_threshold: float = 0.6,
    trend_filter: tuple = None,
    exit_quick_rule: bool = False,
    exit_trend_rule: tuple = None,
    n_estimators: int = 100,
    max_depth: int = 2
) -> pd.DataFrame:
    """
    RSI-based breakout strategy with machine learning metalabeling.
    """

    # Ensure df is a Pandas DataFrame
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

    # No conversion to NumPy arrays here
    # Use Pandas Series directly
    close = df['Close']
    open_prices = df['Open']
    high = df['High']
    low = df['Low']

    # Calculate RSI
    rsi_period, rsi_threshold = rsi_param
    df['RSI'] = compute_rsi(close, rsi_period)

    # Apply trend filter if provided
    if trend_filter is not None:
        tf_period, tf_type = trend_filter
        if tf_type.lower() == 'ema':
            trend_ma = close.ewm(span=tf_period, adjust=False).mean()
        elif tf_type.lower() == 'sma':
            trend_ma = close.rolling(window=tf_period).mean()
        else:
            raise ValueError("Invalid moving average type for trend_filter. Use 'ema' or 'sma'.")
        trend_filter_mask = close > trend_ma
    else:
        trend_filter_mask = pd.Series(True, index=close.index)

    # Apply exit trend rule if provided
    if exit_trend_rule is not None:
        etr_period, etr_type = exit_trend_rule
        if etr_type.lower() == 'ema':
            exit_ma = close.ewm(span=etr_period, adjust=False).mean()
        elif etr_type.lower() == 'sma':
            exit_ma = close.rolling(window=etr_period).mean()
        else:
            raise ValueError("Invalid moving average type for exit_trend_rule. Use 'ema' or 'sma'.")
    else:
        exit_ma = pd.Series(np.nan, index=close.index)

    # Identify potential entry points (setup days)
    rsi = df['RSI']
    entry_conditions = (rsi > rsi_threshold) & trend_filter_mask
    entry_indices = entry_conditions[entry_conditions].index

    # Remove entries on the last day since we cannot enter on the next open
    entry_indices = entry_indices[entry_indices < len(df) - 1]

    # Initialize variables to store trade information
    in_trade = pd.Series(False, index=df.index)
    signals = pd.Series(0, index=df.index)
    trade_entries = pd.Series(np.nan, index=df.index)
    trade_exits = pd.Series(np.nan, index=df.index)

    # Initialize ML feature columns
    ml_feature_names = [
        'mlfeat_volatility',
        'mlfeat_volatility_zscore',
        'mlfeat_rsi_long',
        'mlfeat_rsi_chg3',
        'mlfeat_ewma_diff_short',
        'mlfeat_ewma_diff_long',
        'mlfeat_volume_zscore',
        'mlfeat_adx',
        'mlfeat_carry_change',
        'mlfeat_carry_ema_diff',
        'mlfeat_skew'
    ]
    for feature in ml_feature_names:
        df[feature] = np.nan

    # Initialize ML result columns
    df['mlfeat_result'] = np.nan
    df['mlfeat_probability'] = np.nan

    # Precompute indicators needed for features
    df['ATR'] = compute_atr(df, period=14)
    df['ADX'] = compute_adx(df, period=14)

    # Precompute EWMAs
    df['EWMA_2'] = close.ewm(span=2, adjust=False).mean()
    df['EWMA_8'] = close.ewm(span=8, adjust=False).mean()
    df['EWMA_16'] = close.ewm(span=16, adjust=False).mean()
    df['EWMA_64'] = close.ewm(span=64, adjust=False).mean()

    # Precompute Carry EWMAs if 'Carry' exists
    if 'Carry' in df.columns:
        df['Carry_EMA_16'] = df['Carry'].ewm(span=16, adjust=False).mean()
        df['Carry_EMA_64'] = df['Carry'].ewm(span=64, adjust=False).mean()
        df['Carry_Diff'] = df['Carry'].diff()

    # Precompute Volatility
    vol_window = rsi_param[0] * 2
    df['Pct_Return'] = close.pct_change()
    df['Volatility'] = df['Pct_Return'].rolling(window=vol_window).std() * np.sqrt(252)

    # Precompute Z-scores
    df['Volatility_Zscore'] = (df['Volatility'] - df['Volatility'].rolling(window=vol_window).mean()) / df['Volatility'].rolling(window=vol_window).std()

    # Precompute Skewness
    skew_window = 126
    df['Skew'] = -1 * df['Pct_Return'].rolling(window=skew_window).apply(lambda x: skew(x, bias=False), raw=True)
    df['Skew'] = df['Skew'].ewm(span=max(int(skew_window / 4), 1), adjust=False).mean()

    if 'Volume' in df.columns:
        volume_window = 3 * rsi_param[0]
        df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(window=volume_window).mean()) / df['Volume'].rolling(window=volume_window).std()


    # add func to do fading trades
    trade_sign = np.sign(probability_threshold) 

    # Start the main loop
    past_signals = []

    for idx in entry_indices:
        if in_trade.iloc[idx]:
            continue  # Already in trade, skip

        # Collect features on the setup day
        features = {}

        # 1. Last rsi_param[0]*2 % vol annualized
        features['mlfeat_volatility'] = df['Volatility'].iloc[idx]

        # 2. Z-score of last rsi_param[0]*2 % vol
        features['mlfeat_volatility_zscore'] = df['Volatility_Zscore'].iloc[idx]

        # 3. RSI value with period 3*rsi_param[0]
        rsi_long_period = 3 * rsi_param[0]
        df['RSI_Long'] = compute_rsi(close, rsi_long_period)
        features['mlfeat_rsi_long'] = df['RSI_Long'].iloc[idx]

        # 4. Average of last 3 days RSI
        features['mlfeat_rsi_chg3'] = df['RSI_Long'].diff(periods=3).iloc[idx]

        # 5. (2d EWMA - 8d EWMA) / ATR
        ewma_diff_short = (df['EWMA_2'].iloc[idx] - df['EWMA_8'].iloc[idx]) / df['ATR'].iloc[idx]
        features['mlfeat_ewma_diff_short'] = ewma_diff_short

        # 6. (16d EWMA - 64d EWMA) / ATR
        ewma_diff_long = (df['EWMA_16'].iloc[idx] - df['EWMA_64'].iloc[idx]) / df['ATR'].iloc[idx]
        features['mlfeat_ewma_diff_long'] = ewma_diff_long

        # 7. Z-score of volume for last 3*rsi_param[0] days
        if 'Volume_Zscore' in df.columns:
            features['mlfeat_volume_zscore'] = df['Volume_Zscore'].iloc[idx]
        else:
            features['mlfeat_volume_zscore'] = np.nan

        # 8. ADX indicator
        features['mlfeat_adx'] = df['ADX'].iloc[idx]

        # 9. Carry features if 'Carry' exists
        if 'Carry' in df.columns:
            carry_window = rsi_param[0] * 2
            carry_diff = df['Carry_Diff'].rolling(window=carry_window)
            carry_change = carry_diff.sum().iloc[idx]
            carry_sd = carry_diff.std().iloc[idx]
            features['mlfeat_carry_change'] = carry_change / carry_sd if carry_sd != 0 else np.nan

            carry_ema_diff = (df['Carry_EMA_16'].iloc[idx] - df['Carry_EMA_64'].iloc[idx]) / carry_sd if carry_sd != 0 else np.nan
            features['mlfeat_carry_ema_diff'] = carry_ema_diff
        else:
            features['mlfeat_carry_change'] = np.nan
            features['mlfeat_carry_ema_diff'] = np.nan

        # 10. Skewness of last 126 days
        features['mlfeat_skew'] = df['Skew'].iloc[idx]

        # Assign features to DataFrame
        for feature_name, value in features.items():
            df.at[idx, feature_name] = value

        # Calculate the result vector (t+1 to t+1+N day return)
        if idx + N + 1 < len(df):
            future_return = (close.iloc[idx + N + 1] - open_prices.iloc[idx + 1]) / open_prices.iloc[idx + 1]
            df.at[idx, 'mlfeat_result'] = future_return
        else:
            df.at[idx, 'mlfeat_result'] = np.nan

        # Collect past signals and features for ML training
        past_signals.append(idx)

        # Check if we have enough data to train
        available_dps = len(past_signals) - 1  # Exclude current index
        if available_dps >= min_dps:
            # Prepare training data
            train_indices = past_signals[:-1][-max_dps:]  # Get the last max_dps indices excluding current
            X_train = df.loc[train_indices, ml_feature_names]
            y_train = np.sign(df.loc[train_indices, 'mlfeat_result'])

            # Remove any rows with NaN in features or target
            valid_train = X_train.notnull().all(axis=1) & y_train.notnull()
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]

            if len(y_train) >= min_dps:
                # Train Random Forest model
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight='balanced',  # Penalize to avoid overfitting
                )
                model.fit(X_train_scaled, y_train)

                # Prepare test data (current features)
                X_test = pd.DataFrame([features])
                if X_test.notnull().all(axis=1).iloc[0]:
                    X_test_scaled = scaler.transform(X_test)

                    # Predict probability

                    prob = model.predict_proba(X_test_scaled) # Probability of positive class
                    try:
                        prob = prob[0][1]
                    except:
                        prob = prob[0][0]
                    df.at[idx, 'mlfeat_probability'] = prob
                    
                    # Decide whether to take the trade
                    if prob > abs(probability_threshold) and trade_sign == 1:
                        # Proceed with the trade
                        pass
                    elif prob < abs(probability_threshold) and trade_sign == -1:
                        # Proceed with fading the trade
                        pass
                    else:
                        continue  # Do not take the trade
                else:
                    # Features contain NaN, cannot predict
                    continue
            else:
                # Not enough valid training data
                continue
        else:
            # Not enough data to train
            continue

        # Proceed with the trade
        entry_idx = idx + 1  # Enter at next open
        if in_trade.iloc[entry_idx]:
            continue  # Already in trade, skip

        signals.iloc[entry_idx] = 1 * trade_sign  # Entry signal
        trade_entries.iloc[entry_idx] = open_prices.iloc[entry_idx]
        trade_start = entry_idx

        # Determine exit conditions
        exit_idx = trade_start
        exit_found = False
        while exit_idx < len(df) - 1:
            exit_idx += 1
            days_in_trade = exit_idx - trade_start
            exit_by_N_days = days_in_trade >= N
            if exit_quick_rule and days_in_trade > N_min:
                exit_by_high = close.iloc[exit_idx] > high.iloc[exit_idx - 1]
            else:
                exit_by_high = False

            if exit_trend_rule is not None and not pd.isna(exit_ma.iloc[trade_start]):
                exit_trend_in_effect = close.iloc[trade_start] >= exit_ma.iloc[trade_start]
            else:
                exit_trend_in_effect = False

            if exit_trend_in_effect:
                # Exit when price closes below the exit trend MA
                if close.iloc[exit_idx] < exit_ma.iloc[exit_idx]:
                    exit_found = True
                    break
            else:
                # Normal exit rules
                if exit_by_N_days or exit_by_high:
                    exit_found = True
                    break

        if not exit_found:
            exit_idx = len(df) - 1  # Exit at last available data

        signals.iloc[exit_idx] = -1 * trade_sign # Exit signal
        trade_exits.iloc[exit_idx] = close.iloc[exit_idx]

        # Mark in_trade period
        in_trade.iloc[trade_start:exit_idx + 1] = True 

    # Assign computed values back to DataFrame
    df['Signal'] = signals
    df['TradeEntry'] = trade_entries
    df['TradeExit'] = trade_exits
    df['InTrade'] = in_trade.astype(int) * trade_sign

    # Shift all columns starting with "mlfeat" down one
    mlfeat_columns = [col for col in df.columns if col.startswith('mlfeat_')]
    for col in mlfeat_columns:
        df[col] = df[col].shift(1)
    # If you need to convert back to Polars DataFrame
    df = pl.DataFrame(df)

    return df


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0))
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0))

    atr = compute_atr(df, period)

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx





def compute_atr(df: pd.DataFrame, period: int, atr_type = 'atr') -> pd.Series:

    if atr_type == 'sd':
        close = df['Close']
        close_diff = close.diff().dropna()
        atr = close_diff.rolling(window=period).std()
        return atr
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
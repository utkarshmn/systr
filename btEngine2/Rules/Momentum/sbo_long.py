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
                    df_list[i + 1]['TradeEntry'] = df_list[i + 1]['Open']  # Record the entry price
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
                        df_list[next_day_index]['Holding_Period'] = 1  # Start holding period
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
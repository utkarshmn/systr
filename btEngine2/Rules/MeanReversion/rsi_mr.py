import polars as pl
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime

from btEngine2.Indicators import *
from btEngine2.MarketData import MarketData


def rsi_mrb_short(
    df: pd.DataFrame,
    N: int = 3,
    lmt_atr_ratio: float = 0.75,
    lmt_atr_period: int = 3,
    lmt_epsilon: float = 0.1,
    rsi_period: int = 3,
    consecutive_days: int = 3,
    rsi_threshold: float = 90.0,  # Typically high RSI for short entries
    optional_rsi: tuple = None,    # (rsi_period, cumulative_days, threshold)
    optional_ma: tuple = None,     # (ma_period, ma_type)
    atr_type: str = 'atr'
) -> pd.DataFrame:
    """
    Mean Reversion Short Strategy with RSI and optional moving average filters.

    :param df: Pandas DataFrame containing asset data with 'Date', 'Open', 'High', 'Low', 'Close' columns.
    :param N: Holding period after entry (default: 3).
    :param lmt_atr_ratio: Multiplier for ATR to calculate limit price (default: 0.75).
    :param lmt_atr_period: Period for ATR calculation (default: 3).
    :param lmt_epsilon: Small value to adjust the limit order fill range (default: 0.1).
    :param rsi_period: Period for RSI calculation (default: 3).
    :param consecutive_days: Number of consecutive higher closes required (default: 3).
    :param rsi_threshold: RSI threshold for entry condition (default: 90.0).
    :param optional_rsi: Optional tuple (rsi_period, cumulative_days, threshold).
    :param optional_ma: Optional tuple (ma_period, ma_type), ma_type can be 'ema' or 'sma'.
                         +ma_period for above MA, -ma_period for below MA.
    :param atr_type: Type of ATR calculation ('atr' or 'sd').
    :return: Pandas DataFrame with trading signals appended.
    """

    # Convert Polars DataFrame to Pandas if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Ensure necessary columns are present
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Ensure the DataFrame is sorted by Date in ascending order
    df = df.sort_values('Date').reset_index(drop=True)

    # Step 1: Compute the basic RSI (using helper function)
    df = compute_rsi_df(df, rsi_period)

    # Step 2: Identify consecutive higher closes
    df['Higher_Close'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['Consecutive_Higher_Closes'] = df['Higher_Close'].rolling(window=consecutive_days, min_periods=consecutive_days).sum()

    # Step 3: Optional RSI condition
    if optional_rsi is not None:
        opt_rsi_period, cumulative_days, threshold = optional_rsi
        df = compute_rsi_df(df, opt_rsi_period)
        df['Cumulative_RSI'] = df[f'RSI_{opt_rsi_period}'].rolling(window=cumulative_days, min_periods=cumulative_days).mean()

    # Step 4: Optional MA condition
    if optional_ma is not None:
        ma_period, ma_type = optional_ma
        if ma_type.lower() == 'ema':
            df['MA'] = df['Close'].ewm(span=abs(ma_period), adjust=False).mean()
        elif ma_type.lower() == 'sma':
            df['MA'] = df['Close'].rolling(window=abs(ma_period), min_periods=ma_period).mean()
        else:
            raise ValueError("Invalid ma_type. Use 'ema' or 'sma'.")

        # Determine if price is above or below MA
        if ma_period > 0:
            df['MA_Condition'] = df['Close'] > df['MA']
        else:
            df['MA_Condition'] = df['Close'] < df['MA']

    # Step 5: Calculate ATR using the compute_effective_atr helper function
    df = compute_effective_atr(df, lmt_atr_period, atr_type)

    # Step 6: Define Entry Conditions
    entry_condition = (
        (df[f'RSI_{rsi_period}'] > rsi_threshold) &
        (df['Consecutive_Higher_Closes'] == consecutive_days)
    )

    if optional_rsi is not None:
        entry_condition &= (df['Cumulative_RSI'] > threshold)

    if optional_ma is not None:
        entry_condition &= df['MA_Condition']

    df['Entry_Signal'] = entry_condition

    # Initialize trading signal columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0
    df['Holding_Period'] = 0

    # Initialize position tracking variables
    position_open = False
    holding_period = 0

    # Iterate over the DataFrame to generate signals
    for i in range(len(df) - 1):
        if not position_open:
            # Check for Entry Signal
            if df.loc[i, 'Entry_Signal']:
                # Calculate limit price
                current_close = df.loc[i, 'Close']
                current_atr = df.loc[i, 'ATR']
                lmt_price = current_close + lmt_atr_ratio * current_atr

                # Next day's index
                next_day_index = i + 1
                if next_day_index >= len(df):
                    break  # Reached end of data

                day_low = df.loc[next_day_index, 'Low']
                day_high = df.loc[next_day_index, 'High']

                # Adjusted low and high with lmt_epsilon
                adj_low = day_low + lmt_epsilon
                adj_high = day_high - lmt_epsilon

                # Check if lmt_price is within the next day's range
                if adj_low <= lmt_price <= adj_high:
                    # Limit order is filled
                    df.at[next_day_index, 'Signal'] = -1  # Short signal
                    df.at[next_day_index, 'TradeEntry'] = lmt_price
                    df.at[next_day_index, 'InTrade'] = -1
                    df.at[next_day_index, 'Holding_Period'] = 0
                    position_open = True
                    holding_period = 0
        else:
            # We are in a position
            holding_period += 1
            df.at[i, 'InTrade'] = -1
            df.at[i, 'Holding_Period'] = holding_period

            # Check for Exit Conditions
            exit_condition = (
                (holding_period > N) |
                (df.loc[i, 'Close'] < df.loc[i - 1, 'Low'])
            )

            if exit_condition:
                df.at[i, 'Signal'] = 1  # Exit short position
                df.at[i, 'TradeExit'] = df.loc[i, 'Close']
                df.at[i, 'InTrade'] = -1
                df.at[i, 'Holding_Period'] = 0
                position_open = False
                holding_period = 0

    # Convert back to Polars DataFrame if necessary
    df = pl.DataFrame(df)

    return df


def rsi_mrb_long(
    df: pl.DataFrame,
    N: int = 3,
    lmt_atr_ratio: float = 0.75,
    lmt_atr_period: int = 3,
    lmt_epsilon: float = 0.1,
    rsi_period: int = 3,
    consecutive_days: int = 3,
    rsi_threshold: float = 10.0,  # Typically low RSI for long entries
    optional_rsi: tuple = None,    # (rsi_period, cumulative_days, threshold)
    optional_ma: tuple = None,     # (ma_period, ma_type)
    atr_type: str = 'atr'
) -> pd.DataFrame:
    """
    Mean Reversion Long Strategy with RSI and optional moving average filters.

    :param df: Pandas DataFrame containing asset data with 'Date', 'Open', 'High', 'Low', 'Close' columns.
    :param N: Holding period after entry (default: 3).
    :param lmt_atr_ratio: Multiplier for ATR to calculate limit price (default: 0.75).
    :param lmt_atr_period: Period for ATR calculation (default: 3).
    :param lmt_epsilon: Small value to adjust the limit order fill range (default: 0.1).
    :param rsi_period: Period for RSI calculation (default: 3).
    :param consecutive_days: Number of consecutive lower closes required (default: 3).
    :param rsi_threshold: RSI threshold for entry condition (default: 10.0).
    :param optional_rsi: Optional tuple (rsi_period, cumulative_days, threshold).
    :param optional_ma: Optional tuple (ma_period, ma_type), ma_type can be 'ema' or 'sma'.
                         +ma_period for above MA, -ma_period for below MA.
    :param atr_type: Type of ATR calculation ('atr' or 'sd').
    :return: Pandas DataFrame with trading signals appended.
    """

    if type(df) == pl.DataFrame:
        df = df.to_pandas()
    # Ensure necessary columns are present
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")


    # Ensure the DataFrame is sorted by Date in ascending order
    df = df.sort_values('Date').reset_index(drop=True)

    # Step 1: Compute the basic RSI (using helper function)
    df = compute_rsi_df(df, rsi_period)

    # Step 2: Identify consecutive lower closes
    df['Lower_Close'] = (df['Close'] < df['Close'].shift(1)).astype(int)
    df['Consecutive_Lower_Closes'] = df['Lower_Close'].rolling(window=consecutive_days, min_periods=consecutive_days).sum()

    # Step 3: Optional RSI condition
    if optional_rsi is not None:
        opt_rsi_period, cumulative_days, threshold = optional_rsi
        df = compute_rsi_df(df, opt_rsi_period)
        df['Cumulative_RSI'] = df[f'RSI_{opt_rsi_period}'].rolling(window=cumulative_days, min_periods=cumulative_days).mean()

    # Step 4: Optional MA condition
    if optional_ma is not None:
        ma_period, ma_type = optional_ma
        if ma_type.lower() == 'ema':
            df['MA'] = df['Close'].ewm(span=abs(ma_period), adjust=False).mean()
        elif ma_type.lower() == 'sma':
            df['MA'] = df['Close'].rolling(window=abs(ma_period), min_periods=ma_period).mean()
        else:
            raise ValueError("Invalid ma_type. Use 'ema' or 'sma'.")

        # Determine if price is above or below MA
        if ma_period > 0:
            df['MA_Condition'] = df['Close'] > df['MA']
        else:
            df['MA_Condition'] = df['Close'] < df['MA']

    # Step 5: Calculate ATR using the compute_effective_atr helper function
    df = compute_effective_atr(df, lmt_atr_period, atr_type)

    # Step 6: Define Entry Conditions
    entry_condition = (
        (df[f'RSI_{rsi_period}'] < rsi_threshold) &
        (df['Consecutive_Lower_Closes'] == consecutive_days)
    )

    if optional_rsi is not None:
        entry_condition &= (df['Cumulative_RSI'] < threshold)

    if optional_ma is not None:
        entry_condition &= df['MA_Condition']

    df['Entry_Signal'] = entry_condition

    # Initialize trading signal columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0
    df['Holding_Period'] = 0

    # Initialize position tracking variables
    position_open = False
    holding_period = 0

    # Iterate over the DataFrame to generate signals
    for i in range(len(df) - 1):
        if not position_open:
            # Check for Entry Signal
            if df.loc[i, 'Entry_Signal']:
                # Calculate limit price
                current_close = df.loc[i, 'Close']
                current_atr = df.loc[i, 'ATR']
                lmt_price = current_close - lmt_atr_ratio * current_atr

                # Next day's index
                next_day_index = i + 1
                if next_day_index >= len(df):
                    break  # Reached end of data

                day_low = df.loc[next_day_index, 'Low']
                day_high = df.loc[next_day_index, 'High']

                # Adjusted low and high with lmt_epsilon
                adj_low = day_low + lmt_epsilon
                adj_high = day_high - lmt_epsilon

                # Check if lmt_price is within the next day's range
                if adj_low <= lmt_price <= adj_high:
                    # Limit order is filled
                    df.at[next_day_index, 'Signal'] = 1  # Long signal
                    df.at[next_day_index, 'TradeEntry'] = lmt_price
                    df.at[next_day_index, 'InTrade'] = 1
                    df.at[next_day_index, 'Holding_Period'] = 0
                    position_open = True
                    holding_period = 0
        else:
            # We are in a position
            holding_period += 1
            df.at[i, 'InTrade'] = 1
            df.at[i, 'Holding_Period'] = holding_period

            # Check for Exit Conditions
            exit_condition = (
                (holding_period > N) |
                (df.loc[i, 'Close'] > df.loc[i - 1, 'High'])
            )

            if exit_condition:
                df.at[i, 'Signal'] = -1  # Exit long position
                df.at[i, 'TradeExit'] = df.loc[i, 'Close']
                df.at[i, 'InTrade'] = 1
                df.at[i, 'Holding_Period'] = 0
                position_open = False
                holding_period = 0

    df = pl.DataFrame(df)

    return df

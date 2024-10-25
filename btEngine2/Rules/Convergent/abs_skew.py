import polars as pl
from btEngine2.Indicators import *
from typing import Optional, List
import pandas as pd
import numpy as np
from scipy.stats import skew

def abs_skew_long(df, N, size_factor = 1.0, **kwargs):
    """
    Generate long trading signals based on the negative skewness of returns.

    Parameters:
    - df: pandas DataFrame with at least a 'Close' column.
    - N: int, window size for trailing returns.
    - kwargs: Additional arguments if needed.

    Returns:
    - df: pandas DataFrame with 'signal' column added.
    """

    # Convert Polars DataFrame to Pandas if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate rolling skewness over N-day window
    df['Skew'] = df['Returns'].rolling(window=N).apply(lambda x: skew(x, bias=False), raw=True)
    
    # Multiply skewness by -1 to invert the signal
    df['Skew_Signal'] = -1 * df['Skew']
    
    # Calculate EWMA over N/4 periods
    ewma_span = max(int(N / 4), 1)
    df['Smoothed_Signal'] = df['Skew_Signal'].ewm(span=ewma_span, adjust=False).mean()
    
    df['Signal'] = df['Smoothed_Signal'].apply(lambda x: x if x > 0 else 0).clip(-2.5,2.5)

    # Initialize TradeEntry and TradeExit columns with NaNs
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df.drop(columns=['Returns', 'Skew_Signal'], inplace=True)
    df['InTrade'] = df['Signal'] * size_factor
    df = pl.DataFrame(df)
    
    return df

def abs_skew_short(df, N, size_factor = 1.0, **kwargs):
    """
    Generate short trading signals based on the positive skewness of returns.

    Parameters:
    - df: pandas DataFrame with at least a 'Close' column.
    - N: int, window size for trailing returns.
    - kwargs: Additional arguments if needed.

    Returns:
    - df: pandas DataFrame with 'signal' column added.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate rolling skewness over N-day window
    df['Skew'] = df['Returns'].rolling(window=N).apply(lambda x: skew(x, bias=False), raw=True)
    
    # Use the skewness directly
    df['Skew_Signal'] = -1 * df['Skew']
    
    # Calculate EWMA over N/4 periods
    ewma_span = max(int(N / 4), 1)
    df['Smoothed_Signal'] = df['Skew_Signal'].ewm(span=ewma_span, adjust=False).mean()
    
    # Generate short-only signal: -1 when signal > 0, 0 otherwise
    df['Signal'] = df['Smoothed_Signal'].apply(lambda x: x if x < 0 else 0).clip(-2.5,2.5)
    # Initialize TradeEntry and TradeExit columns with NaNs
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df.drop(columns=['Returns', 'Skew_Signal'], inplace=True)
    df['InTrade'] = df['Signal'] * size_factor
    df = pl.DataFrame(df)

    return df


def abs_skew_combined(df, N, size_factor = 1.0, **kwargs):
    """
    Generate trading signals based on skewness for both long and short positions.

    Parameters:
    - df: pandas DataFrame with at least a 'Close' column.
    - N: int, window size for trailing returns.
    - kwargs: Additional arguments if needed.

    Returns:
    - df: pandas DataFrame with 'signal' column added.
    """

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Skew'] = -1 * df['Returns'].rolling(window=N).apply(lambda x: skew(x, bias=False), raw=True)
    df['Smoothed_Signal'] = df['Skew'].ewm(span=max(int(N / 4), 1), adjust=False).mean()
    df['Signal'] = df['Smoothed_Signal'].clip(-2.5,2.5)
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df.drop(columns=['Returns'], inplace=True)
    df['InTrade'] = df['Signal'] * size_factor

    df = pl.DataFrame(df)

    return df
import polars as pl
from btEngine2.Indicators import *
from typing import Optional, List
import pandas as pd
import numpy as np
from scipy.stats import skew

def abs_skew_long(df, N, size_factor = 1.0, min_sig=0, lag=0, **kwargs):
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
    
    df['Signal'] = df['Smoothed_Signal'].apply(lambda x: x if x > min_sig else 0).clip(-2.5,2.5)

    if lag != 0:
        df['Signal'] = df['Signal'].shift(lag)


    # Initialize TradeEntry and TradeExit columns with NaNs
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df.drop(columns=['Returns', 'Skew_Signal'], inplace=True)
    df['InTrade'] = df['Signal'] * size_factor
    df = pl.DataFrame(df)
    
    return df

def abs_skew_short(df, N, size_factor = 1.0, min_sig = 0, lag=0, **kwargs):
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
    df['Signal'] = df['Smoothed_Signal'].apply(lambda x: x if x < -1*min_sig else 0).clip(-2.5, 2.5)


    if lag != 0:
        df['Signal'] = df['Signal'].shift(lag)
        

    # Initialize TradeEntry and TradeExit columns with NaNs
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df.drop(columns=['Returns', 'Skew_Signal'], inplace=True)
    df['InTrade'] = df['Signal'] * size_factor
    df = pl.DataFrame(df)

    return df


def abs_skew_combined(df, N, size_factor_l = 1.0, size_factor_s=1.0, min_sig_l = 0.0, min_sig_s=0.0, lag=0, **kwargs):
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

    long_min = min_sig_l
    short_min = min_sig_s
    df['long_sig'] = df['Smoothed_Signal'].apply(lambda x: x if x > long_min else 0)
    df['short_sig'] = df['Smoothed_Signal'].apply(lambda x: x if x < -short_min else 0)
    df['Signal'] = df['long_sig'] + df['short_sig']    

    df['Signal'] = df['Smoothed_Signal'].clip(-2.5,2.5)


    if lag != 0:
        df['Signal'] = df['Signal'].shift(lag)
        
        
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    # Clean up
    df['InTrade_l'] = df['long_sig'] * size_factor_l
    df['InTrade_s'] = df['short_sig'] * size_factor_s
    df['InTrade'] = df['InTrade_l'] + df['InTrade_s']


    df.drop(columns=['Returns', 'long_sig', 'short_sig', 'InTrade_l', 'InTrade_s'], inplace=True)

    df = pl.DataFrame(df)

    return df
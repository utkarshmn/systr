import polars as pl
import numpy as np
import pandas as pd

def compute_rsi(series: pl.Series, period: int = 14) -> pl.Series:
    """
    Computes the Relative Strength Index (RSI) of a series using Polars expressions.

    :param series: A Polars Series of price data.
    :param period: The lookback period for RSI calculation.
    :return: A Polars Series containing the RSI values.
    """
    if isinstance(series, pd.Series):
        return calculate_rsi_pd(series, period)
    
    delta = series.diff()
    
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_rsi_pd(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_effective_atr(df, X: int, atr_type: str = 'atr') -> pl.DataFrame:
    """
    Compute the Effective ATR for the given DataFrame.

    :param df: Polars DataFrame containing 'High', 'Low', 'Close', and optionally 'BadOHLC' columns.
    :param X: Lookback period for ATR calculation.
    :param atr_type: Type of ATR calculation ('atr' or 'sd').
    :return: DataFrame with 'RollingMax' and 'Effective_ATR' columns added.
    """
    # Step 2: Calculate True Range components

    if type(df) == pd.DataFrame:
        return compute_effective_atr_pd(df, X, atr_type)


    df = df.with_columns([
        (pl.col('High') - pl.col('Low')).alias('H-L'),
        (pl.col('High') - pl.col('Close').shift(1)).abs().alias('H-PC'),
        (pl.col('Low') - pl.col('Close').shift(1)).abs().alias('L-PC')
    ])
    
    # Step 3: Calculate True Range (TR)
    df = df.with_columns([
        pl.max_horizontal(['H-L', 'H-PC', 'L-PC']).alias('TR'),
    ])
    
    # Step 4: Calculate ATR
    df = df.with_columns([
        pl.col('TR').rolling_mean(window_size=X).alias('ATR')
    ])
    
    # Step 5: Calculate ATR using Standard Deviation method
    df = df.with_columns([
        pl.col('Close').diff().rolling_std(window_size=X).alias('ATR_SD')
    ])
    
    # Step 6: Determine Effective ATR based on atr_type
    if atr_type == 'atr':
        # Use ATR, adjust for 'BadOHLC' if present
        if 'BadOHLC' in df.columns:
            df = df.with_columns([
                pl.when(pl.col('BadOHLC')).then(pl.col('ATR_SD')).otherwise(pl.col('ATR')).alias('Eff_ATR')
            ])
        else:
            df = df.with_columns([
                pl.col('ATR').alias('Eff_ATR')
            ])
    elif atr_type == 'sd':
        df = df.with_columns([
            pl.col('ATR_SD').alias('Eff_ATR')
        ])
    else:
        raise ValueError("atr_type must be either 'atr' or 'sd'")
    
    # Step 7: Drop intermediate columns to clean up the DataFrame
    df = df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'ATR', 'ATR_SD'])
    
    df = df.rename({'Eff_ATR': 'ATR'})
    
    return df

def compute_effective_atr_pd(df, X, atr_type='atr'):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['ATR'] = df['TR'].rolling(window=X).mean()
    
    df['ATR_SD'] = df['Close'].diff().rolling(window=X).std()
    
    if atr_type == 'atr':
        if 'BadOHLC' in df.columns:
            df['Eff_ATR'] = np.where(df['BadOHLC'], df['ATR_SD'], df['ATR'])
        else:
            df['Eff_ATR'] = df['ATR']
    elif atr_type == 'sd':
        df['Eff_ATR'] = df['ATR_SD']
    else:
        raise ValueError("atr_type must be either 'atr' or 'sd'")
    
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR', 'ATR', 'ATR_SD'], inplace=True)
    
    df.rename(columns={'Eff_ATR': 'ATR'}, inplace=True)
    
    return df


def compute_rsi_df(df: pl.DataFrame, period: int, column_name: str = 'Close') -> pl.DataFrame:
    """
    Computes the RSI (Relative Strength Index) for the given period.
    """

    if type(df) == pd.DataFrame:
        return compute_rsi_df_pd(df, period, column_name)
    df = df.with_columns([
        (pl.col(column_name) - pl.col(column_name).shift(1)).alias('Price_Change')
    ])

    df = df.with_columns([
        pl.when(pl.col('Price_Change') > 0).then(pl.col('Price_Change')).otherwise(0).alias('Gain'),
        pl.when(pl.col('Price_Change') < 0).then(-pl.col('Price_Change')).otherwise(0).alias('Loss')
    ])

    df = df.with_columns([
        pl.col('Gain').rolling_mean(window_size=period).alias('Avg_Gain'),
        pl.col('Loss').rolling_mean(window_size=period).alias('Avg_Loss')
    ])

    df = df.with_columns([
        pl.when(pl.col('Avg_Loss') == 0)
        .then(100)
        .otherwise(100 - (100 / (1 + (pl.col('Avg_Gain') / pl.col('Avg_Loss')))))
        .alias(f'RSI_{period}')
    ])
    return df


def compute_rsi_df_pd(df: pd.DataFrame, period: int, column_name: str = 'Close') -> pd.DataFrame:
    """
    Computes the RSI (Relative Strength Index) for the given period.

    :param df: Pandas DataFrame containing the price data.
    :param period: RSI calculation period.
    :param column_name: Column name for price (default: 'Close').
    :return: DataFrame with RSI added as a new column.
    """
    delta = df[column_name].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f'RSI_{period}'] = rsi

    return df
import polars as pl
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime

from btEngine2.Indicators import *
from btEngine2.MarketData import MarketData

def ratioMR_rsi_long(
    df: pl.DataFrame,
    pairs: List[Tuple[str, str]] = [],
    N: Tuple[int, int] = (5, 5),
    rsi_period: int = 2,
    rsi_threshold: float = 10.0,
    lmt_order: bool = False,
    lmt_day: int = 1,
    lmt_day_only: bool = True,
    lmt_atr: float = 1.0,
    lmt_epsilon: float = 0.1,
    atr_period: int = 14,
    atr_type: str = 'atr',
    trend_filter: Tuple[int, int] = None,
    oversold_rsi: Tuple[float, float] = None,
    oversold_cond: float = 10.0,
    corr_cond: Tuple[int,float] = None,
    flip_main: bool = False,
    vol_wgt_ratio: bool = False,
    risk_ratio: Tuple[float, float] = (1.0, 1.0), # risk ratio for sizing.
    market_data: Any = None) -> pl.DataFrame:
    """
    Trading strategy that goes long on the main asset when the RSI of the spread is below a threshold.

    :param df: Polars DataFrame containing data for the main asset.
    :param trading_params: Dictionary of trading parameters, including:
        - 'pairs': List of tuples [(asset_name, comp_name), ...]
        - 'N': Holding period after entry.
        - 'rsi_period': Lookback period for RSI calculation.
        - 'rsi_threshold': RSI threshold for generating buy signals.
        - 'lmt_order': If True, uses limit order logic for entry.
        - 'lmt_day': The day on which the limit order is placed.
        - 'lmt_day_only': If True, limit order is only placed on lmt_day; otherwise, placed every day until lmt_day.
        - 'lmt_atr': Multiplier for ATR to determine limit price.
        - 'lmt_epsilon': Small value to adjust high and low prices for limit order checking.
        - 'atr_period': Lookback period for ATR calculation.
        - 'atr_type': Type of ATR calculation ('atr' or 'sd').
        - 'market_data': Instance of MarketData to fetch comparative asset data.
    :return: Polars DataFrame with trading signals appended for the main asset.
    """
    # Ensure 'Name' column exists
    if 'Name' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Name' column.")

    # Get the main asset's name from the DataFrame
    main_asset_names = df.select(pl.col('Name')).unique().to_series().drop_nulls()
    if main_asset_names.len() == 0:
        raise ValueError("No asset names found in the 'Name' column.")
    main_asset_name = main_asset_names[0]

    # Find if the main asset is in any pair and define asset specific wgts

    comp_name = None
    
    for pair in pairs:
        if pair[0] == main_asset_name:
            comp_name = pair[1]

            if isinstance(N, tuple):
                N = N[0]

            if isinstance(risk_ratio, tuple):
                risk_ratio = risk_ratio[0]
            if isinstance(trend_filter, tuple):
                if isinstance(trend_filter[0], int):
                    trend_filter = trend_filter[0]
            break
        else:
            if isinstance(N, tuple):
                N = N[1]
            if isinstance(risk_ratio, tuple):
                risk_ratio = risk_ratio[1]
            if isinstance(trend_filter, tuple):
                if isinstance(trend_filter[1], int):
                    trend_filter = trend_filter[1]

    if comp_name is None:
        # Main asset not in any pair; return default signals
        df = df.with_columns([
            pl.lit(0).alias('Signal'),
            pl.lit(np.nan).alias('TradeEntry'),
            pl.lit(np.nan).alias('TradeExit'),
            pl.lit(0).alias('InTrade'),
            pl.lit(0).alias('Holding_Period')
        ])
        return df

    # Fetch comparative asset data using market_data
    if market_data is None:
        raise ValueError("market_data must be provided in trading_params.")

    comp_df = market_data.get_ticker_data(comp_name)
    if comp_df is None or comp_df.is_empty():
        # Comparative asset data not found; return default signals
        df = df.with_columns([
            pl.lit(0).alias('Signal'),
            pl.lit(np.nan).alias('TradeEntry'),
            pl.lit(np.nan).alias('TradeExit'),
            pl.lit(0).alias('InTrade'),
            pl.lit(0).alias('Holding_Period')
        ])
        return df
    
    # Ensure both DataFrames are sorted by 'Date'

    df = df.sort('Date')
    comp_df = comp_df.sort('Date')

    if flip_main:
        df_tmp = df.to_pandas().copy()
        df_tmp['Close'] = df['Close'] * -1
        df_tmp['Open'] = df['Open'] * -1
        df_tmp['High'] = df['Low'] * -1
        df_tmp['Low'] = df['High'] * -1
        df = pl.from_pandas(df_tmp)


    '''
    # Merge the two DataFrames on 'Date' using a left join to preserve all main asset rows
    merged_df = df.join(comp_df.select(['Date', 'Close']), on='Date', how='left', suffix="_comp")

    # Optional: Handle missing 'Close_comp' values (e.g., fill forward)
    merged_df = merged_df.with_columns([
        pl.col('Close_comp').fill_null(strategy='forward').alias('Close_comp')
    ])
    '''

    main_pd = df.to_pandas()
    cmp_pd = comp_df.to_pandas()
    
    main_pd.set_index('Date', inplace=True)
    cmp_pd.set_index('Date', inplace=True)

    main_close_df = main_pd[['Close', 'Tick_Value_USD']].rename(
        columns={'Close': 'Main_Close', 'Tick_Value_USD': 'Main_Tick_Value_USD'})
    cmp_close_df = cmp_pd[['Close', 'Tick_Value_USD']].rename(
        columns={'Close': 'Comp_Close', 'Tick_Value_USD': 'Comp_Tick_Value_USD'})

    concat_df = pd.concat([main_close_df, cmp_close_df], axis=1)

    if type(vol_wgt_ratio) == int:
        concat_df['maindiffs'] = concat_df['Main_Close'].diff()
        concat_df['compdiffs'] = concat_df['Comp_Close'].diff()
        concat_df['mainvol'] = concat_df['maindiffs'].rolling(window=vol_wgt_ratio).std()
        concat_df['mainvol'] = concat_df['mainvol'] * concat_df['Main_Tick_Value_USD']
        concat_df['compvol'] = concat_df['compdiffs'].rolling(window=vol_wgt_ratio).std()
        concat_df['compvol'] = concat_df['compvol'] * concat_df['Comp_Tick_Value_USD']        
        voladj = concat_df['mainvol'] / concat_df['compvol']

        concat_df['Main_Close_va'] = concat_df['Main_Close']
        concat_df['Comp_Close_va'] = concat_df['Comp_Close'] * voladj

        concat_df['Ratio'] = concat_df['Main_Close_va'] / concat_df['Comp_Close_va']
    else:
        concat_df['Ratio'] = concat_df['Main_Close'] / concat_df['Comp_Close']
    
    concat_df['RSI_Ratio'] = compute_rsi(concat_df['Ratio'], period=rsi_period)

    merged_df = pd.concat([main_pd, concat_df], axis=1)

    cols_to_drop = ['Main_Close']

    # Calc entry condition dependencies.

    if corr_cond is not None:
        merged_df['maindiffs'] = merged_df['Main_Close'].diff()
        merged_df['compdiffs'] = merged_df['Comp_Close'].diff()
        merged_df['Corr'] = merged_df['maindiffs'].rolling(window=corr_cond[0]).corr(merged_df['compdiffs'])
        merged_df['Corr_Cond'] = merged_df['Corr'] > corr_cond[1]
        cols_to_drop += ['maindiffs', 'compdiffs']
    else:
        merged_df['Corr_Cond'] = True
        cols_to_drop.append('Corr_Cond')


    if trend_filter is not None and type(trend_filter) == int:
        merged_df['MovAvg'] = merged_df['Main_Close'].rolling(window=trend_filter).mean()
        merged_df['Trend_Filter'] = merged_df['Close'] > merged_df['MovAvg']
    else:
        merged_df['Trend_Filter'] = True
        cols_to_drop.append('Trend_Filter')

    
    if oversold_cond is not None and type(oversold_cond) == int:
        merged_df['Oversold_Cond'] = merged_df['Close'].rolling(window=oversold_cond).apply(
            lambda x: all(x[i] < x[i - 1] for i in range(1, len(x))), raw=True
        )
    else:
        merged_df['Oversold_Cond'] = True
        cols_to_drop.append('Oversold_Cond')
    
    if oversold_rsi is not None and type(oversold_rsi) == tuple:
        merged_df['Asst_RSI'] = compute_rsi(merged_df['Close'], period=oversold_rsi[0])
        merged_df['Oversold_RSI'] = merged_df['Asst_RSI'] < oversold_rsi[1]
    else:
        merged_df['Oversold_RSI'] = True
        cols_to_drop.append('Oversold_RSI')


    # Compute yesterday's High for the main asset
    merged_df['Prev_High'] = merged_df['High'].shift(1)
    merged_df['Prev_High_Cond'] = merged_df['Close'] < merged_df['Prev_High']
    cols_to_drop.append('Prev_High_Cond')


    if lmt_order:
        merged_df = compute_effective_atr(merged_df, atr_period, atr_type)

    # Initialize trading signal columns
    merged_df['Signal'] = 0
    merged_df['TradeEntry'] = np.nan
    merged_df['TradeExit'] = np.nan
    merged_df['InTrade'] = 0
    merged_df['Holding_Period'] = 0

    merged_df['Entry_Signal'] = (
        (merged_df['Prev_High_Cond']) &
        (merged_df['RSI_Ratio'] < rsi_threshold) &
        (merged_df['Trend_Filter']) &
        (merged_df['Oversold_Cond']) &
        (merged_df['Oversold_RSI']) &
        (merged_df['Corr_Cond'])
    )

    merged_df.drop(columns=cols_to_drop, inplace=True)
    
    merged_df.reset_index(inplace=True)
    # Convert merged DataFrame to dictionary
    df_list = merged_df.to_dict(orient='records')
    
    # Initialize position variables
    position_open = False
    holding_period = 0

    for i in range(len(df_list)):
        current_day = df_list[i]
        
        if not position_open:
            if current_day.get('Entry_Signal', False):
                if lmt_order:
                    # Calculate limit price
                    lmt_price = current_day['Close'] - lmt_atr * current_day['ATR']

                    # Determine order days
                    if lmt_day_only:
                        order_days = [i + lmt_day]
                    else:
                        order_days = list(range(i + 1, i + lmt_day + 1))

                    order_filled = False

                    for order_day_index in order_days:
                        if order_day_index >= len(df_list):
                            break  # Out of range
                        order_day = df_list[order_day_index]
                        low = order_day['Low'] + lmt_epsilon * current_day['ATR']
                        high = order_day['High'] - lmt_epsilon * current_day['ATR']
                        if low <= lmt_price <= high:
                            # Order is filled
                            df_list[order_day_index]['Signal'] = 1
                            df_list[order_day_index]['TradeEntry'] = lmt_price
                            df_list[order_day_index]['InTrade'] = risk_ratio
                            df_list[order_day_index]['Holding_Period'] = 1
                            position_open = True
                            holding_period = 1
                            order_filled = True
                            break  # Exit after filled

                    if not order_filled:
                        continue  # Order not filled; wait for next signal

                else:
                    # Enter at next day's Open
                    if i + 1 < len(df_list):
                        next_day = df_list[i + 1]
                        df_list[i + 1]['Signal'] = 1
                        df_list[i + 1]['TradeEntry'] = next_day['Open']
                        df_list[i + 1]['InTrade'] = risk_ratio
                        df_list[i + 1]['Holding_Period'] = 1
                        position_open = True
                        holding_period = 1

        else:
            # Update holding period
            holding_period += 1
            df_list[i]['Holding_Period'] = holding_period
            df_list[i]['InTrade'] = risk_ratio

            # Check exit conditions
            exit_condition = False

            # Condition 1: Today's Close > Yesterday's Close
            if i > 0 and df_list[i]['Close'] > df_list[i - 1]['High']:
                exit_condition = True

            # Condition 2: Holding period >= N
            if holding_period >= N:
                exit_condition = True

            if exit_condition:
                # Exit at next day's Open
                if i + 1 < len(df_list):
                    next_day = df_list[i + 1]
                    df_list[i + 1]['Signal'] = -1
                    df_list[i + 1]['TradeExit'] = next_day['Open']
                    df_list[i + 1]['InTrade'] = risk_ratio
                    df_list[i + 1]['Holding_Period'] = 0
                    position_open = False
                    holding_period = 0

    # Replace None with np.nan for consistency
    for row in df_list:
        row['TradeEntry'] = row['TradeEntry'] if row.get('TradeEntry') is not None else np.nan
        row['TradeExit'] = row['TradeExit'] if row.get('TradeExit') is not None else np.nan

    # Convert updated data back to Pandas DataFrame
    df_updated = pd.DataFrame(df_list)

    df_updated = pl.from_pandas(df_updated)


    return df_updated



def ratioMR_rsi_short(
    df: pl.DataFrame,
    pairs: List[Tuple[str, str]] = [],
    N: Tuple[int, int] = (5, 5),
    rsi_period: int = 2,
    rsi_threshold: float = 90.0,
    lmt_order: bool = False,
    lmt_day: int = 1,
    lmt_day_only: bool = True,
    lmt_atr: float = 1.0,
    lmt_epsilon: float = 0.1,
    atr_period: int = 14,
    atr_type: str = 'atr',
    trend_filter: Tuple[int, int] = None,
    overbought_rsi: Tuple[float, float] = None,
    overbought_cond: float = 10.0,
    corr_cond: Tuple[int,float] = None,
    flip_main: bool = False,
    vol_wgt_ratio: bool = False,
    risk_ratio: Tuple[float, float] = (1.0, 1.0), # risk ratio for sizing.
    market_data: Any = None) -> pl.DataFrame:
    """
    Trading strategy that goes short on the main asset when the RSI of the spread is above a threshold.

    :param df: Polars DataFrame containing data for the main asset.
    :param trading_params: Dictionary of trading parameters, including:
        - 'pairs': List of tuples [(asset_name, comp_name), ...]
        - 'N': Holding period after entry.
        - 'rsi_period': Lookback period for RSI calculation.
        - 'rsi_threshold': RSI threshold for generating buy signals.
        - 'lmt_order': If True, uses limit order logic for entry.
        - 'lmt_day': The day on which the limit order is placed.
        - 'lmt_day_only': If True, limit order is only placed on lmt_day; otherwise, placed every day until lmt_day.
        - 'lmt_atr': Multiplier for ATR to determine limit price.
        - 'lmt_epsilon': Small value to adjust high and low prices for limit order checking.
        - 'atr_period': Lookback period for ATR calculation.
        - 'atr_type': Type of ATR calculation ('atr' or 'sd').
        - 'market_data': Instance of MarketData to fetch comparative asset data.
    :return: Polars DataFrame with trading signals appended for the main asset.
    """
    # Ensure 'Name' column exists
    if 'Name' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Name' column.")

    # Get the main asset's name from the DataFrame
    main_asset_names = df.select(pl.col('Name')).unique().to_series().drop_nulls()
    if main_asset_names.len() == 0:
        raise ValueError("No asset names found in the 'Name' column.")
    main_asset_name = main_asset_names[0]

    # Find if the main asset is in any pair and define asset specific wgts

    comp_name = None
    
    for pair in pairs:
        if pair[0] == main_asset_name:
            comp_name = pair[1]

            if isinstance(N, tuple):
                N = N[0]

            if isinstance(risk_ratio, tuple):
                risk_ratio = risk_ratio[0]
            if isinstance(trend_filter, tuple):
                if isinstance(trend_filter[0], int):
                    trend_filter = trend_filter[0]
            break
        else:
            if isinstance(N, tuple):
                N = N[1]
            if isinstance(risk_ratio, tuple):
                risk_ratio = risk_ratio[1]
            if isinstance(trend_filter, tuple):
                if isinstance(trend_filter[1], int):
                    trend_filter = trend_filter[1]

    if comp_name is None:
        # Main asset not in any pair; return default signals
        df = df.with_columns([
            pl.lit(0).alias('Signal'),
            pl.lit(np.nan).alias('TradeEntry'),
            pl.lit(np.nan).alias('TradeExit'),
            pl.lit(0).alias('InTrade'),
            pl.lit(0).alias('Holding_Period')
        ])
        return df

    # Fetch comparative asset data using market_data
    if market_data is None:
        raise ValueError("market_data must be provided in trading_params.")

    comp_df = market_data.get_ticker_data(comp_name)
    if comp_df is None or comp_df.is_empty():
        # Comparative asset data not found; return default signals
        df = df.with_columns([
            pl.lit(0).alias('Signal'),
            pl.lit(np.nan).alias('TradeEntry'),
            pl.lit(np.nan).alias('TradeExit'),
            pl.lit(0).alias('InTrade'),
            pl.lit(0).alias('Holding_Period')
        ])
        return df
    
    # Ensure both DataFrames are sorted by 'Date'

    df = df.sort('Date')
    comp_df = comp_df.sort('Date')

    if flip_main:
        df_tmp = df.to_pandas().copy()
        df_tmp['Close'] = df['Close'] * -1
        df_tmp['Open'] = df['Open'] * -1
        df_tmp['High'] = df['Low'] * -1
        df_tmp['Low'] = df['High'] * -1
        df = pl.from_pandas(df_tmp)


    '''
    # Merge the two DataFrames on 'Date' using a left join to preserve all main asset rows
    merged_df = df.join(comp_df.select(['Date', 'Close']), on='Date', how='left', suffix="_comp")

    # Optional: Handle missing 'Close_comp' values (e.g., fill forward)
    merged_df = merged_df.with_columns([
        pl.col('Close_comp').fill_null(strategy='forward').alias('Close_comp')
    ])
    '''

    main_pd = df.to_pandas()
    cmp_pd = comp_df.to_pandas()
    
    main_pd.set_index('Date', inplace=True)
    cmp_pd.set_index('Date', inplace=True)

    main_close_df = main_pd[['Close', 'Tick_Value_USD']].rename(
        columns={'Close': 'Main_Close', 'Tick_Value_USD': 'Main_Tick_Value_USD'})
    cmp_close_df = cmp_pd[['Close', 'Tick_Value_USD']].rename(
        columns={'Close': 'Comp_Close', 'Tick_Value_USD': 'Comp_Tick_Value_USD'})

    concat_df = pd.concat([main_close_df, cmp_close_df], axis=1)

    if type(vol_wgt_ratio) == int:
        concat_df['maindiffs'] = concat_df['Main_Close'].diff()
        concat_df['compdiffs'] = concat_df['Comp_Close'].diff()
        concat_df['mainvol'] = concat_df['maindiffs'].rolling(window=vol_wgt_ratio).std()
        concat_df['mainvol'] = concat_df['mainvol'] * concat_df['Main_Tick_Value_USD']
        concat_df['compvol'] = concat_df['compdiffs'].rolling(window=vol_wgt_ratio).std()
        concat_df['compvol'] = concat_df['compvol'] * concat_df['Comp_Tick_Value_USD']        
        voladj = concat_df['mainvol'] / concat_df['compvol']

        concat_df['Main_Close_va'] = concat_df['Main_Close']
        concat_df['Comp_Close_va'] = concat_df['Comp_Close'] * voladj

        concat_df['Ratio'] = concat_df['Main_Close_va'] / concat_df['Comp_Close_va']
    else:
        concat_df['Ratio'] = concat_df['Main_Close'] / concat_df['Comp_Close']
    
    concat_df['RSI_Ratio'] = compute_rsi(concat_df['Ratio'], period=rsi_period)

    merged_df = pd.concat([main_pd, concat_df], axis=1)

    cols_to_drop = ['Main_Close']

    # Calc entry condition dependencies.

    if corr_cond is not None:
        merged_df['maindiffs'] = merged_df['Main_Close'].diff()
        merged_df['compdiffs'] = merged_df['Comp_Close'].diff()
        merged_df['Corr'] = merged_df['maindiffs'].rolling(window=corr_cond[0]).corr(merged_df['compdiffs'])
        merged_df['Corr_Cond'] = merged_df['Corr'] > corr_cond[1]
        cols_to_drop += ['maindiffs', 'compdiffs']
    else:
        merged_df['Corr_Cond'] = True
        cols_to_drop.append('Corr_Cond')


    if trend_filter is not None and type(trend_filter) == int:
        merged_df['MovAvg'] = merged_df['Main_Close'].rolling(window=trend_filter).mean()
        merged_df['Trend_Filter'] = merged_df['Close'] < merged_df['MovAvg']
    else:
        merged_df['Trend_Filter'] = True
        cols_to_drop.append('Trend_Filter')

    
    if overbought_cond is not None and type(overbought_cond) == int:
        merged_df['Overbought_Cond'] = merged_df['Close'].rolling(window=overbought_cond).apply(
            lambda x: all(x[i] > x[i - 1] for i in range(1, len(x))), raw=True
        )
    else:
        merged_df['Overbought_Cond'] = True
        cols_to_drop.append('Overbought_Cond')
    
    if overbought_rsi is not None and type(overbought_rsi) == tuple:
        merged_df['Asst_RSI'] = compute_rsi(merged_df['Close'], period=overbought_rsi[0])
        merged_df['Overbought_RSI'] = merged_df['Asst_RSI'] > overbought_rsi[1]
    else:
        merged_df['Overbought_RSI'] = True
        cols_to_drop.append('Overbought_RSI')


    # Compute yesterday's High for the main asset
    merged_df['Prev_High'] = merged_df['High'].shift(1)
    merged_df['Prev_High_Cond'] = merged_df['Close'] < merged_df['Prev_High']
    cols_to_drop.append('Prev_High_Cond')


    if lmt_order:
        merged_df = compute_effective_atr(merged_df, atr_period, atr_type)

    # Initialize trading signal columns
    merged_df['Signal'] = 0
    merged_df['TradeEntry'] = np.nan
    merged_df['TradeExit'] = np.nan
    merged_df['InTrade'] = 0
    merged_df['Holding_Period'] = 0

    merged_df['Entry_Signal'] = (
        (merged_df['Prev_High_Cond']) &
        (merged_df['RSI_Ratio'] > rsi_threshold) &
        (merged_df['Trend_Filter']) &
        (merged_df['Overbought_Cond']) &
        (merged_df['Overbought_RSI']) &
        (merged_df['Corr_Cond'])
    )

    merged_df.drop(columns=cols_to_drop, inplace=True)
    
    merged_df.reset_index(inplace=True)
    # Convert merged DataFrame to dictionary
    df_list = merged_df.to_dict(orient='records')
    
    # Initialize position variables
    position_open = False
    holding_period = 0

    for i in range(len(df_list)):
        current_day = df_list[i]
        
        if not position_open:
            if current_day.get('Entry_Signal', False):
                if lmt_order:
                    # Calculate limit price
                    lmt_price = current_day['Close'] + lmt_atr * current_day['ATR']

                    # Determine order days
                    if lmt_day_only:
                        order_days = [i + lmt_day]
                    else:
                        order_days = list(range(i + 1, i + lmt_day + 1))

                    order_filled = False

                    for order_day_index in order_days:
                        if order_day_index >= len(df_list):
                            break  # Out of range
                        order_day = df_list[order_day_index]
                        low = order_day['Low'] + lmt_epsilon * current_day['ATR']
                        high = order_day['High'] - lmt_epsilon * current_day['ATR']
                        if low <= lmt_price <= high:
                            # Order is filled
                            df_list[order_day_index]['Signal'] = -1
                            df_list[order_day_index]['TradeEntry'] = lmt_price
                            df_list[order_day_index]['InTrade'] = -1 * risk_ratio
                            df_list[order_day_index]['Holding_Period'] = 1
                            position_open = True
                            holding_period = 1
                            order_filled = True
                            break  # Exit after filled

                    if not order_filled:
                        continue  # Order not filled; wait for next signal

                else:
                    # Enter at next day's Open
                    if i + 1 < len(df_list):
                        next_day = df_list[i + 1]
                        df_list[i + 1]['Signal'] = -1
                        df_list[i + 1]['TradeEntry'] = next_day['Open']
                        df_list[i + 1]['InTrade'] = -1 * risk_ratio
                        df_list[i + 1]['Holding_Period'] = 1
                        position_open = True
                        holding_period = 1

        else:
            # Update holding period
            holding_period += 1
            df_list[i]['Holding_Period'] = holding_period
            df_list[i]['InTrade'] = -1 * risk_ratio

            # Check exit conditions
            exit_condition = False

            # Condition 1: Today's Close > Yesterday's Close
            if i > 0 and df_list[i]['Close'] < df_list[i - 1]['Low']:
                exit_condition = True

            # Condition 2: Holding period >= N
            if holding_period >= N:
                exit_condition = True

            if exit_condition:
                # Exit at next day's Open
                if i + 1 < len(df_list):
                    next_day = df_list[i + 1]
                    df_list[i + 1]['Signal'] = 1
                    df_list[i + 1]['TradeExit'] = next_day['Open']
                    df_list[i + 1]['InTrade'] = -1 * risk_ratio
                    df_list[i + 1]['Holding_Period'] = 0
                    position_open = False
                    holding_period = 0

    # Replace None with np.nan for consistency
    for row in df_list:
        row['TradeEntry'] = row['TradeEntry'] if row.get('TradeEntry') is not None else np.nan
        row['TradeExit'] = row['TradeExit'] if row.get('TradeExit') is not None else np.nan

    # Convert updated data back to Pandas DataFrame
    df_updated = pd.DataFrame(df_list)

    df_updated = pl.from_pandas(df_updated)

    

    return df_updated
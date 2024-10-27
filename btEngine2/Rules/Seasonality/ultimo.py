import pandas as pd
import polars as pl
import numpy as np
from datetime import timedelta
import xgboost as xgb
from scipy.stats import skew

def ultimo_long(
    df,
    days_before_end: int,
    days_after_start: int,
    use_ml_model: tuple = None
) -> pd.DataFrame:
    """
    Trades the 'ultimo effect' by entering a long position 'days_before_end' days before month-end
    and exiting 'days_after_start' days after the new month starts.
    
    Parameters:
    - df: DataFrame containing at least 'Date', 'Open', and 'Close' columns.
    - days_before_end: Number of days before month-end to enter the trade.
    - days_after_start: Number of days after month-start to exit the trade.
    - use_ml_model: Tuple (perf_lb, skew_lb, rsi_per) or None.
    
    Returns:
    - df: DataFrame with 'Signal', 'TradeEntry', 'TradeExit', 'InTrade' columns added.
    """

    # Convert Polars DataFrame to Pandas if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Initialize columns
    df['Signal'] = 0
    df['TradeEntry'] = np.nan
    df['TradeExit'] = np.nan
    df['InTrade'] = 0

    # Generate signals based on days_before_end and days_after_start
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['MonthEnd'] = df.index.to_series().shift(-1).dt.month != df['Month']

    # Find potential trade entry and exit dates
    month_ends = df[df['MonthEnd']].index
    trade_entries = []
    trade_exits = []

    for end_date in month_ends:
        # Entry date is 'days_before_end' business days before month end
        idx_end_date = df.index.get_loc(end_date)
        idx_entry_date = idx_end_date - (days_before_end - 1)
        if idx_entry_date >= 0:
            entry_date = df.index[idx_entry_date]
        else:
            continue
        # Exit date is 'days_after_start' business days after month start
        next_month_start_idx = idx_end_date + 1
        idx_exit_date = next_month_start_idx + days_after_start - 1
        if idx_exit_date < len(df):
            exit_date = df.index[idx_exit_date]
        else:
            continue

        trade_entries.append(entry_date)
        trade_exits.append(exit_date)

    # Set Signals at trade entry and exit dates
    df.loc[trade_entries, 'Signal'] = 1  # +1 for entering trade
    df.loc[trade_exits, 'Signal'] = -1  # -1 for exiting trade

    # Update TradeEntry and TradeExit columns according to Signal
    df['TradeEntry'] = np.where(df['Signal'] == 1, df['Open'], np.nan)
    df['TradeExit'] = np.where(df['Signal'] == -1, df['Close'], np.nan)

    if use_ml_model is not None:
        perf_lb, skew_lb, rsi_per, p_thresh = use_ml_model

        # Prepare dataset for ML
        df_ml = df.copy()
        
        df_ml['Returns'] = df['Close'].pct_change()
        df_ml['SetupDay'] = df_ml['Signal'].shift(1) == 1  # Day before Signal == 1
        df_ml['SkewMetric'] = df_ml['Returns'].rolling(window=skew_lb).apply(lambda x: -skew(x), raw=True)
        df_ml['feat_SkewMetric_EWMA'] = df_ml['SkewMetric'].ewm(span=skew_lb // 4).mean()
        df_ml['feat_RSI'] = compute_rsi(df_ml['Close'], rsi_per)

        # Compute feat_mth_perf and feat_seq_perf
        df_ml['Month'] = df_ml.index.month
        df_ml['Year'] = df_ml.index.year

        # Initialize feature columns
        df_ml['feat_mth_perf'] = np.nan
        df_ml['feat_seq_perf'] = np.nan

        # Calculate features on SetupDay
        setup_days = df_ml[df_ml['SetupDay']].index

        for setup_day in setup_days:
            current_month = df_ml.at[setup_day, 'Month']
            current_year = df_ml.at[setup_day, 'Year']

            # Past performance for the same month over past perf_lb years
            past_mth_returns = []
            for y in range(1, perf_lb + 1):
                past_year = current_year - y
                try:
                    past_setup_day = setup_day.replace(year=past_year)
                    past_entry_date = past_setup_day + pd.DateOffset(days=1)
                    past_exit_date = past_entry_date + pd.DateOffset(days=days_before_end + days_after_start - 2)
                    if past_entry_date in df_ml.index and past_exit_date in df_ml.index:
                        ret = df_ml.loc[past_exit_date, 'Close'] / df_ml.loc[past_entry_date, 'Open'] - 1
                        past_mth_returns.append(ret)
                except ValueError:
                    continue  # Handle leap years and date issues
            if past_mth_returns:
                #feat_mth_perf = np.mean(past_mth_returns)
                feat_mth_perf = pd.Series(past_mth_returns).ewm(span=perf_lb).mean().iloc[-1]
                df_ml.at[setup_day, 'feat_mth_perf'] = feat_mth_perf

            # Past performance over past perf_lb * 12 months
            past_seq_returns = []
            for m in range(1, perf_lb * 12 + 1):
                past_setup_day = setup_day - pd.DateOffset(months=m)
                past_entry_date = past_setup_day + pd.DateOffset(days=1)
                past_exit_date = past_entry_date + pd.DateOffset(days=days_before_end + days_after_start - 2)
                if past_entry_date in df_ml.index and past_exit_date in df_ml.index:
                    ret = df_ml.loc[past_exit_date, 'Close'] / df_ml.loc[past_entry_date, 'Open'] - 1
                    past_seq_returns.append(ret)
            if past_seq_returns:
                #feat_seq_perf = np.mean(past_seq_returns)
                feat_seq_perf = pd.Series(past_seq_returns).ewm(span=perf_lb * 12).mean().iloc[-1]
                df_ml.at[setup_day, 'feat_seq_perf'] = feat_seq_perf

        # Prepare features and target
        features = ['feat_mth_perf', 'feat_seq_perf', 'feat_SkewMetric_EWMA', 'feat_RSI']
        df_ml = df_ml[df_ml['Signal'] == 1]
        df_ml.dropna(subset=features, inplace=True)

        # Target: whether the trade was profitable
        df_ml['TradeResult'] = np.nan
        for idx in df_ml.index:
            # Get entry and exit dates
            entry_idx = df_ml.index.get_loc(idx)
            # Find next exit signal
            exit_idx = None
            for i in range(entry_idx + 1, len(df_ml)):
                if df_ml.iloc[i]['Signal'] == -1:
                    exit_idx = df_ml.index.get_loc(df_ml.iloc[i].name)
                    break
            if exit_idx is not None:
                entry_price = df_ml.iloc[entry_idx]['Open']
                exit_price = df_ml.iloc[exit_idx]['Close']
                trade_return = exit_price / entry_price - 1
                df_ml.at[idx, 'TradeResult'] = trade_return
            else:
                continue  # No exit signal found

        df_ml['TradeProfitable'] = np.where(df_ml['TradeResult'] > 0, 1, 0)

        # Only keep rows where TradeProfitable is not nan
        df_ml = df_ml[df_ml['TradeProfitable'].notna()]

        # Identify the last 24 months
        last_date = df_ml.index.max()
        first_date = last_date - pd.DateOffset(months=24)
        df_ml_last_24 = df_ml.loc[first_date:last_date-1]

        train_df = df_ml_last_24.loc[first_date:last_date-1]
        # Prepare data for XGBoost
        X_train = train_df[features]
        y_train = train_df['TradeProfitable']
        X_all = df_ml[features]

        # Train the model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            reg_lambda=10,  # L2 regularization term to prevent overfitting
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Predict probabilities for all trades
        df_ml['ProbProfitable'] = model.predict_proba(X_all)[:, 1]

        # Update 'InTrade' in the original df
        for idx in df_ml.index:
            prob = df_ml.at[idx, 'ProbProfitable']
            entry_date = idx
            # Find exit date
            entry_idx = df.index.get_loc(entry_date)
            exit_idx = None
            for i in range(entry_idx + 1, len(df)):
                if df.iloc[i]['Signal'] == -1:
                    exit_idx = i
                    break
            if exit_idx is not None:
                # Set InTrade between entry_idx and exit_idx
                if prob > p_thresh:
                    adj_intrade_value = 1 + (prob - 0.5)
                else:
                    adj_intrade_value = 0
                df.iloc[entry_idx:exit_idx + 1, df.columns.get_loc('InTrade')] = adj_intrade_value
            else:
                # No exit signal found
                continue

    else:
        # If ML is not used, set InTrade to 1 when in trade
        in_trade = False
        for idx in df.index:
            if df.at[idx, 'Signal'] == 1:
                in_trade = True
            if in_trade:
                df.at[idx, 'InTrade'] = 1
            if df.at[idx, 'Signal'] == -1:
                in_trade = False

    # Reset index
    df.reset_index(inplace=True)

    df = pl.DataFrame(df)
    return df

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
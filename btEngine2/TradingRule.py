import os
import logging
import polars as pl
import pickle
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go  
import plotly.io as pio
import hashlib
import uuid
import plotly.graph_objects as go  
import plotly.io as pio
import hashlib
from datetime import datetime



from typing import Dict, Any, Optional, List, Tuple, Callable

from scipy.stats import skew, norm

def hash_params_compact(params: Dict[str, Any]) -> str:
    """
    Create a compact hash from a dictionary of parameters for unique identification.
    
    :param params: Dictionary of parameters to be hashed.
    :return: A short string representing the hashed parameters (first 8 characters of the hash).
    """
    param_str = str(sorted(params.items()))
    return hashlib.sha256(param_str.encode()).hexdigest()[:10]

def generate_trade_id_compact(trading_rule_name: str, trading_params: Dict[str, Any], asset: str, trade_number: int) -> str:
    """
    Generate a compact unique trade ID based on trading rule, parameters, asset, and trade number.
    
    :param trading_rule_name: The name of the trading rule function.
    :param trading_params: The parameters of the trading rule function.
    :param asset: The name of the asset.
    :param trade_number: The trade number for this asset.
    :return: A compact unique trade ID string.
    """
    rule_hash = hash_params_compact(trading_params)
    asset_short = asset.replace(" ", "")[:6]  # Shorten asset name to first 6 characters
    return f"{rule_hash}_{asset_short}_{trade_number}"

class TradingRule:
    def __init__(
        self, 
        market_data, 
        trading_rule_function: Callable[[pl.DataFrame, Dict[str, Any]], pl.DataFrame], 
        trading_params: Dict[str, Any], 
        position_sizing_params: Dict[str, Any], 
        sizing_rf: float = 1.0,
        excl_ac: List[str] = [],
        incl_ac: List[str] = [],
        excl_assets: List[str] = [],
        incl_assets: List[str] = [],
        cont_rule: bool = False,  # New parameter for continuous rule
        log_level: int = logging.INFO,  # New parameter for log level
        strat_descr: str = None,
        name_label: str = None,
        assign_id: bool = False,
        bt_folder: str = None
    ):
        # Existing initialization code...
        self.market_data = market_data
        self.trading_rule_function = trading_rule_function
        self.trading_params = trading_params
        self.position_sizing_params = position_sizing_params
        
        if incl_assets != []:
            self.incl_assets = incl_assets
            self.market_data = market_data.get_assets_data(self.incl_assets)
        elif excl_assets != []:
            self.incl_assets = [x for x in self.market_data.data.keys() if x not in excl_assets]
            self.market_data = market_data.get_assets_data(self.incl_assets)
        elif incl_ac != []:
            self.incl_assets = []
            for ac in incl_ac:
                self.incl_assets = self.incl_assets + market_data.get_asset_classes()[ac]
            self.market_data = market_data.get_assets_data(self.incl_assets)
        elif excl_ac != []:
            self.incl_assets = []
            for ac in incl_ac:
                self.incl_assets = self.incl_assets + market_data.get_asset_classes()[ac]
            self.incl_assets = [x for x in self.incl_assets if x not in excl_assets]
            self.market_data = market_data.get_assets_data(self.incl_assets)
        else:
            self.market_data = market_data

        # Get the name of the instance
        self.name_lbl = uuid.uuid4().hex[:6]

        if name_label is not None:
            self.name_lbl = name_label
            assign_id = True

        self.strat_descr = strat_descr
        self.cont_rule = cont_rule  # Store the continuous rule flag

        function_name = self.trading_rule_function.__name__
        self.func_name = function_name
        param_hash = hash_params_compact(self.trading_params)
        param_hash2 = hash_params_compact(self.position_sizing_params)
        if assign_id:
            self.folder_name = f"{function_name}_{param_hash}_{param_hash2}_{self.name_lbl}"
        else:
            self.folder_name = f"{function_name}_{param_hash}_{param_hash2}"
        
        self.pretty_funcparams = '_'.join(f'{key}:{value}' for key, value in trading_params.items())
        self.pretty_sizeparams = '_'.join(f'{key}:{value}' for key, value in position_sizing_params.items())

        # Check if bt_folder is a valid directory
        if bt_folder and os.path.isdir(bt_folder):
            self.backtests_folder = bt_folder
        elif bt_folder and not os.path.isdir(bt_folder):
            self.backtests_folder = os.path.join(os.getcwd(), bt_folder)
            os.makedirs(self.backtests_folder, exist_ok=True)
        else:
            self.backtests_folder = os.path.join(os.getcwd(), "BackTests")
            os.makedirs(self.backtests_folder, exist_ok=True)

        # Setup Logger
        self.logger = self.setup_logging(log_level)
        self.logger.info(f"Initialized TradingRule for function '{function_name}' with params hash '{param_hash}'")

        # Process 'AssetVol' parameter
        asset_vol = position_sizing_params.get('AssetVol', None)
        if asset_vol is None:
            self.logger.error("'AssetVol' must be provided in position_sizing_params.")
            raise ValueError("'AssetVol' must be provided in position_sizing_params.")

        self.asset_vol_dict = {}  # Initialize the asset volatility dictionary

        if isinstance(asset_vol, (int, float)):
            # AssetVol is a number; apply the same volatility to all assets
            for asset in self.market_data.data.keys():
                self.asset_vol_dict[asset] = asset_vol * sizing_rf
            self.logger.info(f"Using scalar AssetVol: {asset_vol} for all assets.")
        elif isinstance(asset_vol, str):
            # AssetVol is a file path; read the file to get asset-specific volatilities
            asset_vol_file = asset_vol
            if not os.path.exists(asset_vol_file):
                self.logger.error(f"AssetVol file not found: {asset_vol_file}")
                raise FileNotFoundError(f"AssetVol file not found: {asset_vol_file}")
            else:
                try:
                    asset_vol_df = pl.read_csv(asset_vol_file)
                    # Assuming the file has columns: 'curr' and 'TgtVol'
                    # Remove commas from 'TgtVol' and convert to float
                    try:
                        asset_vol_df = asset_vol_df.with_columns([
                            pl.col('TgtVol').str.replace_all(',', '').cast(pl.Float64)
                        ])
                    except:
                        asset_vol_df = asset_vol_df.with_columns([
                            pl.col('TgtVol').cast(pl.Float64)
                        ])
                    try:
                        asset_vol_df = asset_vol_df.with_columns([
                            (pl.col('TgtVol') * sizing_rf).alias('TgtVol')
                        ])
                    except:
                        self.logger.error('Check the sizing_rf in TradingRule.')
                        raise ValueError('Check the sizing_rf in TradingRule.')
                    # Create a dictionary mapping 'curr' to 'TgtVol'
                    self.asset_vol_dict = dict(zip(asset_vol_df['Asset'], asset_vol_df['TgtVol']))
                    self.logger.info(f"Loaded AssetVol from file: {asset_vol_file}")
                except Exception as e:
                    self.logger.error(f"Error reading AssetVol file: {e}")
                    raise
        else:
            self.logger.error("'AssetVol' must be either a number or a file path.")
            raise ValueError("'AssetVol' must be either a number or a file path.")

    def setup_logging(self, log_level: int) -> logging.Logger:
        """
        Sets up the logger for the TradingRule class.
        
        :param log_level: Logging verbosity level.
        :return: Configured logger instance.
        """
        logger = logging.getLogger(f"TradingRule_{self.func_name}_{hash_params_compact(self.trading_params)}")
        logger.setLevel(log_level)
        
        # Prevent adding multiple handlers if logger already has handlers
        if not logger.handlers:
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            
            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # Add handlers to the logger
            logger.addHandler(ch)
            folder_name = self.folder_name
            folder = os.path.join(self.backtests_folder, folder_name)
            os.makedirs(folder, exist_ok=True)
            # Optionally, add file handler
            file_path = os.path.join(folder, f"{self.folder_name}.log")
            
            fh = logging.FileHandler(file_path)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger
    
    def const_pos_rule(self, asset: str, vol_target: float) -> pl.DataFrame:
        """
        Create a trading rule that is always in trade with a constant position
        targeting a specific annualized volatility.
        
        :param asset: The name of the asset.
        :param vol_target: The annualized volatility target in USD.
        :return: A Polars DataFrame with the trading signals applied.
        """
        self.logger.debug(f"Creating constant position rule for asset '{asset}' with vol_target={vol_target}")
        
        df = self.market_data.get_ticker_data(asset).clone()
        
        # Ensure required columns are present
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Tick_Value_USD']
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Asset '{asset}' data is missing required columns.")
            raise ValueError(f"Asset '{asset}' data is missing required columns.")
        
        df = df.sort('Date')
        self.logger.debug(f"Sorted data for asset '{asset}' by 'Date'")
        
        # Set 'InTrade' to 1 for the whole series
        df = df.with_columns([
            pl.lit(1).alias('InTrade'),
            pl.col('Close').diff().alias('PriceDiff'),
            (pl.col('PriceDiff') * pl.col('Tick_Value_USD')).alias('USD_PnL_1lot')
        ])
        self.logger.debug("Added 'InTrade', 'PriceDiff', and 'USD_PnL_1lot' columns")
        
        # Calculate rolling standard deviation of USD PnL
        lookback_period = self.position_sizing_params['VolLookBack']
        df = df.with_columns([
            pl.col('USD_PnL_1lot').rolling_std(lookback_period).alias('RollingStdDev'),
            (pl.col('RollingStdDev') * np.sqrt(252)).alias('AnnualizedVol_1lot'),
            (vol_target / pl.col('AnnualizedVol_1lot')).round().fill_null(0).alias('Lots_Target_Vol')
        ])
        self.logger.debug(f"Calculated 'RollingStdDev', 'AnnualizedVol_1lot', and 'Lots_Target_Vol' columns")
        
        # Calculate total PnL for the strategy
        df = df.with_columns([
            (pl.col('USD_PnL_1lot') * pl.col('Lots_Target_Vol')).alias('PositionPnL_USD'),
            (pl.col('PositionPnL_USD') * pl.col('InTrade')).alias('Strategy_PnL_USD')
        ])
        self.logger.debug("Calculated 'PositionPnL_USD' and 'Strategy_PnL_USD' columns")
        
        # Calculate cumulative strategy equity
        df = df.with_columns([
            pl.col('Strategy_PnL_USD').cumsum().alias('Strategy_Equity_USD')
        ])
        self.logger.debug("Calculated 'Strategy_Equity_USD' column")
        
        self.logger.info(f"Created constant position rule for asset '{asset}'")
        
        return df
    
    def apply_rule_to_asset(self, asset: str) -> pl.DataFrame:
        """
        Apply the trading rule to a specific asset.
        
        :param asset: The name of the asset.
        :param save: Boolean indicating whether to save the results.
        :return: A Polars DataFrame with the trading signals applied.
        """
        self.logger.debug(f"Applying trading rule to asset '{asset}'")
        
        if asset not in self.market_data.data:
            self.logger.error(f"Asset '{asset}' not found in the MarketData object.")
            raise ValueError(f"Asset '{asset}' not found in the MarketData object.")
        
        try:
            df = self.market_data.get_ticker_data(asset).clone()
            self.logger.debug(f"Fetched data for asset '{asset}': {df.height} records")
        except Exception as e:
            self.logger.error(f"Failed to fetch data for asset '{asset}': {e}")
            raise
        
        # Apply the trading rule function to the DataFrame
        try:
            df = self.trading_rule_function(df, **self.trading_params)
            self.logger.debug(f"Applied trading rule function '{self.trading_rule_function.__name__}' to asset '{asset}'")
        except Exception as e:
            self.logger.error(f"Error applying trading rule function '{self.trading_rule_function.__name__}' to asset '{asset}': {e}")
            raise
        
        # Apply position sizing
        try:
            df = self.apply_position_sizing(df, asset)
            self.logger.debug(f"Applied position sizing to asset '{asset}'")
        except Exception as e:
            self.logger.error(f"Error applying position sizing to asset '{asset}': {e}")
            raise
        return df
    
    def apply_position_sizing(self, df: pl.DataFrame, asset: str) -> pl.DataFrame:
        """
        Apply position sizing to the DataFrame based on the position sizing parameters.

        :param df: The DataFrame with trading signals.
        :param asset: The asset name.
        :return: The DataFrame with position sizes applied.
        """
        self.logger.debug(f"Applying position sizing for asset '{asset}'")

        # Get the target volatility for the asset
        asset_vol_target = self.asset_vol_dict.get(asset, None)
        if asset_vol_target is None:
            self.logger.error(f"No 'AssetVol' specified for asset '{asset}'.")
            raise ValueError(f"No 'AssetVol' specified for asset '{asset}'.")

        # Check for 'VolLookBack' parameter
        lookback_period = self.position_sizing_params.get('VolLookBack', None)
        if lookback_period is None:
            self.logger.error("Position sizing parameter 'VolLookBack' must be provided.")
            raise ValueError("Position sizing parameter 'VolLookBack' must be provided.")

        # Check if 'VolLookBack' is a tuple for blended volatility
        blended_vol = isinstance(lookback_period, tuple)

        if blended_vol:
            if len(lookback_period) != 3:
                self.logger.error("When 'VolLookBack' is a tuple, it must have exactly three elements: (lb_fast, lb_slow, lb_ratio).")
                raise ValueError("When 'VolLookBack' is a tuple, it must have exactly three elements: (lb_fast, lb_slow, lb_ratio).")
         
            lb_fast, lb_slow, lb_ratio = lookback_period
            
            if not (0 <= lb_ratio <= 1):
                self.logger.error("'lb_ratio' must be between 0 and 1.")
                raise ValueError("'lb_ratio' must be between 0 and 1.")

            self.logger.info(f"Blended volatility: lb_fast={lb_fast}, lb_slow={lb_slow}, lb_ratio={lb_ratio}")
        else:
            lb_fast = lookback_period
            lb_slow = None
            lb_ratio = 1.0
            self.logger.info(f"Single volatility: lb_fast={lb_fast}")

        # Get the method to calc vol.
        vol_method = self.position_sizing_params.get('VolMethod', 'std')


        self.logger.debug(f"AssetVol target for '{asset}': {asset_vol_target}, VolLookBack period: {lookback_period}")

        # Step 1: Create the 'UsedPrice' column after converting to pandas
        if type(df) != pd.DataFrame:
            df = df.to_pandas()
        df['UsedPrice'] = np.where(
            pd.notna(df['TradeEntry']), df['TradeEntry'],  # If there's a TradeEntry, use it
            np.where(pd.notna(df['TradeExit']), df['TradeExit'], df['Close'])  # Otherwise, use TradeExit or Close
        )

        self.logger.debug("Converted to pd and created 'UsedPrice' column")

        # Step 2: Calculate the PriceDiff, but account for cases where both TradeEntry and TradeExit exist on the same row
        df['PriceDiff'] = np.where(
            pd.notna(df['TradeEntry']) & pd.notna(df['TradeExit']),  # If both TradeEntry and TradeExit are not NaN
            df['TradeExit'] - df['TradeEntry'],  # Replace PriceDiff with TradeExit - TradeEntry
            df['UsedPrice'].diff()  # Otherwise, use the standard diff of UsedPrice
        )

        df['PriceDiff'] = np.where(
            pd.notna(df['TradeEntry']),
            df['UsedPrice'] - df['TradeEntry'],
            df['PriceDiff']
        )

        # Step 3: Calculate USD PnL for 1 unit of the asset (PriceDiff * Tick_Value_USD)
        df['USD_PnL_1lot'] = df['PriceDiff'] * df['Tick_Value_USD']
        df['USD_PnL_1lot_clean'] = df['Close'].diff() * df['Tick_Value_USD']

        # Step 4: Calculate the rolling standard deviation of the USD PnL
        if blended_vol:
            if vol_method == 'std':
                df['RollingStdDev'] = df['USD_PnL_1lot_clean'].rolling(window=lb_fast).std() * lb_ratio + df['USD_PnL_1lot_clean'].rolling(window=lb_slow).std() * (1 - lb_ratio)
            elif vol_method == 'ewm':
                df['RollingStdDev'] = df['USD_PnL_1lot_clean'].ewm(span=lb_fast).std() * lb_ratio + df['USD_PnL_1lot_clean'].ewm(span=lb_slow).std() * (1 - lb_ratio)
            else:
                self.logger.error('vol_method must be either "std" or "ewm"')
                raise ValueError('vol_method must be either "std" or "ewm"')
        else:
            if vol_method=='std':
                df['RollingStdDev'] = df['USD_PnL_1lot_clean'].rolling(window=lb_fast).std()
            elif vol_method=='ewm':
                df['RollingStdDev'] = df['USD_PnL_1lot_clean'].ewm(span=lb_fast).std()
            else:
                self.logger.error('vol_method must be either "std" or "ewm"')
                raise ValueError('vol_method must be either "std" or "ewm"')

        del df['USD_PnL_1lot_clean']

        # Step 5: Annualize the rolling standard deviation (assuming 252 trading days per year)
        df['AnnualizedVol_1lot'] = df['RollingStdDev'] * np.sqrt(252)

        # Step 6: Calculate the number of units needed to target the desired AssetVol
        df['Lots_Target_Vol'] = np.where(df['AnnualizedVol_1lot'] == 0, 0, asset_vol_target / df['AnnualizedVol_1lot'])

        # Step 7: Round the number of units to the nearest integer
        df['Lots_Target_Vol'] = df['Lots_Target_Vol'].round()

        # Step 8: Fill any NaNs that might result from division by zero or other issues
        df['Lots_Target_Vol'] = df['Lots_Target_Vol'].fillna(0)

        # Step 9: Calculate the USD Notional exposure
        df['USD_Notional'] = df['Lots_Target_Vol'] * df['UsedPrice'] * df['Tick_Value_USD']

        df['USD_Notional'] = df['USD_Notional'].shift(1)
        df['Lots_Target_Vol'] = df['Lots_Target_Vol'].shift(1)

        # We introduce lookahead bias if we don't use the notionals as of when the trades were done.
        self.logger.debug(f"Completed position sizing for asset '{asset}'")

        # Convert back to Polars DataFrame
        #df = pl.DataFrame(df)

        return df 
    
    def backtest_asset(self, asset: str, save: bool = False) -> pl.DataFrame:
        """
        Backtest the trading rule on a specific asset, supporting both discrete and continuous signals.
        
        :param asset: The name of the asset.
        :param save: Boolean indicating whether to save the results.
        :return: A Polars DataFrame with the backtest results.
        """
        self.logger.info(f"Backtesting asset '{asset}'")

        if asset not in self.market_data.data:
            self.logger.error(f"Asset '{asset}' not found in the MarketData object.")
            raise ValueError(f"Asset '{asset}' not found in the MarketData object.")

        # Step 1: Apply the trading rule to the asset
        try:
            df = self.apply_rule_to_asset(asset)
            
            # Drop any nans before the first 'Close' in the dataframe
            df = df.dropna(subset=['Close'])
            self.logger.debug(f"Applied trading rule to asset '{asset}'")
        except Exception as e:
            self.logger.error(f"Error applying trading rule to asset '{asset}': {e}")
            raise

        # Ensure that Position PnL is calculated
        if 'Lots_Target_Vol' in df.columns and 'USD_PnL_1lot' in df.columns:
            df['PositionPnL_USD'] = df['USD_PnL_1lot'] * df['Lots_Target_Vol']
        else:
            raise KeyError("Required columns 'Lots_Target_Vol' or 'USD_PnL_1lot' are missing from DataFrame.")

        # Calculate the strategy's PnL
        df['Strategy_PnL_USD'] = df['PositionPnL_USD'] * df['InTrade']
        df['Strategy_Equity_USD'] = df['Strategy_PnL_USD'].cumsum()
        df['USD_Notional'] = df['USD_Notional'] * df['InTrade']
        df['Num_Lots'] = df['Lots_Target_Vol'] * df['InTrade']
        df['Num_Lots'] = df['Num_Lots'].round()
        
        del (df['Lots_Target_Vol'])
        # Generate compact Trade_ID for each trade
        df['Trade_ID'] = ""  # Initialize Trade_ID column with NaN

        if not self.cont_rule:
            trade_number = 1
            current_trade_id = None  # To keep track of the current Trade_ID

            for idx in df.index:
                signal = df.at[idx, 'Signal']
                if signal != 0 and current_trade_id is None:
                    # Start of a new trade
                    trade_id = generate_trade_id_compact(self.trading_rule_function.__name__, self.trading_params, asset, trade_number)
                    df.at[idx, 'Trade_ID'] = trade_id
                    current_trade_id = trade_id
                elif signal == 0 and current_trade_id is not None:
                    # Continuing the trade
                    df.at[idx, 'Trade_ID'] = current_trade_id
                elif signal != 0 and current_trade_id is not None:
                    # End of the trade
                    df.at[idx, 'Trade_ID'] = current_trade_id
                    current_trade_id = None
                    trade_number += 1  # Increment trade number for the next trade
                else:
                    # No trade ongoing
                    pass  # Do nothing
        else:
            trade_number = 1  # Initialize trade number

            for idx in df.index:
                if df.at[idx, 'USD_Notional'] != 0:
                    trade_id = generate_trade_id_compact(self.trading_rule_function.__name__, self.trading_params, asset, trade_number)
                    df.at[idx, 'Trade_ID'] = trade_id  # Assign Trade_ID to the row
                    trade_number += 1  # Increment trade number for the next trade


        # Clean up the dataframe for backtest results
        #btdf = df[['Trade_ID','Close', 'UsedPrice', 'Signal', 'TradeEntry', 'TradeExit', 'InTrade', 'USD_Notional', 'Num_Lots',
        #            'PositionPnL_USD', 'Strategy_PnL_USD', 'Strategy_Equity_USD']].copy()
        
        # Step 9: Save the outputs if 'save' is True
        if save:
            try:
                # Use function name and hashed parameters for folder naming
                folder_name = self.folder_name
                # Create the folder under 'BackTests'
                folder = os.path.join(self.backtests_folder, folder_name)
                save_dir = os.path.join(folder, 'backtest_results')
                os.makedirs(save_dir, exist_ok=True)
                self.logger.debug(f"Created directory '{save_dir}' for saving results")

                # Save the DataFrame as Parquet for efficiency
                save_path = os.path.join(save_dir, f"{asset}_backtest.parquet")
                pl.DataFrame(df).write_parquet(save_path)
                self.logger.info(f"Backtest data for asset '{asset}' saved to '{save_path}'")
                self._generate_readme(folder)
            except Exception as e:
                self.logger.error(f"Error saving backtest data for asset '{asset}': {e}")
                raise

        self.logger.info(f"Completed backtesting for asset '{asset}'")

        return df

    def backtest_asset_full(self, asset, save=False):
        """
        Backtest the trading rule on a specific asset, adding a unique 'Trade_ID' column
        and generating a summary of trades grouped by Trade_ID.

        :param asset: The name of the asset (as a key in the MarketData data dictionary).
        :param save: Boolean indicating whether to save the results to CSV.
        :return: A tuple with two DataFrames:
                - The detailed backtest results with a 'Trade_ID' column.
                - A summary DataFrame grouped by 'Trade_ID' with the specified columns.
        """
        df = self.backtest_asset(asset, save)  # Get the detailed backtest results from the original function
        df.set_index('Date', inplace=True)
        # Initialize a list to store trade summary rows
        trade_summary = []
        assetclass = self.market_data.get_ac(asset)
        # Group by Trade_ID and create summary for each trade
        grouped = df.groupby('Trade_ID')

        for trade_id, group in grouped:
            if trade_id:  # Exclude any rows where 'Trade_ID' might be missing
                first_row = group.iloc[0]
                last_row = group.iloc[-1]
                first_valid_trade_entry = group['TradeEntry'].dropna().iloc[0] if not group['TradeEntry'].dropna().empty else np.nan
                last_valid_trade_exit = group['TradeExit'].dropna().iloc[-1] if not group['TradeExit'].dropna().empty else np.nan
                
                summary = {
                    'Trade_ID': trade_id,
                    'AssetClass': assetclass,
                    'Asset': first_row['Name'],
                    'FirstDate': group.index[0],
                    'LastDate': group.index[-1],
                    'TradeEntry': first_valid_trade_entry,
                    'TradeExit': last_valid_trade_exit,
                    'USD_Notional': first_row['USD_Notional'],
                    'Num_Lots': first_row['Num_Lots'],
                    'Total PnL': group['PositionPnL_USD'].sum()
                }
                if 'Descr' in first_row:
                    summary['Descr'] = first_row['Descr']
                                
                # Add any elements that begin with "mlfeat_" to the summary
                for col in group.columns:
                    if col.startswith("mlfeat_"):
                        summary[col] = first_row[col]


                trade_summary.append(summary)
                
                # Reorder columns to make 'Descr' the first column if it exists
                if 'Descr' in summary:
                    cols = ['Descr'] + [col for col in summary.keys() if col != 'Descr']
                    summary = {col: summary[col] for col in cols}
        # Create a summary DataFrame from the list of summaries
        trade_summary_df = pd.DataFrame(trade_summary)
        if 'Descr' in trade_summary_df.columns:
            cols = ['Trade_ID', 'Descr'] + [col for col in trade_summary_df.columns if col not in ['Trade_ID', 'Descr']]
            trade_summary_df = trade_summary_df[cols]

        try:
            trade_summary_df = trade_summary_df.sort_values('FirstDate')
        except:
            trade_summary_df = pd.DataFrame(columns=trade_summary_df.columns, index=trade_summary_df.index)
        # Optionally save the summary DataFrame
        if save:


            try:
                # Use function name and hashed parameters for folder naming
                folder_name = self.folder_name

                # Create the folder under 'BackTests'
                folder = os.path.join(self.backtests_folder, folder_name)
                save_dir = os.path.join(folder, 'backtest_results_detail')
                os.makedirs(save_dir, exist_ok=True)
                self.logger.debug(f"Created directory '{save_dir}' for saving results")

                # Save the DataFrame as Parquet for efficiency
                save_path = os.path.join(save_dir, f"{asset}_backtest_trades.parquet")
                pl.DataFrame(trade_summary_df).write_parquet(save_path)
                self.logger.info(f"Backtest data for asset '{asset}' saved to '{save_path}'")
                self._generate_readme(folder)
            except Exception as e:
                self.logger.error(f"Error saving backtest data for asset '{asset}': {e}")
                raise

        self.logger.info(f"Completed backtesting for asset '{asset}'")

        return df, trade_summary_df
    
    def backtest_all_assets(self, save: bool = False):
        """
        Run backtests on all assets in the market data, and save results.

        :param save: Boolean indicating whether to save the results.
        :return: A tuple of two dictionaries:
            - The first dictionary has asset names as keys, and pnldf DataFrames as values.
            - The second dictionary has asset names as keys, and trade summary DataFrames as values.
        """
        self.logger.info("Starting backtest on all assets.")
        pnldf_dict = {}
        trade_summary_dict = {}

        for asset in self.market_data.data.keys():
            try:
                self.logger.info(f"Backtesting asset '{asset}'")
                df_pnl, df_summary = self.backtest_asset_full(asset, save=save)
                pnldf_dict[asset] = df_pnl
                trade_summary_dict[asset] = df_summary

                self.logger.info(f"Completed backtesting for asset '{asset}'")
            except Exception as e:
                self.logger.error(f"Error backtesting asset '{asset}': {e}")
                continue  # Continue with next asset

        # Save the dictionaries to pickle files
        if save:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            self.logger.debug(f"Created directory '{save_dir}' for saving results")
            '''
            # Save the dictionaries to pickle files
            pnldf_save_path = os.path.join(save_dir, 'pnldf_dict.pkl')
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            trade_summary_save_path = os.path.join(save_dir, 'trade_summary_dict.pkl')
            try:
                with open(pnldf_save_path, 'wb') as f:
                    pickle.dump(pnldf_dict, f)
                self.logger.info(f"Saved pnldf_dict to '{pnldf_save_path}'")

                with open(trade_summary_save_path, 'wb') as f:
                    pickle.dump(trade_summary_dict, f)
                self.logger.info(f"Saved trade_summary_dict to '{trade_summary_save_path}'")
            except Exception as e:
                self.logger.error(f"Error saving pickle files: {e}")
                raise
            '''
            full_trades_list = self._full_tradeslist()
            self.tradeslist = full_trades_list
            full_trades_list.to_csv(os.path.join(folder, f'TradesList_{self.folder_name}.csv'), index=True)
            self._generate_readme(folder)

        self.logger.info("Completed backtest on all assets.")

        return pnldf_dict, trade_summary_dict
    
    def bt_res_ts(self, save: bool = False):
        """
        Generate a DataFrame with overall backtest results across all assets.

        The DataFrame will have the following columns:
        - Date
        - Number of Active Positions
        - Daily PnL
        - Strategy_Equity_USD

        :param save: Boolean indicating whether to save the result to a file.
        :return: A pandas DataFrame with the combined backtest results.
        """
        self.logger.info("Generating overall backtest time series.")
        
        # Step 1: Run backtest_all_assets to get pnldf_dict
        self.logger.info("Running backtest on all assets.")
        pnldf_dict, _ = self.backtest_all_assets(save=save)
        
        # Step 2: Initialize a list to collect DataFrames
        list_of_dfs = []

        for asset, df_pnl in pnldf_dict.items():
            # Ensure df_pnl is a pandas DataFrame
            if not isinstance(df_pnl, pd.DataFrame):
                df_pnl = df_pnl.to_pandas()

            # Check if 'Date' is in columns or index
            if 'Date' in df_pnl.columns:
                df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
                df_pnl.set_index('Date', inplace=True)
            else:
                df_pnl.index = pd.to_datetime(df_pnl.index)
            
            # Keep only the necessary columns
            df_asset = df_pnl[['Strategy_PnL_USD', 'InTrade']].copy()
            # Rename columns to include asset name
            df_asset.rename(columns={
                'Strategy_PnL_USD': f'Strategy_PnL_USD_{asset}',
                'InTrade': f'InTrade_{asset}'
            }, inplace=True)
            list_of_dfs.append(df_asset)

        # Step 3: Combine all DataFrames on Date index
        self.logger.info("Combining asset DataFrames.")
        combined_df = pd.concat(list_of_dfs, axis=1).fillna(0)

        # Step 4: Compute the required columns
        self.logger.info("Calculating Number of Active Positions, Daily PnL, and Strategy_Equity_USD.")
        # Get list of InTrade and Strategy_PnL_USD columns
        in_trade_cols = [col for col in combined_df.columns if col.startswith('InTrade_')]
        pnl_cols = [col for col in combined_df.columns if col.startswith('Strategy_PnL_USD_')]

        # Number of Active Positions
        combined_df['Number of Active Positions'] = combined_df[in_trade_cols].astype(bool).sum(axis=1)
        # Daily PnL
        combined_df['Daily PnL'] = combined_df[pnl_cols].sum(axis=1)
        # Strategy_Equity_USD
        combined_df['Strategy_Equity_USD'] = combined_df['Daily PnL'].cumsum()

        # Step 5: Prepare the result DataFrame
        result_df = combined_df[['Number of Active Positions', 'Daily PnL', 'Strategy_Equity_USD']].copy()
        result_df.reset_index(inplace=True)
        result_df.rename(columns={'index': 'Date'}, inplace=True)

        # Optional: Save the result to a file if save=True
        if save:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            self.logger.debug(f"Created directory '{save_dir}' for saving results")

            # Save the DataFrame as CSV
            save_path = os.path.join(save_dir, f'{folder_name}_backtest_timeseries.csv')
            result_df.to_csv(save_path, index=False)
            self.logger.info(f"Overall backtest results saved to '{save_path}'")
            self._generate_readme(folder)
        self.logger.info("Overall backtest time series generation completed.")
        return result_df

    def cum_pnl_byac(self, diffs: bool = False, filter_assets: List[str] = [],
                     excl_assets: List[str] = [], forcebt = False):
        """
        Returns a pandas DataFrame with Date index and columns as asset class names, containing
        the cumulative PnL time series of each asset class. Adds a 'Total' column that is the
        sum of the other columns.

        :param diffs: If True, loads the 'Strategy_PnL_USD' column (daily PnL).
                    If False, loads the 'Strategy_Equity_USD' column (cumulative PnL).
        :param filter_assets: List of assets to include. If empty, includes all assets.
        :return: Pandas DataFrame with Date index and asset class columns.
        """

        if forcebt:
            self.backtest_all_assets(save=True)
        
        # Determine the list of assets to check
        if filter_assets:
            assets_to_check = filter_assets
        else:
            assets_to_check = list(self.market_data.data.keys())

        assets_to_check = [asset for asset in assets_to_check if asset not in excl_assets]
        # Check for missing Parquet files
        missing_assets = []
        for asset in assets_to_check:
            if not self._parquet_file_exists(asset):
                missing_assets.append(asset)

        # If there are missing assets, run backtests for them
        if missing_assets:
            self.logger.info(f"Backtest data missing for assets: {missing_assets}. Running backtest_asset_full.")
            for asset in missing_assets:
                try:
                    self.backtest_asset_full(asset, save=True)
                except Exception as e:
                    self.logger.error(f"Error backtesting asset '{asset}': {e}")
                    continue  # Continue with next asset

        # Load backtest results
        self.logger.info("Loading backtest results from saved parquet files.")
        pnl_df = self._load_backtest_results(diffs=diffs, filter_assets=assets_to_check)

        if pnl_df.empty:
            self.logger.error("No backtest data available to compute cumulative PnL by asset class.")
            return pd.DataFrame()

        # Get asset classes mapping
        asset_classes = self.market_data.get_asset_classes()

        # Reverse mapping from asset to asset class
        asset_to_ac = {}
        for ac_name, assets in asset_classes.items():
            for asset in assets:
                asset_to_ac[asset] = ac_name

        # Filter assets present in pnl_df
        assets_in_pnl = set(pnl_df.columns)

        # Create a mapping of assets in pnl_df to their asset classes
        asset_to_ac_filtered = {asset: asset_to_ac.get(asset, 'Unknown') for asset in assets_in_pnl}

        # Group the assets by asset class
        ac_groups = {}
        for asset, ac in asset_to_ac_filtered.items():
            ac_groups.setdefault(ac, []).append(asset)

        # Compute the PnL per asset class
        ac_pnl = {}
        for ac_name, assets in ac_groups.items():
            # Sum the PnL of assets in the same asset class
            ac_pnl[ac_name] = pnl_df[assets].sum(axis=1)

        # Create DataFrame from ac_pnl dictionary
        ac_pnl_df = pd.DataFrame(ac_pnl).sort_index().fillna(0)

        # If diffs is True, and the data is daily PnL, we need to cumsum to get cumulative PnL
        if diffs:
            ac_pnl_df = ac_pnl_df.cumsum()

        # Add 'Total' column as sum of asset class columns
        ac_pnl_df['Total'] = ac_pnl_df.sum(axis=1)

        self.logger.info("Generated cumulative PnL DataFrame by asset class.")

        return ac_pnl_df

    def cum_pnl_byassets(self, diffs: bool = False, filter_assets: List[str] = [],
                         excl_assets: List[str] = [], forcebt=False):
        """
        Returns a pandas DataFrame with Date index and columns as asset names, containing
        the cumulative PnL time series of each asset. Adds a 'Total' column that is the
        sum of the other columns.

        :param diffs: If True, loads the 'Strategy_PnL_USD' column (daily PnL).
                    If False, loads the 'Strategy_Equity_USD' column (cumulative PnL).
        :param filter_assets: List of assets to include. If empty, includes all assets.
        :return: Pandas DataFrame with Date index and asset columns.
        """

        if forcebt:
            self.backtest_all_assets(save=True)
        
        # Determine the list of assets to check
        if filter_assets:
            assets_to_check = filter_assets
        else:
            assets_to_check = list(self.market_data.data.keys())

        assets_to_check = [asset for asset in assets_to_check if asset not in excl_assets]

        # Check for missing Parquet files
        missing_assets = []
        for asset in assets_to_check:
            if not self._parquet_file_exists(asset):
                missing_assets.append(asset)

        # If there are missing assets, run backtests for them
        if missing_assets:
            self.logger.info(f"Backtest data missing for assets: {missing_assets}. Running backtest_asset_full.")
            for asset in missing_assets:
                try:
                    self.backtest_asset_full(asset, save=True)
                except Exception as e:
                    self.logger.error(f"Error backtesting asset '{asset}': {e}")
                    continue  # Continue with next asset

        # Load backtest results
        self.logger.info("Loading backtest results from saved parquet files.")
        pnl_df = self._load_backtest_results(diffs=diffs, filter_assets=assets_to_check)

        if pnl_df.empty:
            self.logger.error("No backtest data available to compute cumulative PnL by assets.")
            return pd.DataFrame()

        # If diffs is True, the data represents daily PnL and needs to be cumsummed
        if diffs:
            pnl_df = pnl_df.cumsum()

        # Add 'Total' column as sum of asset columns
        pnl_df['Total'] = pnl_df.sum(axis=1)

        self.logger.info("Generated cumulative PnL DataFrame by assets.")
        return pnl_df

    def plot_equity(self, byac: bool = False, byassets: bool = False, totalsys: bool = True,
                    filter_assets: List[str] = [], filter_ac: List[str] = [],
                    excl_ac: List[str] = [], excl_assets: List[str] = [],
                    start_date: str = '', end_date: str = '',
                    save_fig: bool = False, file_format: str = 'html',
                    naming = 'descr'):
        """
        Plots the Strategy_Equity_USD using Plotly and optionally saves the figure.

        :param byac: If True, plots equity curves by asset classes.
        :param byassets: If True, plots equity curves by assets.
        :param totalsys: If True, includes the plot of the total system PnL.
        :param filter_assets: List of assets to include. If empty, includes all assets.
        :param filter_ac: List of asset classes to include. If empty, includes all asset classes.
        :param start_date: Start date in 'DDMMYYYY' format. If provided, rebases the equity curve from this date.
        :param end_date: End date in 'DDMMYYYY' format. If provided, ends the equity curve on this date.
        :param save_fig: If True, saves the figure to a file.
        :param file_format: The format to save the figure ('html' or 'png').
        :return: The DataFrame containing the equity data.
        """
        import pandas as pd
        import plotly.graph_objects as go
        from datetime import datetime
        import os

        # Handle date parsing and error checking
        try:
            if start_date:
                start_date_parsed = datetime.strptime(start_date, '%d%m%Y')
            else:
                start_date_parsed = None
            if end_date:
                end_date_parsed = datetime.strptime(end_date, '%d%m%Y')
            else:
                end_date_parsed = None
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return

        if excl_ac:
            asset_classes = self.market_data.get_asset_classes()
            for ac in excl_ac:
                assets_in_ac = asset_classes.get(ac, [])
                excl_assets.extend(assets_in_ac)
            excl_assets = list(set(excl_assets))
        
        # Load cumulative PnL data
        if byac:
            # If filtering by asset classes, get assets belonging to those classes
            if filter_ac:
                asset_classes = self.market_data.get_asset_classes()
                filter_assets = []
                for ac in filter_ac:
                    assets_in_ac = asset_classes.get(ac, [])
                    filter_assets.extend(assets_in_ac)
            pnl_df = self.cum_pnl_byac(diffs=False, filter_assets=filter_assets, excl_assets=excl_assets)
        elif byassets:
            # Get cumulative PnL by assets
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets, excl_assets=excl_assets)
        else:
            # Total system PnL
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets, excl_assets=excl_assets)
            pnl_df = pnl_df[['Total']]

        if pnl_df.empty:
            self.logger.error("No data available to plot.")
            return

        # Apply date filters
        if start_date_parsed:
            pnl_df = pnl_df[pnl_df.index >= start_date_parsed]
        if end_date_parsed:
            pnl_df = pnl_df[pnl_df.index <= end_date_parsed]

        if pnl_df.empty:
            self.logger.error("No data available after applying date filters.")
            return

        # Rebase equity curves from start date
        if start_date:
               pnl_df.iloc[0] = pnl_df.iloc[0].fillna(0)
               pnl_df = pnl_df - pnl_df.iloc[0]
        # Create Plotly figure
        fig = go.Figure()

        # Plot each column
        for column in pnl_df.columns:
            if column == 'Total' and not totalsys:
                continue  # Skip 'Total' if totalsys is False
            fig.add_trace(go.Scatter(
                x=pnl_df.index,
                y=pnl_df[column],
                mode='lines',
                name=column
            ))

        plt_tit = self.folder_name
        if naming == 'params':
            plt_tit = f'{self.func_name} with params:{self.pretty_funcparams}'
        elif naming == 'descr':
            if self.strat_descr is None:
                plt_tit = f'{self.func_name} with params:{self.pretty_funcparams}'
            else:
                plt_tit = self.strat_descr
        
        # Update layout
        fig.update_layout(
            title=f'{plt_tit} Equity Curves',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL (USD)',
            hovermode='x unified'
        )

        # Show the plot
        fig.show()

        # Save the figure if requested
        if save_fig:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            self.logger.debug(f"Created directory '{save_dir}' for saving results")

            # Construct the file name
            current_time = datetime.now().strftime('%d%m%Y-%H%M%S')
            
            file_name = f'{folder_name}_equity_curve_{current_time}.{file_format.lower()}'
            save_path = os.path.join(save_dir, file_name)
            # Save the figure
            if file_format.lower() == 'html':
                fig.write_html(save_path)
                self.logger.info(f"Figure saved as '{save_path}'")
            elif file_format.lower() == 'png':
                # Save as static PNG image
                # You may need to install the 'kaleido' package for this to work
                try:
                    pio.write_image(fig, save_path, scale=1)
                    self.logger.info(f"Figure saved as '{save_path}'")
                except ValueError as e:
                    self.logger.error(f"Error saving figure as PNG: {e}")
                    self.logger.info("Ensure that 'kaleido' is installed (pip install kaleido).")
            elif file_format.lower() == 'svg':
                # Save as static PNG image
                # You may need to install the 'kaleido' package for this to work
                try:
                    fig.write_image(save_path, scale=1)
                    self.logger.info(f"Figure saved as '{save_path}'")
                except ValueError as e:
                    self.logger.error(f"Error saving figure as SVG: {e}")
                    self.logger.info("Ensure that 'kaleido' is installed (pip install kaleido).")
            else:
                self.logger.error("Invalid file format. Choose 'html' or 'png' or 'svg'.")

        return pnl_df

    def calculate_statistics(self, byac: bool = False, byassets: bool = False, totalsys: bool = True,
                            filter_assets: List[str] = [], filter_ac: List[str] = [],
                            start_date: str = '', end_date: str = '',
                            by_actual_trade: bool = True, save: bool = False) -> pd.DataFrame:


        # Handle date parsing and error checking
        try:
            if start_date:
                start_date_parsed = datetime.strptime(start_date, '%d%m%Y')
            else:
                start_date_parsed = None
            if end_date:
                end_date_parsed = datetime.strptime(end_date, '%d%m%Y')
            else:
                end_date_parsed = None
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()

        # Load daily PnL data using diffs=True
        if byac:
            if filter_ac:
                asset_classes = self.market_data.get_asset_classes()
                filter_assets = []
                for ac in filter_ac:
                    assets_in_ac = asset_classes.get(ac, [])
                    filter_assets.extend(assets_in_ac)
            pnl_df = self.cum_pnl_byac(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
        elif byassets:
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
        else:
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
            pnl_df = pnl_df[['Total']]

        if pnl_df.empty:
            self.logger.error("No data available to calculate statistics.")
            return pd.DataFrame()

        # Apply date filters
        if start_date_parsed:
            pnl_df = pnl_df[pnl_df.index >= start_date_parsed]
        if end_date_parsed:
            pnl_df = pnl_df[pnl_df.index <= end_date_parsed]

        # Find the first valid value in 'Total' column
        first_valid_index = pnl_df['Total'].ne(0).idxmax()
        #first_valid_index = pnl_df['Total'].first_valid_index()

        # Drop rows before the first valid value
        if first_valid_index is not None:
            pnl_df = pnl_df.loc[first_valid_index:]

        if pnl_df.empty:
            self.logger.error("No data available after applying date filters.")
            return pd.DataFrame()

        # Store pnl_df for use in trade statistics
        self.pnl_df = pnl_df

        # Initialize a DataFrame to hold statistics
        stats_df = pd.DataFrame()

        # Calculate statistics for each column
        for column in pnl_df.columns:
            if column == 'Total' and not totalsys:
                continue

            daily_pnl = pnl_df[column]
            if daily_pnl.empty:
                continue

            # Calculate statistics
            stats = self._calculate_column_statistics(daily_pnl, column_name=column, by_actual_trade=by_actual_trade)
            
            # Convert stats dict to DataFrame
            stats_row = pd.DataFrame([stats])
            
            # Concatenate to stats_df
            stats_df = pd.concat([stats_df, stats_row], ignore_index=True)

        # Set the index to the column names
        stats_df.set_index('Column', inplace=True)
        result_df = stats_df.copy()

        if save:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            
            # Add 2 empty rows and then a row with the input settings
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)

            settings_row = {
                'byac': byac,
                'byassets': byassets,
                'totalsys': totalsys,
                'filter_assets': filter_assets,
                'filter_ac': filter_ac,
                'start_date': start_date,
                'end_date': end_date
            }
            settings_names = ['byac', 'byassets', 'totalsys', 'filter_assets', 'filter_ac', 'start_date', 'end_date']
            settings_values = [settings_row[name] for name in settings_names]
            result_df.loc[len(result_df)] = ['Settings: '] + settings_names + [np.nan] * (len(result_df.columns)-len(settings_names)-1)
            result_df.loc[len(result_df)] = ['Settings: '] + settings_values + [np.nan] * (len(result_df.columns)-len(settings_values)-1)


            # Save the DataFrame as CSV
            current_time = datetime.now().strftime('%d%m%Y_%H%M')
            result_df.loc[len(result_df)] = [current_time] + [np.nan] * (len(result_df.columns)-1)            

            # Relabel any index less than 1000 to ""
            result_df.index = result_df.index.map(lambda x: "" if type(x) is int else x)
            
            save_path = os.path.join(save_dir, f'StratMetrics_{folder_name}_{current_time}.csv')
            result_df.T.to_csv(save_path, index=True)
            self.logger.info(f"Strategy stats saved to '{save_path}'")
            self._generate_readme(folder)
        self.logger.info("Strategy stats calculation complete.")
        # Return the final DataFrame

        return stats_df

    def perf_table(
        self,
        byac: bool = False,
        byassets: bool = True,
        filter_assets: List[str] = [],
        filter_ac: List[str] = [],
        start_date: str = '',
        end_date: str = '',
        metric: str = 'pnl',
        period: str = 'y',  # New parameter
        save: bool = False,
        table_detail: str = 'full'
    ) -> pd.DataFrame:
        """
        Calculate performance metrics based on the specified detail level and period.

        :param byac: Aggregate PnL by Asset Class.
        :param byassets: Aggregate PnL by Assets.
        :param filter_assets: List of assets to include.
        :param filter_ac: List of asset classes to include.
        :param start_date: Start date in '%d%m%Y' format.
        :param end_date: End date in '%d%m%Y' format.
        :param metric: Performance metric to calculate ('pnl', 'sharpe', 'hr', 'maxdd', 'skew').
        :param period: Time period granularity ('m' for month, 'q' for quarter, 'y' for year).
        :param save: Whether to save the statistics to a CSV file.
        :param table_detail: Detail level of the output table ('full', 'assets', 'ac').
        :return: DataFrame with performance metrics.
        """

        metric = metric.lower()
        period = period.lower()
        table_detail = table_detail.lower()

        # Validate parameters
        if table_detail not in ['full', 'assets', 'ac']:
            self.logger.error("Parameter 'table_detail' must be one of ['full', 'assets', 'ac'].")
            raise ValueError("Parameter 'table_detail' must be one of ['full', 'assets', 'ac'].")

        if metric not in ['pnl', 'sharpe', 'hr', 'maxdd', 'skew']:
            self.logger.error("Parameter 'metric' must be one of ['pnl', 'sharpe', 'hr', 'maxdd', 'skew'].")
            raise ValueError("Parameter 'metric' must be one of ['pnl', 'sharpe', 'hr', 'maxdd', 'skew'].")

        if period not in ['m', 'q', 'y']:
            self.logger.error("Parameter 'period' must be one of ['m', 'q', 'y'].")
            raise ValueError("Parameter 'period' must be one of ['m', 'q', 'y'].")

        # Handle date parsing and error checking
        try:
            if start_date:
                start_date_parsed = datetime.strptime(start_date, '%d%m%Y')
            else:
                start_date_parsed = None
            if end_date:
                end_date_parsed = datetime.strptime(end_date, '%d%m%Y')
            else:
                end_date_parsed = None
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()

        # Load daily PnL data using diffs=True
        if byac:
            if filter_ac:
                asset_classes = self.market_data.get_asset_classes()
                filter_assets = []
                for ac in filter_ac:
                    assets_in_ac = asset_classes.get(ac, [])
                    filter_assets.extend(assets_in_ac)
            pnl_df = self.cum_pnl_byac(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
        elif byassets:
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
        else:
            pnl_df = self.cum_pnl_byassets(diffs=False, filter_assets=filter_assets)
            pnl_df = pnl_df.diff()
            pnl_df = pnl_df[['Total']]

        if pnl_df.empty:
            self.logger.error("No data available to calculate statistics.")
            return pd.DataFrame()

        # Apply date filters
        if start_date_parsed:
            pnl_df = pnl_df[pnl_df.index >= start_date_parsed]
        if end_date_parsed:
            pnl_df = pnl_df[pnl_df.index <= end_date_parsed]

        if pnl_df.empty:
            self.logger.error("No data available after applying date filters.")
            return pd.DataFrame()

        # Store pnl_df for use in trade statistics
        self.pnl_df = pnl_df

        # Prepare the data for aggregation
        pnl_df.index = pd.to_datetime(pnl_df.index)
        pnl_long = pnl_df.reset_index().melt(id_vars='Date', var_name='Asset', value_name='PnL')

        # Extract Period based on the 'period' parameter
        if period == 'y':
            pnl_long['Period'] = pnl_long['Date'].dt.year.astype(str)
        elif period == 'q':
            pnl_long['Year'] = pnl_long['Date'].dt.year.astype(str)
            pnl_long['Quarter'] = pnl_long['Date'].dt.quarter.astype(str)
            pnl_long['Period'] = pnl_long['Year'] + ' Q' + pnl_long['Quarter']
        elif period == 'm':
            pnl_long['Year'] = pnl_long['Date'].dt.year.astype(str)
            pnl_long['Month'] = pnl_long['Date'].dt.month.astype(str).str.zfill(2)
            pnl_long['Period'] = pnl_long['Year'] + '-' + pnl_long['Month']


        # Map assets to asset classes if needed
        if table_detail in ['full', 'ac']:
            asset_classes = self.market_data.get_asset_classes()
            asset_to_ac = {}
            for ac, assets in asset_classes.items():
                for asset in assets:
                    asset_to_ac[asset] = ac
            pnl_long['AssetClass'] = pnl_long['Asset'].map(asset_to_ac)
            pnl_long['AssetClass'] = pnl_long['AssetClass'].fillna('Unknown')

        # Define group keys based on table_detail
        if table_detail == 'full':
            if not byassets:
                self.logger.error("Parameter 'table_detail=full' requires byassets=True.")
                raise ValueError("Parameter 'table_detail=full' requires byassets=True.")
            group_keys = ['AssetClass', 'Asset', 'Period']
        elif table_detail == 'assets':
            if not byassets:
                self.logger.error("Parameter 'table_detail=assets' requires byassets=True.")
                raise ValueError("Parameter 'table_detail=assets' requires byassets=True.")
            group_keys = ['Asset', 'Period']
        elif table_detail == 'ac':
            if not byac:
                self.logger.error("Parameter 'table_detail=ac' requires byac=True.")
                raise ValueError("Parameter 'table_detail=ac' requires byac=True.")
            group_keys = ['AssetClass', 'Period']

        # Aggregate the data based on the metric
        def aggregate_metric(df, metric_column):
            if metric == 'pnl':
                # Sum of PnL
                result = df.groupby(group_keys)[metric_column].sum().reset_index(name='Metric')
            elif metric == 'sharpe':
                # Sharpe ratio
                def sharpe(x):
                    mean = x.mean()
                    std = x.std()
                    if std == 0 or np.isnan(std):
                        return np.nan
                    return (mean / std) * np.sqrt(252)
                result = df.groupby(group_keys)[metric_column].apply(sharpe).reset_index(name='Metric')
            elif metric == 'hr':
                # Hit rate
                def hit_rate(x):
                    return (x > 0).sum() / (x != 0).sum() if len(x!=0) > 0 else np.nan
                
                result = df.groupby(group_keys)[metric_column].apply(hit_rate).reset_index(name='Metric')
            elif metric == 'maxdd':
                # Max drawdown
                def max_drawdown(x):
                    cumulative = x.cumsum()
                    max_cum = np.maximum.accumulate(cumulative)
                    drawdown = cumulative - max_cum
                    return drawdown.min()
                result = df.groupby(group_keys)[metric_column].apply(max_drawdown).reset_index(name='Metric')
            elif metric == 'skew':
                # Skewness
                result = df.groupby(group_keys)[metric_column].apply(lambda x: skew(x)).reset_index(name='Metric')
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            return result

        # Perform aggregation
        grouped = aggregate_metric(pnl_long, 'PnL')

        # Pivot the table to have Periods as columns
        if table_detail == 'full':
            index_cols = ['AssetClass', 'Asset']
        elif table_detail == 'assets':
            index_cols = ['Asset']
        elif table_detail == 'ac':
            index_cols = ['AssetClass']

        pivot_table = grouped.pivot_table(index=index_cols, columns='Period', values='Metric', aggfunc='first')

        # Optional: Sort the index and columns
        pivot_table = pivot_table.sort_index()
        pivot_table = pivot_table.sort_index(axis=1, ascending=False)

        result_df = pivot_table

        if metric=='pnl':
            result_df['Total'] = result_df.sum(axis=1)
        else:
            result_df['Total'] = result_df.mean(axis=1)
        

        # Optional: Save the DataFrame as CSV
        if save:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)

            # Add 2 empty rows and then a row with the input settings
            result_df = result_df.copy()
            for _ in range(3):
                empty_row = pd.Series([np.nan] * len(result_df.columns), index=result_df.columns, name='')
                result_df = pd.concat([result_df, empty_row])

            settings_row = {
                'byac': byac,
                'byassets': byassets,
                'filter_assets': filter_assets,
                'filter_ac': filter_ac,
                'start_date': start_date,
                'end_date': end_date,
                'metric': metric,
                'period': period,
                'table_detail': table_detail
            }
            settings_names = list(settings_row.keys())
            settings_values = list(settings_row.values())
            settings_series = pd.Series(['Settings: '] + settings_names + [np.nan] * (len(result_df.columns) - len(settings_names) - 1),
                                        index=result_df.columns, name='')
            result_df = pd.concat([result_df, settings_series], ignore_index=False)

            settings_values_series = pd.Series(['Settings: '] + settings_values + [np.nan] * (len(result_df.columns) - len(settings_values) - 1),
                                            index=result_df.columns, name='')
            result_df = pd.concat([result_df, settings_values_series], ignore_index=False)

            # Save the DataFrame as CSV
            current_time = datetime.now().strftime('%d%m%Y_%H%M')
            timestamp_series = pd.Series([current_time] + [np.nan] * (len(result_df.columns) - 1),
                                        index=result_df.columns, name='')
            result_df = pd.concat([result_df, timestamp_series], ignore_index=False)

            # Save path
            save_path = os.path.join(save_dir, f'PerfTable_{metric}_{folder_name}_{current_time}.csv')
            result_df.T.to_csv(save_path, index=True)
            self.logger.info(f"Performance {metric} saved to '{save_path}'")
            self._generate_readme(folder)

        self.logger.info(f"Performance {metric} calculation complete.")

        return result_df

    def strat_monthly_pnl(self, byac: bool = False, byassets: bool = False, totalsys: bool = True,
                            filter_assets: List[str] = [], filter_ac: List[str] = [],
                            start_date: str = '', end_date: str = '',
                            save: bool = False, name: str = None) -> pd.DataFrame:
        import pandas as pd
        import numpy as np
        from datetime import datetime

        # Handle date parsing and error checking
        try:
            if start_date:
                start_date_parsed = datetime.strptime(start_date, '%d%m%Y')
            else:
                start_date_parsed = None
            if end_date:
                end_date_parsed = datetime.strptime(end_date, '%d%m%Y')
            else:
                end_date_parsed = None
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()

        # Load daily PnL data using diffs=True
        if byac:
            if filter_ac:
                asset_classes = self.market_data.get_asset_classes()
                filter_assets = []
                for ac in filter_ac:
                    assets_in_ac = asset_classes.get(ac, [])
                    filter_assets.extend(assets_in_ac)
            pnl_df = self.cum_pnl_byac(diffs=True, filter_assets=filter_assets)
        elif byassets:
            pnl_df = self.cum_pnl_byassets(diffs=True, filter_assets=filter_assets)
        else:
            pnl_df = self.cum_pnl_byassets(diffs=True, filter_assets=filter_assets)
            pnl_df = pnl_df[['Total']]

        if pnl_df.empty:
            self.logger.error("No data available to calculate statistics.")
            return pd.DataFrame()

        # Apply date filters
        if start_date_parsed:
            pnl_df = pnl_df[pnl_df.index >= start_date_parsed]
        if end_date_parsed:
            pnl_df = pnl_df[pnl_df.index <= end_date_parsed]

        if pnl_df.empty:
            self.logger.error("No data available after applying date filters.")
            return pd.DataFrame()

        # Store pnl_df for use in trade statistics
        self.pnl_df = pnl_df

        # Ensure 'Total' column exists
        if 'Total' not in pnl_df.columns:
            self.logger.error("'Total' column not found in PnL DataFrame.")
            return pd.DataFrame()

        # Start of the modified code

        # Step 1: Prepare the DataFrame for grouping
        pnl_df = pnl_df.copy()
        pnl_df['Year'] = pnl_df.index.year
        pnl_df['Month'] = pnl_df.index.month

        # Step 2: Group by Year and Month and sum the PnL
        monthly_pnl_grouped = pnl_df.groupby(['Year', 'Month'])['Total'].sum()

        # Step 3: Pivot the table to have Years as indices and Months as columns
        monthly_pnl_pivot = monthly_pnl_grouped.unstack(level='Month')

        # Ensure months from 1 to 12 are present as columns
        all_months = range(1, 13)
        monthly_pnl_pivot = monthly_pnl_pivot.reindex(columns=all_months, fill_value=0)

        # Step 4: Initialize a list to hold statistics
        stats_list = []

        # Step 5: Calculate statistics for each year
        for year in monthly_pnl_pivot.index:
            # Filter daily PnL data for the year
            daily_pnl_year = pnl_df[pnl_df['Year'] == year]['Total']
            total_days = len(daily_pnl_year)
            if total_days == 0:
                continue

            # Annualization factor (assume 252 trading days)
            annual_factor = np.sqrt(252)

            # Annual PnL
            annual_pnl = daily_pnl_year.sum()

            # Daily mean and std deviation
            daily_mean = daily_pnl_year.mean()
            daily_std = daily_pnl_year.std()

            # Handle case when daily_std is zero
            if daily_std == 0:
                sharpe_ratio = np.nan
            else:
                sharpe_ratio = (daily_mean / daily_std) * annual_factor

            # Annual realized volatility
            annual_vol = daily_std * np.sqrt(252)

            # Hit rate
            positive_days = (daily_pnl_year > 0).sum()
            non_zero_days = (daily_pnl_year != 0).sum()
            hit_rate = positive_days / non_zero_days if non_zero_days > 0 else np.nan
            


            # Profit factor
            total_positive_pnl = daily_pnl_year[daily_pnl_year > 0].sum()
            total_negative_pnl = daily_pnl_year[daily_pnl_year < 0].sum()
            if total_negative_pnl == 0:
                profit_factor = np.nan
            else:
                profit_factor = total_positive_pnl / -total_negative_pnl

            # Worst drawdown
            cumulative_pnl = daily_pnl_year.cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = cumulative_pnl - running_max
            worst_drawdown = drawdown.min()

            # Store the statistics in a dictionary
            stats = {
                'Year': year,
                'Annual_PnL': annual_pnl,
                'Annual_Vol': annual_vol,
                'Worst_Drawdown': worst_drawdown,
                'Sharpe_Ratio': sharpe_ratio,
                'Hit_Rate': hit_rate,
                'Profit_Factor': profit_factor,
            }

            stats_list.append(stats)

        # Step 6: Create a DataFrame from the stats list
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('Year', inplace=True)

        # Step 7: Merge the statistics DataFrame with the monthly PnL pivot table
        result_df = monthly_pnl_pivot.merge(stats_df, left_index=True, right_index=True, how='left')

        # Optional: Sort columns (Months from 1 to 12, followed by statistics)
        month_cols = list(range(1, 13))
        stats_cols = ['Annual_PnL', 'Sharpe_Ratio', 'Annual_Vol', 'Hit_Rate', 'Profit_Factor', 'Worst_Drawdown']
        result_df = result_df[month_cols + stats_cols]
        output_df = result_df.copy()

        if save:
            # Use function name and hashed parameters for folder naming
            folder_name = self.folder_name

            # Create the folder under 'BackTests'
            folder = os.path.join(self.backtests_folder, folder_name)
            save_dir = folder
            os.makedirs(save_dir, exist_ok=True)
            
            # Add 2 empty rows and then a row with the input settings
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)
            result_df.loc[len(result_df)] = [np.nan] * len(result_df.columns)

            settings_row = {
                'byac': byac,
                'byassets': byassets,
                'totalsys': totalsys,
                'filter_assets': filter_assets,
                'filter_ac': filter_ac,
                'start_date': start_date,
                'end_date': end_date
            }
            settings_names = ['byac', 'byassets', 'totalsys', 'filter_assets', 'filter_ac', 'start_date', 'end_date']
            settings_values = [settings_row[name] for name in settings_names]
            result_df.loc[len(result_df)] = ['Settings: '] + settings_names + [np.nan] * (len(result_df.columns)-len(settings_names)-1)
            result_df.loc[len(result_df)] = ['Settings: '] + settings_values + [np.nan] * (len(result_df.columns)-len(settings_values)-1)


            # Save the DataFrame as CSV
            current_time = datetime.now().strftime('%d%m%Y_%H%M')
            result_df.loc[len(result_df)] = [current_time] + [np.nan] * (len(result_df.columns)-1)            
            # Relabel any index less than 1000 to ""
            result_df.index = result_df.index.map(lambda x: "" if x < 1800 else x)
            if name is None:
                name = f'MonthlyPnL_{folder_name}_{current_time}.csv'
            save_path = os.path.join(save_dir, name)
            result_df.to_csv(save_path, index=True)
            self.logger.info(f"Monthly PnLs saved to '{save_path}'")
            self._generate_readme(folder)
        self.logger.info("Monthly PnL calculation complete.")
        # Return the final DataFrame

        return output_df

    #Stats Helper Functions

    def _calculate_column_statistics(self, daily_pnl: pd.Series, column_name: str, by_actual_trade: bool) -> Dict[str, Any]:
        """
        Calculates statistics for a single PnL column.

        :param daily_pnl: Pandas Series containing daily PnL values.
        :param column_name: Name of the column (asset or 'Total').
        :param by_actual_trade: If True, uses actual trades; if False, approximates trades.
        :return: Dictionary containing the calculated statistics.
        """
        import pandas as pd
        import numpy as np

        daily_pnl = daily_pnl.dropna()
        if daily_pnl.empty:
            self.logger.warning(f"No PnL data available for {column_name}.")
            return {}

        # Store daily_pnl for use in trade statistics (if needed)
        self.pnl_df = pd.DataFrame({column_name: daily_pnl})

        # Total PnL
        total_pnl = daily_pnl.sum()

        # Number of days
        num_days = daily_pnl.shape[0]

        # Number of years
        num_years = num_days / 252  # Assuming 252 trading days per year

        # Average Annual PnL
        avg_annual_pnl = (daily_pnl.mean()) * 252

        # Average Annual Volatility
        avg_annual_vol = (daily_pnl.std()) * np.sqrt(252)

        # Sharpe Ratio (annualized)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_pnl)

        # Sortino Ratio (annualized)
        sortino_ratio = self._calculate_sortino_ratio(daily_pnl)

        # Max Drawdown and related statistics
        cumulative_pnl = daily_pnl.cumsum()
        drawdown_info = self._calculate_drawdowns(cumulative_pnl)

        # Exposure Time (assuming every day is 'in trade' for time series data)
        exposure_time = num_days

        # Trade statistics
        asset = None if column_name == 'Total' else column_name
        trade_stats = self._calculate_trade_statistics(column_name=column_name, asset=asset, by_actual_trade=by_actual_trade)

        # Skewness and Tail Ratios
        skew_and_tails = self._calculate_pnl_skew_and_tails(daily_pnl)

        # Compile all statistics
        stats = {
            'Column': column_name,
            'Total PnL': total_pnl,
            'Average Ann. PnL': avg_annual_pnl,
            'Average Ann. Vol': avg_annual_vol,
            'Sharpe Ratio (ann.)': sharpe_ratio,
            'Sortino Ratio (ann.)': sortino_ratio,
            'Max drawdown': drawdown_info['Max Drawdown'],
            'Average drawdown': drawdown_info['Average Drawdown'],
            'Max drawdown Duration': drawdown_info['Max Drawdown Duration'],
            'Avg Drawdown Duration': drawdown_info['Average Drawdown Duration'],
            'Total Number of Drawdowns': drawdown_info['Total Number of Drawdowns'],
            'Drawdowns per year': drawdown_info['Total Number of Drawdowns'] / num_years,
            'HWM PnL': drawdown_info['High Water Mark'],
            'Exposure Time': exposure_time,
        }

        # Merge trade statistics
        stats.update(trade_stats)

        # Merge skewness and tail ratios
        stats.update(skew_and_tails)

        return stats

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the annualized Sharpe Ratio of a return series.

        :param returns: Daily returns series.
        :param risk_free_rate: Risk-free rate, default is 0.0.
        :return: Annualized Sharpe Ratio.
        """
        import numpy as np

        excess_returns = returns - risk_free_rate / 252
        avg_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()
        if std_excess_return == 0:
            return np.nan
        daily_sharpe = avg_excess_return / std_excess_return
        annual_sharpe = daily_sharpe * np.sqrt(252)
        return annual_sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the annualized Sortino Ratio of a return series.

        :param returns: Daily returns series.
        :param risk_free_rate: Risk-free rate, default is 0.0.
        :return: Annualized Sortino Ratio.
        """
        import numpy as np

        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        std_negative = negative_returns.std()
        if std_negative == 0:
            return np.nan
        avg_excess_return = excess_returns.mean()
        daily_sortino = avg_excess_return / std_negative
        annual_sortino = daily_sortino * np.sqrt(252)
        return annual_sortino

    def _calculate_drawdowns(self, cumulative_pnl: pd.Series) -> Dict[str, Any]:
        """
        Calculates drawdown statistics from a cumulative PnL series.

        :param cumulative_pnl: Cumulative PnL series.
        :return: Dictionary containing drawdown statistics.
        """
        import pandas as pd
        import numpy as np

        hwm = cumulative_pnl.cummax()
        drawdowns = hwm - cumulative_pnl

        # Max Drawdown
        max_drawdown = drawdowns.max()

        # Drawdown Durations
        drawdown_durations = []
        duration = 0
        for dd in drawdowns:
            if dd > 0:
                duration += 1
            else:
                if duration > 0:
                    drawdown_durations.append(duration)
                duration = 0
        if duration > 0:
            drawdown_durations.append(duration)

        # Average Drawdown Duration
        avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0

        # Total Number of Drawdowns
        total_drawdowns = len(drawdown_durations)

        # Average Drawdown
        average_drawdown = drawdowns[drawdowns > 0].mean()

        # Max Drawdown Duration
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

        return {
            'Max Drawdown': max_drawdown,
            'Average Drawdown': average_drawdown,
            'Max Drawdown Duration': max_drawdown_duration,
            'Average Drawdown Duration': avg_drawdown_duration,
            'Total Number of Drawdowns': total_drawdowns,
            'High Water Mark': hwm.max(),
        }

    def _calculate_trade_statistics(self, column_name: str, asset: str = None, by_actual_trade: bool = True) -> Dict[str, Any]:
        """
        Calculates trade statistics using actual trade data from tradeslist or approximates trades.

        :param column_name: Name of the column ('Total' or asset name).
        :param asset: Asset name (if applicable).
        :param by_actual_trade: If True, uses actual trades; if False, approximates trades.
        :return: Dictionary containing trade statistics.
        """
        import pandas as pd
        import numpy as np

        if not by_actual_trade or self.cont_rule:
            # Use approximate trade statistics
            self.logger.info("Approximating trade statistics.")
            daily_pnl = self.pnl_df[column_name]
            trade_stats = self._approximate_trade_statistics(daily_pnl)
            return trade_stats
        self.logger.info("Calculating trade statistics using actual trades.")
        # Proceed with actual trade statistics
        # Check if self.tradeslist exists
        if hasattr(self, 'tradeslist'):
            trades_df = self.tradeslist.copy()
        else:
            # Load trades from parquet files
            self.logger.info("Loading trades from backtest_results_detail folder.")
            trades_df = self._load_tradeslist()

        if trades_df.empty:
            self.logger.error("No trade data available to calculate trade statistics.")
            return {}

        # Filter by asset if applicable
        if asset and asset != 'Total':
            trades_df = trades_df[trades_df['Asset'] == asset]
            if trades_df.empty:
                self.logger.warning(f"No trades found for asset '{asset}'.")
                return {}
        else:
            # For 'Total', use all trades
            pass

        # Number of trades
        num_trades = len(trades_df)

        # Total PnL
        total_pnl = trades_df['Total PnL'].sum()

        # Number of winning and losing trades
        num_wins = (trades_df['Total PnL'] > 0).sum()
        num_losses = (trades_df['Total PnL'] < 0).sum()

        # Hit Rate (%)
        hit_rate = (num_wins / num_trades) * 100 if num_trades > 0 else np.nan

        # Average Win and Loss
        avg_win = trades_df.loc[trades_df['Total PnL'] > 0, 'Total PnL'].mean()
        avg_loss = trades_df.loc[trades_df['Total PnL'] < 0, 'Total PnL'].mean()

        # Win/Loss Ratio
        win_loss_ratio = avg_win / -avg_loss if avg_loss != 0 else np.nan

        # Total PnL Winners and Losers
        total_pnl_winners = trades_df.loc[trades_df['Total PnL'] > 0, 'Total PnL'].sum()
        total_pnl_losers = trades_df.loc[trades_df['Total PnL'] < 0, 'Total PnL'].sum()

        # Profit Factor
        profit_factor = total_pnl_winners / -total_pnl_losers if total_pnl_losers != 0 else np.nan

        # PnL Skew (calculated separately in skewness functions)
        pnl_skew = trades_df['Total PnL'].skew()

        # Trade Expected Value
        trade_ev = trades_df['Total PnL'].mean()

        # Best and Worst Trades
        best_trade = trades_df['Total PnL'].max()
        worst_trade = trades_df['Total PnL'].min()

        # Percentiles
        percentile_25 = trades_df['Total PnL'].quantile(0.25)
        percentile_50 = trades_df['Total PnL'].quantile(0.5)
        percentile_75 = trades_df['Total PnL'].quantile(0.75)

        # Pareto Number (%)
        sorted_pnl = trades_df['Total PnL'].sort_values(ascending=False)
        cumulative_pnl = sorted_pnl.cumsum()
        total_pnl_cumulative = cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0
        if total_pnl_cumulative != 0:
            pareto_threshold = 0.8 * total_pnl_cumulative
            pareto_trades = cumulative_pnl[cumulative_pnl <= pareto_threshold]
            pareto_num = (len(pareto_trades) / num_trades) * 100 if num_trades > 0 else np.nan
        else:
            pareto_num = np.nan

        # Compile statistics
        stats = {
            'Num_Trades': num_trades,
            'Hit Rate (%)': hit_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'W/L ratio': win_loss_ratio,
            'Total Wins': num_wins,
            'Total Losses': num_losses,
            'Total PnL Winners': total_pnl_winners,
            'Total PnL Losers': total_pnl_losers,
            'Profit Factor': profit_factor,
            'PnL Skew': pnl_skew,
            'Trade EV': trade_ev,
            'Best Trade': best_trade,
            'Worst Trade': worst_trade,
            '25p Trade': percentile_25,
            '50p Trade': percentile_50,
            '75p Trade': percentile_75,
            'Pareto Number (%)': pareto_num,
        }

        return stats

    def _approximate_trade_statistics(self, daily_pnl: pd.Series) -> Dict[str, Any]:
        """
        Approximates trade statistics from daily PnL series for continuous trading rules.

        :param daily_pnl: Daily PnL series.
        :return: Dictionary containing trade statistics.
        """
        # Identify trades based on sign changes in daily PnL
        pnl_sign = daily_pnl.apply(lambda x: 'win' if x > 0 else ('loss' if x < 0 else 'zero'))
        trade_segments = pnl_sign.ne(pnl_sign.shift()).cumsum()

        trade_pnl = daily_pnl.groupby(trade_segments).sum()
        trade_types = pnl_sign.groupby(trade_segments).first()

        # Filter out zero PnL trades
        trade_pnl = trade_pnl[trade_types != 'zero']
        trade_types = trade_types[trade_types != 'zero']

        num_trades = len(trade_pnl)
        num_wins = (trade_pnl > 0).sum()
        num_losses = (trade_pnl < 0).sum()
        hit_rate = num_wins / num_trades * 100 if num_trades > 0 else np.nan

        avg_win = trade_pnl[trade_pnl > 0].mean()
        avg_loss = trade_pnl[trade_pnl < 0].mean()
        win_loss_ratio = avg_win / -avg_loss if avg_loss != 0 else np.nan

        total_pnl_winners = trade_pnl[trade_pnl > 0].sum()
        total_pnl_losers = trade_pnl[trade_pnl < 0].sum()
        profit_factor = total_pnl_winners / -total_pnl_losers if total_pnl_losers != 0 else np.nan

        pnl_skew = trade_pnl.skew()
        trade_ev = trade_pnl.mean()

        best_trade = trade_pnl.max()
        worst_trade = trade_pnl.min()
        percentile_25 = trade_pnl.quantile(0.25)
        percentile_50 = trade_pnl.quantile(0.5)
        percentile_75 = trade_pnl.quantile(0.75)

        # Pareto Number (%)
        cumulative_pnl = trade_pnl.cumsum()
        total_pnl = cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0
        if total_pnl != 0:
            pareto_pnl = cumulative_pnl[cumulative_pnl <= 0.8 * total_pnl]
            pareto_num = len(pareto_pnl) / num_trades * 100 if num_trades > 0 else np.nan
        else:
            pareto_num = np.nan

        return {
            'Num_Trades': num_trades,
            'Hit Rate (%)': hit_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'W/L ratio': win_loss_ratio,
            'Total Wins': num_wins,
            'Total Losses': num_losses,
            'Total PnL Winners': total_pnl_winners,
            'Total PnL Losers': total_pnl_losers,
            'Profit Factor': profit_factor,
            'PnL Skew': pnl_skew,
            'Trade EV': trade_ev,
            'Best Trade': best_trade,
            'Worst Trade': worst_trade,
            '25p Trade': percentile_25,
            '50p Trade': percentile_50,
            '75p Trade': percentile_75,
            'Pareto Number (%)': pareto_num,
        }

    def _calculate_pnl_skew_and_tails(self, daily_pnl: pd.Series) -> Dict[str, Any]:
        """
        Calculates skewness and tail ratios for different time frames.

        :param daily_pnl: Daily PnL series.
        :return: Dictionary containing skewness and tail ratios.
        """


        # Define periods in trading days
        periods = {
            'Daily': 1,
            'Weekly': 5,
            'Monthly': 21,
            'Quarterly': 63,
            'Yearly': 252
        }

        skew_dict = {}
        tail_dict = {}

        for period_name, period_days in periods.items():
            if period_days == 1:
                period_pnl = daily_pnl
            else:
                # Resample to business days and sum over the period
                period_pnl = daily_pnl.resample('B').sum().rolling(window=period_days).sum().dropna()

            if period_pnl.empty:
                continue

            # Skewness
            pnl_skew = skew(period_pnl)
            skew_dict[f'{period_name} PnL Skew'] = pnl_skew

            # Left and Right Tail Ratios
            left_tail, right_tail = self._calculate_tail_ratios(period_pnl)
            tail_dict[f'{period_name} Left Tail'] = left_tail
            tail_dict[f'{period_name} Right Tail'] = right_tail


        # Combine the dictionaries
        skew_and_tails = {**skew_dict, **tail_dict}

        return skew_and_tails

    def _calculate_tail_ratios(self, pnl_series: pd.Series):
        """
        Calculates the left tail and right tail ratios.

        :param pnl_series: PnL series for a given period.
        :return: Tuple containing (left_tail_ratio, right_tail_ratio).
        """
        # Demean the PnL time series
        pnl_demeaned = pnl_series - pnl_series.mean()

        # Calculate percentiles
        p1 = np.percentile(pnl_demeaned, 1)
        p30 = np.percentile(pnl_demeaned, 30)
        p70 = np.percentile(pnl_demeaned, 70)
        p99 = np.percentile(pnl_demeaned, 99)
        


        # Avoid division by zero
        if p30 == 0 or p70 == 0:
            left_p_ratio = np.nan
            right_p_ratio = np.nan
        else:
            left_p_ratio = p1 / p30
            right_p_ratio = p99 / p70

        # Calculate Gaussian expected value for the ratios
        gauss_left_ratio = norm.ppf(0.01) / norm.ppf(0.30)
        gauss_right_ratio = norm.ppf(0.99) / norm.ppf(0.70)

        # Left and Right Tail Ratios
        left_tail = left_p_ratio / gauss_left_ratio if gauss_left_ratio != 0 else np.nan
        right_tail = right_p_ratio / gauss_right_ratio if gauss_right_ratio != 0 else np.nan

        return left_tail, right_tail

    # Other Helper Functions

    def _generate_readme(self, folder: str):
        """
        Generates a README.txt file in the specified folder containing trading rule information.

        :param folder: The path to the folder where the README.txt file will be created.
        """
        readme_path = os.path.join(folder, 'README.txt')


            # Use function name and hashed parameters for folder naming
        function_name = self.trading_rule_function.__name__
        readme_content = f'{self.name_lbl} Backtest \n'
        if self.strat_descr is not None:
            readme_content += f'{self.strat_descr}\n\n'

        # Prepare the content for the README file
        readme_content += f"\nTrading Rule Function: {function_name}\n\n"

        # Include Trading Rule Parameters
        readme_content += "Trading Rule Parameters:\n"
        for key, value in self.trading_params.items():
            readme_content += f"  {key}: {value}\n"

        # Include Position Sizing Parameters
        readme_content += "\nPosition Sizing Parameters:\n"
        for key, value in self.position_sizing_params.items():
            readme_content += f"  {key}: {value}\n"
        

        # Include Market Data Assets and Start Dates
        readme_content += "\nMarket Data Assets and Start Dates:\n"
        for asset, df in self.market_data.data.items():
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            vol_tgt = self.asset_vol_dict.get(asset, None)
            readme_content += f"  {asset} ({vol_tgt}): {start_date} - {end_date}\n"
        # Write the README file 
        with open(readme_path, 'w') as readme_file:
            readme_file.write(readme_content)

        # Log only if the README file is created
        self.logger.info(f"README.txt created at {readme_path}")

    def _get_parquet_file_path(self, asset: str, detail: bool = False) -> str:
        """
        Constructs the expected Parquet file path for the given asset.

        :param asset: The asset name.
        :param detail: If True, constructs the path for detailed backtest results.
        :return: The full path to the Parquet file.
        """

        # Use function name and hashed parameters for folder naming
        folder_name = self.folder_name

        # Construct the folder path
        folder = os.path.join(self.backtests_folder, folder_name)
        if detail:
            save_dir = os.path.join(folder, 'backtest_results_detail')
            filename = f"{asset}_backtest_trades.parquet"
        else:
            save_dir = os.path.join(folder, 'backtest_results')
            filename = f"{asset}_backtest.parquet"

        # Construct the file path
        file_path = os.path.join(save_dir, filename)

        return file_path

    def _parquet_file_exists(self, asset: str, detail: bool = False) -> bool:
        """
        Checks if the Parquet file for the given asset exists.

        :param asset: The asset name.
        :param detail: If True, checks for detailed backtest results file.
        :return: True if the Parquet file exists, False otherwise.
        """
        file_path = self._get_parquet_file_path(asset, detail)
        return os.path.exists(file_path)

    def _load_tradeslist(self) -> pd.DataFrame:
        """
        Loads the tradeslist from parquet files in the backtest_results_detail folder.

        :return: Pandas DataFrame containing all trades.
        """
        # Use function name and hashed parameters for folder naming
        folder_name = self.folder_name

        # Construct the folder path
        folder = os.path.join(self.backtests_folder, folder_name)
        trades_dir = os.path.join(folder, 'backtest_results_detail')

        # Check if the directory exists
        if not os.path.exists(trades_dir):
            self.logger.error(f"Trades results directory '{trades_dir}' not found.")
            return pd.DataFrame()

        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(trades_dir, '*_backtest_trades.parquet'))

        if not parquet_files:
            self.logger.error(f"No parquet files found in '{trades_dir}'.")
            return pd.DataFrame()

        # Initialize a list to hold DataFrames
        df_list = []

        for file_path in parquet_files:
            # Read the parquet file using polars
            df_pl = pl.read_parquet(file_path)
            # Convert to pandas DataFrame
            df_pd = df_pl.to_pandas()
            df_list.append(df_pd)

        if not df_list:
            self.logger.error("No trade data loaded. Check if the assets exist or if the files are correct.")
            return pd.DataFrame()

        # Concatenate the DataFrames vertically
        trades_df = pd.concat(df_list, axis=0, ignore_index=True)

        # Ensure 'Trade_ID' is set as index
        if 'Trade_ID' in trades_df.columns:
            trades_df.set_index('Trade_ID', inplace=True)

        # Store tradeslist for future use
        self.tradeslist = trades_df

        return trades_df    

    def _full_tradeslist(self, filter_assets: List[str] = []):
        """
        Loads and combines trade details from the backtest_results_detail subfolder.

        :param filter_assets: List of assets to include. If empty, includes all assets.
        :return: Pandas DataFrame with Trade_ID as the index, sorted by FirstDate.
        """
        # Use function name and hashed parameters for folder naming
        folder_name = self.folder_name

        # Construct the folder path
        folder = os.path.join(self.backtests_folder, folder_name)
        trades_dir = os.path.join(folder, 'backtest_results_detail')
        
        # Check if the directory exists
        if not os.path.exists(trades_dir):
            self.logger.error(f"Trades results directory '{trades_dir}' not found.")
            raise FileNotFoundError(f"Trades results directory '{trades_dir}' not found.")

        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(trades_dir, '*_backtest_trades.parquet'))

        if not parquet_files:
            self.logger.error(f"No parquet files found in '{trades_dir}'.")
            raise FileNotFoundError(f"No parquet files found in '{trades_dir}'.")

        # Initialize a list to hold DataFrames
        df_list = []

        for file_path in parquet_files:
            # Extract the asset name from the file name
            asset_name = os.path.basename(file_path).replace('_backtest_trades.parquet', '')

            # If filter_assets is not empty, skip assets not in filter_assets
            if filter_assets and asset_name not in filter_assets:
                continue

            # Read the parquet file using polars
            df_pl = pl.read_parquet(file_path)
            # Convert to pandas DataFrame
            df_pd = df_pl.to_pandas()
            # Set 'Trade_ID' as index
            if 'Trade_ID' in df_pd.columns:
                df_pd.set_index('Trade_ID', inplace=True)
            else:
                self.logger.warning(f"'Trade_ID' column not found in file '{file_path}'. Skipping asset '{asset_name}'.")
                continue

            # Add an 'Asset' column to identify the asset
            df_pd['Asset'] = asset_name

            df_list.append(df_pd)

        if not df_list:
            self.logger.error("No trade data loaded. Check if the assets exist or if the files are correct.")
            return pd.DataFrame()

        # Concatenate the DataFrames vertically
        trades_df = pd.concat(df_list, axis=0)

        # Convert 'FirstDate' to datetime for sorting
        if 'FirstDate' in trades_df.columns:
            trades_df['FirstDate'] = pd.to_datetime(trades_df['FirstDate'])
            trades_df.sort_values('FirstDate', inplace=True)
        else:
            self.logger.warning("'FirstDate' column not found in trades data.")

        return trades_df

    def _load_backtest_results(self, diffs: bool = False, filter_assets: List[str] = [],
                               excl_assets: List[str] = []):
        """
        Helper method to load backtest results from parquet files in the backtest_results folder.

        :param diffs: If True, loads the 'Strategy_PnL_USD' column (daily PnL).
                    If False, loads the 'Strategy_Equity_USD' column (cumulative PnL).
        :param filter_assets: List of assets to load. If empty, loads all assets.
        :return: Pandas DataFrame with Date index and asset columns.
        """

        # Use function name and hashed parameters for folder naming
        folder_name = self.folder_name

        # Construct the folder path
        folder = os.path.join(self.backtests_folder, folder_name)
        save_dir = os.path.join(folder, 'backtest_results')

        # Check if the directory exists
        if not os.path.exists(save_dir):
            self.logger.error(f"Backtest results directory '{save_dir}' not found.")
            raise FileNotFoundError(f"Backtest results directory '{save_dir}' not found.")

        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(save_dir, '*_backtest.parquet'))

        if not parquet_files:
            self.logger.error(f"No parquet files found in '{save_dir}'.")
            raise FileNotFoundError(f"No parquet files found in '{save_dir}'.")

        # Initialize a list to hold DataFrames
        df_list = []

        # Determine which column to load
        column_name = 'Strategy_PnL_USD' if diffs else 'Strategy_Equity_USD'

        # Process each parquet file
        for file_path in parquet_files:
            # Extract the asset name from the file name
            asset_name = os.path.basename(file_path).replace('_backtest.parquet', '')

            # If filter_assets is not empty, skip assets not in filter_assets
            if filter_assets and asset_name not in filter_assets and asset_name not in excl_assets:
                continue
            
            # Read the parquet file using polars
            df_pl = pl.read_parquet(file_path)
            # Convert to pandas DataFrame
            df_pd = df_pl.to_pandas()
            # Ensure 'Date' is datetime
            df_pd['Date'] = pd.to_datetime(df_pd['Date'])
            # Set 'Date' as index
            df_pd.set_index('Date', inplace=True)
            # Select the desired column
            if column_name in df_pd.columns:
                df_asset = df_pd[[column_name]].rename(columns={column_name: asset_name})
                df_list.append(df_asset)
            else:
                self.logger.warning(f"Column '{column_name}' not found in file '{file_path}'. Skipping asset '{asset_name}'.")

        if not df_list:
            self.logger.error("No data loaded. Check if the assets exist or if the files are correct.")
            return pd.DataFrame()

        # Concatenate the DataFrames along columns
        result_df = pd.concat(df_list, axis=1).sort_index().ffill()
        if diffs:
            result_df = result_df.diff().ffill().dropna()
        return result_df

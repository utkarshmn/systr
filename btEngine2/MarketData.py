import os
from typing import Dict, Any, Optional, List, Tuple
import polars as pl
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from btEngine2.TradingRule import *
import datetime

class MarketData:
    def __init__(
        self,
        base_directory: str,
        tick_values_path: str,
        fx_rates_path: str,
        instrument_type: str = "Futures",
        n_threads: int = 4,
        log_level: int = logging.INFO,
        load_data: bool = True
    ):
        """
        Initializes the MarketData instance by loading asset data, tick values, and FX rates.

        :param base_directory: Base directory where asset class folders and Parquet files are stored.
        :param tick_values_path: Path to the fut_val_pt.parquet file.
        :param fx_rates_path: Path to the fxHist.parquet file.
        :param instrument_type: Type of instruments (default: "Futures").
        :param n_threads: Number of threads for parallel data loading.
        :param log_level: Logging verbosity level.
        """
        self.base_directory = base_directory
        self.tick_values_path = tick_values_path
        self.fx_rates_path = fx_rates_path
        self.instrument_type = instrument_type
        self.n_threads = n_threads

        # Initialize dictionaries to store data
        self.data: Dict[str, pl.DataFrame] = {}          # Maps ticker symbols to their data
        self.asset_classes: Dict[str, List[str]] = {}    # Maps asset classes to lists of tickers
        self.assets = []
        self.tick_values_df: Optional[pl.DataFrame] = None
        self.fx_rates_df: Optional[pl.DataFrame] = None

        # Setup logging
        self.logger = self.setup_logging(log_level)
        if load_data:
            # Load auxiliary data: futures value per point and FX rates
            self.tick_values_df = self.load_tick_values()
            self.fx_rates_df = self.load_fx_rates()

            # Load all asset data from Parquet files
            self.load_all_data()

            # Process data: correct bad OHLC entries and add Tick_Value_USD
            self.correct_bad_data()
            self.add_usd_tick_value()

    def setup_logging(self, log_level: int) -> logging.Logger:
        """
        Sets up the logging configuration.

        :param log_level: Logging verbosity level.
        :return: Configured logger.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        # Prevent adding multiple handlers in interactive environments
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_tick_values(self) -> Optional[pl.DataFrame]:
        """
        Loads the futures value per point data from the Parquet file.

        :return: Polars DataFrame containing fut_val_pt data or None if failed.
        """
        try:
            self.logger.info(f"Loading tick values from {self.tick_values_path}")
            tick_values_df = pl.read_parquet(self.tick_values_path)
            self.logger.info(f"Loaded tick values for {tick_values_df.height} assets.")
            return tick_values_df
        except Exception as e:
            self.logger.error(f"Error loading tick values from {self.tick_values_path}: {e}")
            return None

    def load_fx_rates(self) -> Optional[pl.DataFrame]:
        """
        Loads the FX rates history data from the Parquet file.

        :return: Polars DataFrame containing FX rates or None if failed.
        """
        try:
            self.logger.info(f"Loading FX rates from {self.fx_rates_path}")
            fx_rates_df = pl.read_parquet(self.fx_rates_path)
            self.logger.info(f"Loaded FX rates with shape {fx_rates_df.shape}.")
            return fx_rates_df
        except Exception as e:
            self.logger.error(f"Error loading FX rates from {self.fx_rates_path}: {e}")
            return None

    def load_all_data(self):
        """
        Loads all asset data from Parquet files organized by asset classes.
        Utilizes multi-threading for faster loading.
        """
        self.logger.info(f"Loading all asset data from {self.base_directory}")
        # List all directories in the base directory (each representing an asset class)
        asset_class_dirs = [
            os.path.join(self.base_directory, d) for d in os.listdir(self.base_directory)
            if os.path.isdir(os.path.join(self.base_directory, d))
        ]

        self.logger.info(f"Found {len(asset_class_dirs)} asset class directories.")

        # Prepare a list of tuples (asset_class, file_path) for all Parquet files
        asset_files = []
        for asset_class_dir in asset_class_dirs:
            asset_class = os.path.basename(asset_class_dir)  # Extract folder name
            for file in os.listdir(asset_class_dir):
                if file.endswith(".parquet"):
                    asset_files.append((asset_class, os.path.join(asset_class_dir, file)))

        self.logger.info(f"Found {len(asset_files)} asset Parquet files to load.")

        # Use ThreadPoolExecutor to load files in parallel
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Map the _load_single_parquet method over all asset_files
            results = list(executor.map(self._load_single_parquet, asset_files))

        # Populate data and asset_classes dictionaries based on loaded data
        for asset_class, ticker, df in results:
            if df is not None:
                self.data[ticker] = df
                self.asset_classes.setdefault(asset_class, []).append(ticker)
                self.assets.append(ticker)

        self.logger.info(f"Successfully loaded data for {len(self.data)} tickers across {len(self.asset_classes)} asset classes.")

    def _load_single_parquet(self, asset_info):
        """
        Helper method to load a single Parquet file.
        Returns a tuple of (asset_class, ticker, DataFrame)
        """
        asset_class, file_path = asset_info
        try:
            df = pl.read_parquet(file_path)
            ticker = os.path.splitext(os.path.basename(file_path))[0]  # Extract ticker from file name
            return (asset_class, ticker, df)
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return (asset_class, None, None)


    def correct_bad_data(self):
        """
        Corrects data where Open, High, and Low prices are the same and different from Close.
        Adds a 'BadOHLC' column to flag corrected rows.
        """
        self.logger.info("Correcting bad OHLC data.")
        for ticker, df in self.data.items():
            try:
                # Create a boolean mask where Open == High == Low and Close != Open
                mask = (
                    (df["Open"] == df["High"]) &
                    (df["High"] == df["Low"]) &
                    (df["Close"] != df["Open"])
                )

                bad_ohlc_count = mask.sum()  # Count the number of bad OHLC entries

                if bad_ohlc_count > 0:
                    # Correct Open, High, Low to Close where mask is True
                    corrected_df = df.with_columns([
                        pl.when(mask).then(pl.col("Close")).otherwise(pl.col("Open")).alias("Open"),
                        pl.when(mask).then(pl.col("Close")).otherwise(pl.col("High")).alias("High"),
                        pl.when(mask).then(pl.col("Close")).otherwise(pl.col("Low")).alias("Low"),
                        pl.lit(mask).alias("BadOHLC")  # Flag the corrected rows
                    ])
                    self.data[ticker] = corrected_df
                    self.logger.debug(f"Corrected {bad_ohlc_count} bad OHLC entries for {ticker}.")

                if ticker == 'SI1 Comdty':
                    start_date = datetime.date(1987, 1, 1)
                    df = df.filter(pl.col('Date') >= start_date)
                    self.data[ticker] = df
                else:
                    # Ensure 'BadOHLC' column exists and is False if no corrections were made
                    if "BadOHLC" not in df.columns:
                        self.data[ticker] = df.with_columns(pl.lit(False).alias("BadOHLC"))

                # Filter the DataFrame for ticker 'SI1 Comdty' to start from 01/01/1987
                
                
            except Exception as e:
                self.logger.error(f"Error correcting data for {ticker}: {e}")

    def add_usd_tick_value(self):
        """
        Adds a 'Tick_Value_USD' column to each asset's DataFrame by merging with tick values and FX rates.
        """
        self.logger.info("Adding Tick_Value_USD to asset data.")
        
        # Check if required DataFrames are loaded
        if self.tick_values_df is None:
            self.logger.error("Tick values DataFrame is not loaded.")
            return
        if self.fx_rates_df is None:
            self.logger.error("FX rates DataFrame is not loaded.")
            return
        
        try:
            # Select necessary columns and convert to dictionary
            tick_values_dict = self.tick_values_df.select([
                pl.col("Asset"),
                pl.col("fut_val_pt"),
                pl.col("ccy")
            ]).to_dict(as_series=False)
            
            # Unpack tuples correctly in the dictionary comprehension
            asset_val_dict = {
                asset: {"fut_val_pt": fut_val_pt, "ccy": ccy} 
                for asset, fut_val_pt, ccy in zip(
                    tick_values_dict["Asset"],
                    tick_values_dict["fut_val_pt"],
                    tick_values_dict["ccy"]
                )
            }
            

            # Iterate over each ticker and its corresponding DataFrame
            for ticker, df in self.data.items():
                self.logger.info(f"Processing ticker: {ticker}")
                try:
                    if ticker in asset_val_dict:
                        # Retrieve fut_val_pt and currency for the ticker
                        fut_val_pt = asset_val_dict[ticker]["fut_val_pt"]
                        ccy = asset_val_dict[ticker]["ccy"]
                        
                        self.logger.debug(f"Ticker: {ticker}, fut_val_pt: {fut_val_pt}, ccy: {ccy}")
                        

                        
                        if df["Date"].dtype == str:
                            df = df.with_columns(
                                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date")
                            )
                        
                        
                        
                        # Get FX rate for the currency from fx_rates_df
                        if ccy in self.fx_rates_df.columns:
                            # Select 'Date' and the specific currency FX rate, rename to 'FX_Rate'
                            fx_rate = self.fx_rates_df.select([
                                pl.col("Date"),
                                pl.col(ccy).alias("FX_Rate")
                            ])
                            
                            self.logger.debug(f"FX Rate DataFrame for {ccy}:\n{fx_rate.head()}")
                        else:
                            # Default FX rate to 1.0 if currency is USD or not found
                            self.logger.warning(f"Currency '{ccy}' not found in FX rates. Defaulting FX_Rate to 1.0 for ticker {ticker}.")
                            fx_rate = pl.DataFrame({
                                "Date": df["Date"],
                                "FX_Rate": [1.0] * df.height
                            })
                        
                        # Merge FX rate with asset data on 'Date' using left join
                        # Ensure both Date columns are of the same type
                        df = df.with_columns(pl.col("Date").cast(pl.Date))
                        fx_rate = fx_rate.with_columns(pl.col("Date").cast(pl.Date))
                        merged_df = df.join(fx_rate, on="Date", how="left")
                        
                        self.logger.debug(f"Merged DataFrame for {ticker}:\n{merged_df.head()}")
                        
                        # Handle missing FX rates by filling them with 1.0
                        merged_df = merged_df.with_columns([
                            pl.col("FX_Rate").fill_null(1.0)
                        ])
                        
                        # Calculate Tick_Value_USD as fut_val_pt * FX_Rate
                        merged_df = merged_df.with_columns([
                            (pl.lit(fut_val_pt)).alias("Tick_Value_Base"),
                            (pl.lit(fut_val_pt) * pl.col("FX_Rate")).alias("Tick_Value_USD")
                        ])
                        
                        self.logger.debug(f"Tick_Value_USD for {ticker} calculated.")
                        
                        # Update the data dictionary with the merged DataFrame
                        self.data[ticker] = merged_df
                    else:
                        # If no tick value info is found, set Tick_Value_USD to NaN
                        self.logger.warning(f"No tick value information found for {ticker}. Tick_Value_USD set to NaN.")
                        df = df.with_columns([
                            pl.lit(np.nan).alias("Tick_Value_USD")
                        ])
                        self.data[ticker] = df
                except Exception as e:
                    self.logger.error(f"Error adding Tick_Value_USD for {ticker}: {e}")
        except Exception as e:
            self.logger.error(f"Error in add_usd_tick_value: {e}")

    def date_filter(self, start_date=None, end_date=None):
        """
        Returns a new MarketData object with all underlying time series sliced based on the provided start and/or end dates.

        :param start_date: Optional start date in 'DDMMYYYY' format.
        :param end_date: Optional end date in 'DDMMYYYY' format.
        :return: A new MarketData object with sliced data.
        """
        # Ensure at least one date is provided
        if start_date is None and end_date is None:
            self.logger.error("At least one of start_date or end_date must be provided.")
            raise ValueError("At least one of start_date or end_date must be provided.")

        # Function to parse dates
        def parse_date(date_str):
            try:
                return datetime.datetime.strptime(date_str, "%d%m%Y")
            except ValueError as e:
                self.logger.error(f"Incorrect date format for '{date_str}'. Expected 'DDMMYYYY'.")
                raise ValueError(f"Incorrect date format for '{date_str}'. Expected 'DDMMYYYY'.") from e

        # Parse dates if provided
        start_date_parsed = parse_date(start_date) if start_date else None
        end_date_parsed = parse_date(end_date) if end_date else None

        # Create a new MarketData object without loading data
        new_market_data = MarketData(
            base_directory=self.base_directory,
            tick_values_path=self.tick_values_path,
            fx_rates_path=self.fx_rates_path,
            n_threads=self.n_threads,
            load_data=False  # Prevent loading data during initialization
        )

        # Slice each DataFrame in self.data
        for asset, df in self.data.items():
            # Ensure 'Date' column is of datetime type
            if 'Date' not in df.columns:
                self.logger.error(f"'Date' column not found in data for asset '{asset}'.")
                raise KeyError(f"'Date' column not found in data for asset '{asset}'.")

            # Convert 'Date' column to datetime if necessary
            if df['Date'].dtype != pl.Datetime and df['Date'].dtype != pl.Date:
                df = df.with_columns(
                    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False)
                )

            # Apply slicing based on start_date and end_date
            if start_date_parsed and end_date_parsed:
                df_sliced = df.filter(
                    (pl.col('Date') >= start_date_parsed.date()) & (pl.col('Date') <= end_date_parsed.date())
                )
            elif start_date_parsed:
                df_sliced = df.filter(pl.col('Date') >= start_date_parsed.date())
            elif end_date_parsed:
                df_sliced = df.filter(pl.col('Date') <= end_date_parsed.date())

            # Add the sliced DataFrame to the new data dictionary
            new_market_data.data[asset] = df_sliced

        # Copy asset_classes
        new_market_data.asset_classes = self.asset_classes.copy()

        # Copy other necessary attributes
        new_market_data.fx_rates_df = self.fx_rates_df
        new_market_data.tick_values_df = self.tick_values_df

        # If fx_rates and tick_values have 'Date' columns, slice them as well
        if hasattr(new_market_data, 'fx_rates') and 'Date' in new_market_data.fx_rates_df.columns:
            # Convert 'Date' column to datetime if necessary
            if new_market_data.fx_rates_df['Date'].dtype != pl.Datetime and new_market_data.fx_rates_df['Date'].dtype != pl.Date:
                new_market_data.fx_rates_df = new_market_data.fx_rates_df.with_columns(
                    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False)
                )
            # Slice fx_rates
            if start_date_parsed and end_date_parsed:
                new_market_data.fx_rates_df = new_market_data.fx_rates_df.filter(
                    (pl.col('Date') >= start_date_parsed.date()) & (pl.col('Date') <= end_date_parsed.date())
                )
            elif start_date_parsed:
                new_market_data.fx_rates_df = new_market_data.fx_rates_df.filter(pl.col('Date') >= start_date_parsed.date())
            elif end_date_parsed:
                new_market_data.fx_rates_df = new_market_data.fx_rates_df.filter(pl.col('Date') <= end_date_parsed.date())

        return new_market_data

    def get_ticker_data(self, ticker: str) -> pl.DataFrame:
        """
        Retrieves the DataFrame for a specific ticker.

        :param ticker: Ticker symbol.
        :return: Polars DataFrame containing the ticker's data.
        :raises ValueError: If the ticker is not found.
        """
        if ticker in self.data:
            return self.data[ticker]
        else:
            raise ValueError(f"Ticker '{ticker}' not found in the market data.")

    def get_data(self) -> Dict[str, pl.DataFrame]:
        """
        Retrieves all preprocessed data.

        :return: Dictionary mapping tickers to their Polars DataFrames.
        """
        return self.data

    def get_fx_rates(self) -> Optional[pl.DataFrame]:
        """
        Retrieves the FX rates DataFrame.

        :return: Polars DataFrame containing FX rates or None if not loaded.
        """
        return self.fx_rates_df

    def get_tick_values(self) -> Optional[pl.DataFrame]:
        """
        Retrieves the tick values DataFrame.

        :return: Polars DataFrame containing tick values or None if not loaded.
        """
        return self.tick_values_df

    def get_asset_classes(self) -> Dict[str, List[str]]:
        """
        Retrieves the asset classes and their associated tickers.

        :return: Dictionary mapping asset classes to lists of tickers.
        """
        return self.asset_classes

    def get_asset_class_data(self, asset_class_names):
        """
        Returns a new MarketData object containing only the data for the specified asset class(es).

        :param asset_class_names: The name of the asset class or a list of asset class names.
        :return: A new MarketData object with data for the specified asset class(es).
        """
        # Ensure asset_class_names is a list
        if isinstance(asset_class_names, str):
            asset_class_names = [asset_class_names]
        elif not isinstance(asset_class_names, list):
            self.logger.error("asset_class_names must be a string or a list of strings.")
            raise TypeError("asset_class_names must be a string or a list of strings.")

        # Validate asset class names
        invalid_classes = [ac for ac in asset_class_names if ac not in self.asset_classes]
        if invalid_classes:
            self.logger.error(f"Asset classes not found: {invalid_classes}")
            raise ValueError(f"Asset classes not found: {invalid_classes}")

        # Get the list of assets in the specified asset classes
        assets_in_classes = []
        for asset_class in asset_class_names:
            assets_in_classes.extend(self.asset_classes[asset_class])

        # Remove duplicates, if any
        assets_in_classes = list(set(assets_in_classes))

        # Create a new MarketData object without loading data
        new_market_data = MarketData(
            base_directory=self.base_directory,
            tick_values_path=self.tick_values_path,
            fx_rates_path=self.fx_rates_path,
            n_threads=self.n_threads,
            load_data=False  # Prevent loading data during initialization
        )

        # Overwrite the data attribute
        new_market_data.data = {asset: self.data[asset] for asset in assets_in_classes}

        # Set the asset_classes attribute
        new_market_data.asset_classes = {ac: self.asset_classes[ac] for ac in asset_class_names}

        # Copy other necessary attributes
        new_market_data.fx_rates_df = self.fx_rates_df
        new_market_data.tick_values_df = self.tick_values_df

        # Since we've already loaded the data, we don't need to load it again
        return new_market_data
    
    def get_assets_data(self, asset_list):
        """
        Returns a new MarketData object containing only the data for the specified list of assets.

        :param asset_list: A list of asset names.
        :return: A new MarketData object with data for the specified assets.
        """
        if isinstance(asset_list, str):
            asset_list = [asset_list]
        elif not isinstance(asset_list, list):
            self.logger.error("asset_list must be a string or a list of strings.")
            raise TypeError("asset_list must be a string or a list of strings.")

        # Validate asset names
        missing_assets = [asset for asset in asset_list if asset not in self.data]
        if missing_assets:
            self.logger.error(f"Assets not found in data: {missing_assets}")
            raise ValueError(f"Assets not found in data: {missing_assets}")

        # Create a new MarketData object without loading data
        new_market_data = MarketData(
            base_directory=self.base_directory,
            tick_values_path=self.tick_values_path,
            fx_rates_path=self.fx_rates_path,
            n_threads=self.n_threads,
            load_data=False  # Prevent loading data during initialization
        )

        # Overwrite the data attribute
        new_market_data.data = {asset: self.data[asset] for asset in asset_list}

        # Update asset_classes to include only the specified assets
        new_market_data.asset_classes = {}
        for asset_class, assets in self.asset_classes.items():
            # Get the intersection of assets in this class and the specified asset_list
            assets_in_class = list(set(assets) & set(asset_list))
            if assets_in_class:
                new_market_data.asset_classes[asset_class] = assets_in_class

        # Copy other necessary attributes
        new_market_data.fx_rates_df = self.fx_rates_df
        new_market_data.tick_values_df = self.tick_values_df

        return new_market_data    

    def get_ac(self, asst):
        for asset_class, tickers in self.asset_classes.items():
            if asst in tickers:
                return asset_class
        raise ValueError(f"Asset '{asst}' not found in any asset class.")

    def gen_custom_assets(
        self,
        input_csv: str,
        output_folder: str,
        trading_rule_config: Optional[Dict[str, Any]] = None,
        default_tick_value_base: float = 1.0,
        default_tick_value_usd: float = 1.0,
        n_threads: int = 4
    ):
        """
        Creates custom portfolio Parquet files based on the provided input CSV.

        :param input_csv: Path to the input CSV file containing portfolio definitions.
        :param output_folder: Path to the folder where the output Parquet files will be saved.
        :param trading_rule_config: Configuration dictionary for TradingRule.
        :param default_tick_value_base: Default value for Tick_Value_Base if not specified.
        :param default_tick_value_usd: Default value for Tick_Value_USD if not specified.
        :param n_threads: Number of threads for parallel processing.
        """
        self.logger.info(f"Generating custom assets from {input_csv} into {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

        # Read the input CSV using Polars for consistency and performance
        try:
            portfolios = pl.read_csv(input_csv)
            self.logger.info(f"Loaded {portfolios.height} portfolios from {input_csv}")
        except Exception as e:
            self.logger.error(f"Error reading custom assets CSV file: {e}")
            return

        custom_files = []  # List to keep track of generated custom portfolio files

        # Initialize TradingRule once, which can be reused for all portfolios
        trading_rule = TradingRule(
            market_data=self,
            config=trading_rule_config or {},
            params={'VolLookBack': 252}  # Example parameter
        )

        # Define a helper function for processing each portfolio
        def process_portfolio(row: pl.Series):
            """
            Processes a single portfolio based on the provided row.

            :param row: Polars Series containing portfolio information.
            """
            portfolio_name = row["Asset Name"]   # Name of the custom portfolio
            method = row["Method"]               # Method type (e.g., "vol", "ratio")
            tick_value_base = row.get("Tick_Value_Base", default_tick_value_base)  # Base tick value
            tick_value_usd = row.get("Tick_Value_USD", default_tick_value_usd)     # USD tick value

            portfolio_df = None  # Initialize an empty DataFrame for the portfolio

            try:
                if method == "vol":
                    # Extract securities and their weights from the row
                    securities = [
                        (row.get("Sec1"), row.get("Wgt1")),
                        (row.get("Sec2"), row.get("Wgt2")),
                        (row.get("Sec3"), row.get("Wgt3", 0))  # Default weight to 0 if not specified
                    ]
                    pnl_series = []  # List to hold PnL series for each security

                    for sec, wgt in securities:
                        if sec and wgt != 0:
                            # Apply the constant position trading rule
                            df = trading_rule.const_pos_rule(sec, wgt)
                            if df is not None:
                                pnl_series.append(df["Strategy_PnL_USD"])

                    if pnl_series:
                        # Combine all PnL series by summing them horizontally
                        combined_pnl = pl.concat(pnl_series, how="horizontal").sum(axis=1)
                        cumulative_pnl = combined_pnl.cumsum()  # Calculate cumulative PnL

                        # Create the portfolio DataFrame
                        # Assuming all PnL series have the same 'Date' column
                        first_sec = securities[0][0]
                        date_column = trading_rule.market_data.get_ticker_data(first_sec)["Date"]
                        portfolio_df = pl.DataFrame({
                            "Date": date_column,
                            "Close": cumulative_pnl,
                            "Tick_Value_Base": pl.lit(tick_value_base),
                            "FX Rate": pl.lit(1.0),               # Assuming FX Rate is 1.0 for simplicity
                            "Tick_Value_USD": pl.lit(tick_value_usd)
                        })
                    else:
                        # If no PnL series were generated, create an empty DataFrame
                        portfolio_df = pl.DataFrame({
                            "Date": [],
                            "Close": [],
                            "Tick_Value_Base": [],
                            "FX Rate": [],
                            "Tick_Value_USD": []
                        })

                elif method == "ratio":
                    # Extract securities and their weights
                    sec1, wgt1 = row["Sec1"], row["Wgt1"]
                    sec2, wgt2 = row["Sec2"], row["Wgt2"]

                    try:
                        # Load data for the first security and apply its weight
                        sec1_df = self.get_ticker_data(sec1)[["Date", "Open", "High", "Low", "Close"]].with_columns([
                            (pl.col("Open") * wgt1).alias("Open"),
                            (pl.col("High") * wgt1).alias("High"),
                            (pl.col("Low") * wgt1).alias("Low"),
                            (pl.col("Close") * wgt1).alias("Close")
                        ])

                        # Load data for the second security and apply its weight
                        sec2_df = self.get_ticker_data(sec2)[["Date", "Open", "High", "Low", "Close"]].with_columns([
                            (pl.col("Open") * wgt2).alias("Open"),
                            (pl.col("High") * wgt2).alias("High"),
                            (pl.col("Low") * wgt2).alias("Low"),
                            (pl.col("Close") * wgt2).alias("Close")
                        ])

                        # Align data by 'Date' using inner join to ensure matching dates
                        joined_df = sec1_df.join(sec2_df, on="Date", how="inner", suffix="_sec2")

                        # Calculate ratio for each price column
                        portfolio_df = joined_df.with_columns([
                            (pl.col("Open") / pl.col("Open_sec2")).alias("Open"),
                            (pl.col("High") / pl.col("High_sec2")).alias("High"),
                            (pl.col("Low") / pl.col("Low_sec2")).alias("Low"),
                            (pl.col("Close") / pl.col("Close_sec2")).alias("Close")
                        ]).select([
                            "Date", "Open", "High", "Low", "Close"
                        ])

                        # Add tick value columns
                        portfolio_df = portfolio_df.with_columns([
                            pl.lit(tick_value_base).alias("Tick_Value_Base"),
                            pl.lit(1.0).alias("FX Rate"),               # Assuming FX Rate is 1.0 for simplicity
                            pl.lit(tick_value_usd).alias("Tick_Value_USD")
                        ])
                    except Exception as e:
                        # Log error and create an empty DataFrame in case of failure
                        self.logger.error(f"Error processing ratio method for portfolio '{portfolio_name}': {e}")
                        portfolio_df = pl.DataFrame({
                            "Date": [],
                            "Open": [],
                            "High": [],
                            "Low": [],
                            "Close": [],
                            "Tick_Value_Base": [],
                            "FX Rate": [],
                            "Tick_Value_USD": []
                        })
                else:
                    # Log a warning for unknown method types and skip processing
                    self.logger.warning(f"Unknown method '{method}' for portfolio '{portfolio_name}'. Skipping.")
                    return

                # Save the resulting DataFrame to a Parquet file
                output_file = os.path.join(output_folder, f"{portfolio_name}.parquet")
                portfolio_df.write_parquet(output_file)
                custom_files.append(f"{portfolio_name}.parquet")
                self.logger.info(f"Created portfolio file: {output_file}")

                # Add the portfolio DataFrame to the self.data dictionary
                if portfolio_df.height > 0:
                    self.data[portfolio_name] = portfolio_df.drop_nulls()
            except Exception as e:
                # Log any unexpected errors during portfolio processing
                self.logger.error(f"Error processing portfolio '{portfolio_name}': {e}")

        # Use ThreadPoolExecutor for parallel processing of portfolios
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Iterate over each row in the portfolios DataFrame and process it
            executor.map(process_portfolio, [row for row in portfolios.iter_rows(named=True)])

        # Update the asset_classes dictionary to include custom portfolios under a "Custom" asset class
        self.asset_classes.setdefault("Custom", []).extend(custom_files)
        self.logger.info(f"Generated {len(custom_files)} custom portfolio files.")
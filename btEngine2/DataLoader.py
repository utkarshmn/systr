import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import polars as pl
from xbbg import blp
import logging
from tqdm import tqdm 

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class DataLoader:
    def __init__(
        self, 
        ticker_csv_path: str,
        base_directory: str,
        loadonly: Optional[List[str]] = None,
        sleep_time: int = 0.5,
        log_level: int = logging.INFO,
        max_retries: int = 10,
        loadCarry: bool = False  # New parameter to enable carry calculation
    ):
        """
        Initializes the DataLoader with necessary configurations.

        :param ticker_csv_path: Path to the CSV file containing ticker information.
        :param base_directory: Base directory where Parquet files will be saved.
        :param loadonly: List of tickers to load. If None, all tickers are loaded.
        :param sleep_time: Time in seconds to pause between API requests.
        :param log_level: Logging verbosity level.
        :param max_retries: Maximum number of retries for Bloomberg API calls.
        :param loadCarry: Boolean flag to enable carry calculation. Defaults to False.
        """
        self.ticker_csv_path = ticker_csv_path
        self.base_directory = base_directory
        self.loadonly = loadonly or []  # If loadonly is None, load all tickers
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.loadCarry = loadCarry  # Store the loadCarry flag
        self.all_data_rt: Dict[str, pl.DataFrame] = {}
        self.logger = self.setup_logging(log_level)
        self.df = self.read_ticker_csv()
        self.helper_files_folder = self.setup_helper_files_folder()
        self.fut_val_pt_path = os.path.join(self.helper_files_folder, 'fut_val_pt.parquet')
        self.fx_hist_path = os.path.join(self.helper_files_folder, 'fxHist.parquet')


        self.bloomberg_lock = Lock()

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

    def setup_helper_files_folder(self) -> str:
        """
        Sets up a separate helper files folder one directory above the base_directory.

        :return: Path to the helper files folder.
        """
        parent_directory = os.path.dirname(self.base_directory)
        helper_files_folder = os.path.join(parent_directory, 'HelperFiles')
        os.makedirs(helper_files_folder, exist_ok=True)
        self.logger.info(f"Helper files will be stored in: {helper_files_folder}")
        return helper_files_folder

    def read_ticker_csv(self) -> pd.DataFrame:
        """
        Reads the ticker CSV file.

        :return: DataFrame containing ticker information.
        """
        try:
            df = pd.read_csv(self.ticker_csv_path)
            df.dropna(how='all', inplace=True)
            self.logger.info(f"Loaded {len(df)} tickers from {self.ticker_csv_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading ticker CSV file: {e}")
            raise e

    def load_data(self, ticker: str, to_load = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'VOLUME'],
                  renames = ['Open', 'High', 'Low', 'Close', 'Volume'], start_date ='1980-01-01') -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a given ticker using XBBG with retry mechanism.

        :param ticker: Ticker symbol to fetch data for.
        :return: DataFrame containing historical data or None if failed.
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Fetch data for open, high, low, close, and volume
                data = blp.bdh(
                    ticker, 
                    to_load, 
                    start_date=start_date
                )
                if data.empty:
                    # Log a warning and raise an error to trigger a retry
                    self.logger.warning(f"No data returned for ticker {ticker}. Attempt {attempt + 1}/{self.max_retries}")
                    raise ValueError(f"No data returned for ticker {ticker}.")

                # Flatten multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(1)
                
                # Reset index to have 'Date' as a column
                data.reset_index(inplace=True)
                
                renames = ['Date'] + renames
                data.columns = renames
                
                # Ensure 'Date' is in datetime format and sort by Date
                data['Date'] = pd.to_datetime(data['Date'])
                data.sort_values('Date', inplace=True)
                
                # Log successful data load
                self.logger.info(f"Successfully loaded data for ticker {ticker} with {len(data)} records.")
                return data
            except Exception as e:
                attempt += 1
                self.logger.error(f"Error loading data for {ticker} (Attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.sleep_time)  # Wait before retrying
        # Log failure after max retries
        self.logger.error(f"Failed to load data for {ticker} after {self.max_retries} attempts.")
        return None

    def load_fut_val_pt(self) -> Optional[pl.DataFrame]:
        """
        Loads futures value per point data from Bloomberg with retry mechanism.

        :return: Polars DataFrame containing fut_val_pt data or None if failed.
        """
        input_csv = self.ticker_csv_path
        output_csv = os.path.join(self.helper_files_folder, 'fut_val_pt.csv')
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Step 1: Load the list of futures from a CSV file
                futures_df = pd.read_csv(input_csv)
                self.logger.info(f"Loaded {len(futures_df)} futures from {input_csv}")
                
                # Step 2: Get the list of futures tickers and remove any duplicates in the input
                futures_tickers = futures_df['curr'].drop_duplicates().tolist()
                self.logger.debug(f"Unique futures tickers: {futures_tickers}")
                
                # Step 3: Make a single Bloomberg API call to get the FUT_VAL_PT for all futures
                data = blp.bdp(
                    tickers=futures_tickers, 
                    flds=['fut_tick_size','fut_tick_val', 'Crncy']
                )
                
                if data.empty:
                    # Log a warning and raise an error to trigger a retry
                    self.logger.warning(f"No data returned for fut_val_pt. Attempt {attempt + 1}/{self.max_retries}")
                    raise ValueError("No data returned for fut_val_pt.")

                # Step 4: Handle duplicates by dropping them (if any exist)
                data.reset_index(inplace=True)
                data.columns = ['Asset', 'fut_tick_size', 'fut_tick_val', 'ccy']
                data.drop_duplicates(subset='Asset', inplace=True)
                data['fut_val_pt'] = data['fut_tick_val'] / data['fut_tick_size']
                
                # Log successful data retrieval
                self.logger.info(f"Successfully retrieved fut_val_pt data for {len(data)} assets.")
                
                # Step 5: Save the results to a CSV file
                data.to_csv(output_csv, index=False)
                self.logger.info(f"Futures value per point data saved to {output_csv}")
                
                # Convert to Polars DataFrame for further processing
                fut_val_pt_df = pl.from_pandas(data)
                return fut_val_pt_df
            except Exception as e:
                attempt += 1
                self.logger.error(f"Error retrieving fut_val_pt data (Attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.sleep_time)  # Wait before retrying
        
        # Log failure after max retries
        self.logger.error(f"Failed to retrieve fut_val_pt data after {self.max_retries} attempts.")
        return None

    def load_fx_history(self, fut_val_pt_df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Loads FX rate history data from Bloomberg with retry mechanism.

        :param fut_val_pt_df: Polars DataFrame containing futures value per point data.
        :return: Polars DataFrame containing FX rates or None if failed.
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Step 1: Extract unique currencies excluding 'USD'
                ccyList = fut_val_pt_df['ccy'].unique().to_list()
                if 'USD' in ccyList:
                    ccyList.remove('USD')  # USD does not need an FX rate
                self.logger.info(f"Currencies to fetch FX rates for: {ccyList}")
                
                if not ccyList:
                    # If no additional currencies are found, create an FX history DataFrame with only USD
                    self.logger.info("No additional currencies found beyond USD. Skipping FX rate loading.")
                    fxHist_df = pl.DataFrame({
                        'Date': [],
                        'USD': [1.0]
                    })
                    return fxHist_df
                
                # Step 2: Generate currency tickers (e.g., EURUSD Curncy)
                ccyTickers = [x + 'USD Curncy' for x in ccyList]
                self.logger.debug(f"FX tickers: {ccyTickers}")
                
                # Step 3: Make Bloomberg API call to get PX_LAST for FX rates
                fxHist = blp.bdh(
                    tickers=ccyTickers, 
                    flds=['PX_LAST'], 
                    start_date='1980-01-01'
                )
                
                if fxHist.empty:
                    # Log a warning and raise an error to trigger a retry
                    self.logger.warning(f"No FX rate data returned. Attempt {attempt + 1}/{self.max_retries}")
                    raise ValueError("No FX rate data returned.")
                
                # Step 4: Flatten multi-index columns if present
                if isinstance(fxHist.columns, pd.MultiIndex):
                    fxHist.columns = fxHist.columns.get_level_values(0)
                
                # Step 5: Reset index to have 'Date' as a column
                fxHist.reset_index(inplace=True)
                
                # Step 6: Rename columns to match expected format
                fxHist.rename(columns={'date': 'Date'}, inplace=True)
                
                # Step 7: Rename currency columns appropriately (e.g., EURUSD Curncy -> EUR)
                for ccy in ccyList:
                    fx_column = ccy + 'USD Curncy'
                    if fx_column in fxHist.columns:
                        fxHist.rename(columns={fx_column: ccy}, inplace=True)
                    else:
                        self.logger.warning(f"FX column {fx_column} not found in returned data.")
                
                # Step 8: Add 'USD' as a currency with rate 1.0
                fxHist['USD'] = 1.0
                
                # Step 9: Forward fill missing values to maintain data continuity
                # Convert to Polars DataFrame before using fill_null
                fxHist_pl = pl.from_pandas(fxHist)
                fxHist_pl = fxHist_pl.fill_null(strategy='forward')
                # Rename the index column to 'Date'
                fxHist_pl = fxHist_pl.rename({'index': 'Date'})
                
                # Log successful FX data retrieval
                self.logger.info(f"Successfully retrieved FX history data for {len(ccyList)} currencies.")
                
                # Step 10: Save to CSV (optional, as Parquet is preferred)
                fxHist_pl.to_pandas().to_csv(os.path.join(self.helper_files_folder, 'fxHist.csv'), index=False)
                self.logger.info(f"FX rate history data saved to {os.path.join(self.helper_files_folder, 'fxHist.csv')}")
                
                # Return the Polars DataFrame
                return fxHist_pl
            except Exception as e:
                attempt += 1
                self.logger.error(f"Error retrieving FX history data (Attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.sleep_time)  # Wait before retrying
    
    def save_parquet(self, data: pd.DataFrame, file_path: str):
        """
        Saves the DataFrame as a Parquet file using Polars.

        :param data: DataFrame to save.
        :param file_path: Path where the Parquet file will be saved.
        """
        try:
            pl_df = pl.from_pandas(data)
            pl_df.write_parquet(file_path)
            self.logger.info(f"Saved data to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")

    def stitch_data(self, curr_data: pd.DataFrame, old_data: Optional[pd.DataFrame], curr: str) -> pd.DataFrame:
        """
        Stitches current and old data to create a continuous time series.

        :param curr_data: DataFrame containing current data.
        :param old_data: DataFrame containing old data.
        :param curr: Current ticker symbol.
        :return: Combined DataFrame.
        """
        if old_data is not None:
            try:
                # Find the last day of old data
                last_day_old = old_data['Date'].max()
                # Calculate the spread based on the last day of old data
                spread = curr_data.loc[curr_data['Date'] == last_day_old, 'Close'].values[0] - \
                            old_data.loc[old_data['Date'] == last_day_old, 'Close'].values[0]
                
                # Adjust old_data prices by the spread to align with current data
                for price_col in ['Open', 'High', 'Low', 'Close']:
                    old_data[price_col] += spread
                
                # Combine old_data with curr_data after last_day_old
                combined_data = pd.concat([
                    old_data, 
                    curr_data[curr_data['Date'] > last_day_old]
                ], ignore_index=True)
                
                # Log successful stitching
                self.logger.info(f"Stitched data for {curr} using old data.")
            except KeyError as e:
                # Log specific KeyError and use current data only
                self.logger.error(f"Key error during stitching for {curr}: {e}")
                combined_data = curr_data
            except IndexError as e:
                # Log specific IndexError and use current data only
                self.logger.error(f"Index error during stitching for {curr}: {e}")
                combined_data = curr_data
        else:
            # If no old data is available, use current data only
            combined_data = curr_data
        
        return combined_data

    def process_ticker_fut(self, row: pd.Series, current_time: datetime):
        """
        Processes a single ticker: loads data, stitches if necessary, calculates carry (if enabled),
        and saves as Parquet.

        :param row: Series containing ticker information.
        :param current_time: Current datetime for reference.
        """
        curr = row['curr']               # Current ticker symbol
        old = row['old']                 # Old ticker symbol for stitching (if any)
        carry_ctrct = row.get('carry ctrct')  # Carry contract ticker
        carry_old = row.get('carry old')      # Old carry contract ticker
        folder = row['folder']           # Asset class folder name
        file_path = os.path.join(self.base_directory, folder, f"{curr}.parquet")  # Full path to save Parquet
        folder_path = os.path.dirname(file_path)  # Directory path

        # Check if ticker is in the loadonly list (if specified)
        if self.loadonly and curr not in self.loadonly:
            self.logger.info(f"---- Skipping {curr} as it is not in the loadonly list. ----")
            return
        
        # Check if the Parquet file exists
        if os.path.exists(file_path):
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Load the Parquet file to check if the "Carry" column is present
            try:
                loaded_df = pl.read_parquet(file_path)
                if "Carry" in loaded_df.columns:
                    # If the "Carry" column exists and the file was modified within the last day, skip it
                    if current_time - last_modified_time < timedelta(days=1):
                        self.logger.info(f"Skipping {curr} as it was already updated within the last day and has a Carry column.")
                        return
                else:
                    # If the "Carry" column is missing, continue to calculate Carry
                    self.logger.info(f"Carry column missing for {curr}. Proceeding with carry calculation.")
            except Exception as e:
                # Handle errors during file loading
                self.logger.error(f"Error reading Parquet file for {curr}: {e}. Proceeding with carry calculation.")
        
        # Load current and old data
        curr_data = self.load_data(curr)
        old_data = self.load_data(old) if pd.notna(old) else None
        
        if curr_data is None and old_data is None:
            # If both current and old data are unavailable, skip processing
            self.logger.warning(f"No data available for {curr}. Skipping.")
            return
        
        # Stitch data if both curr and old data are available
        combined_data = self.stitch_data(curr_data, old_data, curr)
        
        # Forward fill missing values to ensure data continuity
        # Since combined_data is a pandas DataFrame, use pandas' ffill()
        combined_data = combined_data.ffill()
        
        # Add 'Name' column for identification
        combined_data['Name'] = curr
        combined_data['Descr'] = row['Descr']
        
        # Validate that the DataFrame has the expected columns
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name', 'Descr']
        if combined_data.shape[1] != len(expected_columns) or not all(col in combined_data.columns for col in expected_columns):
            self.logger.warning(f"Unexpected columns for {curr}: {combined_data.columns.tolist()}. Skipping.")
            return
        
        # Convert relevant columns to numeric types, coercing errors to NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
        
        # If loadCarry is enabled and carry_ctrct is specified, perform carry calculation
        if self.loadCarry and pd.notna(carry_ctrct):
            self.logger.info(f"Calculating Carry for {curr} using carry contract {carry_ctrct}")
            try:
                # Fetch LAST_TRADEABLE_DT for 'curr' and 'carry_ctrct'
                # Assumption here is that current time difference in contract expiries applied throughout history.
                curr_exp_series = blp.bdp(tickers=curr, flds='LAST_TRADEABLE_DT')
                carry_exp_series = blp.bdp(tickers=carry_ctrct, flds='LAST_TRADEABLE_DT')
                curr_exp_series.columns = [x.upper() for x in curr_exp_series.columns]
                carry_exp_series.columns = [x.upper() for x in carry_exp_series.columns]

                if curr_exp_series.empty or carry_exp_series.empty:
                    self.logger.warning(f"Could not retrieve expiration dates for {curr} or {carry_ctrct}. Skipping Carry calculation.")
                else:
                    curr_exp = pd.to_datetime(curr_exp_series['LAST_TRADEABLE_DT'].iloc[0])
                    carry_exp = pd.to_datetime(carry_exp_series['LAST_TRADEABLE_DT'].iloc[0])
                    
                    self.logger.debug(f"{curr} expiration date: {curr_exp}")
                    self.logger.debug(f"{carry_ctrct} expiration date: {carry_exp}")
                    
                    # Determine the direction of carry based on expiration dates
                    if curr_exp > carry_exp:
                        carry_direction = 'curr_gt_carry'
                    else:
                        carry_direction = 'curr_lt_carry'
                    
                    self.logger.debug(f"Carry direction: {carry_direction}")
                    
                    # Load carry contract price history, including stitching if necessary
                    carry_data = self.load_data(carry_ctrct)
                    carry_old_data = self.load_data(carry_old) if pd.notna(carry_old) else None
                    if carry_data is None:
                        self.logger.warning(f"No data available for carry contract {carry_ctrct}. Skipping Carry calculation.")
                    else:
                        carry_combined_data = self.stitch_data(carry_data, carry_old_data, carry_ctrct)
                        carry_combined_data = carry_combined_data.ffill()
                        
                        # We only need the 'Close' prices
                        carry_close = carry_combined_data[['Date', 'Close']].rename(columns={'Close': 'Carry_Close'})
                        
                        # Merge carry_close with combined_data on 'Date'
                        merged = pd.merge(combined_data, carry_close, on='Date', how='left')
                        merged = merged.ffill()
                        # Calculate raw carry
                        if carry_direction == 'curr_gt_carry':
                            merged['Raw_Carry'] = merged['Carry_Close'] - merged['Close']
                        else:
                            merged['Raw_Carry'] = merged['Close'] - merged['Carry_Close']
                        
                        # Calculate the period in days between expirations
                        period_days = abs((carry_exp - curr_exp).days)
                        if period_days == 0:
                            self.logger.warning(f"Expiration dates for {curr} and {carry_ctrct} are the same. Skipping annualization.")
                            merged['Carry'] = merged['Raw_Carry']
                        else:
                            # Annualize the raw carry
                            merged['Carry'] = merged['Raw_Carry'] / period_days * 365
                        
                        # Fill any NaN Carry values with 0 or appropriate default
                        merged['Carry'] = merged['Carry'].ffill()
                        merged['Carry'] = merged['Carry'].rolling(window=8).mean()
                        
                        # Add the 'Carry' column to combined_data
                        combined_data = merged.drop(columns=['Raw_Carry'])
                        
                        self.logger.info(f"Carry calculated and added for {curr}.")
            except Exception as e:
                self.logger.error(f"Error calculating Carry for {curr}: {e}")
        
        # Ensure the target directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Save the DataFrame as a Parquet file
        self.save_parquet(combined_data, file_path)
        
        # Optionally, store the data in the all_data_rt dictionary for in-memory access
        self.all_data_rt[curr] = pl.from_pandas(combined_data)
        
        # Pause to avoid overloading the Bloomberg server
        time.sleep(self.sleep_time)

    def bbgloader(self):
        """
        Executes the data loading process for all tickers, including futures values and FX history.
        """
        current_time = datetime.now()
        self.logger.info("Starting data loading process...")
        
        # Step 1: Load futures value per point data
        fut_val_pt_df = self.load_fut_val_pt()
        if fut_val_pt_df is None:
            self.logger.error("Failed to load futures value per point data. Aborting FX history loading.")
            return
        
        # Step 2: Load FX rate history data
        fx_hist_df = self.load_fx_history(fut_val_pt_df)
        if fx_hist_df is None:
            self.logger.error("Failed to load FX rate history data.")
            return
        
        # Step 3: Save auxiliary data as Parquet
        if not fx_hist_df.is_empty():
            fx_hist_df.write_parquet(self.fx_hist_path)
            self.logger.info(f"FX rate history data saved to {self.fx_hist_path}")
        else:
            self.logger.warning("FX history DataFrame is empty. Skipping saving.")
        
        if not fut_val_pt_df.is_empty():
            fut_val_pt_df.write_parquet(self.fut_val_pt_path)
            self.logger.info(f"Futures value per point data saved to {self.fut_val_pt_path}")
        else:
            self.logger.warning("Futures value per point DataFrame is empty. Skipping saving.")
        
        # Step 4: Process each ticker in the ticker list with progress bar
        with tqdm(total=len(self.df), desc="Processing tickers", unit="ticker") as pbar:
            for index, row in self.df.iterrows():
                self.process_ticker_fut(row, current_time)
                pbar.update(1)  # Update progress bar after processing each ticker
        
        self.logger.info("All data updated successfully.")

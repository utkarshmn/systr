import pandas as pd
import numpy as np
import yaml
import logging
from typing import List, Dict, Any, Union, Callable
import os
import importlib
import pkgutil

from btEngine2.TradingRule import TradingRule
from btEngine2.MarketData import MarketData


class trPortfolio:
    def __init__(
        self,
        market_data,
        trading_rules: List[Any] = None,
        readme_files: List[str] = None,
        yaml_file: str = None,
        position_sizing_params: Dict[str, Any] = None,
        log_level=logging.INFO
    ):
        """
        Initializes the trPortfolio instance.

        :param market_data: MarketData instance.
        :param trading_rules: List of TradingRule instances.
        :param readme_files: List of paths to README files to initialize TradingRule instances.
        :param yaml_file: Path to YAML configuration file.
        :param position_sizing_params: Default position sizing parameters to use if not specified in individual TradingRules.
        :param log_level: Logging level.
        """
        self.market_data = market_data
        self.position_sizing_params = position_sizing_params or {}
        self.trading_rules = []
        self.logger = logging.getLogger('trPortfolio')
        self.logger.setLevel(log_level)

        if trading_rules is not None:
            self._initialize_from_trading_rules(trading_rules)

        if readme_files is not None:
            self._initialize_from_readmes(readme_files)

        if yaml_file is not None:
            self._initialize_from_yaml(yaml_file)

        self.portfolio_pnl = None
        self.statistics = None
        self.correlation_matrix = None


    def _initialize_from_trading_rules(self, trading_rules: List[Any]):
        """
        Initializes the portfolio with provided TradingRule instances.

        :param trading_rules: List of TradingRule instances.
        """
        for tr in trading_rules:
            # If a TradingRule doesn't have position sizing parameters, use the portfolio's default
            if not tr.position_sizing_params:
                tr.position_sizing_params = self.position_sizing_params
            tr.log_level = self.logger.level
            self.trading_rules.append(tr)
            self.logger.info(f"TradingRule '{tr.name_lbl}' added to portfolio.")

    def _initialize_from_readmes(self, readme_files: List[str]):
        """
        Initializes TradingRule instances from README files and adds them to the portfolio.

        :param readme_files: List of README file paths.
        """
        for readme_path in readme_files:
            try:
                trading_rule = TradingRule.from_readme(
                    readme_path=readme_path,
                    market_data=self.market_data,
                    log_level=self.logger.level
                )
                # Use portfolio's position sizing params if not defined
                if not trading_rule.position_sizing_params:
                    trading_rule.position_sizing_params = self.position_sizing_params
                self.trading_rules.append(trading_rule)
                self.logger.info(f"TradingRule initialized from {readme_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize TradingRule from {readme_path}: {e}")

    def _initialize_from_yaml(self, yaml_file: str):
        """
        Initializes TradingRule instances from a YAML configuration file.

        :param yaml_file: Path to YAML configuration file.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        for rule_config in config.get('trading_rules', []):
            try:
                # Extract parameters
                function_name = rule_config['function_name']
                trading_params = rule_config.get('trading_params', {})
                position_sizing_params = rule_config.get('position_sizing_params', self.position_sizing_params)
                trading_rule_function = self._get_trading_rule_function(function_name)

                if trading_rule_function is None:
                    self.logger.error(f"Trading rule function '{function_name}' not found.")
                    continue

                # Use the portfolio's position sizing params if not provided
                if not position_sizing_params:
                    position_sizing_params = self.position_sizing_params

                trading_rule = TradingRule(
                    market_data=self.market_data,
                    log_level=self.logger.level,
                    trading_rule_function=trading_rule_function,
                    trading_params=trading_params,
                    position_sizing_params=position_sizing_params,
                    **rule_config.get('additional_params', {})
                )
                self.trading_rules.append(trading_rule)
                self.logger.info(f"TradingRule '{function_name}' initialized from YAML.")
            except Exception as e:
                self.logger.error(f"Failed to initialize TradingRule from YAML: {e}")

    @staticmethod
    def _get_trading_rule_function(function_name: str) -> Callable:
        """
        Searches for the trading rule function in all modules under btEngine2.Rules.

        :param function_name: Name of the trading rule function.
        :return: The trading rule function callable, or None if not found.
        """
        import importlib
        import pkgutil

        # Base package name
        base_package = 'btEngine2.Rules'

        # List to keep track of modules where we've searched
        searched_modules = set()

        # Function to recursively search modules
        def recursive_search(package_name):
            nonlocal function_name

            # Get the package
            try:
                package = importlib.import_module(package_name)
            except ImportError:
                return None

            # Iterate over all modules in the package
            for _, modname, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
                if modname in searched_modules:
                    continue
                searched_modules.add(modname)

                try:
                    module = importlib.import_module(modname)
                except ImportError:
                    continue

                # Check if the function exists in the module
                if hasattr(module, function_name):
                    return getattr(module, function_name)

                # If it's a package, recurse into it
                if ispkg:
                    result = recursive_search(modname)
                    if result:
                        return result

            return None

        # Start the search from the base package
        trading_rule_function = recursive_search(base_package)

        return trading_rule_function

    def backtest_all_rules(self, save: bool = False):
        """
        Backtests all trading rules in the portfolio.

        :param save: Whether to save the backtest results.
        """
        for trading_rule in self.trading_rules:
            self.logger.info(f"Backtesting {trading_rule.name_lbl}")
            trading_rule.backtest_all_assets(save=save)

    def aggregate_pnl(self):
        """
        Aggregates PnL from all trading rules to create the portfolio PnL.

        :return: DataFrame with portfolio cumulative PnL.
        """
        pnl_dfs = []
        for trading_rule in self.trading_rules:
            pnl_df = trading_rule.cum_pnl_byac()[['Total']]
            if pnl_df is None or pnl_df.empty:
                self.logger.warning(f"No PnL data for {trading_rule.name_lbl}. Skipping.")
                continue
            pnl_df.rename(columns={'Total': trading_rule.name_lbl}, inplace=True)
            pnl_dfs.append(pnl_df[[trading_rule.name_lbl]])

        if not pnl_dfs:
            self.logger.error("No PnL data available from trading rules.")
            return None

        # Merge all PnL DataFrames on Date
        portfolio_df = pnl_dfs[0]
        for df in pnl_dfs[1:]:
            portfolio_df = pd.concat([df, portfolio_df], axis=0)

        portfolio_df.fillna(method='ffill', inplace=True)
        portfolio_df.fillna(0, inplace=True)

        # Calculate portfolio total PnL
        pnl_columns = [tr.name_lbl for tr in self.trading_rules if tr.name_lbl in portfolio_df.columns]
        portfolio_df['PortfolioPnL'] = portfolio_df[pnl_columns].sum(axis=1)
        self.portfolio_pnl = portfolio_df
        return portfolio_df

    def calculate_portfolio_statistics(self):
        """
        Calculates portfolio-level statistics such as Sharpe ratio and drawdowns.

        :return: DataFrame with statistics.
        """
        if self.portfolio_pnl is None:
            self.aggregate_pnl()

        if self.portfolio_pnl is None or self.portfolio_pnl.empty:
            self.logger.error("Portfolio PnL data is not available.")
            return None

        returns = self.portfolio_pnl['PortfolioPnL'].diff().fillna(0)
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else np.nan  # Assuming daily returns

        # Calculate drawdowns
        cumulative_returns = self.portfolio_pnl['PortfolioPnL']
        running_max = cumulative_returns.cummax()
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = drawdown.max()

        stats = pd.DataFrame({
            'Mean Return': [mean_return],
            'Std Dev': [std_return],
            'Sharpe Ratio': [sharpe_ratio],
            'Max Drawdown': [max_drawdown]
        })
        self.statistics = stats
        return stats

    def calculate_correlations(self):
        """
        Calculates correlations between the PnL of different trading rules.

        :return: Correlation matrix DataFrame.
        """
        if self.portfolio_pnl is None:
            self.aggregate_pnl()

        if self.portfolio_pnl is None or self.portfolio_pnl.empty:
            self.logger.error("Portfolio PnL data is not available.")
            return None

        pnl_columns = [tr.name_lbl for tr in self.trading_rules if tr.name_lbl in self.portfolio_pnl.columns]
        returns_df = self.portfolio_pnl[pnl_columns].diff().fillna(0)
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix

    def plot_portfolio_pnl(self):
        """
        Plots the cumulative PnL of the portfolio.
        """
        if self.portfolio_pnl is None:
            self.aggregate_pnl()

        if self.portfolio_pnl is None or self.portfolio_pnl.empty:
            self.logger.error("Portfolio PnL data is not available.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_pnl['Date'], self.portfolio_pnl['PortfolioPnL'], label='Portfolio PnL')
        plt.title('Portfolio Cumulative PnL')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.show()

    def save_portfolio_results(self, folder: str):
        """
        Saves portfolio PnL and statistics to CSV files.

        :param folder: Folder path to save the results.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.portfolio_pnl is not None:
            self.portfolio_pnl.to_csv(os.path.join(folder, 'portfolio_pnl.csv'), index=False)
            self.logger.info(f"Portfolio PnL saved to {folder}/portfolio_pnl.csv")

        if self.statistics is not None:
            self.statistics.to_csv(os.path.join(folder, 'portfolio_statistics.csv'), index=False)
            self.logger.info(f"Portfolio statistics saved to {folder}/portfolio_statistics.csv")

        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(os.path.join(folder, 'portfolio_correlations.csv'))
            self.logger.info(f"Portfolio correlations saved to {folder}/portfolio_correlations.csv")
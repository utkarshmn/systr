TestMR_WN1 Comdty Backtest 
rsi mr strategy NQ1 Index vs. WN1 Comdty


Trading Rule Function: ratioMR_rsi_long

Trading Rule Parameters:
  pairs: [('NQ1 Index', 'WN1 Comdty'), ('WN1 Comdty', 'NQ1 Index')]
  N: (5, 2)
  rsi_period: (3, 3)
  rsi_threshold: (10.0, 8.0)
  lmt_order: False
  lmt_day: 1
  lmt_day_only: True
  lmt_atr: 1.0
  lmt_epsilon: 0.1
  atr_period: 14
  atr_type: atr
  market_data: <btEngine2.MarketData.MarketData object at 0x00000173630F6DD0>
  risk_ratio: (1.0, 1.2)

Position Sizing Parameters:
  AssetVol: 5000000
  VolLookBack: 21

Market Data Assets and Start Dates:
  NQ1 Index (5000000.0): 1999-06-21 - 2024-10-11
  WN1 Comdty (5000000.0): 2010-01-11 - 2024-10-11

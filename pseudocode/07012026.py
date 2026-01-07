# Pseudocode for Strategy 07012026

# --- STRATEGY OVERVIEW ---
#
# Name: Dynamic Multi-Indicator Strategy
# Timeframe: 15m
# Asset Type: Futures (Can short)
#
# This strategy combines multiple trend and momentum indicators to identify
# entry and exit points. It uses dual Bollinger Bands and Keltner Channels
# with different lengths to adapt to varying market volatility, along with
# multiple moving averages to confirm the underlying trend.
#
# All parameters are designed to be hyperoptimizable, allowing for robust
# optimization and backtesting.

# --- INDICATORS ---
#
# 1. Bollinger Bands 1 (BB1):
#    - Period: Optimizable integer (e.g., 10-50)
#    - Standard Deviation: Optimizable decimal (e.g., 1.0-3.0)
#
# 2. Bollinger Bands 2 (BB2):
#    - Period: Optimizable integer (e.g., 10-50)
#    - Standard Deviation: Optimizable decimal (e.g., 1.0-3.0)
#
# 3. Keltner Channel 1 (KC1):
#    - Period: Optimizable integer (e.g., 10-50)
#    - ATR Multiplier: Optimizable decimal (e.g., 1.0-4.0)
#
# 4. Keltner Channel 2 (KC2):
#    - Period: Optimizable integer (e.g., 10-50)
#    - ATR Multiplier: Optimizable decimal (e.g., 1.0-4.0)
#
# 5. Exponential Moving Averages (EMAs):
#    - Short Period: Optimizable integer (e.g., 5-25)
#    - Medium Period: Optimizable integer (e.g., 20-50)
#    - Long Period: Optimizable integer (e.g., 50-200)
#
# 6. Relative Strength Index (RSI):
#    - Standard 14-period RSI is used for buy/sell thresholds.
#
# 7. Average True Range (ATR):
#    - Period: Optimizable integer (e.g., 7-21)
#    - Used for risk management and position sizing.

# --- ENTRY LOGIC ---
#
# LONG ENTRY:
#   IF the following conditions are met:
#     - Short-term EMA is GREATER THAN Medium-term EMA (confirming short-term uptrend)
#     - Medium-term EMA is GREATER THAN Long-term EMA (confirming long-term uptrend)
#     - Close price is LESS THAN the lower band of BB1 (potential oversold)
#     - Close price is LESS THAN the lower band of KC1 (further oversold confirmation)
#     - Close price is LESS THAN the lower band of BB2 (stronger oversold signal)
#     - Close price is LESS THAN the lower band of KC2 (strongest oversold confirmation)
#     - RSI is LESS THAN the optimizable `buy_rsi_value` (e.g., 30)
#   THEN enter a LONG position.
#
# SHORT ENTRY:
#   IF the following conditions are met:
#     - Short-term EMA is LESS THAN Medium-term EMA (confirming short-term downtrend)
#     - Medium-term EMA is LESS THAN Long-term EMA (confirming long-term downtrend)
#     - Close price is GREATER THAN the upper band of BB1 (potential overbought)
#     - Close price is GREATER THAN the upper band of KC1 (further overbought confirmation)
#     - Close price is GREATER THAN the upper band of BB2 (stronger overbought signal)
#     - Close price is GREATER THAN the upper band of KC2 (strongest overbought confirmation)
#     - RSI is GREATER THAN the optimizable `sell_rsi_value` (e.g., 70)
#   THEN enter a SHORT position.

# --- EXIT LOGIC ---
#
# LONG EXIT:
#   IF the following condition is met:
#     - Close price is GREATER THAN the upper band of BB1 multiplied by an optimizable threshold (e.g., 1.1)
#   THEN exit the LONG position.
#
# SHORT EXIT:
#   IF the following condition is met:
#     - Close price is LESS THAN the lower band of BB1 multiplied by an optimizable threshold (e.g., 0.9)
#   THEN exit the SHORT position.

# --- RISK MANAGEMENT ---
#
# The strategy incorporates the following protections, which are non-restrictive
# and hyperoptimizable:
#
# - StoplossGuard: Stops trading after a specified number of stoplosses
#   occur within a defined time window.
# - CooldownPeriod: Enforces a cooldown period after each trade to prevent
#   overtrading in choppy markets.
# - LowProfitPairs: Stops trading on pairs that have consistently low
#   profitability.
# - MaxDrawdown: Stops trading if the maximum allowed drawdown is reached,
#   protecting capital.

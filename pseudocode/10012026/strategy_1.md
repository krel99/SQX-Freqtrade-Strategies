# Strategy 1: Volatility Squeeze Breakout

This strategy identifies periods of low volatility, known as a "squeeze," and enters a trade when the price breaks out of this consolidation range. The core of the strategy is the use of Bollinger Bands and Keltner Channels to detect the squeeze.

## Indicators

The strategy will use the following indicators with configurable parameters:

1.  **Bollinger Bands (BB):**
    *   `bb_period`: Time period for the moving average.
    *   `bb_stddev`: Standard deviation for the upper and lower bands.

2.  **Keltner Channels (KC):**
    *   `kc_period`: Time period for the Exponential Moving Average (EMA).
    *   `kc_atr_period`: Time period for the Average True Range (ATR).
    *   `kc_multiplier`: Multiplier for the ATR to define the channel width.

3.  **Average Directional Index (ADX):**
    *   `adx_period`: Time period for calculating ADX.

4.  **Relative Strength Index (RSI):**
    *   `rsi_period`: Time period for calculating RSI.

5.  **Moving Average Convergence Divergence (MACD):**
    *   `macd_fast_period`: Time period for the fast EMA.
    *   `macd_slow_period`: Time period for the slow EMA.
    *   `macd_signal_period`: Time period for the signal line EMA.

## Entry Conditions (Long)

A long position is entered when the following conditions are met:

1.  **Squeeze Identification:**
    *   The lower Bollinger Band is above the lower Keltner Channel.
    *   The upper Bollinger Band is below the upper Keltner Channel.
    *   This indicates the market is in a state of consolidation or "squeeze."

2.  **Breakout Confirmation:**
    *   The squeeze condition is no longer true in the current candle.
    *   The closing price is above the upper Bollinger Band.

3.  **Trend and Momentum Confirmation:**
    *   `ADX` is above a configurable threshold (`adx_threshold`), indicating a strong trend.
    *   `RSI` is above a configurable threshold (`rsi_buy_threshold`), indicating bullish momentum.
    *   The MACD line is above the MACD signal line (`macd > macd_signal`), confirming the upward momentum.

## Exit Conditions (Long)

The long position is exited based on one of the following conditions:

1.  **Take Profit:**
    *   The price closes a certain percentage above the upper Bollinger Band, defined by a `take_profit_pct` parameter.

2.  **Stop Loss:**
    *   A trailing stop loss is implemented, set at a configurable percentage (`stop_loss_pct`) below the entry price.

3.  **RSI-Based Exit:**
    *   The RSI crosses below a configurable `rsi_sell_threshold`, suggesting that the bullish momentum is fading.

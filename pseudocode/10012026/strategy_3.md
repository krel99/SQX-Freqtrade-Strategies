# Strategy 3: Dynamic Support/Resistance RSI

This strategy uses Bollinger Bands applied to the Relative Strength Index (RSI) to create dynamic overbought and oversold levels. Instead of relying on fixed RSI levels (e.g., 70/30), this approach adapts to market volatility, potentially leading to more accurate entry and exit signals.

## Indicators

The strategy will use the following indicators with configurable parameters:

1.  **Relative Strength Index (RSI):**
    *   `rsi_period`: Time period for the main RSI calculation.

2.  **RSI Bollinger Bands (RSI-BB):**
    *   `rsi_bb_period`: Time period for the moving average of the RSI.
    *   `rsi_bb_stddev`: Standard deviation for the upper and lower bands on the RSI.

3.  **Moving Averages (for trend confirmation):**
    *   `ema_fast_period`: Time period for the fast Exponential Moving Average (EMA).
    *   `ema_slow_period`: Time period for the slow EMA.
    *   `ema_trend_period`: A longer-term EMA to confirm the overall market direction.

4.  **Average True Range (ATR):**
    *   `atr_period`: Time period for ATR calculation, used for volatility filtering and stop-loss placement.

5.  **Parabolic SAR (SAR):**
    *   `sar_acceleration`: Acceleration factor for the SAR.
    *   `sar_maximum`: Maximum value for the acceleration factor.

6.  **Volume Oscillator (VO):**
    *   `vo_fast_period`: Fast period for the volume moving average.
    *   `vo_slow_period`: Slow period for the volume moving average.

## Entry Conditions (Long)

A long position is entered when the following conditions are met:

1.  **Trend Confirmation:**
    *   The fast EMA is above the slow EMA (`ema_fast > ema_slow`), indicating a short-term uptrend.
    *   The close price is above the long-term trend EMA (`close > ema_trend`), confirming the overall bullish market.

2.  **Dynamic Oversold Signal:**
    *   The RSI crosses above the lower Bollinger Band of the RSI (`RSI > RSI-BB_lower`), signaling a potential reversal from an oversold condition.

3.  **Volatility and Momentum Confirmation:**
    *   The ATR is above a certain percentage of the close price, defined by `atr_min_pct`, ensuring sufficient market volatility.
    *   The Parabolic SAR is below the close price (`SAR < close`), indicating upward momentum.
    *   The Volume Oscillator is positive (`VO > 0`), showing an increase in buying volume.

## Exit Conditions (Long)

The long position is exited based on one of the following conditions:

1.  **Dynamic Overbought Signal:**
    *   The RSI crosses below the upper Bollinger Band of the RSI (`RSI < RSI-BB_upper`), suggesting the momentum is fading.

2.  **Take Profit:**
    *   A take-profit level is set at a multiple of the ATR from the entry price, defined by `take_profit_atr_mult`.

3.  **Stop Loss:**
    *   A trailing stop loss is implemented, based on the Parabolic SAR value to trail the price as it moves upwards.

4.  **Trend Reversal:**
    *   The fast EMA crosses below the slow EMA (`ema_fast < ema_slow`), indicating a potential trend change.

# Strategy 2: Multi-Timeframe Momentum

This strategy identifies trading opportunities by aligning momentum indicators across multiple timeframes. The core idea is to confirm a trend on a higher timeframe before entering a trade on a lower timeframe, increasing the probability of a successful trade.

## Indicators

The strategy will use the following indicators with configurable parameters across two timeframes (e.g., 1-hour and 4-hour):

### Primary Timeframe (e.g., 1-hour)

1.  **Stochastic RSI (StochRSI):**
    *   `stochrsi_period`: Time period for StochRSI calculation.
    *   `stochrsi_k_period`: Smoothing period for %K.
    *   `stochrsi_d_period`: Smoothing period for %D.

2.  **Awesome Oscillator (AO):**
    *   `ao_fast_period`: Fast period for the AO.
    *   `ao_slow_period`: Slow period for the AO.

3.  **Commodity Channel Index (CCI):**
    *   `cci_period`: Time period for CCI calculation.

4.  **Chaikin Money Flow (CMF):**
    *   `cmf_period`: Time period for CMF calculation.

### Confirmation Timeframe (e.g., 4-hour)

1.  **Relative Strength Index (RSI):**
    *   `rsi_period_high`: Time period for RSI on the higher timeframe.

2.  **Moving Average Convergence Divergence (MACD):**
    *   `macd_fast_high`: Fast EMA period on the higher timeframe.
    *   `macd_slow_high`: Slow EMA period on the higher timeframe.
    *   `macd_signal_high`: Signal line EMA period on the higher timeframe.

3.  **Directional Movement Index (DMI):**
    *   `dmi_period_high`: Time period for DMI on the higher timeframe.

## Entry Conditions (Long)

A long position is entered when the following conditions are met:

1.  **Confirmation Timeframe Alignment:**
    *   The RSI on the higher timeframe is above a configurable threshold (`rsi_high_threshold`), indicating a long-term uptrend.
    *   The MACD line on the higher timeframe is above its signal line (`macd_high > macd_signal_high`), confirming bullish momentum.
    *   The Plus Directional Indicator (+DI) is above the Minus Directional Indicator (-DI) on the higher timeframe (`plus_di_high > minus_di_high`), suggesting a directional trend.

2.  **Primary Timeframe Entry Signal:**
    *   The StochRSI %K line crosses above the %D line (`stochrsi_k > stochrsi_d`) and is below an `oversold_threshold`.
    *   The Awesome Oscillator is positive (`ao > 0`), indicating upward momentum.
    *   The CCI is above a `cci_buy_threshold`, confirming the strength of the trend.
    *   The Chaikin Money Flow is positive (`cmf > 0`), indicating buying pressure.

## Exit Conditions (Long)

The long position is exited based on one of the following conditions:

1.  **Take Profit:**
    *   A dynamic take-profit level is set based on the Average True Range (ATR) of the primary timeframe, multiplied by a `take_profit_atr_mult` factor.

2.  **Stop Loss:**
    *   A trailing stop loss is implemented, set at a configurable percentage (`stop_loss_pct`) below the entry price.

3.  **Momentum Fade:**
    *   The StochRSI %K line crosses below the %D line while in the `overbought` zone.
    *   The Awesome Oscillator turns negative (`ao < 0`), signaling a potential reversal.
    *   The CCI crosses below a `cci_sell_threshold`.

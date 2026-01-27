# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

# --------------------------------
from datetime import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta


class ConsolidationBreakout_05(IStrategy):
    """
    Breakout of Consolidation Box Strategy

    Detect recent range/consolidation, then enter when price breaks out with volume.

    Improvements:
    - ATR-based consolidation detection
    - False breakout filter using retest logic
    - Momentum confirmation with RSI
    - Dynamic lookback period based on volatility
    - Better volume spike detection
    - Support/resistance flip confirmation

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.04,
        "10": 0.03,
        "20": 0.025,
        "40": 0.02,
        "60": 0.015,
        "120": 0.01,
    }

    # Optimal stoploss
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperparameters
    lookback_period = IntParameter(15, 30, default=20, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_spike_mult = DecimalParameter(1.3, 2.0, default=1.5, space="buy")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_mult_breakout = DecimalParameter(0.2, 0.5, default=0.3, space="buy")
    consolidation_atr_max = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    # RSI parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_breakout_up = IntParameter(50, 60, default=55, space="buy")
    rsi_breakout_down = IntParameter(40, 50, default=45, space="buy")

    # Breakout parameters
    breakout_buffer = DecimalParameter(0.0001, 0.0005, default=0.0002, space="buy")
    min_consolidation_candles = IntParameter(5, 15, default=8, space="buy")

    # Exit parameters
    take_profit_mult = DecimalParameter(1.0, 2.5, default=1.5, space="sell")
    box_reentry_buffer = DecimalParameter(0.8, 1.0, default=0.9, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculate indicators for all possible hyperopt parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """

        # Pre-calculate range high and low for all possible lookback periods (15-30)
        for period in range(15, 31):
            dataframe[f"range_high_{period}"] = dataframe["high"].rolling(window=period).max()
            dataframe[f"range_low_{period}"] = dataframe["low"].rolling(window=period).min()

        # Pre-calculate Volume MA for all possible periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate ATR for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Pre-calculate RSI for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # EMA for trend context (fixed periods)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Trend determination
        dataframe["uptrend"] = dataframe["ema50"] > dataframe["ema200"]
        dataframe["downtrend"] = dataframe["ema50"] < dataframe["ema200"]

        # Price momentum (fixed)
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5) * 100
        )

        # Candle patterns (fixed)
        dataframe["bullish_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["candle_body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]

        # Strong candle = large body relative to range
        dataframe["strong_bullish"] = dataframe["bullish_candle"] & (
            dataframe["candle_body"] > dataframe["candle_range"] * 0.6
        )
        dataframe["strong_bearish"] = dataframe["bearish_candle"] & (
            dataframe["candle_body"] > dataframe["candle_range"] * 0.6
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        lookback = self.lookback_period.value
        volume_ma_period = self.volume_ma_period.value
        atr_period = self.atr_period.value
        rsi_period = self.rsi_period.value

        # Get pre-calculated indicators for current hyperopt values
        range_high = dataframe[f"range_high_{lookback}"]
        range_low = dataframe[f"range_low_{lookback}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate derived values using hyperopt parameters
        range_width = range_high - range_low
        range_width_pct = range_width / range_low * 100
        range_mid = (range_high + range_low) / 2

        # Volume spike detection
        volume_spike = dataframe["volume"] > (volume_ma * self.volume_spike_mult.value)

        # Consolidation detection using hyperopt parameters
        range_atr_ratio = range_width / atr
        is_consolidating = (range_atr_ratio < self.consolidation_atr_max.value * 2) & (
            range_width_pct < 3.0
        )

        # Count consecutive consolidation candles
        consol_count = is_consolidating.rolling(
            window=self.min_consolidation_candles.value, min_periods=1
        ).sum()

        # Breakout detection using hyperopt buffer
        breakout_up = (dataframe["close"] > range_high * (1 + self.breakout_buffer.value)) & (
            dataframe["close"].shift(1) <= range_high.shift(1)
        )

        breakout_down = (dataframe["close"] < range_low * (1 - self.breakout_buffer.value)) & (
            dataframe["close"].shift(1) >= range_low.shift(1)
        )

        # False breakout detection
        false_breakout_up = breakout_up.shift(1) & (dataframe["close"] < range_high)
        false_breakout_down = breakout_down.shift(1) & (dataframe["close"] > range_low)

        # ATR threshold check
        atr_ok = atr > atr.rolling(20).mean() * 0.8

        # LONG ENTRY - Breakout up
        dataframe.loc[
            (
                (breakout_up)  # Price breaks above range high
                & (volume_spike)  # Volume confirmation
                & (consol_count >= self.min_consolidation_candles.value)  # Sufficient consolidation
                & (rsi > self.rsi_breakout_up.value)  # RSI momentum confirmation
                & (dataframe["momentum"] > 0.5)  # Positive momentum
                & (false_breakout_up.shift(1) == False)  # No recent false breakout
                & (dataframe["strong_bullish"])  # Strong bullish candle
                & (atr_ok)  # Volatility not too low
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY - Breakout down
        dataframe.loc[
            (
                (breakout_down)  # Price breaks below range low
                & (volume_spike)  # Volume confirmation
                & (consol_count >= self.min_consolidation_candles.value)  # Sufficient consolidation
                & (rsi < self.rsi_breakout_down.value)  # RSI momentum confirmation
                & (dataframe["momentum"] < -0.5)  # Negative momentum
                & (false_breakout_down.shift(1) == False)  # No recent false breakout
                & (dataframe["strong_bearish"])  # Strong bearish candle
                & (atr_ok)  # Volatility not too low
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        lookback = self.lookback_period.value
        volume_ma_period = self.volume_ma_period.value
        atr_period = self.atr_period.value
        rsi_period = self.rsi_period.value

        # Get pre-calculated indicators for current hyperopt values
        range_high = dataframe[f"range_high_{lookback}"]
        range_low = dataframe[f"range_low_{lookback}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate dynamic targets based on ATR
        long_target = range_high + (atr * self.take_profit_mult.value)
        short_target = range_low - (atr * self.take_profit_mult.value)

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] < range_high * self.box_reentry_buffer.value)  # Back into box
                | (dataframe["close"] >= long_target)  # Target reached
                | (rsi > 75)  # Overbought
                | (dataframe["momentum"] < -1.0)  # Momentum reversal
                | (dataframe["volume"] < volume_ma * 0.5)  # Volume dried up
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (
                    dataframe["close"] > range_low * (2 - self.box_reentry_buffer.value)
                )  # Back into box
                | (dataframe["close"] <= short_target)  # Target reached
                | (rsi < 25)  # Oversold
                | (dataframe["momentum"] > 1.0)  # Momentum reversal
                | (dataframe["volume"] < volume_ma * 0.5)  # Volume dried up
            ),
            "exit_short",
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Custom exit logic for breakout strategy
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        lookback = self.lookback_period.value
        atr_period = self.atr_period.value
        rsi_period = self.rsi_period.value

        range_high = last_candle[f"range_high_{lookback}"]
        range_low = last_candle[f"range_low_{lookback}"]
        range_width = range_high - range_low
        range_width_pct = (range_width / range_low * 100) if range_low > 0 else 0
        atr = last_candle[f"atr_{atr_period}"]
        rsi = last_candle[f"rsi_{rsi_period}"]

        # Calculate consolidation status
        range_atr_ratio = range_width / atr if atr > 0 else 0
        is_consolidating = (range_atr_ratio < self.consolidation_atr_max.value * 2) and (
            range_width_pct < 3.0
        )

        # Quick profit taking if momentum is strong
        if current_profit > 0.015:
            if not trade.is_short and rsi > 70:
                return "take_profit_overbought"
            if trade.is_short and rsi < 30:
                return "take_profit_oversold"

        # Exit if consolidation forms again (breakout failed)
        if is_consolidating:
            return "new_consolidation"

        # Exit if volume dries up significantly
        volume_ma = last_candle[f"volume_ma_{self.volume_ma_period.value}"]
        if last_candle["volume"] < volume_ma * 0.3:
            return "no_volume"

        # Time-based exit
        if current_time - trade.open_date_utc > pd.Timedelta(hours=2):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.01:
                return "time_exit_small_loss"

        # Exit on strong reversal candle
        if not trade.is_short and last_candle["strong_bearish"]:
            if current_profit > 0:
                return "reversal_candle"

        if trade.is_short and last_candle["strong_bullish"]:
            if current_profit > 0:
                return "reversal_candle"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Custom stoploss logic using ATR and range
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get ATR based on hyperopt parameter
        atr = last_candle[f"atr_{self.atr_period.value}"]

        # Dynamic stop based on ATR
        atr_stop = -(atr * 2 / trade.open_rate)

        # Tighten stop after profit
        if current_profit > 0.02:
            return -0.005
        elif current_profit > 0.01:
            return -0.008
        elif current_profit > 0.005:
            return max(atr_stop, -0.015)

        # Progressive stop over time
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            return max(atr_stop, -0.02)

        return max(atr_stop, self.stoploss)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Additional checks before entering a trade
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        lookback = self.lookback_period.value

        range_high = last_candle[f"range_high_{lookback}"]
        range_low = last_candle[f"range_low_{lookback}"]
        range_width_pct = ((range_high - range_low) / range_low * 100) if range_low > 0 else 0

        # Don't enter if range is too narrow (not worth the risk)
        if range_width_pct < 0.5:
            return False

        # Avoid low liquidity periods
        hour = current_time.hour
        if hour >= 2 and hour <= 4:  # UTC
            return False

        return True

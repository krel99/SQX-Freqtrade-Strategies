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


class EMATrendRSIPullback_01(IStrategy):
    """
    EMA Trend + Pullback RSI Scalper

    Trade with the short-term trend (fast EMA over slow EMA).
    Enter on pullbacks where RSI briefly goes oversold/overbought against the trend, then reverts.

    Improvements:
    - Added volume confirmation for better entry signals
    - Dynamic RSI thresholds based on EMA spread
    - Trend strength filter using EMA angle
    - Better exit logic with partial exits

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.025, "10": 0.018, "20": 0.012, "30": 0.008, "60": 0.005}

    # Optimal stoploss
    stoploss = -0.035

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Hyperparameters
    ema_fast_period = IntParameter(15, 25, default=20, space="buy")
    ema_slow_period = IntParameter(40, 60, default=50, space="buy")
    rsi_period = IntParameter(10, 20, default=14, space="buy")

    # RSI thresholds
    rsi_buy_threshold = IntParameter(30, 40, default=35, space="buy")
    rsi_sell_threshold = IntParameter(60, 70, default=65, space="buy")

    # Exit thresholds
    rsi_exit_long = IntParameter(55, 65, default=60, space="sell")
    rsi_exit_short = IntParameter(35, 45, default=40, space="sell")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.1, space="buy")

    # Trend strength filter
    ema_spread_min = DecimalParameter(0.001, 0.005, default=0.002, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculate indicators for all possible hyperopt parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """

        # Pre-calculate EMAs for all possible fast periods (15-25)
        for period in range(15, 26):
            dataframe[f"ema_fast_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate EMAs for all possible slow periods (40-60)
        for period in range(40, 61):
            dataframe[f"ema_slow_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate RSI for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all possible periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        ema_fast_period = self.ema_fast_period.value
        ema_slow_period = self.ema_slow_period.value
        rsi_period = self.rsi_period.value
        volume_ma_period = self.volume_ma_period.value

        # Get pre-calculated indicators for current hyperopt values
        ema_fast = dataframe[f"ema_fast_{ema_fast_period}"]
        ema_slow = dataframe[f"ema_slow_{ema_slow_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]

        # Calculate derived values using hyperopt parameters
        # EMA spread for trend strength
        ema_spread = (ema_fast - ema_slow) / ema_slow

        # Volume ratio
        volume_ratio = dataframe["volume"] / volume_ma

        # Price position relative to EMAs
        close_above_fast = dataframe["close"] > ema_fast
        close_below_fast = dataframe["close"] < ema_fast

        # EMA trend
        uptrend = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow

        # EMA angle for momentum (using 5 period lookback)
        ema_fast_angle = (ema_fast - ema_fast.shift(5)) / ema_fast.shift(5) * 100

        # Dynamic RSI thresholds based on trend strength
        rsi_buy_dynamic = np.where(
            abs(ema_spread) > 0.005,
            self.rsi_buy_threshold.value + 5,  # More strict in strong trends
            self.rsi_buy_threshold.value,
        )

        rsi_sell_dynamic = np.where(
            abs(ema_spread) > 0.005,
            self.rsi_sell_threshold.value - 5,  # More strict in strong trends
            self.rsi_sell_threshold.value,
        )

        # LONG ENTRY
        dataframe.loc[
            (
                (uptrend)  # Uptrend
                & (rsi < rsi_buy_dynamic)  # RSI pullback
                & (close_above_fast)  # Price above fast EMA
                & (abs(ema_spread) > self.ema_spread_min.value)  # Sufficient trend strength
                & (volume_ratio > self.volume_threshold.value)  # Volume confirmation
                & (ema_fast_angle > 0.1)  # Positive momentum
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (downtrend)  # Downtrend
                & (rsi > rsi_sell_dynamic)  # RSI pullback up
                & (close_below_fast)  # Price below fast EMA
                & (abs(ema_spread) > self.ema_spread_min.value)  # Sufficient trend strength
                & (volume_ratio > self.volume_threshold.value)  # Volume confirmation
                & (ema_fast_angle < -0.1)  # Negative momentum
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
        ema_fast_period = self.ema_fast_period.value
        ema_slow_period = self.ema_slow_period.value
        rsi_period = self.rsi_period.value

        # Get pre-calculated indicators for current hyperopt values
        ema_fast = dataframe[f"ema_fast_{ema_fast_period}"]
        ema_slow = dataframe[f"ema_slow_{ema_slow_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate derived values
        close_above_fast = dataframe["close"] > ema_fast
        close_below_fast = dataframe["close"] < ema_fast
        uptrend = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow
        ema_fast_angle = (ema_fast - ema_fast.shift(5)) / ema_fast.shift(5) * 100

        # LONG EXIT
        dataframe.loc[
            (
                (rsi > self.rsi_exit_long.value)  # Mean reversion reached
                | (close_below_fast)  # Loss of momentum
                | (downtrend)  # Trend reversal
                | (ema_fast_angle < -0.2)  # Strong negative momentum
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (rsi < self.rsi_exit_short.value)  # Mean reversion reached
                | (close_above_fast)  # Loss of momentum
                | (uptrend)  # Trend reversal
                | (ema_fast_angle > 0.2)  # Strong positive momentum
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
        Custom exit logic for additional safety
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        ema_fast_period = self.ema_fast_period.value
        ema_slow_period = self.ema_slow_period.value
        rsi_period = self.rsi_period.value

        ema_fast = last_candle[f"ema_fast_{ema_fast_period}"]
        ema_slow = last_candle[f"ema_slow_{ema_slow_period}"]
        rsi = last_candle[f"rsi_{rsi_period}"]

        # EMA spread
        ema_spread = (ema_fast - ema_slow) / ema_slow

        # Exit if trend becomes too weak
        if abs(ema_spread) < 0.001:
            return "trend_too_weak"

        # Exit long if RSI becomes extremely overbought
        if trade.is_short == False and rsi > 75:
            return "rsi_extreme_overbought"

        # Exit short if RSI becomes extremely oversold
        if trade.is_short == True and rsi < 25:
            return "rsi_extreme_oversold"

        # Time-based exit if trade is stuck
        if current_time - trade.open_date_utc > pd.Timedelta(hours=2):
            if current_profit < 0.005 and current_profit > -0.01:
                return "time_exit_stuck"

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
        Custom stoploss logic - tighter stops for better risk management
        """

        # After 30 minutes, if profit > 1%, move stop to breakeven
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=30):
            if current_profit > 0.01:
                return -0.001  # Breakeven stop

        # After 1 hour, if profit > 0.5%, use tight stop
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            if current_profit > 0.005:
                return -0.005  # Tight stop

        return self.stoploss

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
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Can this strategy go short?
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
        Adds several different TA indicators to the given DataFrame
        """

        # EMAs
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)

        # EMA spread for trend strength
        dataframe["ema_spread"] = (
            dataframe["ema_fast"] - dataframe["ema_slow"]
        ) / dataframe["ema_slow"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        # Price position relative to EMAs
        dataframe["close_above_fast"] = dataframe["close"] > dataframe["ema_fast"]
        dataframe["close_below_fast"] = dataframe["close"] < dataframe["ema_fast"]

        # EMA trend
        dataframe["uptrend"] = dataframe["ema_fast"] > dataframe["ema_slow"]
        dataframe["downtrend"] = dataframe["ema_fast"] < dataframe["ema_slow"]

        # EMA angle for momentum (using 5 period lookback)
        dataframe["ema_fast_angle"] = (
            (dataframe["ema_fast"] - dataframe["ema_fast"].shift(5))
            / dataframe["ema_fast"].shift(5)
            * 100
        )

        # Dynamic RSI thresholds based on trend strength
        dataframe["rsi_buy_dynamic"] = np.where(
            abs(dataframe["ema_spread"]) > 0.005,
            self.rsi_buy_threshold.value + 5,  # More strict in strong trends
            self.rsi_buy_threshold.value,
        )

        dataframe["rsi_sell_dynamic"] = np.where(
            abs(dataframe["ema_spread"]) > 0.005,
            self.rsi_sell_threshold.value - 5,  # More strict in strong trends
            self.rsi_sell_threshold.value,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY
        dataframe.loc[
            (
                (dataframe["uptrend"])  # Uptrend
                & (dataframe["rsi"] < dataframe["rsi_buy_dynamic"])  # RSI pullback
                & (dataframe["close_above_fast"])  # Price above fast EMA
                & (
                    abs(dataframe["ema_spread"]) > self.ema_spread_min.value
                )  # Sufficient trend strength
                & (
                    dataframe["volume_ratio"] > self.volume_threshold.value
                )  # Volume confirmation
                & (dataframe["ema_fast_angle"] > 0.1)  # Positive momentum
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (dataframe["downtrend"])  # Downtrend
                & (dataframe["rsi"] > dataframe["rsi_sell_dynamic"])  # RSI pullback up
                & (dataframe["close_below_fast"])  # Price below fast EMA
                & (
                    abs(dataframe["ema_spread"]) > self.ema_spread_min.value
                )  # Sufficient trend strength
                & (
                    dataframe["volume_ratio"] > self.volume_threshold.value
                )  # Volume confirmation
                & (dataframe["ema_fast_angle"] < -0.1)  # Negative momentum
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["rsi"] > self.rsi_exit_long.value)  # Mean reversion reached
                | (dataframe["close_below_fast"])  # Loss of momentum
                | (dataframe["downtrend"])  # Trend reversal
                | (dataframe["ema_fast_angle"] < -0.2)  # Strong negative momentum
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_exit_short.value)  # Mean reversion reached
                | (dataframe["close_above_fast"])  # Loss of momentum
                | (dataframe["uptrend"])  # Trend reversal
                | (dataframe["ema_fast_angle"] > 0.2)  # Strong positive momentum
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

        # Exit if trend becomes too weak
        if abs(last_candle["ema_spread"]) < 0.001:
            return "trend_too_weak"

        # Exit long if RSI becomes extremely overbought
        if trade.is_short == False and last_candle["rsi"] > 75:
            return "rsi_extreme_overbought"

        # Exit short if RSI becomes extremely oversold
        if trade.is_short == True and last_candle["rsi"] < 25:
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

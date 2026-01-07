# --- Do not remove these imports ---
from functools import reduce
from typing import Dict, List

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


class FlexibleScoringStrategy(IStrategy):
    """
    Dynamic Multi-Indicator Strategy - Improved Version
    Uses a combination of Bollinger Bands, Keltner Channels, EMAs, and RSI
    with more logical entry/exit conditions
    """

    # Strategy interface version
    INTERFACE_VERSION = 3
    # Timeframe
    timeframe = "15m"

    # Can short
    can_short = True

    # ROI table
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0.01}

    # Stoploss
    stoploss = -0.10

    # Trailing stop
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Exit signal
    use_exit_signal = CategoricalParameter([True, False], default=True, space="sell")

    # Indicator Parameters
    # Bollinger Bands 1 (Primary)
    bb1_period = IntParameter(10, 50, default=20, space="buy")
    bb1_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # Bollinger Bands 2 (Secondary)
    bb2_period = IntParameter(15, 60, default=30, space="buy")
    bb2_std = DecimalParameter(1.5, 3.5, default=2.5, decimals=1, space="buy")

    # Keltner Channel 1 (Primary)
    kc1_period = IntParameter(10, 50, default=20, space="buy")
    kc1_multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy")

    # Keltner Channel 2 (Secondary)
    kc2_period = IntParameter(15, 60, default=30, space="buy")
    kc2_multiplier = DecimalParameter(1.5, 4.0, default=2.0, decimals=1, space="buy")

    # Moving Averages
    ema_short_period = IntParameter(5, 25, default=9, space="buy")
    ema_medium_period = IntParameter(20, 50, default=21, space="buy")
    ema_long_period = IntParameter(50, 200, default=50, space="buy")

    # ATR
    atr_period = IntParameter(7, 21, default=14, space="buy")

    # RSI Parameters
    rsi_period = IntParameter(7, 21, default=14, space="buy")
    buy_rsi_value = IntParameter(20, 45, default=35, space="buy")
    sell_rsi_value = IntParameter(55, 80, default=65, space="sell")

    # Band proximity thresholds (how close to bands price needs to be)
    bb_proximity = DecimalParameter(0.98, 1.02, default=1.0, decimals=3, space="buy")
    kc_proximity = DecimalParameter(0.98, 1.02, default=1.0, decimals=3, space="buy")

    # Trend strength parameter (0 = no trend filter, 1 = strict trend filter)
    trend_strength = DecimalParameter(0.0, 1.0, default=0.5, decimals=2, space="buy")

    # Exit Thresholds
    exit_long_bb_mult = DecimalParameter(0.95, 1.05, default=1.0, decimals=3, space="sell")
    exit_short_bb_mult = DecimalParameter(0.95, 1.05, default=1.0, decimals=3, space="sell")
    exit_rsi_long = IntParameter(60, 85, default=70, space="sell")
    exit_rsi_short = IntParameter(15, 40, default=30, space="sell")

    # Minimum required conditions for entry (1-4)
    min_conditions_long = IntParameter(1, 4, default=2, space="buy")
    min_conditions_short = IntParameter(1, 4, default=2, space="buy")

    def _get_param_value(self, param):
        """Helper function to get parameter value"""
        return param.value if hasattr(param, "value") else param

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands 1
        bb1 = ta.BBANDS(
            dataframe,
            timeperiod=self._get_param_value(self.bb1_period),
            nbdevup=self._get_param_value(self.bb1_std),
            nbdevdn=self._get_param_value(self.bb1_std),
        )
        dataframe["bb1_lower"] = bb1["lowerband"]
        dataframe["bb1_middle"] = bb1["middleband"]
        dataframe["bb1_upper"] = bb1["upperband"]
        dataframe["bb1_width"] = (bb1["upperband"] - bb1["lowerband"]) / bb1["middleband"]

        # Bollinger Bands 2
        bb2 = ta.BBANDS(
            dataframe,
            timeperiod=self._get_param_value(self.bb2_period),
            nbdevup=self._get_param_value(self.bb2_std),
            nbdevdn=self._get_param_value(self.bb2_std),
        )
        dataframe["bb2_lower"] = bb2["lowerband"]
        dataframe["bb2_middle"] = bb2["middleband"]
        dataframe["bb2_upper"] = bb2["upperband"]
        dataframe["bb2_width"] = (bb2["upperband"] - bb2["lowerband"]) / bb2["middleband"]

        # Keltner Channel 1
        kc1_period = self._get_param_value(self.kc1_period)
        kc1_multiplier = self._get_param_value(self.kc1_multiplier)
        dataframe["kc1_atr"] = ta.ATR(dataframe, timeperiod=kc1_period)
        dataframe["kc1_middle"] = ta.EMA(dataframe, timeperiod=kc1_period)
        dataframe["kc1_upper"] = dataframe["kc1_middle"] + dataframe["kc1_atr"] * kc1_multiplier
        dataframe["kc1_lower"] = dataframe["kc1_middle"] - dataframe["kc1_atr"] * kc1_multiplier

        # Keltner Channel 2
        kc2_period = self._get_param_value(self.kc2_period)
        kc2_multiplier = self._get_param_value(self.kc2_multiplier)
        dataframe["kc2_atr"] = ta.ATR(dataframe, timeperiod=kc2_period)
        dataframe["kc2_middle"] = ta.EMA(dataframe, timeperiod=kc2_period)
        dataframe["kc2_upper"] = dataframe["kc2_middle"] + dataframe["kc2_atr"] * kc2_multiplier
        dataframe["kc2_lower"] = dataframe["kc2_middle"] - dataframe["kc2_atr"] * kc2_multiplier

        # EMAs
        dataframe["ema_short"] = ta.EMA(
            dataframe, timeperiod=self._get_param_value(self.ema_short_period)
        )
        dataframe["ema_medium"] = ta.EMA(
            dataframe, timeperiod=self._get_param_value(self.ema_medium_period)
        )
        dataframe["ema_long"] = ta.EMA(
            dataframe, timeperiod=self._get_param_value(self.ema_long_period)
        )

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self._get_param_value(self.rsi_period))

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self._get_param_value(self.atr_period))

        # Volume indicators
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        # Price position relative to bands (normalized 0-1)
        dataframe["bb1_position"] = (dataframe["close"] - dataframe["bb1_lower"]) / (
            dataframe["bb1_upper"] - dataframe["bb1_lower"]
        )
        dataframe["kc1_position"] = (dataframe["close"] - dataframe["kc1_lower"]) / (
            dataframe["kc1_upper"] - dataframe["kc1_lower"]
        )

        # Trend indicators
        dataframe["trend_ema"] = (
            (dataframe["ema_short"] > dataframe["ema_medium"]).astype(int)
            + (dataframe["ema_medium"] > dataframe["ema_long"]).astype(int)
        ) / 2.0  # 0 = strong down, 0.5 = neutral, 1 = strong up

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get parameter values
        bb_prox = self._get_param_value(self.bb_proximity)
        kc_prox = self._get_param_value(self.kc_proximity)
        trend_str = self._get_param_value(self.trend_strength)
        min_long = self._get_param_value(self.min_conditions_long)
        min_short = self._get_param_value(self.min_conditions_short)
        buy_rsi = self._get_param_value(self.buy_rsi_value)
        sell_rsi = self._get_param_value(self.sell_rsi_value)

        # LONG CONDITIONS - Count how many conditions are met
        long_conditions = []

        # Condition 1: Price near or below BB1 lower band
        long_conditions.append((dataframe["close"] <= dataframe["bb1_lower"] * bb_prox))

        # Condition 2: Price near or below KC1 lower band
        long_conditions.append((dataframe["close"] <= dataframe["kc1_lower"] * kc_prox))

        # Condition 3: RSI oversold
        long_conditions.append((dataframe["rsi"] < buy_rsi))

        # Condition 4: Trend alignment (weighted by trend_strength parameter)
        # When trend_strength is 0, this condition is always True
        # When trend_strength is 1, requires full uptrend
        long_conditions.append((dataframe["trend_ema"] >= (0.5 - trend_str * 0.5)))

        # Condition 5: Volume above average (optional boost)
        long_conditions.append((dataframe["volume"] > dataframe["volume_mean"] * 0.8))

        # Count met conditions and require minimum
        long_score = reduce(
            lambda x, y: x + y.astype(int), long_conditions[1:], long_conditions[0].astype(int)
        )

        # SHORT CONDITIONS - Count how many conditions are met
        short_conditions = []

        # Condition 1: Price near or above BB1 upper band
        short_conditions.append((dataframe["close"] >= dataframe["bb1_upper"] / bb_prox))

        # Condition 2: Price near or above KC1 upper band
        short_conditions.append((dataframe["close"] >= dataframe["kc1_upper"] / kc_prox))

        # Condition 3: RSI overbought
        short_conditions.append((dataframe["rsi"] > sell_rsi))

        # Condition 4: Trend alignment (weighted by trend_strength parameter)
        short_conditions.append((dataframe["trend_ema"] <= (0.5 + trend_str * 0.5)))

        # Condition 5: Volume above average (optional boost)
        short_conditions.append((dataframe["volume"] > dataframe["volume_mean"] * 0.8))

        # Count met conditions and require minimum
        short_score = reduce(
            lambda x, y: x + y.astype(int), short_conditions[1:], short_conditions[0].astype(int)
        )

        # Set entry signals based on minimum conditions met
        dataframe.loc[long_score >= min_long, "enter_long"] = 1
        dataframe.loc[short_score >= min_short, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self._get_param_value(self.use_exit_signal):
            exit_long_bb = self._get_param_value(self.exit_long_bb_mult)
            exit_short_bb = self._get_param_value(self.exit_short_bb_mult)
            exit_rsi_l = self._get_param_value(self.exit_rsi_long)
            exit_rsi_s = self._get_param_value(self.exit_rsi_short)

            # Exit long conditions (any of these)
            exit_long_conditions = [
                (
                    dataframe["close"] > dataframe["bb1_upper"] * exit_long_bb
                ),  # Price above BB upper
                (dataframe["close"] > dataframe["kc1_upper"]),  # Price above KC upper
                (dataframe["rsi"] > exit_rsi_l),  # RSI overbought
                (dataframe["ema_short"] < dataframe["ema_medium"]),  # Trend reversal
            ]

            # Exit short conditions (any of these)
            exit_short_conditions = [
                (
                    dataframe["close"] < dataframe["bb1_lower"] * exit_short_bb
                ),  # Price below BB lower
                (dataframe["close"] < dataframe["kc1_lower"]),  # Price below KC lower
                (dataframe["rsi"] < exit_rsi_s),  # RSI oversold
                (dataframe["ema_short"] > dataframe["ema_medium"]),  # Trend reversal
            ]

            # Use OR logic for exits (any condition triggers exit)
            dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), "exit_long"] = 1
            dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), "exit_short"] = 1

        return dataframe

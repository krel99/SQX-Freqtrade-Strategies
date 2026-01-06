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


class Strategy07012026(IStrategy):
    """
    Dynamic Multi-Indicator Strategy
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
    # Bollinger Bands 1
    bb1_period = IntParameter(10, 50, default=20, space="buy")
    bb1_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # Bollinger Bands 2
    bb2_period = IntParameter(10, 50, default=20, space="buy")
    bb2_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # Keltner Channel 1
    kc1_period = IntParameter(10, 50, default=20, space="buy")
    kc1_multiplier = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="buy")

    # Keltner Channel 2
    kc2_period = IntParameter(10, 50, default=20, space="buy")
    kc2_multiplier = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="buy")

    # Moving Averages
    ema_short_period = IntParameter(5, 25, default=9, space="buy")
    ema_medium_period = IntParameter(20, 50, default=21, space="buy")
    ema_long_period = IntParameter(50, 200, default=50, space="buy")

    # ATR
    atr_period = IntParameter(7, 21, default=14, space="buy")

    # Entry/Exit Thresholds
    buy_rsi_value = IntParameter(20, 50, default=30, space="buy")
    sell_rsi_value = IntParameter(50, 80, default=70, space="sell")
    exit_long_threshold = DecimalParameter(1.0, 1.2, default=1.1, decimals=2, space="sell")
    exit_short_threshold = DecimalParameter(0.8, 1.0, default=0.9, decimals=2, space="sell")

    # Protection Parameters
    protection_stoploss_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_stoploss_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_stoploss_lookback_period = IntParameter(
        10, 1440, default=60, space="protection"
    )
    protection_stoploss_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )

    protection_cooldown_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_cooldown_period = IntParameter(1, 20, default=5, space="protection")

    protection_lowprofit_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_lowprofit_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_lowprofit_lookback_period = IntParameter(
        10, 1440, default=360, space="protection"
    )
    protection_lowprofit_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )
    protection_lowprofit_required_profit = DecimalParameter(
        -0.05, 0.05, default=0.0, decimals=3, space="protection"
    )

    protection_maxdrawdown_enabled = CategoricalParameter(
        [True, False], default=False, space="protection"
    )
    protection_maxdrawdown_trade_limit = IntParameter(3, 20, default=8, space="protection")
    protection_maxdrawdown_lookback_period = IntParameter(
        10, 1440, default=200, space="protection"
    )
    protection_maxdrawdown_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )
    protection_maxdrawdown_allowed_drawdown = DecimalParameter(
        0.01, 0.30, default=0.10, decimals=2, space="protection"
    )

    def _get_param_value(self, param):
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
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self._get_param_value(self.ema_short_period))
        dataframe["ema_medium"] = ta.EMA(dataframe, timeperiod=self._get_param_value(self.ema_medium_period))
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self._get_param_value(self.ema_long_period))

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self._get_param_value(self.atr_period))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = [
            (dataframe["ema_short"] > dataframe["ema_medium"]),
            (dataframe["ema_medium"] > dataframe["ema_long"]),
            (dataframe["close"] < dataframe["bb1_lower"]),
            (dataframe["close"] < dataframe["kc1_lower"]),
            (dataframe["close"] < dataframe["bb2_lower"]),
            (dataframe["close"] < dataframe["kc2_lower"]),
            (dataframe["rsi"] < self._get_param_value(self.buy_rsi_value)),
        ]

        conditions_short = [
            (dataframe["ema_short"] < dataframe["ema_medium"]),
            (dataframe["ema_medium"] < dataframe["ema_long"]),
            (dataframe["close"] > dataframe["bb1_upper"]),
            (dataframe["close"] > dataframe["kc1_upper"]),
            (dataframe["close"] > dataframe["bb2_upper"]),
            (dataframe["close"] > dataframe["kc2_upper"]),
            (dataframe["rsi"] > self._get_param_value(self.sell_rsi_value)),
        ]

        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), "enter_long"] = 1
        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self._get_param_value(self.use_exit_signal):
            dataframe.loc[
                (dataframe["close"] > dataframe["bb1_upper"] * self._get_param_value(self.exit_long_threshold)), "exit_long"
            ] = 1
            dataframe.loc[
                (dataframe["close"] < dataframe["bb1_lower"] * self._get_param_value(self.exit_short_threshold)), "exit_short"
            ] = 1
        return dataframe

    @property
    def protections(self):
        prot = []
        if self._get_param_value(self.protection_stoploss_enabled):
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_stoploss_lookback_period
                    ),
                    "trade_limit": self._get_param_value(
                        self.protection_stoploss_trade_limit
                    ),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_stoploss_stop_duration
                    ),
                    "only_per_pair": False,
                }
            )
        if self._get_param_value(self.protection_cooldown_enabled):
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self._get_param_value(
                        self.protection_cooldown_period
                    ),
                }
            )
        if self._get_param_value(self.protection_lowprofit_enabled):
            prot.append(
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_lowprofit_lookback_period
                    ),
                    "trade_limit": self._get_param_value(
                        self.protection_lowprofit_trade_limit
                    ),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_lowprofit_stop_duration
                    ),
                    "required_profit": self._get_param_value(
                        self.protection_lowprofit_required_profit
                    ),
                    "only_per_pair": True,
                }
            )
        if self._get_param_value(self.protection_maxdrawdown_enabled):
            prot.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_maxdrawdown_lookback_period
                    ),
                    "trade_limit": self._get_param_value(
                        self.protection_maxdrawdown_trade_limit
                    ),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_maxdrawdown_stop_duration
                    ),
                    "max_allowed_drawdown": self._get_param_value(
                        self.protection_maxdrawdown_allowed_drawdown
                    ),
                }
            )
        return prot

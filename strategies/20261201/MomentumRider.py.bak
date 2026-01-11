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
    merge_informative_pair,
)

# --------------------------------
from datetime import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta


class MomentumRider(IStrategy):
    """
    Momentum Rider Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "15m"
    info_timeframe = "1h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.15, "30": 0.05, "60": 0.01}

    # Stoploss
    stoploss = -0.1

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Process only new candles
    process_only_new_candles = True

    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Startup candle count
    startup_candle_count: int = 30

    # --- Hyperparameters ---

    # -- 1h Timeframe --
    buy_ema_1h_period = IntParameter(20, 100, default=50, space="buy")

    # -- 15m Timeframe --
    buy_rsi_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_15m_threshold = IntParameter(40, 70, default=55, space="buy")
    buy_macd_15m_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_15m_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_15m_signal = IntParameter(5, 18, default=9, space="buy")
    buy_bb_15m_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_15m_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    # -- Exit Parameters --
    sell_rsi_15m_threshold = IntParameter(60, 90, default=75, space="sell")

    def informative_pairs(self):
        pairs = self.config["exchange"]["pair_whitelist"]
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.info_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Indicators for 1h timeframe --
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.info_timeframe
        )
        informative["ema_slow"] = ta.EMA(informative, timeperiod=self.buy_ema_1h_period.value)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
        )

        # -- Indicators for 15m timeframe --
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_15m_period.value)
        macd_15m = ta.MACD(
            dataframe,
            fastperiod=self.buy_macd_15m_fast.value,
            slowperiod=self.buy_macd_15m_slow.value,
            signalperiod=self.buy_macd_15m_signal.value,
        )
        dataframe["macd"] = macd_15m["macd"]
        dataframe["macdsignal"] = macd_15m["macdsignal"]
        bb_15m = ta.BBANDS(
            dataframe,
            timeperiod=self.buy_bb_15m_period.value,
            nbdevup=self.buy_bb_15m_stddev.value,
            nbdevdn=self.buy_bb_15m_stddev.value,
        )
        dataframe["bb_upper"] = bb_15m["upperband"]
        dataframe["bb_middle"] = bb_15m["middleband"]
        dataframe["bb_lower"] = bb_15m["lowerband"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 1h Momentum Conditions --
        momentum_1h = (
            dataframe[f"close_{self.info_timeframe}"] > dataframe[f"ema_slow_{self.info_timeframe}"]
        )

        # -- 15m Momentum Conditions --
        momentum_15m = (dataframe["rsi"] > self.buy_rsi_15m_threshold.value) & (
            dataframe["macd"] > dataframe["macdsignal"]
        )

        # -- 15m Entry Trigger --
        entry_trigger = dataframe["close"] > dataframe["bb_upper"]

        # -- Combine Conditions --
        dataframe.loc[
            momentum_1h & momentum_15m & entry_trigger,
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Overbought Signal --
        rsi_exit = dataframe["rsi"] > self.sell_rsi_15m_threshold.value

        dataframe.loc[rsi_exit, "exit_long"] = 1

        return dataframe

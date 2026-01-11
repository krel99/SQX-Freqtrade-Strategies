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
from technical.indicators import vortex


class TrendRider_MultiFrame(IStrategy):
    """
    Multi-Timeframe Trend Riding Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "15m"
    info_timeframe = "4h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.3, "60": 0.15, "120": 0.05}

    # Stoploss
    stoploss = -0.15

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Process only new candles
    process_only_new_candles = True

    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Startup candle count
    startup_candle_count: int = 50

    # --- Hyperparameters ---

    # -- 4h Timeframe --
    buy_ema_4h_fast_period = IntParameter(10, 50, default=20, space="buy")
    buy_ema_4h_slow_period = IntParameter(20, 100, default=50, space="buy")
    buy_psar_4h_acceleration = DecimalParameter(0.01, 0.1, default=0.02, space="buy")
    buy_psar_4h_maximum = DecimalParameter(0.1, 0.5, default=0.2, space="buy")
    buy_vortex_4h_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_4h_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_4h_threshold = IntParameter(15, 50, default=25, space="buy")

    # -- 15m Timeframe --
    buy_ema_15m_fast_period = IntParameter(5, 20, default=10, space="buy")
    buy_ema_15m_slow_period = IntParameter(10, 50, default=20, space="buy")
    buy_rsi_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_15m_threshold = IntParameter(40, 60, default=50, space="buy")
    buy_rsi_15m_upper_threshold = IntParameter(60, 80, default=70, space="buy")
    buy_macd_15m_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_15m_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_15m_signal = IntParameter(5, 18, default=9, space="buy")
    buy_adx_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_15m_threshold = IntParameter(15, 50, default=20, space="buy")
    buy_willr_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_willr_15m_threshold = IntParameter(-100, -50, default=-80, space="buy")

    # -- Exit Parameters --
    sell_rsi_15m_threshold = IntParameter(70, 90, default=75, space="sell")
    sell_ema_15m_fast_period = IntParameter(5, 20, default=10, space="sell")
    sell_ema_15m_slow_period = IntParameter(10, 50, default=20, space="sell")


    def informative_pairs(self):
        pairs = self.config['exchange']['pair_whitelist']
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.info_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Indicators for 4h timeframe --
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.info_timeframe
        )

        informative["ema_fast"] = ta.EMA(informative, timeperiod=self.buy_ema_4h_fast_period.value)
        informative["ema_slow"] = ta.EMA(informative, timeperiod=self.buy_ema_4h_slow_period.value)
        informative["psar"] = ta.SAR(
            informative,
            acceleration=self.buy_psar_4h_acceleration.value,
            maximum=self.buy_psar_4h_maximum.value,
        )
        vortex_4h = vortex(informative, period=self.buy_vortex_4h_period.value)
        informative["vortex_plus"] = vortex_4h["vi_plus"]
        informative["vortex_minus"] = vortex_4h["vi_minus"]
        informative["adx"] = ta.ADX(informative, timeperiod=self.buy_adx_4h_period.value)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
            suffix=f"_{self.info_timeframe}",
        )

        # -- Indicators for 15m timeframe --
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.buy_ema_15m_fast_period.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.buy_ema_15m_slow_period.value)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_15m_period.value)
        macd_15m = ta.MACD(
            dataframe,
            fastperiod=self.buy_macd_15m_fast.value,
            slowperiod=self.buy_macd_15m_slow.value,
            signalperiod=self.buy_macd_15m_signal.value,
        )
        dataframe["macd"] = macd_15m["macd"]
        dataframe["macdsignal"] = macd_15m["macdsignal"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.buy_adx_15m_period.value)
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=self.buy_willr_15m_period.value)

        # -- Exit Indicators --
        dataframe["sell_ema_fast"] = ta.EMA(dataframe, timeperiod=self.sell_ema_15m_fast_period.value)
        dataframe["sell_ema_slow"] = ta.EMA(dataframe, timeperiod=self.sell_ema_15m_slow_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 4h Uptrend Confirmation --
        uptrend_4h = (
            (dataframe[f"ema_fast_{self.info_timeframe}"] > dataframe[f"ema_slow_{self.info_timeframe}"])
            & (dataframe[f"close_{self.info_timeframe}"] > dataframe[f"psar_{self.info_timeframe}"])
            & (dataframe[f"vortex_plus_{self.info_timeframe}"] > dataframe[f"vortex_minus_{self.info_timeframe}"])
            & (dataframe[f"adx_{self.info_timeframe}"] > self.buy_adx_4h_threshold.value)
        )

        # -- 15m Entry Trigger (Pullback) --
        entry_15m = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["rsi"] > self.buy_rsi_15m_threshold.value)
            & (dataframe["rsi"] < self.buy_rsi_15m_upper_threshold.value)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["adx"] > self.buy_adx_15m_threshold.value)
            & (dataframe["willr"] < self.buy_willr_15m_threshold.value)
        )

        # -- Combine Conditions --
        dataframe.loc[uptrend_4h & entry_15m, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Trend Reversal Signal on 15m --
        ema_cross_exit = dataframe["sell_ema_fast"] < dataframe["sell_ema_slow"]

        # -- Overbought Signal on 15m --
        rsi_exit = dataframe["rsi"] > self.sell_rsi_15m_threshold.value

        dataframe.loc[ema_cross_exit | rsi_exit, "exit_long"] = 1

        return dataframe

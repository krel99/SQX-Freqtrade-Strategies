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
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MeanReversion_MultiFrame(IStrategy):
    """
    Multi-Timeframe Mean Reversion Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "5m"
    info_timeframe = "1h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.1, "15": 0.05, "30": 0.01}

    # Stoploss
    stoploss = -0.1

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
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

    # -- 1h Timeframe --
    buy_kc_1h_ema_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_1h_atr_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_1h_mult = DecimalParameter(1.0, 3.0, default=2.0, space="buy")
    buy_stochrsi_1h_period = IntParameter(10, 50, default=14, space="buy")
    buy_stochrsi_1h_fastk = IntParameter(3, 10, default=3, space="buy")
    buy_stochrsi_1h_fastd = IntParameter(3, 10, default=3, space="buy")
    buy_stochrsi_1h_threshold = IntParameter(10, 40, default=20, space="buy")
    buy_willr_1h_period = IntParameter(10, 50, default=14, space="buy")
    buy_willr_1h_threshold = IntParameter(-100, -70, default=-80, space="buy")

    # -- 5m Timeframe --
    buy_bb_5m_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_5m_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    buy_rsi_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_5m_threshold = IntParameter(20, 40, default=30, space="buy")
    buy_macd_5m_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_5m_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_5m_signal = IntParameter(5, 18, default=9, space="buy")
    buy_cci_5m_period = IntParameter(10, 50, default=20, space="buy")
    buy_cci_5m_threshold = IntParameter(-150, -50, default=-100, space="buy")
    buy_mfi_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_mfi_5m_threshold = IntParameter(10, 40, default=20, space="buy")

    # -- Exit Parameters --
    sell_rsi_5m_threshold = IntParameter(60, 90, default=70, space="sell")


    def informative_pairs(self):
        pairs = self.config['exchange']['pair_whitelist']
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.info_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Indicators for 1h timeframe --
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.info_timeframe
        )

        kc_1h_ema = ta.EMA(informative, timeperiod=self.buy_kc_1h_ema_period.value)
        kc_1h_atr = ta.ATR(informative, timeperiod=self.buy_kc_1h_atr_period.value)
        informative["kc_lower"] = kc_1h_ema - kc_1h_atr * self.buy_kc_1h_mult.value

        stoch_rsi = ta.STOCHRSI(
            informative,
            timeperiod=self.buy_stochrsi_1h_period.value,
            fastk_period=self.buy_stochrsi_1h_fastk.value,
            fastd_period=self.buy_stochrsi_1h_fastd.value,
        )
        informative["fastk"] = stoch_rsi["fastk"]
        informative["willr"] = ta.WILLR(informative, timeperiod=self.buy_willr_1h_period.value)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
            suffix=f"_{self.info_timeframe}",
        )

        # -- Indicators for 5m timeframe --
        bb_5m = ta.BBANDS(
            dataframe,
            timeperiod=self.buy_bb_5m_period.value,
            nbdevup=self.buy_bb_5m_stddev.value,
            nbdevdn=self.buy_bb_5m_stddev.value,
        )
        dataframe["bb_lower"] = bb_5m["lowerband"]
        dataframe["bb_middle"] = bb_5m["middleband"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_5m_period.value)
        macd_5m = ta.MACD(
            dataframe,
            fastperiod=self.buy_macd_5m_fast.value,
            slowperiod=self.buy_macd_5m_slow.value,
            signalperiod=self.buy_macd_5m_signal.value,
        )
        dataframe["macd"] = macd_5m["macd"]
        dataframe["macdsignal"] = macd_5m["macdsignal"]
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.buy_cci_5m_period.value)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=self.buy_mfi_5m_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 1h Oversold Condition --
        oversold_1h = (
            (dataframe[f"close_{self.info_timeframe}"] < dataframe[f"kc_lower_{self.info_timeframe}"])
            & (dataframe[f"fastk_{self.info_timeframe}"] < self.buy_stochrsi_1h_threshold.value)
            & (dataframe[f"willr_{self.info_timeframe}"] < self.buy_willr_1h_threshold.value)
        )

        # -- 5m Entry Trigger --
        entry_5m = (
            qtpylib.crossed_above(dataframe["close"], dataframe["bb_lower"])
            & (dataframe["rsi"] < self.buy_rsi_5m_threshold.value)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["cci"] < self.buy_cci_5m_threshold.value)
            & (dataframe["mfi"] < self.buy_mfi_5m_threshold.value)
        )

        # -- Combine Conditions --
        dataframe.loc[oversold_1h & entry_5m, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Price reaches the 5m middle Bollinger Band --
        price_reaches_middle_bb = dataframe["close"] > dataframe["bb_middle"]

        # -- RSI on 5m becomes overbought --
        rsi_exit = dataframe["rsi"] > self.sell_rsi_5m_threshold.value

        dataframe.loc[price_reaches_middle_bb | rsi_exit, "exit_long"] = 1

        return dataframe

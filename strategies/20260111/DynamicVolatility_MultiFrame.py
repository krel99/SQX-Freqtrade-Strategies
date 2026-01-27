# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
#
#
# HAS ERRORS
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
from technical.indicators.volatility import choppiness


class DynamicVolatility_MultiFrame(IStrategy):
    """
    Multi-Timeframe Dynamic Volatility Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "15m"
    info_timeframe = "4h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.25, "45": 0.1, "90": 0.05}

    # Stoploss
    stoploss = -0.12

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
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
    buy_atr_4h_period = IntParameter(10, 50, default=14, space="buy")
    buy_atr_4h_ma_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_4h_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_4h_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    buy_bbw_4h_threshold = DecimalParameter(0.01, 0.1, default=0.03, space="buy")
    buy_chop_4h_period = IntParameter(10, 50, default=14, space="buy")
    buy_chop_4h_threshold = IntParameter(50, 80, default=62, space="buy")

    # -- 15m Timeframe --
    buy_bb_15m_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_15m_stddev = DecimalParameter(1.5, 3.0, default=2.5, space="buy")
    buy_rsi_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_15m_threshold = IntParameter(40, 70, default=55, space="buy")
    buy_macd_15m_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_15m_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_15m_signal = IntParameter(5, 18, default=9, space="buy")
    buy_adx_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_15m_threshold = IntParameter(15, 50, default=25, space="buy")
    buy_di_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_aroon_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_cmo_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_cmo_15m_threshold = IntParameter(0, 50, default=20, space="buy")

    # -- Exit Parameters --
    sell_rsi_15m_threshold = IntParameter(65, 95, default=80, space="sell")

    def informative_pairs(self):
        pairs = self.config["exchange"]["pair_whitelist"]
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.info_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Indicators for 4h timeframe --
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.info_timeframe
        )

        informative["atr"] = ta.ATR(informative, timeperiod=self.buy_atr_4h_period.value)
        informative["atr_ma"] = ta.SMA(
            informative["atr"], timeperiod=self.buy_atr_4h_ma_period.value
        )

        bb_4h = ta.BBANDS(
            informative,
            timeperiod=self.buy_bb_4h_period.value,
            nbdevup=self.buy_bb_4h_stddev.value,
            nbdevdn=self.buy_bb_4h_stddev.value,
        )
        informative["bb_width"] = (bb_4h["upperband"] - bb_4h["lowerband"]) / bb_4h["middleband"]
        informative["chop"] = choppiness(informative, period=self.buy_chop_4h_period.value)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
            append_timeframe=False,
            suffix=self.info_timeframe,
        )

        # -- Indicators for 15m timeframe --
        bb_15m = ta.BBANDS(
            dataframe,
            timeperiod=self.buy_bb_15m_period.value,
            nbdevup=self.buy_bb_15m_stddev.value,
            nbdevdn=self.buy_bb_15m_stddev.value,
        )
        dataframe["bb_upper"] = bb_15m["upperband"]
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
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.buy_di_15m_period.value)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.buy_di_15m_period.value)
        aroon = ta.AROON(dataframe, timeperiod=self.buy_aroon_15m_period.value)
        dataframe["aroon_up"] = aroon["aroonup"]
        dataframe["aroon_down"] = aroon["aroondown"]
        dataframe["cmo"] = ta.CMO(dataframe, timeperiod=self.buy_cmo_15m_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 4h Low Volatility Confirmation --
        low_volatility_4h = (
            (dataframe[f"atr_{self.info_timeframe}"] < dataframe[f"atr_ma_{self.info_timeframe}"])
            & (dataframe[f"bb_width_{self.info_timeframe}"] < self.buy_bbw_4h_threshold.value)
            & (dataframe[f"chop_{self.info_timeframe}"] > self.buy_chop_4h_threshold.value)
        )

        # -- 15m Breakout and Momentum Entry --
        breakout_15m = dataframe["close"] > dataframe["bb_upper"]
        momentum_15m = (
            (dataframe["rsi"] > self.buy_rsi_15m_threshold.value)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["adx"] > self.buy_adx_15m_threshold.value)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["aroon_up"] > dataframe["aroon_down"])
            & (dataframe["cmo"] > self.buy_cmo_15m_threshold.value)
        )

        # -- Combine Conditions --
        dataframe.loc[low_volatility_4h & breakout_15m & momentum_15m, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- RSI Overbought --
        rsi_exit = dataframe["rsi"] > self.sell_rsi_15m_threshold.value

        dataframe.loc[rsi_exit, "exit_long"] = 1

        return dataframe

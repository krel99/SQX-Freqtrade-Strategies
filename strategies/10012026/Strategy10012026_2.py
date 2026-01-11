# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
# PROBLEM - TOO LITTLE TRADES
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
    informative,
)

# --------------------------------
from datetime import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta


class Strategy10012026_2(IStrategy):
    """
    Multi-Timeframe Momentum Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Define timeframes
    timeframe = "15m"
    info_timeframe = "1h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}

    # Stoploss
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Other settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 30

    # --- Hyperparameters ---

    # -- Primary Timeframe Parameters --
    # StochRSI
    buy_stochrsi_period = IntParameter(5, 30, default=14, space="buy")
    buy_stochrsi_k_period = IntParameter(2, 10, default=3, space="buy")
    buy_stochrsi_d_period = IntParameter(2, 10, default=3, space="buy")
    buy_stochrsi_oversold = DecimalParameter(0.1, 0.5, default=0.2, space="buy")

    # Awesome Oscillator (AO)
    buy_ao_fast = IntParameter(3, 10, default=5, space="buy")
    buy_ao_slow = IntParameter(20, 50, default=34, space="buy")

    # Commodity Channel Index (CCI) 1
    buy_cci_period = IntParameter(10, 40, default=20, space="buy")
    buy_cci_threshold = IntParameter(-50, 100, default=0, space="buy")

    # Commodity Channel Index (CCI) 2
    buy_cci_period2 = IntParameter(20, 60, default=40, space="buy")
    buy_cci_threshold2 = IntParameter(-100, 50, default=-50, space="buy")

    # Chaikin Money Flow (CMF)
    buy_cmf_period = IntParameter(10, 40, default=20, space="buy")
    buy_cmf_threshold = DecimalParameter(-0.1, 0.2, default=0.0, space="buy")

    # -- Confirmation Timeframe Parameters --
    # RSI
    inf_rsi_period = IntParameter(10, 30, default=14, space="buy")
    inf_rsi_threshold = IntParameter(40, 60, default=50, space="buy")

    # MACD
    inf_macd_fast = IntParameter(6, 24, default=12, space="buy")
    inf_macd_slow = IntParameter(13, 52, default=26, space="buy")
    inf_macd_signal = IntParameter(5, 18, default=9, space="buy")

    # DMI (ADX)
    inf_dmi_period = IntParameter(10, 30, default=14, space="buy")
    inf_adx_threshold = IntParameter(15, 35, default=25, space="buy")

    # EMA
    inf_ema_fast_period = IntParameter(20, 80, default=50, space="buy")
    inf_ema_slow_period = IntParameter(80, 240, default=100, space="buy")

    # Williams %R
    inf_willr_period = IntParameter(10, 50, default=20, space="buy")
    inf_willr_threshold = IntParameter(-40, -10, default=-20, space="buy")

    # -- Exit Parameters --
    sell_stochrsi_overbought = DecimalParameter(0.6, 0.9, default=0.7, space="sell")
    sell_cci_threshold = IntParameter(-150, -50, default=-100, space="sell")
    sell_atr_mult = DecimalParameter(1.0, 4.0, default=2.5, space="sell")
    sell_atr_period = IntParameter(10, 30, default=14, space="sell")

    @informative("1h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.inf_rsi_period.value)

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.inf_macd_fast.value,
            slowperiod=self.inf_macd_slow.value,
            signalperiod=self.inf_macd_signal.value,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        # DMI
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.inf_dmi_period.value)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.inf_dmi_period.value)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.inf_dmi_period.value)

        # EMA
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.inf_ema_fast_period.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.inf_ema_slow_period.value)

        # Williams %R
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=self.inf_willr_period.value)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Primary timeframe indicators --
        # StochRSI
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=self.buy_stochrsi_period.value)
        dataframe["stochrsi_k"] = stoch_rsi["fastk"]
        dataframe["stochrsi_d"] = stoch_rsi["fastd"]

        # Awesome Oscillator
        dataframe["ao"] = qtpylib.awesome_oscillator(
            dataframe, fast=self.buy_ao_fast.value, slow=self.buy_ao_slow.value
        )

        # CCI 1 & 2
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.buy_cci_period.value)
        dataframe["cci2"] = ta.CCI(dataframe, timeperiod=self.buy_cci_period2.value)

        # CMF
        dataframe["cmf"] = pandas_ta.cmf(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            volume=dataframe["volume"],
            length=self.buy_cmf_period.value,
        )

        # ATR for exits
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.sell_atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Higher timeframe confirmation - require at least 3 out of 6 conditions
        htf_rsi = dataframe[f"rsi_{self.info_timeframe}"] > self.inf_rsi_threshold.value
        htf_macd = (
            dataframe[f"macd_{self.info_timeframe}"]
            > dataframe[f"macdsignal_{self.info_timeframe}"]
        )
        htf_dmi = (
            dataframe[f"plus_di_{self.info_timeframe}"]
            > dataframe[f"minus_di_{self.info_timeframe}"]
        )
        htf_adx = dataframe[f"adx_{self.info_timeframe}"] > self.inf_adx_threshold.value
        htf_ema = (
            dataframe[f"ema_fast_{self.info_timeframe}"]
            > dataframe[f"ema_slow_{self.info_timeframe}"]
        )
        htf_willr = dataframe[f"willr_{self.info_timeframe}"] > self.inf_willr_threshold.value

        # Count true conditions
        htf_score = (
            htf_rsi.astype(int)
            + htf_macd.astype(int)
            + htf_dmi.astype(int)
            + htf_adx.astype(int)
            + htf_ema.astype(int)
            + htf_willr.astype(int)
        )
        confirm_trend = htf_score >= 3

        # StochRSI signal - either oversold bounce or momentum cross
        stoch_signal = (
            # Oversold bounce
            (dataframe["stochrsi_k"] < self.buy_stochrsi_oversold.value)
            & (qtpylib.crossed_above(dataframe["stochrsi_k"], dataframe["stochrsi_d"]))
        ) | (
            # Momentum cross in mid-range
            (dataframe["stochrsi_k"] > self.buy_stochrsi_oversold.value)
            & (dataframe["stochrsi_k"] < 0.7)
            & (qtpylib.crossed_above(dataframe["stochrsi_k"], dataframe["stochrsi_d"]))
        )

        # CCI signals - at least one should be favorable
        cci_signal = (dataframe["cci"] > self.buy_cci_threshold.value) | (
            dataframe["cci2"] > self.buy_cci_threshold2.value
        )

        # Core momentum conditions
        momentum_conditions = (
            stoch_signal
            & (dataframe["ao"] > 0)
            & cci_signal
            & (dataframe["cmf"] > self.buy_cmf_threshold.value)
        )

        dataframe.loc[confirm_trend & momentum_conditions, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit conditions based on primary timeframe
        stoch_exit = (qtpylib.crossed_below(dataframe["stochrsi_k"], dataframe["stochrsi_d"])) & (
            dataframe["stochrsi_k"] > self.sell_stochrsi_overbought.value
        )

        ao_exit = dataframe["ao"] < 0

        cci_exit = dataframe["cci"] < self.sell_cci_threshold.value

        dataframe.loc[stoch_exit | ao_exit | cci_exit, "exit_long"] = 1

        return dataframe

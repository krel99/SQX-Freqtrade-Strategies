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
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        # Pre-calculate RSI for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate MACD for common combinations
        for fast in range(6, 25):
            for slow in range(13, 53):
                for signal in range(5, 19):
                    if fast < slow:
                        macd = ta.MACD(
                            dataframe, fastperiod=fast, slowperiod=slow, signalperiod=signal
                        )
                        dataframe[f"macd_{fast}_{slow}_{signal}"] = macd["macd"]
                        dataframe[f"macdsignal_{fast}_{slow}_{signal}"] = macd["macdsignal"]

        # Pre-calculate DMI for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"plus_di_{period}"] = ta.PLUS_DI(dataframe, timeperiod=period)
            dataframe[f"minus_di_{period}"] = ta.MINUS_DI(dataframe, timeperiod=period)
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Pre-calculate EMA for all periods (20-240)
        for period in range(20, 241):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate Williams %R for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"willr_{period}"] = ta.WILLR(dataframe, timeperiod=period)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        # Pre-calculate StochRSI for all periods (5-30)
        for period in range(5, 31):
            stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=period)
            dataframe[f"stochrsi_k_{period}"] = stoch_rsi["fastk"]
            dataframe[f"stochrsi_d_{period}"] = stoch_rsi["fastd"]

        # Pre-calculate Awesome Oscillator for all combinations
        for fast in range(3, 11):
            for slow in range(20, 51):
                if fast < slow:
                    dataframe[f"ao_{fast}_{slow}"] = qtpylib.awesome_oscillator(
                        dataframe, fast=fast, slow=slow
                    )

        # Pre-calculate CCI for all periods (10-60)
        for period in range(10, 61):
            dataframe[f"cci_{period}"] = ta.CCI(dataframe, timeperiod=period)

        # Pre-calculate CMF for all periods (10-40)
        for period in range(10, 41):
            dataframe[f"cmf_{period}"] = pandas_ta.cmf(
                high=dataframe["high"],
                low=dataframe["low"],
                close=dataframe["close"],
                volume=dataframe["volume"],
                length=period,
            )

        # Pre-calculate ATR for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        inf_rsi_period = self.inf_rsi_period.value
        inf_macd_fast = self.inf_macd_fast.value
        inf_macd_slow = self.inf_macd_slow.value
        inf_macd_signal = self.inf_macd_signal.value
        inf_dmi_period = self.inf_dmi_period.value
        inf_ema_fast_period = self.inf_ema_fast_period.value
        inf_ema_slow_period = self.inf_ema_slow_period.value
        inf_willr_period = self.inf_willr_period.value

        stochrsi_period = self.buy_stochrsi_period.value
        ao_fast = self.buy_ao_fast.value
        ao_slow = self.buy_ao_slow.value
        cci_period = self.buy_cci_period.value
        cci_period2 = self.buy_cci_period2.value
        cmf_period = self.buy_cmf_period.value

        # Select pre-calculated indicators from 1h timeframe
        htf_rsi = (
            dataframe[f"rsi_{inf_rsi_period}_{self.info_timeframe}"] > self.inf_rsi_threshold.value
        )
        htf_macd = (
            dataframe[
                f"macd_{inf_macd_fast}_{inf_macd_slow}_{inf_macd_signal}_{self.info_timeframe}"
            ]
            > dataframe[
                f"macdsignal_{inf_macd_fast}_{inf_macd_slow}_{inf_macd_signal}_{self.info_timeframe}"
            ]
        )
        htf_dmi = (
            dataframe[f"plus_di_{inf_dmi_period}_{self.info_timeframe}"]
            > dataframe[f"minus_di_{inf_dmi_period}_{self.info_timeframe}"]
        )
        htf_adx = (
            dataframe[f"adx_{inf_dmi_period}_{self.info_timeframe}"] > self.inf_adx_threshold.value
        )
        htf_ema = (
            dataframe[f"ema_{inf_ema_fast_period}_{self.info_timeframe}"]
            > dataframe[f"ema_{inf_ema_slow_period}_{self.info_timeframe}"]
        )
        htf_willr = (
            dataframe[f"willr_{inf_willr_period}_{self.info_timeframe}"]
            > self.inf_willr_threshold.value
        )

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

        # Select pre-calculated indicators from primary timeframe
        stochrsi_k = dataframe[f"stochrsi_k_{stochrsi_period}"]
        stochrsi_d = dataframe[f"stochrsi_d_{stochrsi_period}"]
        ao = dataframe[f"ao_{ao_fast}_{ao_slow}"]
        cci = dataframe[f"cci_{cci_period}"]
        cci2 = dataframe[f"cci_{cci_period2}"]
        cmf = dataframe[f"cmf_{cmf_period}"]

        # StochRSI signal - either oversold bounce or momentum cross
        stoch_signal = (
            # Oversold bounce
            (stochrsi_k < self.buy_stochrsi_oversold.value)
            & (qtpylib.crossed_above(stochrsi_k, stochrsi_d))
        ) | (
            # Momentum cross in mid-range
            (stochrsi_k > self.buy_stochrsi_oversold.value)
            & (stochrsi_k < 0.7)
            & (qtpylib.crossed_above(stochrsi_k, stochrsi_d))
        )

        # CCI signals - at least one should be favorable
        cci_signal = (cci > self.buy_cci_threshold.value) | (cci2 > self.buy_cci_threshold2.value)

        # Core momentum conditions
        momentum_conditions = (
            stoch_signal & (ao > 0) & cci_signal & (cmf > self.buy_cmf_threshold.value)
        )

        dataframe.loc[confirm_trend & momentum_conditions, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        stochrsi_period = self.buy_stochrsi_period.value
        ao_fast = self.buy_ao_fast.value
        ao_slow = self.buy_ao_slow.value
        cci_period = self.buy_cci_period.value

        # Select pre-calculated indicators
        stochrsi_k = dataframe[f"stochrsi_k_{stochrsi_period}"]
        stochrsi_d = dataframe[f"stochrsi_d_{stochrsi_period}"]
        ao = dataframe[f"ao_{ao_fast}_{ao_slow}"]
        cci = dataframe[f"cci_{cci_period}"]

        # Exit conditions based on primary timeframe
        stoch_exit = (qtpylib.crossed_below(stochrsi_k, stochrsi_d)) & (
            stochrsi_k > self.sell_stochrsi_overbought.value
        )

        ao_exit = ao < 0

        cci_exit = cci < self.sell_cci_threshold.value

        dataframe.loc[stoch_exit | ao_exit | cci_exit, "exit_long"] = 1

        return dataframe

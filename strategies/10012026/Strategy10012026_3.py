# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
# # PROBLEM - TOO LITTLE TRADES
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
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Strategy10012026_3(IStrategy):
    """
    Dynamic Support/Resistance RSI Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.2, "30": 0.15, "60": 0.1}

    # Stoploss
    stoploss = -0.15

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Other settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 30

    # --- Hyperparameters ---
    # RSI
    buy_rsi_period = IntParameter(10, 30, default=14, space="buy")

    # RSI Bollinger Bands 1
    buy_rsi_bb_period = IntParameter(10, 40, default=20, space="buy")
    buy_rsi_bb_stddev = DecimalParameter(1.0, 2.5, default=1.8, space="buy")

    # RSI Bollinger Bands 2
    buy_rsi_bb_period2 = IntParameter(20, 60, default=40, space="buy")
    buy_rsi_bb_stddev2 = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    # Trend Confirmation EMAs
    buy_ema_fast_period = IntParameter(10, 40, default=20, space="buy")
    buy_ema_slow_period = IntParameter(25, 75, default=50, space="buy")
    buy_ema_trend_period = IntParameter(50, 150, default=100, space="buy")

    # ATR
    buy_atr_period = IntParameter(10, 30, default=14, space="buy")
    buy_atr_min_pct = DecimalParameter(0.005, 0.03, default=0.01, space="buy")

    # Parabolic SAR
    buy_sar_accel = DecimalParameter(0.01, 0.05, default=0.02, space="buy")
    buy_sar_max = DecimalParameter(0.1, 0.5, default=0.2, space="buy")

    # Volume Oscillator
    buy_vo_fast = IntParameter(3, 10, default=5, space="buy")
    buy_vo_slow = IntParameter(10, 30, default=14, space="buy")

    # Additional confirmations
    buy_mfi_period = IntParameter(10, 30, default=14, space="buy")
    buy_mfi_threshold = IntParameter(15, 45, default=25, space="buy")
    buy_cci_period = IntParameter(10, 40, default=20, space="buy")
    buy_cci_threshold = IntParameter(-150, 0, default=-80, space="buy")
    buy_adx_period = IntParameter(10, 30, default=14, space="buy")
    buy_adx_threshold = IntParameter(15, 35, default=18, space="buy")

    # --- Exit Parameters ---
    sell_rsi_bb_stddev = DecimalParameter(1.0, 3.0, default=2.0, space="sell")
    sell_take_profit_atr_mult = DecimalParameter(1.5, 5.0, default=3.0, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Main RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)

        # RSI Bollinger Bands 1
        rsi_bb1 = qtpylib.bollinger_bands(
            dataframe["rsi"], window=self.buy_rsi_bb_period.value, stds=self.buy_rsi_bb_stddev.value
        )
        dataframe["rsi_bb1_lower"] = rsi_bb1["lower"]
        dataframe["rsi_bb1_upper"] = rsi_bb1["upper"]

        # RSI Bollinger Bands 2
        rsi_bb2 = qtpylib.bollinger_bands(
            dataframe["rsi"],
            window=self.buy_rsi_bb_period2.value,
            stds=self.buy_rsi_bb_stddev2.value,
        )
        dataframe["rsi_bb2_lower"] = rsi_bb2["lower"]
        dataframe["rsi_bb2_upper"] = rsi_bb2["upper"]

        # EMAs
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.buy_ema_fast_period.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.buy_ema_slow_period.value)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.buy_ema_trend_period.value)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.buy_atr_period.value)

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(
            dataframe, acceleration=self.buy_sar_accel.value, maximum=self.buy_sar_max.value
        )

        # Volume Oscillator
        dataframe["vo"] = ta.PPO(
            dataframe["volume"],
            fastperiod=self.buy_vo_fast.value,
            slowperiod=self.buy_vo_slow.value,
        )

        # Other indicators
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=self.buy_mfi_period.value)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.buy_cci_period.value)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.buy_adx_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend confirmation - at least one of the two conditions
        trend_confirmation = (dataframe["ema_fast"] > dataframe["ema_slow"]) | (
            dataframe["close"] > dataframe["ema_trend"]
        )

        # RSI dynamic oversold - RSI below at least one BB lower band or recently crossed above
        dynamic_oversold = (
            # Currently oversold
            (dataframe["rsi"] < dataframe["rsi_bb1_lower"])
            | (dataframe["rsi"] < dataframe["rsi_bb2_lower"])
            # Or recently crossed above from oversold
            | (qtpylib.crossed_above(dataframe["rsi"], dataframe["rsi_bb1_lower"]))
            | (qtpylib.crossed_above(dataframe["rsi"], dataframe["rsi_bb2_lower"]))
        )

        # Volume/momentum confirmations - require at least 3 out of 6
        atr_cond = dataframe["atr"] > dataframe["close"] * self.buy_atr_min_pct.value
        sar_cond = dataframe["sar"] < dataframe["close"]
        vo_cond = dataframe["vo"] > -5  # More lenient than > 0
        mfi_cond = dataframe["mfi"] > self.buy_mfi_threshold.value
        cci_cond = dataframe["cci"] > self.buy_cci_threshold.value
        adx_cond = dataframe["adx"] > self.buy_adx_threshold.value

        # Count true conditions
        momentum_score = (
            atr_cond.astype(int)
            + sar_cond.astype(int)
            + vo_cond.astype(int)
            + mfi_cond.astype(int)
            + cci_cond.astype(int)
            + adx_cond.astype(int)
        )
        vol_momentum_confirmation = momentum_score >= 3

        dataframe.loc[
            trend_confirmation & dynamic_oversold & vol_momentum_confirmation, "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit when RSI is overbought (either above upper band or crossing below it)
        rsi_overbought = (dataframe["rsi"] > dataframe["rsi_bb1_upper"]) & (dataframe["rsi"] > 70)

        rsi_reversal = qtpylib.crossed_below(dataframe["rsi"], dataframe["rsi_bb1_upper"]) & (
            dataframe["rsi"] > 60
        )

        # Trend reversal with volume confirmation
        trend_reversal = qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"]) | (
            dataframe["close"] < dataframe["sar"]
        )

        # Bearish divergence - price up but indicators weakening
        bearish_divergence = (
            (dataframe["close"] > dataframe["close"].shift(5))
            & (dataframe["mfi"] < dataframe["mfi"].shift(5))
            & (dataframe["rsi"] < dataframe["rsi"].shift(5))
        )

        # Volume declining on price rise
        volume_weakness = (
            (dataframe["close"] > dataframe["close"].shift(3))
            & (dataframe["volume"] < dataframe["volume"].shift(3))
            & (dataframe["vo"] < -5)
        )

        # Combine exit conditions
        exit_signal = (
            rsi_overbought | rsi_reversal | trend_reversal | (bearish_divergence & volume_weakness)
        )

        dataframe.loc[exit_signal, "exit_long"] = 1

        return dataframe

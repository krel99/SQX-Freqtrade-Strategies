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
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        # Pre-calculate RSI for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate EMAs for all periods (10-150 covers all EMA params)
        for period in range(10, 151):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate ATR for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Pre-calculate MFI for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"mfi_{period}"] = ta.MFI(dataframe, timeperiod=period)

        # Pre-calculate CCI for all periods (10-40)
        for period in range(10, 41):
            dataframe[f"cci_{period}"] = ta.CCI(dataframe, timeperiod=period)

        # Pre-calculate ADX for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Pre-calculate Volume Oscillator (PPO) for all fast/slow combinations
        for fast in range(3, 11):
            for slow in range(10, 31):
                if fast < slow:
                    dataframe[f"vo_{fast}_{slow}"] = ta.PPO(
                        dataframe["volume"], fastperiod=fast, slowperiod=slow
                    )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        rsi_period = self.buy_rsi_period.value
        rsi_bb_period = self.buy_rsi_bb_period.value
        rsi_bb_stddev = self.buy_rsi_bb_stddev.value
        rsi_bb_period2 = self.buy_rsi_bb_period2.value
        rsi_bb_stddev2 = self.buy_rsi_bb_stddev2.value
        ema_fast_period = self.buy_ema_fast_period.value
        ema_slow_period = self.buy_ema_slow_period.value
        ema_trend_period = self.buy_ema_trend_period.value
        atr_period = self.buy_atr_period.value
        vo_fast = self.buy_vo_fast.value
        vo_slow = self.buy_vo_slow.value
        mfi_period = self.buy_mfi_period.value
        cci_period = self.buy_cci_period.value
        adx_period = self.buy_adx_period.value

        # Select pre-calculated indicators
        rsi = dataframe[f"rsi_{rsi_period}"]
        ema_fast = dataframe[f"ema_{ema_fast_period}"]
        ema_slow = dataframe[f"ema_{ema_slow_period}"]
        ema_trend = dataframe[f"ema_{ema_trend_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        vo = dataframe[f"vo_{vo_fast}_{vo_slow}"]
        mfi = dataframe[f"mfi_{mfi_period}"]
        cci = dataframe[f"cci_{cci_period}"]
        adx = dataframe[f"adx_{adx_period}"]

        # Calculate RSI Bollinger Bands dynamically (since they depend on RSI values)
        rsi_bb1 = qtpylib.bollinger_bands(rsi, window=rsi_bb_period, stds=rsi_bb_stddev)
        rsi_bb1_lower = rsi_bb1["lower"]
        rsi_bb1_upper = rsi_bb1["upper"]

        rsi_bb2 = qtpylib.bollinger_bands(rsi, window=rsi_bb_period2, stds=rsi_bb_stddev2)
        rsi_bb2_lower = rsi_bb2["lower"]
        rsi_bb2_upper = rsi_bb2["upper"]

        # Calculate Parabolic SAR dynamically (too many combinations to pre-calculate)
        sar = ta.SAR(
            dataframe, acceleration=self.buy_sar_accel.value, maximum=self.buy_sar_max.value
        )

        # Trend confirmation - at least one of the two conditions
        trend_confirmation = (ema_fast > ema_slow) | (dataframe["close"] > ema_trend)

        # RSI dynamic oversold - RSI below at least one BB lower band or recently crossed above
        dynamic_oversold = (
            # Currently oversold
            (rsi < rsi_bb1_lower)
            | (rsi < rsi_bb2_lower)
            # Or recently crossed above from oversold
            | (qtpylib.crossed_above(rsi, rsi_bb1_lower))
            | (qtpylib.crossed_above(rsi, rsi_bb2_lower))
        )

        # Volume/momentum confirmations - require at least 3 out of 6
        atr_cond = atr > dataframe["close"] * self.buy_atr_min_pct.value
        sar_cond = sar < dataframe["close"]
        vo_cond = vo > -5  # More lenient than > 0
        mfi_cond = mfi > self.buy_mfi_threshold.value
        cci_cond = cci > self.buy_cci_threshold.value
        adx_cond = adx > self.buy_adx_threshold.value

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

        # Store RSI BB values for exit trend
        dataframe["rsi"] = rsi
        dataframe["rsi_bb1_lower"] = rsi_bb1_lower
        dataframe["rsi_bb1_upper"] = rsi_bb1_upper
        dataframe["ema_fast"] = ema_fast
        dataframe["ema_slow"] = ema_slow
        dataframe["sar"] = sar
        dataframe["mfi"] = mfi
        dataframe["vo"] = vo

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

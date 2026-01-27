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
from technical.indicators import ichimoku


class VolSqueezeBreakout_MultiFrame(IStrategy):
    """
    Multi-Timeframe Volatility Squeeze Breakout Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "5m"
    info_timeframe = "1h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.2, "20": 0.1, "40": 0.05}

    # Stoploss
    stoploss = -0.1

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.015
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

    # -- 1h Timeframe --
    buy_bb_1h_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb_1h_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    buy_kc_1h_ema_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_1h_atr_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_1h_mult = DecimalParameter(1.0, 3.0, default=1.5, space="buy")
    buy_ichi_1h_tenkan = IntParameter(5, 20, default=9, space="buy")
    buy_ichi_1h_kijun = IntParameter(20, 50, default=26, space="buy")
    buy_ichi_1h_senkou = IntParameter(40, 100, default=52, space="buy")
    buy_squeeze_1h_lookback = IntParameter(1, 10, default=3, space="buy")

    # -- 5m Timeframe --
    buy_bb_5m_period = IntParameter(10, 50, default=30, space="buy")
    buy_bb_5m_stddev = DecimalParameter(1.5, 3.0, default=2.5, space="buy")
    buy_rsi_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_5m_threshold = IntParameter(40, 70, default=55, space="buy")
    buy_macd_5m_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_5m_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_5m_signal = IntParameter(5, 18, default=9, space="buy")
    buy_adx_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_5m_threshold = IntParameter(15, 50, default=25, space="buy")
    buy_volume_ma_5m_period = IntParameter(10, 50, default=20, space="buy")
    buy_volume_spike_5m_factor = DecimalParameter(1.1, 3.0, default=1.5, space="buy")
    buy_mfi_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_mfi_5m_threshold = IntParameter(30, 70, default=50, space="buy")
    buy_cci_5m_period = IntParameter(10, 50, default=20, space="buy")
    buy_cci_5m_threshold = IntParameter(50, 150, default=100, space="buy")

    # -- Exit Parameters --
    sell_rsi_5m_threshold = IntParameter(60, 90, default=75, space="sell")

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

        bb_1h = ta.BBANDS(
            informative,
            timeperiod=self.buy_bb_1h_period.value,
            nbdevup=self.buy_bb_1h_stddev.value,
            nbdevdn=self.buy_bb_1h_stddev.value,
        )
        informative["bb_lower"] = bb_1h["lowerband"]
        informative["bb_upper"] = bb_1h["upperband"]

        kc_1h_ema = ta.EMA(informative, timeperiod=self.buy_kc_1h_ema_period.value)
        kc_1h_atr = ta.ATR(informative, timeperiod=self.buy_kc_1h_atr_period.value)
        informative["kc_lower"] = kc_1h_ema - kc_1h_atr * self.buy_kc_1h_mult.value
        informative["kc_upper"] = kc_1h_ema + kc_1h_atr * self.buy_kc_1h_mult.value

        informative["squeeze_on"] = (informative["bb_lower"] > informative["kc_lower"]) & (
            informative["bb_upper"] < informative["kc_upper"]
        )

        ichi = ichimoku(
            informative,
            conversion_line_period=self.buy_ichi_1h_tenkan.value,
            base_line_periods=self.buy_ichi_1h_kijun.value,
            laggin_span=self.buy_ichi_1h_senkou.value,
            displacement=26,  # Standard displacement
        )
        informative["senkou_a"] = ichi["senkou_a"]
        informative["senkou_b"] = ichi["senkou_b"]

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
            append_timeframe=False,
            suffix=self.info_timeframe,
        )

        # -- Indicators for 5m timeframe --
        bb_5m = ta.BBANDS(
            dataframe,
            timeperiod=self.buy_bb_5m_period.value,
            nbdevup=self.buy_bb_5m_stddev.value,
            nbdevdn=self.buy_bb_5m_stddev.value,
        )
        dataframe["bb_upper"] = bb_5m["upperband"]
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
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.buy_adx_5m_period.value)
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.buy_volume_ma_5m_period.value
        )
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=self.buy_mfi_5m_period.value)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.buy_cci_5m_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 1h Squeeze and Trend Conditions --
        squeeze_1h = (
            dataframe[f"squeeze_on_{self.info_timeframe}"]
            .rolling(self.buy_squeeze_1h_lookback.value)
            .sum()
            == self.buy_squeeze_1h_lookback.value
        )
        above_cloud_1h = (
            dataframe[f"close_{self.info_timeframe}"] > dataframe[f"senkou_a_{self.info_timeframe}"]
        ) & (
            dataframe[f"close_{self.info_timeframe}"] > dataframe[f"senkou_b_{self.info_timeframe}"]
        )

        # -- 5m Breakout and Momentum Confirmation --
        breakout_5m = dataframe["close"] > dataframe["bb_upper"]
        confirmation_5m = (
            (dataframe["rsi"] > self.buy_rsi_5m_threshold.value)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["adx"] > self.buy_adx_5m_threshold.value)
            & (dataframe["volume"] > dataframe["volume_ma"] * self.buy_volume_spike_5m_factor.value)
            & (dataframe["mfi"] > self.buy_mfi_5m_threshold.value)
            & (dataframe["cci"] > self.buy_cci_5m_threshold.value)
        )

        # -- Combine Conditions --
        dataframe.loc[
            squeeze_1h.shift(1)  # Squeeze was on in the previous 1h candle
            & ~dataframe[f"squeeze_on_{self.info_timeframe}"]  # Squeeze is now off
            & above_cloud_1h
            & breakout_5m
            & confirmation_5m,
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Overbought Signal --
        rsi_exit = dataframe["rsi"] > self.sell_rsi_5m_threshold.value

        # -- Price falls below middle Bollinger Band --
        price_below_middle_bb = dataframe["close"] < dataframe["bb_middle"]

        dataframe.loc[rsi_exit | price_below_middle_bb, "exit_long"] = 1

        return dataframe

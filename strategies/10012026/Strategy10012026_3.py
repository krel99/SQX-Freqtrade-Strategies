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

    timeframe = '1h'
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
    buy_rsi_bb_stddev = DecimalParameter(1.0, 3.0, default=2.0, space="buy")

    # RSI Bollinger Bands 2
    buy_rsi_bb_period2 = IntParameter(20, 60, default=40, space="buy")
    buy_rsi_bb_stddev2 = DecimalParameter(1.5, 4.0, default=2.5, space="buy")

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
    buy_mfi_threshold = IntParameter(20, 50, default=30, space="buy")
    buy_cci_period = IntParameter(10, 40, default=20, space="buy")
    buy_cci_threshold = IntParameter(-150, -50, default=-100, space="buy")
    buy_adx_period = IntParameter(10, 30, default=14, space="buy")
    buy_adx_threshold = IntParameter(15, 40, default=20, space="buy")

    # --- Exit Parameters ---
    sell_rsi_bb_stddev = DecimalParameter(1.0, 3.0, default=2.0, space="sell")
    sell_take_profit_atr_mult = DecimalParameter(1.5, 5.0, default=3.0, space="sell")


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Main RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)

        # RSI Bollinger Bands 1
        rsi_bb1 = ta.BBANDS(dataframe['rsi'], timeperiod=self.buy_rsi_bb_period.value, nbdevup=self.buy_rsi_bb_stddev.value, nbdevdn=self.buy_rsi_bb_stddev.value)
        dataframe['rsi_bb1_lower'] = rsi_bb1['lowerband']
        dataframe['rsi_bb1_upper'] = rsi_bb1['upperband']

        # RSI Bollinger Bands 2
        rsi_bb2 = ta.BBANDS(dataframe['rsi'], timeperiod=self.buy_rsi_bb_period2.value, nbdevup=self.buy_rsi_bb_stddev2.value, nbdevdn=self.buy_rsi_bb_stddev2.value)
        dataframe['rsi_bb2_lower'] = rsi_bb2['lowerband']
        dataframe['rsi_bb2_upper'] = rsi_bb2['upperband']

        # EMAs
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.buy_ema_fast_period.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.buy_ema_slow_period.value)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.buy_ema_trend_period.value)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.buy_atr_period.value)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe, acceleration=self.buy_sar_accel.value, maximum=self.buy_sar_max.value)

        # Volume Oscillator
        dataframe['vo'] = ta.PPO(dataframe['volume'], fastperiod=self.buy_vo_fast.value, slowperiod=self.buy_vo_slow.value) - ta.SMA(ta.PPO(dataframe['volume'], fastperiod=self.buy_vo_fast.value, slowperiod=self.buy_vo_slow.value), timeperiod=9)

        # Other indicators
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=self.buy_mfi_period.value)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=self.buy_cci_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.buy_adx_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        trend_confirmation = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_trend'])
        )

        dynamic_oversold = (
            qtpylib.crossed_above(dataframe['rsi'], dataframe['rsi_bb1_lower']) &
            qtpylib.crossed_above(dataframe['rsi'], dataframe['rsi_bb2_lower'])
        )

        vol_momentum_confirmation = (
            (dataframe['atr'] > dataframe['close'] * self.buy_atr_min_pct.value) &
            (dataframe['sar'] < dataframe['close']) &
            (dataframe['vo'] > 0) &
            (dataframe['mfi'] > self.buy_mfi_threshold.value) &
            (dataframe['cci'] > self.buy_cci_threshold.value) &
            (dataframe['adx'] > self.buy_adx_threshold.value)
        )

        dataframe.loc[
            trend_confirmation &
            dynamic_oversold &
            vol_momentum_confirmation,
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dynamic_overbought = qtpylib.crossed_below(dataframe['rsi'], dataframe['rsi_bb1_upper'])

        trend_reversal = qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])

        dataframe.loc[
            dynamic_overbought | trend_reversal,
            'exit_long'
        ] = 1

        return dataframe

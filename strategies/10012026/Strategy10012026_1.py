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


class Strategy10012026_1(IStrategy):
    """
    Volatility Squeeze Breakout Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.1}

    # Optimal stoploss
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # --- Hyperparameters ---
    # Bollinger Bands 1
    buy_bb1_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb1_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    # Keltner Channel 1
    buy_kc1_ema_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc1_atr_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc1_mult = DecimalParameter(1.0, 3.0, default=1.5, space="buy")

    # Bollinger Bands 2
    buy_bb2_period = IntParameter(10, 50, default=30, space="buy")
    buy_bb2_stddev = DecimalParameter(1.5, 3.0, default=2.5, space="buy")

    # Keltner Channel 2
    buy_kc2_ema_period = IntParameter(10, 50, default=30, space="buy")
    buy_kc2_atr_period = IntParameter(10, 50, default=30, space="buy")
    buy_kc2_mult = DecimalParameter(1.0, 3.0, default=2.0, space="buy")

    # Squeeze Lookback
    buy_squeeze_lookback = IntParameter(1, 10, default=3, space="buy")

    # ADX
    buy_adx_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_threshold = IntParameter(15, 50, default=25, space="buy")

    # RSI
    buy_rsi_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_threshold = IntParameter(40, 70, default=55, space="buy")

    # MACD
    buy_macd_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_signal = IntParameter(5, 18, default=9, space="buy")

    # Volume
    buy_volume_ma_period = IntParameter(10, 50, default=20, space="buy")
    buy_volume_spike_factor = DecimalParameter(1.1, 3.0, default=1.5, space="buy")

    # Breakout
    buy_breakout_confirmation_period = IntParameter(1, 5, default=1, space="buy")
    buy_breakout_price_source = CategoricalParameter(['close', 'high', 'low'], default='close', space='buy')

    # MFI
    buy_mfi_period = IntParameter(10, 50, default=14, space="buy")
    buy_mfi_threshold = IntParameter(30, 70, default=50, space="buy")

    # CCI
    buy_cci_period = IntParameter(10, 50, default=20, space="buy")
    buy_cci_threshold = IntParameter(50, 150, default=100, space="buy")

    # Exit Parameters
    sell_rsi_threshold = IntParameter(60, 90, default=75, space="sell")
    sell_take_profit_pct = DecimalParameter(0.01, 0.1, default=0.03, space="sell")


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands 1
        bb1 = ta.BBANDS(dataframe, timeperiod=self.buy_bb1_period.value, nbdevup=self.buy_bb1_stddev.value, nbdevdn=self.buy_bb1_stddev.value)
        dataframe['bb1_lower'] = bb1['lowerband']
        dataframe['bb1_upper'] = bb1['upperband']

        # Keltner Channel 1
        kc1_ema = ta.EMA(dataframe, timeperiod=self.buy_kc1_ema_period.value)
        kc1_atr = ta.ATR(dataframe, timeperiod=self.buy_kc1_atr_period.value)
        dataframe['kc1_lower'] = kc1_ema - kc1_atr * self.buy_kc1_mult.value
        dataframe['kc1_upper'] = kc1_ema + kc1_atr * self.buy_kc1_mult.value

        # Bollinger Bands 2
        bb2 = ta.BBANDS(dataframe, timeperiod=self.buy_bb2_period.value, nbdevup=self.buy_bb2_stddev.value, nbdevdn=self.buy_bb2_stddev.value)
        dataframe['bb2_lower'] = bb2['lowerband']
        dataframe['bb2_upper'] = bb2['upperband']

        # Keltner Channel 2
        kc2_ema = ta.EMA(dataframe, timeperiod=self.buy_kc2_ema_period.value)
        kc2_atr = ta.ATR(dataframe, timeperiod=self.buy_kc2_atr_period.value)
        dataframe['kc2_lower'] = kc2_ema - kc2_atr * self.buy_kc2_mult.value
        dataframe['kc2_upper'] = kc2_ema + kc2_atr * self.buy_kc2_mult.value

        # Squeeze detection
        dataframe['squeeze_on_1'] = (dataframe['bb1_lower'] > dataframe['kc1_lower']) & (dataframe['bb1_upper'] < dataframe['kc1_upper'])
        dataframe['squeeze_on_2'] = (dataframe['bb2_lower'] > dataframe['kc2_lower']) & (dataframe['bb2_upper'] < dataframe['kc2_upper'])
        dataframe['squeeze_on'] = dataframe['squeeze_on_1'] & dataframe['squeeze_on_2']

        # Other indicators
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.buy_adx_period.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)
        macd = ta.MACD(dataframe, fastperiod=self.buy_macd_fast.value, slowperiod=self.buy_macd_slow.value, signalperiod=self.buy_macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=self.buy_volume_ma_period.value)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=self.buy_mfi_period.value)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=self.buy_cci_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_squeeze = (dataframe['squeeze_on'].rolling(self.buy_squeeze_lookback.value).sum() == self.buy_squeeze_lookback.value)

        breakout_source = dataframe[self.buy_breakout_price_source.value]
        is_breakout = breakout_source > dataframe['bb1_upper']

        confirmation = (
            (dataframe['adx'] > self.buy_adx_threshold.value) &
            (dataframe['rsi'] > self.buy_rsi_threshold.value) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['volume'] > dataframe['volume_ma'] * self.buy_volume_spike_factor.value) &
            (dataframe['mfi'] > self.buy_mfi_threshold.value) &
            (dataframe['cci'] > self.buy_cci_threshold.value)
        )

        dataframe.loc[
            is_squeeze.shift(self.buy_breakout_confirmation_period.value) &
            ~dataframe['squeeze_on'] &
            is_breakout &
            confirmation,
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        rsi_exit = dataframe['rsi'] > self.sell_rsi_threshold.value

        # Take profit when price is a certain percentage above the upper BB
        bb_take_profit = dataframe['close'] > dataframe['bb1_upper'] * (1 + self.sell_take_profit_pct.value)

        dataframe.loc[
            rsi_exit | bb_take_profit,
            'exit_long'
        ] = 1

        return dataframe

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


class MT_ESAK_1(IStrategy):
    """
    Multi-Timeframe Strategy with EMA, Stochastic Oscillator, ADX, and Keltner Channels
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"
    informative_timeframe = "1h"
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
    startup_candle_count: int = 100

    # --- Hyperparameters ---
    # EMA (Higher Timeframe)
    buy_ema_htf_period = IntParameter(10, 50, default=20, space="buy")

    # Stochastic Oscillator (Lower Timeframe)
    buy_stoch_ltf_k = IntParameter(5, 20, default=14, space="buy")
    buy_stoch_ltf_d = IntParameter(3, 10, default=3, space="buy")
    buy_stoch_ltf_threshold = IntParameter(10, 40, default=30, space="buy")

    # Keltner Channels (Lower Timeframe)
    buy_kc_ltf_ema_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_ltf_atr_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc_ltf_mult = DecimalParameter(1.0, 3.0, default=1.5, space="buy")

    # Exit Parameters
    sell_stoch_ltf_threshold = IntParameter(60, 90, default=80, space="sell")


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Higher Timeframe Indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        informative['ema_htf'] = ta.EMA(informative, timeperiod=self.buy_ema_htf_period.value)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

        # Lower Timeframe Indicators
        stoch = ta.STOCH(dataframe, fastk_period=self.buy_stoch_ltf_k.value, slowk_period=self.buy_stoch_ltf_d.value, slowd_period=self.buy_stoch_ltf_d.value)
        dataframe['slowk_ltf'] = stoch['slowk']
        dataframe['slowd_ltf'] = stoch['slowd']
        kc_ema = ta.EMA(dataframe, timeperiod=self.buy_kc_ltf_ema_period.value)
        kc_atr = ta.ATR(dataframe, timeperiod=self.buy_kc_ltf_atr_period.value)
        dataframe['kc_lower_ltf'] = kc_ema - kc_atr * self.buy_kc_ltf_mult.value
        dataframe['kc_upper_ltf'] = kc_ema + kc_atr * self.buy_kc_ltf_mult.value

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe[f'ema_htf_{self.informative_timeframe}']) &
            (dataframe['slowk_ltf'] < self.buy_stoch_ltf_threshold.value) &
            (dataframe['close'] < dataframe['kc_lower_ltf']),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['slowk_ltf'] > self.sell_stoch_ltf_threshold.value),
            'exit_long'
        ] = 1

        return dataframe

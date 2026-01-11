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


class VolatilityCatcher(IStrategy):
    """
    Volatility Catcher Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframes
    timeframe = "15m"
    info_timeframe = "1h"

    # Can short
    can_short = False

    # Minimal ROI
    minimal_roi = {"0": 0.15, "30": 0.05, "60": 0.01}

    # Stoploss
    stoploss = -0.1

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Process only new candles
    process_only_new_candles = True

    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Startup candle count
    startup_candle_count: int = 30

    # --- Hyperparameters ---

    # -- 1h Timeframe --
    buy_ema_1h_period = IntParameter(20, 100, default=50, space="buy")

    # -- 15m Timeframe --
    buy_atr_15m_period = IntParameter(10, 50, default=14, space="buy")
    buy_atr_15m_threshold = DecimalParameter(0.5, 2.0, default=1.0, space="buy")
    buy_stoch_15m_fastk = IntParameter(5, 20, default=14, space="buy")
    buy_stoch_15m_slowk = IntParameter(3, 10, default=3, space="buy")
    buy_stoch_15m_slowd = IntParameter(3, 10, default=3, space="buy")
    buy_ao_15m_threshold = DecimalParameter(0.0, 1.0, default=0.0, space="buy") # Awesome Oscillator threshold

    # -- Exit Parameters --
    sell_stoch_15m_threshold = IntParameter(60, 90, default=75, space="sell")


    def informative_pairs(self):
        pairs = self.dp.get_pair_dataframe(self.timeframe, self.config['exchange']['pair_whitelist'])
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.info_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Indicators for 1h timeframe --
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.info_timeframe
        )
        informative["ema_slow"] = ta.EMA(informative, timeperiod=self.buy_ema_1h_period.value)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.info_timeframe,
            ffill=True,
            suffix=f"_{self.info_timeframe}",
        )

        # -- Indicators for 15m timeframe --
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.buy_atr_15m_period.value)
        stoch = ta.STOCH(
            dataframe,
            fastk_period=self.buy_stoch_15m_fastk.value,
            slowk_period=self.buy_stoch_15m_slowk.value,
            slowd_period=self.buy_stoch_15m_slowd.value,
        )
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]
        dataframe["ao"] = ta.APO(dataframe, fastperiod=5, slowperiod=34) # Awesome Oscillator

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 1h Momentum Conditions --
        momentum_1h = (
            (dataframe[f"close_{self.info_timeframe}"] > dataframe[f"ema_slow_{self.info_timeframe}"])
        )

        # -- 15m Volatility Conditions --
        volatility_15m = (
            (dataframe["atr"] > self.buy_atr_15m_threshold.value)
            & (dataframe["slowk"] > dataframe["slowd"])
            & (dataframe["ao"] > self.buy_ao_15m_threshold.value)
        )

        # -- Combine Conditions --
        dataframe.loc[
            momentum_1h & volatility_15m,
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Overbought Signal --
        stoch_exit = dataframe["slowk"] > self.sell_stoch_15m_threshold.value

        dataframe.loc[stoch_exit, "exit_long"] = 1

        return dataframe

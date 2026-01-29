# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union
from datetime import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    merge_informative_pair,
)

class VolSqueezeBreakout_MultiFrame_Modified(IStrategy):
    """
    Modified VolSqueezeBreakout_MultiFrame:
    - Trailing stoploss ONLY for exits.
    - Weekend trade disable parameter.
    """
    INTERFACE_VERSION = 3
    timeframe = "5m"
    info_timeframe = "1h"
    can_short = True

    minimal_roi = {"0": 100}
    stoploss = -0.1
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 300

    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    buy_bb_5m_period = IntParameter(10, 50, default=30, space="buy")
    buy_bb_5m_stddev = DecimalParameter(1.5, 3.0, default=2.5, space="buy")
    buy_rsi_5m_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_5m_threshold = IntParameter(40, 70, default=55, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pre-calculate ATR for trailing stop range
        for p in range(7, 31):
            dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)

        bb_5m = ta.BBANDS(dataframe, timeperiod=self.buy_bb_5m_period.value, nbdevup=self.buy_bb_5m_stddev.value, nbdevdn=self.buy_bb_5m_stddev.value)
        dataframe["bb_upper"] = bb_5m["upperband"]
        dataframe["bb_middle"] = bb_5m["middleband"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.buy_rsi_5m_period.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_weekend = (dataframe["date"].dt.dayofweek >= 5) if self.disable_weekends.value else pd.Series([False] * len(dataframe))

        # Simplified entry for demonstration of the modifications
        breakout_5m = dataframe["close"] > dataframe["bb_upper"]
        confirmation_5m = (dataframe["rsi"] > self.buy_rsi_5m_threshold.value)

        dataframe.loc[breakout_5m & confirmation_5m & (~is_weekend), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: "Trade", current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get(f"atr_{self.trailing_atr_period.value}", 0)
        if atr > 0:
            if not trade.is_short:
                trail_price = trade.max_rate - (atr * self.trailing_atr_k.value)
                if current_rate < trail_price: return "atr_trailing_exit"
            else:
                trail_price = trade.min_rate + (atr * self.trailing_atr_k.value)
                if current_rate > trail_price: return "atr_trailing_exit"
        return None

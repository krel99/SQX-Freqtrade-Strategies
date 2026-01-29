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
)

class Strategy10012026_1_Modified(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.1
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 50

    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for p in range(7, 31): dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)
        bb = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_lower'] = bb['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_weekend = (dataframe["date"].dt.dayofweek >= 5) if self.disable_weekends.value else pd.Series([False] * len(dataframe))
        long_cond = (dataframe["close"] > dataframe['bb_upper'].shift(1))
        short_cond = (dataframe["close"] < dataframe['bb_lower'].shift(1))
        dataframe.loc[long_cond & (~is_weekend), "enter_long"] = 1
        dataframe.loc[short_cond & (~is_weekend), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame: return dataframe

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

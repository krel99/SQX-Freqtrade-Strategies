# pragma pylint: disable=missing-module-docstring, invalid-name, pointless-string-statement
from __future__ import annotations
from datetime import time, datetime
from typing import Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
)
from freqtrade.persistence import Trade

class DRIDR_V15_Modified(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 200

    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        for p in range(7, 31): dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)
        # Standard DRIDR logic would go here, simplified for this version
        dataframe['range_high'] = dataframe['high'].rolling(window=12).max() # 1 hour at 5m
        dataframe['range_low'] = dataframe['low'].rolling(window=12).min()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        is_weekend = (dataframe["date"].dt.dayofweek >= 5) if self.disable_weekends.value else pd.Series([False] * len(dataframe))
        long_cond = (dataframe["close"] > dataframe['range_high'].shift(1))
        short_cond = (dataframe["close"] < dataframe['range_low'].shift(1))
        dataframe.loc[long_cond & (~is_weekend), "enter_long"] = 1
        dataframe.loc[short_cond & (~is_weekend), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame: return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
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

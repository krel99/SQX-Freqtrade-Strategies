# --- Do not remove these libs ---
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)

if TYPE_CHECKING:
    from freqtrade.persistence import Trade

# --------------------------------

class FractalStrategyV3_Modified(IStrategy):
    """
    Modified FractalStrategyV3:
    - Trailing stoploss ONLY for exits.
    - Weekend trade disable parameter.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)

    long_fractal_window = CategoricalParameter([3, 5], default=5, space="buy", optimize=True)
    long_breakout_threshold = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="buy", optimize=True)
    short_fractal_window = CategoricalParameter([3, 5], default=5, space="sell", optimize=True)
    short_breakout_threshold = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="sell", optimize=True)

    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    minimal_roi = {"0": 100}
    stoploss = -0.08
    trailing_stop = False
    timeframe = "15m"
    process_only_new_candles = True

    def _calculate_fractals(self, dataframe: DataFrame, window_size: int) -> tuple:
        if window_size == 5:
            top = (dataframe['high'].shift(2) > dataframe['high'].shift(3)) & (dataframe['high'].shift(2) > dataframe['high'].shift(4)) & (dataframe['high'].shift(2) > dataframe['high'].shift(1)) & (dataframe['high'].shift(2) > dataframe['high'])
            bottom = (dataframe['low'].shift(2) < dataframe['low'].shift(3)) & (dataframe['low'].shift(2) < dataframe['low'].shift(4)) & (dataframe['low'].shift(2) < dataframe['low'].shift(1)) & (dataframe['low'].shift(2) < dataframe['low'])
            fractal_tops, fractal_bottoms = np.where(top, dataframe['high'].shift(2), np.nan), np.where(bottom, dataframe['low'].shift(2), np.nan)
        elif window_size == 3:
            top = (dataframe['high'].shift(1) > dataframe['high'].shift(2)) & (dataframe['high'].shift(1) > dataframe['high'])
            bottom = (dataframe['low'].shift(1) < dataframe['low'].shift(2)) & (dataframe['low'].shift(1) < dataframe['low'])
            fractal_tops, fractal_bottoms = np.where(top, dataframe['high'].shift(1), np.nan), np.where(bottom, dataframe['low'].shift(1), np.nan)
        else: fractal_tops, fractal_bottoms = np.full(len(dataframe), np.nan), np.full(len(dataframe), np.nan)
        return fractal_tops, fractal_bottoms

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for window_size in [3, 5]:
            tops, bottoms = self._calculate_fractals(dataframe, window_size)
            dataframe[f"fractal_top_{window_size}"] = pd.Series(tops).ffill()
            dataframe[f"fractal_bottom_{window_size}"] = pd.Series(bottoms).ffill()
        for p in range(7, 31): dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_weekend = (dataframe["date"].dt.dayofweek >= 5) if self.disable_weekends.value else pd.Series([False] * len(dataframe))
        long_cond = (dataframe["close"] > dataframe[f"fractal_top_{self.long_fractal_window.value}"] * (1 + self.long_breakout_threshold.value))
        short_cond = (dataframe["close"] < dataframe[f"fractal_bottom_{self.short_fractal_window.value}"] * (1 - self.short_breakout_threshold.value))
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

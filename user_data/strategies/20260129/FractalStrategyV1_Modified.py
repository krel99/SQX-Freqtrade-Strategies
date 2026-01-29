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

class FractalStrategyV1_Modified(IStrategy):
    """
    Modified FractalStrategyV1:
    - Trailing stoploss ONLY for exits.
    - Weekend trade disable parameter.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    # === GLOBAL PARAMETERS ===
    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)

    # === LONG ENTRY PARAMETERS (buy space) ===
    long_fractal_window = CategoricalParameter([3, 5], default=5, space="buy", optimize=True)
    long_breakout_threshold = DecimalParameter(
        0.001, 0.05, default=0.01, decimals=3, space="buy", optimize=True
    )
    long_ma_period = IntParameter(20, 500, default=100, space="buy", optimize=True)
    long_ma_type = CategoricalParameter(["EMA", "SMA", "WMA"], default="EMA", space="buy", optimize=True)
    long_use_ma_filter = BooleanParameter(default=True, space="buy", optimize=True)
    long_rsi_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    long_rsi_min = IntParameter(20, 50, default=30, space="buy", optimize=True)
    long_use_rsi_filter = BooleanParameter(default=True, space="buy", optimize=True)
    long_volume_ma_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    long_volume_threshold = DecimalParameter(0.5, 3.0, default=1.2, decimals=1, space="buy", optimize=True)
    long_use_volume_filter = BooleanParameter(default=True, space="buy", optimize=True)
    long_adx_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    long_adx_min = IntParameter(15, 40, default=25, space="buy", optimize=True)
    long_use_adx_filter = BooleanParameter(default=False, space="buy", optimize=True)

    # === SHORT ENTRY PARAMETERS (sell space) ===
    short_fractal_window = CategoricalParameter([3, 5], default=5, space="sell", optimize=True)
    short_breakout_threshold = DecimalParameter(
        0.001, 0.05, default=0.01, decimals=3, space="sell", optimize=True
    )
    short_ma_period = IntParameter(20, 500, default=100, space="sell", optimize=True)
    short_ma_type = CategoricalParameter(["EMA", "SMA", "WMA"], default="EMA", space="sell", optimize=True)
    short_use_ma_filter = BooleanParameter(default=True, space="sell", optimize=True)
    short_rsi_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_rsi_max = IntParameter(50, 80, default=70, space="sell", optimize=True)
    short_use_rsi_filter = BooleanParameter(default=True, space="sell", optimize=True)
    short_volume_ma_period = IntParameter(10, 50, default=20, space="sell", optimize=True)
    short_volume_threshold = DecimalParameter(0.5, 3.0, default=1.2, decimals=1, space="sell", optimize=True)
    short_use_volume_filter = BooleanParameter(default=True, space="sell", optimize=True)
    short_adx_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_adx_min = IntParameter(15, 40, default=25, space="sell", optimize=True)
    short_use_adx_filter = BooleanParameter(default=False, space="sell", optimize=True)

    # === EXIT PARAMETERS (profit space) ===
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    # === STRATEGY SETTINGS ===
    minimal_roi = {"0": 100}
    stoploss = -0.08
    trailing_stop = False
    timeframe = "15m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def _calculate_fractals(self, dataframe: DataFrame, window_size: int) -> tuple:
        if window_size == 5:
            top = (dataframe['high'].shift(2) > dataframe['high'].shift(3)) & \
                  (dataframe['high'].shift(2) > dataframe['high'].shift(4)) & \
                  (dataframe['high'].shift(2) > dataframe['high'].shift(1)) & \
                  (dataframe['high'].shift(2) > dataframe['high'])
            bottom = (dataframe['low'].shift(2) < dataframe['low'].shift(3)) & \
                     (dataframe['low'].shift(2) < dataframe['low'].shift(4)) & \
                     (dataframe['low'].shift(2) < dataframe['low'].shift(1)) & \
                     (dataframe['low'].shift(2) < dataframe['low'])
            fractal_tops = np.where(top, dataframe['high'].shift(2), np.nan)
            fractal_bottoms = np.where(bottom, dataframe['low'].shift(2), np.nan)
        elif window_size == 3:
            top = (dataframe['high'].shift(1) > dataframe['high'].shift(2)) & \
                  (dataframe['high'].shift(1) > dataframe['high'])
            bottom = (dataframe['low'].shift(1) < dataframe['low'].shift(2)) & \
                     (dataframe['low'].shift(1) < dataframe['low'])
            fractal_tops = np.where(top, dataframe['high'].shift(1), np.nan)
            fractal_bottoms = np.where(bottom, dataframe['low'].shift(1), np.nan)
        else:
            fractal_tops = np.full(len(dataframe), np.nan)
            fractal_bottoms = np.full(len(dataframe), np.nan)
        return fractal_tops, fractal_bottoms

    def _calculate_ma(self, dataframe: DataFrame, period: int, ma_type: str) -> np.ndarray:
        if ma_type == "EMA": return ta.EMA(dataframe["close"], timeperiod=period)
        elif ma_type == "SMA": return ta.SMA(dataframe["close"], timeperiod=period)
        elif ma_type == "WMA": return ta.WMA(dataframe["close"], timeperiod=period)
        else: return ta.EMA(dataframe["close"], timeperiod=period)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for window_size in [3, 5]:
            tops, bottoms = self._calculate_fractals(dataframe, window_size)
            dataframe[f"fractal_top_{window_size}"] = pd.Series(tops).ffill()
            dataframe[f"fractal_bottom_{window_size}"] = pd.Series(bottoms).ffill()

        # Pre-calculate ATR for trailing stop range
        for p in range(7, 31):
            dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.disable_weekends.value:
            is_weekend = dataframe["date"].dt.dayofweek >= 5
        else:
            is_weekend = pd.Series([False] * len(dataframe))

        # LONG
        long_fractal_col_top = f"fractal_top_{self.long_fractal_window.value}"
        long_conditions = (dataframe["close"] > dataframe[long_fractal_col_top] * (1 + self.long_breakout_threshold.value))

        if self.long_use_ma_filter.value:
            ma = self._calculate_ma(dataframe, self.long_ma_period.value, self.long_ma_type.value)
            long_conditions &= (dataframe["close"] > ma)
        if self.long_use_rsi_filter.value:
            rsi = ta.RSI(dataframe["close"], timeperiod=self.long_rsi_period.value)
            long_conditions &= (rsi > self.long_rsi_min.value)
        if self.long_use_volume_filter.value:
            vol_ma = ta.SMA(dataframe["volume"], timeperiod=self.long_volume_ma_period.value)
            long_conditions &= (dataframe["volume"] > vol_ma * self.long_volume_threshold.value)
        if self.long_use_adx_filter.value:
            adx = ta.ADX(dataframe, timeperiod=self.long_adx_period.value)
            long_conditions &= (adx > self.long_adx_min.value)

        dataframe.loc[long_conditions & (~is_weekend), "enter_long"] = 1

        # SHORT
        short_fractal_col_bottom = f"fractal_bottom_{self.short_fractal_window.value}"
        short_conditions = (dataframe["close"] < dataframe[short_fractal_col_bottom] * (1 - self.short_breakout_threshold.value))

        if self.short_use_ma_filter.value:
            ma = self._calculate_ma(dataframe, self.short_ma_period.value, self.short_ma_type.value)
            short_conditions &= (dataframe["close"] < ma)
        if self.short_use_rsi_filter.value:
            rsi = ta.RSI(dataframe["close"], timeperiod=self.short_rsi_period.value)
            short_conditions &= (rsi < self.short_rsi_max.value)
        if self.short_use_volume_filter.value:
            vol_ma = ta.SMA(dataframe["volume"], timeperiod=self.short_volume_ma_period.value)
            short_conditions &= (dataframe["volume"] > vol_ma * self.short_volume_threshold.value)
        if self.short_use_adx_filter.value:
            adx = ta.ADX(dataframe, timeperiod=self.short_adx_period.value)
            short_conditions &= (adx > self.short_adx_min.value)

        dataframe.loc[short_conditions & (~is_weekend), "enter_short"] = 1

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

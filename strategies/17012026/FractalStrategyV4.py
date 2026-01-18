# --- Do not remove these libs ---
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

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


class FractalStrategyV4(IStrategy):
    """
    V3: Hyperopt-safe + safer ATR trailing exit.

    Fixes:
    - Hyperopt KeyError (e.g., 'WMA_159') by computing MA/RSI/ADX/Volume-MA on-demand
      in populate_entry_trend if the needed column is missing.
    - Keeps V2 ATR trailing improvements (arming + ATR distance stabilization).
    - Stores trade info per trade.id.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    # === LONG ENTRY PARAMETERS (buy space) ===
    long_fractal_window = CategoricalParameter([3, 5], default=5, space="buy", optimize=True)
    long_breakout_threshold = DecimalParameter(
        0.001, 0.05, default=0.01, decimals=3, space="buy", optimize=True
    )

    long_ma_period = IntParameter(20, 500, default=100, space="buy", optimize=True)
    long_ma_type = CategoricalParameter(
        ["EMA", "SMA", "WMA"], default="EMA", space="buy", optimize=True
    )
    long_use_ma_filter = BooleanParameter(default=True, space="buy", optimize=True)

    long_rsi_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    long_rsi_min = IntParameter(20, 50, default=30, space="buy", optimize=True)
    long_use_rsi_filter = BooleanParameter(default=True, space="buy", optimize=True)

    long_volume_ma_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    long_volume_threshold = DecimalParameter(
        0.5, 3.0, default=1.2, decimals=1, space="buy", optimize=True
    )
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
    short_ma_type = CategoricalParameter(
        ["EMA", "SMA", "WMA"], default="EMA", space="sell", optimize=True
    )
    short_use_ma_filter = BooleanParameter(default=True, space="sell", optimize=True)

    short_rsi_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_rsi_max = IntParameter(50, 80, default=70, space="sell", optimize=True)
    short_use_rsi_filter = BooleanParameter(default=True, space="sell", optimize=True)

    short_volume_ma_period = IntParameter(10, 50, default=20, space="sell", optimize=True)
    short_volume_threshold = DecimalParameter(
        0.5, 3.0, default=1.2, decimals=1, space="sell", optimize=True
    )
    short_use_volume_filter = BooleanParameter(default=True, space="sell", optimize=True)

    short_adx_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_adx_min = IntParameter(15, 40, default=25, space="sell", optimize=True)
    short_use_adx_filter = BooleanParameter(default=False, space="sell", optimize=True)

    # === EXIT PARAMETERS (profit space) ===
    volatility_tp_X = DecimalParameter(
        2.0, 3.0, default=2.5, decimals=1, space="profit", optimize=True
    )

    atr_trailing_k = DecimalParameter(
        1.0, 2.0, default=1.5, decimals=1, space="profit", optimize=True
    )

    # Arm ATR trailing only after trade proves itself
    atr_trail_activate_profit = DecimalParameter(
        0.0, 0.02, default=0.004, decimals=3, space="profit", optimize=True
    )
    atr_trail_activate_atr = DecimalParameter(
        0.5, 3.0, default=1.0, decimals=1, space="profit", optimize=True
    )
    atr_trail_allow_on_loss = BooleanParameter(default=False, space="profit", optimize=True)

    # Stabilize ATR distance (entry ATR weight)
    atr_distance_entry_weight = DecimalParameter(
        0.0, 1.0, default=0.7, decimals=2, space="profit", optimize=True
    )

    # === STRATEGY SETTINGS ===
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = "15m"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    # Per trade state
    custom_info = {}

    # ---------- helpers ----------

    def _calculate_fractals(
        self, dataframe: DataFrame, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        mid_point = window_size // 2
        tops = np.full(len(dataframe), np.nan)
        bottoms = np.full(len(dataframe), np.nan)

        for i in range(mid_point, len(dataframe) - mid_point):
            is_top = True
            for j in range(i - mid_point, i):
                if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                    is_top = False
                    break
            if is_top:
                for j in range(i + 1, i + mid_point + 1):
                    if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                        is_top = False
                        break
            if is_top:
                tops[i] = dataframe["high"].iloc[i]

            is_bottom = True
            for j in range(i - mid_point, i):
                if dataframe["low"].iloc[j] <= dataframe["low"].iloc[i]:
                    is_bottom = False
                    break
            if is_bottom:
                for j in range(i + 1, i + mid_point + 1):
                    if dataframe["low"].iloc[j] <= dataframe["low"].iloc[i]:
                        is_bottom = False
                        break
            if is_bottom:
                bottoms[i] = dataframe["low"].iloc[i]

        return tops, bottoms

    def _calculate_ma(self, dataframe: DataFrame, period: int, ma_type: str) -> np.ndarray:
        if ma_type == "EMA":
            return ta.EMA(dataframe["close"], timeperiod=period)
        if ma_type == "SMA":
            return ta.SMA(dataframe["close"], timeperiod=period)
        if ma_type == "WMA":
            return ta.WMA(dataframe["close"], timeperiod=period)
        return ta.EMA(dataframe["close"], timeperiod=period)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fractals for 3 and 5 (as before)
        for w in [3, 5]:
            tops, bottoms = self._calculate_fractals(dataframe, w)
            dataframe[f"fractal_top_{w}"] = tops
            dataframe[f"fractal_bottom_{w}"] = bottoms
            dataframe[f"fractal_top_{w}"] = dataframe[f"fractal_top_{w}"].ffill()
            dataframe[f"fractal_bottom_{w}"] = dataframe[f"fractal_bottom_{w}"].ffill()

        # Always-present indicators used by exits / structure
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["bb_upper"], dataframe["bb_middle"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === LONG ===
        long_top = f"fractal_top_{self.long_fractal_window.value}"
        long_bot = f"fractal_bottom_{self.long_fractal_window.value}"

        long_conditions = dataframe["close"] > dataframe[long_top] * (
            1 + self.long_breakout_threshold.value
        )

        # MA filter (hyperopt-safe: compute if missing)
        if self.long_use_ma_filter.value:
            ma_col = f"{self.long_ma_type.value}_{self.long_ma_period.value}"
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.long_ma_period.value, self.long_ma_type.value
                )
            long_conditions &= dataframe["close"] > dataframe[ma_col]

        # RSI filter
        if self.long_use_rsi_filter.value:
            rsi_col = f"rsi_{self.long_rsi_period.value}"
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.long_rsi_period.value
                )
            long_conditions &= dataframe[rsi_col] > self.long_rsi_min.value

        # Volume filter
        if self.long_use_volume_filter.value:
            vma_col = f"volume_ma_{self.long_volume_ma_period.value}"
            if vma_col not in dataframe.columns:
                dataframe[vma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.long_volume_ma_period.value
                )
            long_conditions &= (
                dataframe["volume"] > dataframe[vma_col] * self.long_volume_threshold.value
            )

        # ADX filter
        if self.long_use_adx_filter.value:
            adx_col = f"adx_{self.long_adx_period.value}"
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.long_adx_period.value)
            long_conditions &= dataframe[adx_col] > self.long_adx_min.value

        dataframe.loc[long_conditions, "enter_long"] = 1
        dataframe.loc[long_conditions, "fractal_top_entry"] = dataframe[long_top]
        dataframe.loc[long_conditions, "fractal_bottom_entry"] = dataframe[long_bot]

        # === SHORT ===
        short_top = f"fractal_top_{self.short_fractal_window.value}"
        short_bot = f"fractal_bottom_{self.short_fractal_window.value}"

        short_conditions = dataframe["close"] < dataframe[short_bot] * (
            1 - self.short_breakout_threshold.value
        )

        if self.short_use_ma_filter.value:
            ma_col = f"{self.short_ma_type.value}_{self.short_ma_period.value}"
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.short_ma_period.value, self.short_ma_type.value
                )
            short_conditions &= dataframe["close"] < dataframe[ma_col]

        if self.short_use_rsi_filter.value:
            rsi_col = f"rsi_{self.short_rsi_period.value}"
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.short_rsi_period.value
                )
            short_conditions &= dataframe[rsi_col] < self.short_rsi_max.value

        if self.short_use_volume_filter.value:
            vma_col = f"volume_ma_{self.short_volume_ma_period.value}"
            if vma_col not in dataframe.columns:
                dataframe[vma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.short_volume_ma_period.value
                )
            short_conditions &= (
                dataframe["volume"] > dataframe[vma_col] * self.short_volume_threshold.value
            )

        if self.short_use_adx_filter.value:
            adx_col = f"adx_{self.short_adx_period.value}"
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.short_adx_period.value)
            short_conditions &= dataframe[adx_col] > self.short_adx_min.value

        dataframe.loc[short_conditions, "enter_short"] = 1
        dataframe.loc[short_conditions, "fractal_top_entry"] = dataframe[short_top]
        dataframe.loc[short_conditions, "fractal_bottom_entry"] = dataframe[short_bot]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def _trade_key(self, pair: str, trade: "Trade") -> str:
        tid = getattr(trade, "id", None)
        return f"{pair}:{tid}" if tid is not None else f"{pair}:{trade.open_date_utc.isoformat()}"

    def _get_entry_candle(self, dataframe: DataFrame, trade_open_time: datetime):
        if dataframe is None or dataframe.empty:
            return None

        # If index is datetime-like
        try:
            idx = dataframe.index
            if hasattr(idx, "tz") or hasattr(idx, "to_pydatetime"):
                mask = idx <= trade_open_time
                if mask.any():
                    return dataframe.loc[mask].iloc[-1]
        except Exception:
            pass

        # If "date" column exists
        if "date" in dataframe.columns:
            try:
                mask = dataframe["date"] <= trade_open_time
                if mask.any():
                    return dataframe.loc[mask].iloc[-1]
            except Exception:
                pass

        return dataframe.iloc[-1]

    def _ensure_trade_info(self, pair: str, trade: "Trade") -> dict:
        key = self._trade_key(pair, trade)
        if key in self.custom_info:
            return self.custom_info[key]

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        entry_candle = self._get_entry_candle(dataframe, trade.open_date_utc)

        info = {
            "fractal_top_entry": None,
            "fractal_bottom_entry": None,
            "atr_entry": 0.0,
            "highest_price": trade.open_rate,
            "lowest_price": trade.open_rate,
            "entry_time": trade.open_date_utc,
        }

        if entry_candle is not None:
            info["fractal_top_entry"] = entry_candle.get("fractal_top_entry", None)
            info["fractal_bottom_entry"] = entry_candle.get("fractal_bottom_entry", None)
            info["atr_entry"] = float(entry_candle.get("atr", 0.0) or 0.0)

        self.custom_info[key] = info
        return info

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1] if dataframe is not None and not dataframe.empty else None
        atr_now = float(last_candle.get("atr", 0.0)) if last_candle is not None else 0.0

        info = self._ensure_trade_info(pair, trade)
        entry_atr = float(info.get("atr_entry", 0.0) or 0.0) or atr_now

        info["highest_price"] = max(float(info.get("highest_price", trade.open_rate)), current_rate)
        info["lowest_price"] = min(float(info.get("lowest_price", trade.open_rate)), current_rate)

        # ---- Arm trailing only after favorable excursion ----
        favorable_move = 0.0
        if entry_atr and entry_atr > 0:
            if trade.is_short:
                favorable_move = (trade.open_rate - info["lowest_price"]) / entry_atr
            else:
                favorable_move = (info["highest_price"] - trade.open_rate) / entry_atr

        armed_by_profit = current_profit >= float(self.atr_trail_activate_profit.value)
        armed_by_atr = favorable_move >= float(self.atr_trail_activate_atr.value)
        trailing_armed = armed_by_profit or armed_by_atr

        if trailing_armed:
            w = float(self.atr_distance_entry_weight.value)
            effective_atr = (w * entry_atr) + ((1.0 - w) * atr_now) if atr_now > 0 else entry_atr
            distance = effective_atr * float(self.atr_trailing_k.value)

            if not trade.is_short:
                trail_price = info["highest_price"] - distance
                if current_rate < trail_price:
                    if self.atr_trail_allow_on_loss.value or current_profit >= 0:
                        return "atr_trailing_exit"
            else:
                trail_price = info["lowest_price"] + distance
                if current_rate > trail_price:
                    if self.atr_trail_allow_on_loss.value or current_profit >= 0:
                        return "atr_trailing_exit"

        # Volatility-normalized TP
        if atr_now > 0:
            normalized_profit = abs(current_rate - trade.open_rate) / atr_now
            if normalized_profit >= float(self.volatility_tp_X.value):
                return "volatility_tp"

        # Time-decay loss exit
        trade_age_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if trade_age_hours > 3 and current_profit < -0.01:
            return "time_decay_loss"

        # Volatility collapse exit (same as original intent)
        if entry_atr > 0 and atr_now > 0 and (atr_now / entry_atr) < 0.6 and current_profit < -0.02:
            return "volatility_collapse_exit"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        info = self._ensure_trade_info(pair, trade)

        fractal_top_entry = info.get("fractal_top_entry")
        fractal_bottom_entry = info.get("fractal_bottom_entry")
        atr_entry = float(info.get("atr_entry", 0.0) or 0.0)

        if fractal_top_entry is None or fractal_bottom_entry is None or atr_entry <= 0:
            return -1

        if not trade.is_short:
            stoploss_price = float(fractal_bottom_entry) - (atr_entry * 0.5)
            stoploss_pct = (trade.open_rate - stoploss_price) / trade.open_rate
            if current_profit > 0.015:
                return max(-0.002, -stoploss_pct)
        else:
            stoploss_price = float(fractal_top_entry) + (atr_entry * 0.5)
            stoploss_pct = (stoploss_price - trade.open_rate) / trade.open_rate
            if current_profit > 0.015:
                return max(-0.002, -stoploss_pct)

        return max(-stoploss_pct, self.stoploss)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        return min(3.0, max_leverage)

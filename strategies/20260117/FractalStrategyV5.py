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


class FractalStrategyV5(IStrategy):
    """
    V5 goals:
    1) Fix volatility_tp so it cannot be a loss exit (direction-aware, profit-gated).
    2) Make critical "loss bucket" controls optimizable:
       - time_decay_hours
       - time_decay_min_profit
       - atr_trail_allow_on_loss
       - atr_trail_min_profit_if_allow_on_loss (safety gate)
    3) Keep everything hyperopt-friendly and avoid KeyError for MA columns.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    # === LONG ENTRY PARAMETERS (buy) ===
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

    # === SHORT ENTRY PARAMETERS (sell) ===
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

    # === EXIT PARAMETERS (profit) ===
    volatility_tp_X = DecimalParameter(
        1.0, 5.0, default=2.5, decimals=1, space="profit", optimize=True
    )
    volatility_tp_min_profit = DecimalParameter(
        0.0, 0.02, default=0.005, decimals=3, space="profit", optimize=True
    )

    atr_trailing_k = DecimalParameter(
        1.0, 2.5, default=1.5, decimals=1, space="profit", optimize=True
    )
    atr_trail_activate_profit = DecimalParameter(
        0.0, 0.05, default=0.01, decimals=3, space="profit", optimize=True
    )
    atr_trail_activate_atr = DecimalParameter(
        0.5, 5.0, default=1.0, decimals=1, space="profit", optimize=True
    )
    atr_distance_entry_weight = DecimalParameter(
        0.0, 1.0, default=0.7, decimals=2, space="profit", optimize=True
    )

    # critical: make it optimizable
    atr_trail_allow_on_loss = BooleanParameter(default=False, space="profit", optimize=True)

    # safety gate if allow_on_loss=True
    atr_trail_min_profit_if_allow_on_loss = DecimalParameter(
        -0.03, 0.01, default=0.0, decimals=3, space="profit", optimize=True
    )

    # critical: time-decay controls (optimizable)
    time_decay_hours = DecimalParameter(
        2.0, 12.0, default=6.0, decimals=1, space="profit", optimize=True
    )
    time_decay_min_profit = DecimalParameter(
        -0.05, -0.005, default=-0.02, decimals=3, space="profit", optimize=True
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

    custom_info = {}

    # ---------- helpers ----------

    def _calculate_fractals(
        self, dataframe: DataFrame, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        mid = window_size // 2
        tops = np.full(len(dataframe), np.nan)
        bottoms = np.full(len(dataframe), np.nan)

        for i in range(mid, len(dataframe) - mid):
            is_top = True
            for j in range(i - mid, i):
                if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                    is_top = False
                    break
            if is_top:
                for j in range(i + 1, i + mid + 1):
                    if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                        is_top = False
                        break
            if is_top:
                tops[i] = dataframe["high"].iloc[i]

            is_bottom = True
            for j in range(i - mid, i):
                if dataframe["low"].iloc[j] <= dataframe["low"].iloc[i]:
                    is_bottom = False
                    break
            if is_bottom:
                for j in range(i + 1, i + mid + 1):
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
        for w in [3, 5]:
            tops, bottoms = self._calculate_fractals(dataframe, w)
            dataframe[f"fractal_top_{w}"] = tops
            dataframe[f"fractal_bottom_{w}"] = bottoms
            dataframe[f"fractal_top_{w}"] = dataframe[f"fractal_top_{w}"].ffill()
            dataframe[f"fractal_bottom_{w}"] = dataframe[f"fractal_bottom_{w}"].ffill()

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG
        long_top = f"fractal_top_{self.long_fractal_window.value}"
        long_bot = f"fractal_bottom_{self.long_fractal_window.value}"
        long_cond = dataframe["close"] > dataframe[long_top] * (
            1 + self.long_breakout_threshold.value
        )

        if self.long_use_ma_filter.value:
            ma_col = f"{self.long_ma_type.value}_{self.long_ma_period.value}"
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.long_ma_period.value, self.long_ma_type.value
                )
            long_cond &= dataframe["close"] > dataframe[ma_col]

        if self.long_use_rsi_filter.value:
            rsi_col = f"rsi_{self.long_rsi_period.value}"
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.long_rsi_period.value
                )
            long_cond &= dataframe[rsi_col] > self.long_rsi_min.value

        if self.long_use_volume_filter.value:
            vma_col = f"volume_ma_{self.long_volume_ma_period.value}"
            if vma_col not in dataframe.columns:
                dataframe[vma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.long_volume_ma_period.value
                )
            long_cond &= dataframe["volume"] > dataframe[vma_col] * self.long_volume_threshold.value

        if self.long_use_adx_filter.value:
            adx_col = f"adx_{self.long_adx_period.value}"
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.long_adx_period.value)
            long_cond &= dataframe[adx_col] > self.long_adx_min.value

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[long_cond, "fractal_top_entry"] = dataframe[long_top]
        dataframe.loc[long_cond, "fractal_bottom_entry"] = dataframe[long_bot]

        # SHORT
        short_top = f"fractal_top_{self.short_fractal_window.value}"
        short_bot = f"fractal_bottom_{self.short_fractal_window.value}"
        short_cond = dataframe["close"] < dataframe[short_bot] * (
            1 - self.short_breakout_threshold.value
        )

        if self.short_use_ma_filter.value:
            ma_col = f"{self.short_ma_type.value}_{self.short_ma_period.value}"
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.short_ma_period.value, self.short_ma_type.value
                )
            short_cond &= dataframe["close"] < dataframe[ma_col]

        if self.short_use_rsi_filter.value:
            rsi_col = f"rsi_{self.short_rsi_period.value}"
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.short_rsi_period.value
                )
            short_cond &= dataframe[rsi_col] < self.short_rsi_max.value

        if self.short_use_volume_filter.value:
            vma_col = f"volume_ma_{self.short_volume_ma_period.value}"
            if vma_col not in dataframe.columns:
                dataframe[vma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.short_volume_ma_period.value
                )
            short_cond &= (
                dataframe["volume"] > dataframe[vma_col] * self.short_volume_threshold.value
            )

        if self.short_use_adx_filter.value:
            adx_col = f"adx_{self.short_adx_period.value}"
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.short_adx_period.value)
            short_cond &= dataframe[adx_col] > self.short_adx_min.value

        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[short_cond, "fractal_top_entry"] = dataframe[short_top]
        dataframe.loc[short_cond, "fractal_bottom_entry"] = dataframe[short_bot]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = dataframe.iloc[-1]

        if not hasattr(self, "custom_info"):
            self.custom_info = {}

        # Key by pair only (simple) — but avoids missing data at entry
        self.custom_info[pair] = {
            "fractal_top_entry": float(last.get("fractal_top_entry", np.nan)),
            "fractal_bottom_entry": float(last.get("fractal_bottom_entry", np.nan)),
            "atr_entry": float(last.get("atr", 0.0) or 0.0),
            "entry_time": current_time,
            "side": side,
            "highest_price": rate,
            "lowest_price": rate,
        }
        return True

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
        last = dataframe.iloc[-1]
        atr_now = float(last.get("atr", 0.0) or 0.0)

        if not hasattr(self, "custom_info") or pair not in self.custom_info:
            return None

        info = self.custom_info[pair]
        entry_atr = float(info.get("atr_entry", 0.0) or 0.0) or atr_now

        info["highest_price"] = max(float(info.get("highest_price", trade.open_rate)), current_rate)
        info["lowest_price"] = min(float(info.get("lowest_price", trade.open_rate)), current_rate)

        # Favorable excursion (ATR units) using entry ATR (more stable)
        favorable_atr = 0.0
        if entry_atr and entry_atr > 0:
            if trade.is_short:
                favorable_atr = (trade.open_rate - info["lowest_price"]) / entry_atr
            else:
                favorable_atr = (info["highest_price"] - trade.open_rate) / entry_atr

        # ---- ATR trailing (armed) ----
        armed = (current_profit >= float(self.atr_trail_activate_profit.value)) or (
            favorable_atr >= float(self.atr_trail_activate_atr.value)
        )

        if armed and entry_atr > 0:
            w = float(self.atr_distance_entry_weight.value)
            effective_atr = entry_atr if atr_now <= 0 else (w * entry_atr) + ((1.0 - w) * atr_now)
            distance = effective_atr * float(self.atr_trailing_k.value)

            allow_on_loss = bool(self.atr_trail_allow_on_loss.value)
            min_profit_gate = float(self.atr_trail_min_profit_if_allow_on_loss.value)

            def atr_allowed() -> bool:
                if allow_on_loss:
                    return current_profit >= min_profit_gate
                return current_profit >= 0.0

            if not trade.is_short:
                trail_price = info["highest_price"] - distance
                if current_rate < trail_price and atr_allowed():
                    return "atr_trailing_exit"
            else:
                trail_price = info["lowest_price"] + distance
                if current_rate > trail_price and atr_allowed():
                    return "atr_trailing_exit"

        # ---- Volatility TP (FIXED: profit-only, direction-aware) ----
        if atr_now > 0:
            if not trade.is_short:
                move_atr = (current_rate - trade.open_rate) / atr_now
            else:
                move_atr = (trade.open_rate - current_rate) / atr_now

            if current_profit >= float(self.volatility_tp_min_profit.value) and move_atr >= float(
                self.volatility_tp_X.value
            ):
                return "volatility_tp"

        # ---- Time-decay loss (OPTIMIZABLE) ----
        age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if age_h >= float(self.time_decay_hours.value) and current_profit <= float(
            self.time_decay_min_profit.value
        ):
            return "time_decay_loss"

        # Volatility collapse emergency (unchanged)
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
        if not hasattr(self, "custom_info") or pair not in self.custom_info:
            return -1

        info = self.custom_info[pair]
        top = info.get("fractal_top_entry")
        bot = info.get("fractal_bottom_entry")
        atr_entry = float(info.get("atr_entry", 0.0) or 0.0)

        if top is None or bot is None or np.isnan(top) or np.isnan(bot):
            return -1

        if not trade.is_short:
            stop_price = float(bot) - (atr_entry * 0.5)
            stop_pct = (trade.open_rate - stop_price) / trade.open_rate
            if current_profit > 0.015:
                return max(-0.002, -stop_pct)
        else:
            stop_price = float(top) + (atr_entry * 0.5)
            stop_pct = (stop_price - trade.open_rate) / trade.open_rate
            if current_profit > 0.015:
                return max(-0.002, -stop_pct)

        return max(-stop_pct, self.stoploss)

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

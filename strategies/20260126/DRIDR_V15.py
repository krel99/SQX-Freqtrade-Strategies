# pragma pylint: disable=missing-module-docstring, invalid-name, pointless-string-statement
from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib


class DRIDR_V15(IStrategy):
    """
    DR/IDR (Regular session) strategy inspired by TradingView "DR/IDR V1.5".

    Core idea replicated from the TV script:
      - During Regular Defining Range (RDR) time: 09:30-10:30 New York
          DR High/Low  = max(high) / min(low) during that window
          IDR High/Low = max(max(open,close)) / min(min(open,close)) during that window
      - After RDR ends, those levels remain "active" for the rest of the Regular day session.

    Hyperopt note (as requested):
      - populate_indicators computes ONLY non-optimizable columns.
      - All optimizable logic is applied in populate_entry_trend / populate_exit_trend.
    """

    # ========= Freqtrade base settings =========
    timeframe = "5m"  # The TV script uses 5m when chart TF < 5m. Use 5m here for clean DR/IDR construction.
    can_short = True

    minimal_roi = {"0": 0.10}  # will usually be overridden by exits; keep sane default
    stoploss = -0.99  # strategy-managed exits (std-based) - keep wide; you can tighten later
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 200  # plenty for stable session computations

    # ========= Session settings (not hyperopt) =========
    # TradingView: TIMEZONE = 'America/New_York'
    NY_TZ = "America/New_York"

    # Regular session windows (TV defaults)
    # RDR: 09:30-10:30 ; Regular extend: 10:30-16:00
    RDR_START = time(9, 30)
    RDR_END = time(10, 30)
    REG_EXT_END = time(16, 0)

    # ========= Hyperopt parameters (DO NOT use in populate_indicators) =========
    entry_mode = CategoricalParameter(
        ["breakout", "reversion"],
        default="breakout",
        space="buy",
        optimize=True,
    )

    # Entry threshold in "STD units" where 1.0 == IDR High/Low (because std_step = 0.5*IDR range)
    entry_std = DecimalParameter(0.0, 4.0, default=0.0, decimals=1, space="buy", optimize=True)

    # Exits as STD units away from IDR mid
    tp_std = DecimalParameter(0.5, 6.0, default=2.0, decimals=1, space="sell", optimize=True)
    sl_std = DecimalParameter(0.5, 6.0, default=2.0, decimals=1, space="sell", optimize=True)

    # Optional filter: require IDR range > X * price (avoid tiny ranges)
    min_idr_bps = IntParameter(0, 50, default=5, space="buy", optimize=True)  # basis points (0.01%)

    # Optional filter: only trade for N candles after RDR ends
    trade_window_candles = IntParameter(6, 120, default=72, space="buy", optimize=True)  # 5m candles

    # ========= Helpers =========
    @staticmethod
    def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure 'date' is tz-aware UTC for reliable conversion.
        Freqtrade typically provides tz-aware UTC, but handle both cases.
        """
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column.")
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], utc=True)

        # If naive, localize to UTC; if aware, convert to UTC
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")
        return df

    @staticmethod
    def _time_in_range(t: pd.Series, start: time, end: time) -> pd.Series:
        """
        True for times in [start, end) assuming start < end within same day.
        """
        return (t >= start) & (t < end)

    # ========= Indicator construction (NO hyperopt params used here) =========
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        df = self._ensure_utc_index(df)

        # Convert to New York time for session logic
        ny_dt = df["date"].dt.tz_convert(self.NY_TZ)
        df["ny_date"] = ny_dt.dt.date
        df["ny_time"] = ny_dt.dt.time

        # Session masks
        df["in_rdr"] = self._time_in_range(df["ny_time"], self.RDR_START, self.RDR_END)
        df["in_reg_extend"] = self._time_in_range(df["ny_time"], self.RDR_END, self.REG_EXT_END)

        # Define "active trading region" == after RDR ends, until regular extend end
        df["idr_active"] = df["in_reg_extend"].astype("int8")

        # --- DR / IDR levels per NY day (computed only from RDR window) ---
        # DR = high/low extremes
        # IDR = open/close extremes
        oc_high = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
        oc_low = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
        df["oc_high"] = oc_high
        df["oc_low"] = oc_low

        # Use groupby transform, but only on rows inside RDR.
        # Outside RDR: NaN, then forward-fill within each day for the rest of the session.
        g = df.groupby("ny_date", sort=False)

        df["dr_high_raw"] = np.where(df["in_rdr"], df["high"], np.nan)
        df["dr_low_raw"] = np.where(df["in_rdr"], df["low"], np.nan)
        df["idr_high_raw"] = np.where(df["in_rdr"], df["oc_high"], np.nan)
        df["idr_low_raw"] = np.where(df["in_rdr"], df["oc_low"], np.nan)

        # Per-day final levels (same value on all rows of the day, but only meaningful if that day had RDR candles)
        df["dr_high_day"] = g["dr_high_raw"].transform("max")
        df["dr_low_day"] = g["dr_low_raw"].transform("min")
        df["idr_high_day"] = g["idr_high_raw"].transform("max")
        df["idr_low_day"] = g["idr_low_raw"].transform("min")

        # Keep levels only after RDR ends (during extend window), otherwise NaN to prevent pre-session usage
        df["dr_high"] = np.where(df["in_reg_extend"], df["dr_high_day"], np.nan)
        df["dr_low"] = np.where(df["in_reg_extend"], df["dr_low_day"], np.nan)
        df["idr_high"] = np.where(df["in_reg_extend"], df["idr_high_day"], np.nan)
        df["idr_low"] = np.where(df["in_reg_extend"], df["idr_low_day"], np.nan)

        # Mid and step (STD step factor = 0.5 * IDR range) -> matches TV logic
        df["idr_mid"] = (df["idr_high"] + df["idr_low"]) / 2.0
        df["idr_range"] = (df["idr_high"] - df["idr_low"]).abs()
        df["std_step"] = df["idr_range"] * 0.5

        # "std_units" position relative to IDR mid, where:
        #   idr_high corresponds to +1.0, idr_low corresponds to -1.0
        # Guard zero step
        df["std_units"] = np.where(
            df["std_step"] > 0,
            (df["close"] - df["idr_mid"]) / df["std_step"],
            np.nan,
        )

        # Candle index since RDR ended (per day) to enforce a limited trade window
        # We count only extend-window candles.
        df["ext_candle_idx"] = np.where(df["in_reg_extend"], 1, 0)
        df["ext_candle_idx"] = g["ext_candle_idx"].cumsum()
        df.loc[~df["in_reg_extend"], "ext_candle_idx"] = 0

        # Clean temp columns
        df.drop(columns=["oc_high", "oc_low"], inplace=True, errors="ignore")

        return df

    # ========= Entries (uses hyperopt params) =========
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()

        # Basic validity: IDR must exist and step must be > 0
        valid = (
            df["idr_active"].eq(1)
            & df["idr_high"].notna()
            & df["idr_low"].notna()
            & df["std_step"].gt(0)
        )

        # Range filter in basis points (bps)
        # idr_range / close >= min_idr_bps * 0.0001
        min_range_ok = (df["idr_range"] / df["close"]) >= (self.min_idr_bps.value * 0.0001)

        # Limit to first N candles of the regular extend
        win_ok = df["ext_candle_idx"].between(1, self.trade_window_candles.value)

        base = valid & min_range_ok & win_ok

        # Thresholds in price space (but derived from std_step and hyperopt params)
        long_break_level = df["idr_high"] + (self.entry_std.value * df["std_step"])
        short_break_level = df["idr_low"] - (self.entry_std.value * df["std_step"])

        # Breakout mode: take cross beyond threshold
        long_breakout = base & qtpylib.crossed_above(df["close"], long_break_level)
        short_breakout = base & qtpylib.crossed_below(df["close"], short_break_level)

        # Reversion mode:
        #   - Long: price pushes below lower threshold then crosses back above IDR low (or threshold)
        #   - Short: price pushes above upper threshold then crosses back below IDR high (or threshold)
        long_revert = base & qtpylib.crossed_above(df["close"], df["idr_low"])
        long_revert &= df["low"].shift(1) < short_break_level.shift(1)

        short_revert = base & qtpylib.crossed_below(df["close"], df["idr_high"])
        short_revert &= df["high"].shift(1) > long_break_level.shift(1)

        if self.entry_mode.value == "breakout":
            df.loc[long_breakout, "enter_long"] = 1
            df.loc[short_breakout, "enter_short"] = 1
        else:
            df.loc[long_revert, "enter_long"] = 1
            df.loc[short_revert, "enter_short"] = 1

        return df

    # ========= Exits (uses hyperopt params) =========
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()

        valid = (
            df["idr_high"].notna()
            & df["idr_low"].notna()
            & df["idr_mid"].notna()
            & df["std_step"].gt(0)
        )

        # Targets/Stops relative to IDR mid in "std units"
        # For longs:
        #   TP at idr_mid + tp_std*std_step
        #   SL at idr_mid - sl_std*std_step
        long_tp = df["idr_mid"] + (self.tp_std.value * df["std_step"])
        long_sl = df["idr_mid"] - (self.sl_std.value * df["std_step"])

        # For shorts:
        #   TP at idr_mid - tp_std*std_step
        #   SL at idr_mid + sl_std*std_step
        short_tp = df["idr_mid"] - (self.tp_std.value * df["std_step"])
        short_sl = df["idr_mid"] + (self.sl_std.value * df["std_step"])

        # Exit signals
        df.loc[valid & qtpylib.crossed_above(df["close"], long_tp), "exit_long"] = 1
        df.loc[valid & qtpylib.crossed_below(df["close"], long_sl), "exit_long"] = 1

        df.loc[valid & qtpylib.crossed_below(df["close"], short_tp), "exit_short"] = 1
        df.loc[valid & qtpylib.crossed_above(df["close"], short_sl), "exit_short"] = 1

        return df

    # Optional: ensure we only trade during regular extend (safety net)
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        df = kwargs.get("dataframe")
        if df is None or df.empty:
            return True
        last = df.iloc[-1]
        return bool(last.get("idr_active", 0) == 1)

    # Optional: custom exit hook (not required)
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        return None

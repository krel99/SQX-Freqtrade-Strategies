# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy


class ChannelBreakoutOvernightV1(IStrategy):
    """
    Robust Version 1 (works with RangeIndex dataframes in backtesting):
    - Uses dataframe['date'] (Freqtrade standard) as the time source.
    - Builds a daily 1-hour channel starting at (channel_start_hour:channel_start_minute).
    - Measures "overnight" window before channel start (ATR% + directionality proxy).
    - Skip day if channel width outside bounds.
    - After channel ends, allow breakout entries for entry_expiry_minutes.
    - TP at proj_takeprofit_mult * channel_width.
    - Stop management via custom_stoploss (3 modes).
    - Time expiry via custom_exit (min 3h, max 36h).
    """

    can_short = True
    timeframe = "5m"
    startup_candle_count = 600

    process_only_new_candles = True
    use_exit_signal = True

    minimal_roi = {"0": 0.0}
    stoploss = -0.99

    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ====== Optimizable parameters ======
    channel_start_hour = IntParameter(0, 23, default=8, space="buy", optimize=True)
    channel_start_minute = IntParameter(0, 55, default=0, space="buy", optimize=True)  # 5m steps

    channel_min_width = DecimalParameter(
        0.001, 0.03, default=0.004, decimals=4, space="buy", optimize=True
    )
    channel_max_width = DecimalParameter(
        0.003, 0.08, default=0.02, decimals=4, space="buy", optimize=True
    )

    entry_expiry_minutes = IntParameter(15, 360, default=120, space="buy", optimize=True)

    overnight_lookback_hours = IntParameter(3, 24, default=12, space="buy", optimize=True)

    atr_len = IntParameter(50, 400, default=200, space="buy", optimize=True)
    overnight_vol_threshold = DecimalParameter(
        0.001, 0.04, default=0.008, decimals=4, space="buy", optimize=True
    )
    directionality_threshold = DecimalParameter(
        0.2, 0.95, default=0.55, decimals=2, space="buy", optimize=True
    )

    allow_trade_flat = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    allow_trade_trend_with = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=True
    )
    allow_trade_trend_against = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=True
    )

    breakout_buffer_frac = DecimalParameter(
        0.0, 0.01, default=0.001, decimals=4, space="buy", optimize=True
    )

    proj_takeprofit_mult = DecimalParameter(
        0.5, 5.0, default=2.0, decimals=2, space="sell", optimize=True
    )
    initial_stop_mult = DecimalParameter(
        0.2, 2.0, default=1.0, decimals=2, space="sell", optimize=True
    )

    exit_mode = CategoricalParameter(
        ["step_stop", "be_after_l1", "trail_atr"], default="step_stop", space="sell", optimize=True
    )
    trail_atr_mult = DecimalParameter(
        0.5, 6.0, default=3.0, decimals=2, space="sell", optimize=True
    )

    min_hold_hours = IntParameter(3, 12, default=3, space="sell", optimize=True)
    max_hold_hours = IntParameter(6, 36, default=24, space="sell", optimize=True)

    # ====== Time handling (critical fix) ======
    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure df has a timezone-aware UTC datetime column 'dt'.
        Freqtrade usually provides a 'date' column. We'll use that.
        """
        if "date" not in df.columns:
            raise ValueError("DataFrame has no 'date' column. Cannot compute channel windows.")

        out = df.copy()
        dt = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out["dt"] = dt

        # Drop rows with invalid dt (should be rare)
        out = out[out["dt"].notna()]
        return out

    @staticmethod
    def _day_start_utc(dt: pd.Series) -> pd.Series:
        """
        Normalize to day start in UTC (00:00).
        dt must be tz-aware UTC.
        """
        # dt.dt.floor('D') works on tz-aware series
        return dt.dt.floor("D")

    # ====== Indicators ======
    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        return pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    @classmethod
    def _atr(cls, df: pd.DataFrame, length: int) -> pd.Series:
        tr = cls._true_range(df)
        return tr.rolling(length, min_periods=length).mean()

    def _compute_channel_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
        - ch_start, ch_end (tz-aware UTC timestamps)
        - in_channel, post_channel
        - ch_high, ch_low, ch_width, ch_width_frac
        - entry_allowed (expiry after channel end)
        """
        df = self._ensure_dt(df)

        day0 = self._day_start_utc(df["dt"])
        ch_start = (
            day0
            + pd.to_timedelta(int(self.channel_start_hour.value), unit="h")
            + pd.to_timedelta(int(self.channel_start_minute.value), unit="m")
        )
        ch_end = ch_start + pd.to_timedelta(1, unit="h")

        df["ch_start"] = ch_start
        df["ch_end"] = ch_end

        df["in_channel"] = (df["dt"] >= df["ch_start"]) & (df["dt"] < df["ch_end"])
        df["post_channel"] = df["dt"] >= df["ch_end"]

        # compute channel high/low per day (key = ch_start)
        tmp_high = df["high"].where(df["in_channel"])
        tmp_low = df["low"].where(df["in_channel"])

        ch_high = tmp_high.groupby(df["ch_start"]).transform("max")
        ch_low = tmp_low.groupby(df["ch_start"]).transform("min")

        # ffill inside each day so post-channel rows have values
        df["ch_high"] = ch_high.groupby(df["ch_start"]).transform(lambda s: s.ffill())
        df["ch_low"] = ch_low.groupby(df["ch_start"]).transform(lambda s: s.ffill())

        df["ch_width"] = df["ch_high"] - df["ch_low"]
        df["ch_mid"] = (df["ch_high"] + df["ch_low"]) / 2.0
        df["ch_width_frac"] = df["ch_width"] / df["ch_mid"]

        expiry = pd.to_timedelta(int(self.entry_expiry_minutes.value), unit="m")
        df["entry_allowed"] = (df["dt"] >= df["ch_end"]) & (df["dt"] < (df["ch_end"] + expiry))

        return df

    def _compute_overnight_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes overnight stats per day (keyed by ch_start):
        - overnight_vol: mean ATR% during overnight window
        - overnight_ret: return over overnight window
        - overnight_dir_strength: |ret| / (vol + eps)
        """
        length = int(self.atr_len.value)
        df["atr"] = self._atr(df, length)
        df["atrp"] = df["atr"] / df["close"]

        overnight_vol = np.full(len(df), np.nan, dtype=float)
        overnight_ret = np.full(len(df), np.nan, dtype=float)
        overnight_dir_strength = np.full(len(df), np.nan, dtype=float)
        overnight_dir = np.zeros(len(df), dtype=int)

        unique_days = pd.Index(df["ch_start"].unique()).sort_values()
        lb_hours = int(self.overnight_lookback_hours.value)

        for ch_start in unique_days:
            if pd.isna(ch_start):
                continue

            o_start = ch_start - pd.to_timedelta(lb_hours, unit="h")
            o_end = ch_start

            mask = (df["dt"] >= o_start) & (df["dt"] < o_end)
            if mask.sum() < max(10, length // 4):
                continue

            vol_mean = float(df.loc[mask, "atrp"].mean())
            first_close = float(df.loc[mask, "close"].iloc[0])
            last_close = float(df.loc[mask, "close"].iloc[-1])
            ret = (last_close / first_close) - 1.0

            eps = 1e-9
            dir_strength = abs(ret) / (vol_mean + eps)

            daymask = df["ch_start"] == ch_start
            idxs = np.where(daymask.values)[0]

            overnight_vol[idxs] = vol_mean
            overnight_ret[idxs] = ret
            overnight_dir_strength[idxs] = dir_strength
            overnight_dir[idxs] = 1 if ret > 0 else (-1 if ret < 0 else 0)

        df["overnight_vol"] = overnight_vol
        df["overnight_ret"] = overnight_ret
        df["overnight_dir_strength"] = overnight_dir_strength
        df["overnight_trend_dir"] = overnight_dir

        df["overnight_is_volatile"] = df["overnight_vol"] > float(
            self.overnight_vol_threshold.value
        )
        df["overnight_is_trending"] = df["overnight_dir_strength"] > float(
            self.directionality_threshold.value
        )
        df["overnight_flat_volatile"] = df["overnight_is_volatile"] & (~df["overnight_is_trending"])
        return df

    # ====== Freqtrade required methods ======
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()

        df = self._compute_channel_cols(df)
        df = self._compute_overnight_regime(df)

        # helper: 24h rolling range zscore (works on row count; 288 for 5m)
        df["range24"] = (df["high"].rolling(288).max() - df["low"].rolling(288).min()) / df["close"]
        df["range24_z"] = (df["range24"] - df["range24"].rolling(288).mean()) / (
            df["range24"].rolling(288).std() + 1e-9
        )

        return df

    def _day_is_tradable(self, df: pd.DataFrame) -> pd.Series:
        width_ok = (df["ch_width_frac"] >= float(self.channel_min_width.value)) & (
            df["ch_width_frac"] <= float(self.channel_max_width.value)
        )

        trending = df["overnight_is_trending"].fillna(False)
        flat_volatile = df["overnight_flat_volatile"].fillna(False)
        low_vol = ~df["overnight_is_volatile"].fillna(False)

        allow_flat = bool(self.allow_trade_flat.value)
        allow_with = bool(self.allow_trade_trend_with.value)
        allow_against = bool(self.allow_trade_trend_against.value)

        regime_ok = (
            (flat_volatile & allow_flat)
            | (low_vol & allow_flat)
            | (trending & (allow_with | allow_against))
        )
        return width_ok & regime_ok

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()

        tradable = self._day_is_tradable(df)
        allowed = tradable & df["entry_allowed"].fillna(False)

        buf = float(self.breakout_buffer_frac.value)
        long_trigger = df["close"] > (df["ch_high"] * (1.0 + buf))
        short_trigger = df["close"] < (df["ch_low"] * (1.0 - buf))

        trending = df["overnight_is_trending"].fillna(False)
        dirn = df["overnight_trend_dir"].fillna(0).astype(int)

        allow_flat = bool(self.allow_trade_flat.value)
        allow_with = bool(self.allow_trade_trend_with.value)
        allow_against = bool(self.allow_trade_trend_against.value)

        long_ok = np.full(len(df), False, dtype=bool)
        short_ok = np.full(len(df), False, dtype=bool)

        flat_like = ~trending
        long_ok |= flat_like & allow_flat
        short_ok |= flat_like & allow_flat

        if allow_with:
            long_ok |= trending & (dirn > 0)
            short_ok |= trending & (dirn < 0)
        if allow_against:
            long_ok |= trending & (dirn < 0)
            short_ok |= trending & (dirn > 0)

        df.loc[allowed & long_ok & long_trigger, "enter_long"] = 1
        df.loc[allowed & short_ok & short_trigger, "enter_short"] = 1

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Required by your freqtrade version when use_exit_signal=True.
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df

    # ====== Custom exits / stops ======
    @staticmethod
    def _trade_age_hours(trade: Trade, now: datetime) -> float:
        open_dt = trade.open_date_utc
        if open_dt.tzinfo is None:
            open_dt = open_dt.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - open_dt).total_seconds() / 3600.0

    @staticmethod
    def _get_df_trade_width(df: pd.DataFrame | None, trade: Trade) -> float | None:
        """
        Locate channel width near the trade open time using df['dt'].
        """
        if df is None or df.empty:
            return None
        if "dt" not in df.columns:
            # If Freqtrade doesn't pass the preprocessed df, give up.
            return None

        ts = pd.Timestamp(trade.open_date_utc, tz=timezone.utc)
        # nearest by absolute time difference
        idx = (df["dt"] - ts).abs().idxmin()
        w = float(df.loc[idx, "ch_width"]) if "ch_width" in df.columns else np.nan
        if not np.isfinite(w) or w <= 0:
            return None
        return w

    def _profit_target_hit(
        self, trade: Trade, current_rate: float, df: pd.DataFrame | None
    ) -> bool:
        w = self._get_df_trade_width(df, trade)
        if w is None:
            return False
        tp_mult = float(self.proj_takeprofit_mult.value)
        entry = float(trade.open_rate)
        tp_price = (entry + tp_mult * w) if trade.is_long else (entry - tp_mult * w)
        return (current_rate >= tp_price) if trade.is_long else (current_rate <= tp_price)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        df = kwargs.get("dataframe", None)

        # Projection TP
        if self._profit_target_hit(trade, current_rate, df):
            return "proj_tp"

        # Time expiry
        age_h = self._trade_age_hours(trade, current_time)
        min_h = max(3, min(12, int(self.min_hold_hours.value)))
        max_h = max(6, min(36, int(self.max_hold_hours.value)))
        if max_h < min_h:
            max_h = min_h

        if age_h >= max_h:
            return "time_expiry"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        df = kwargs.get("dataframe", None)
        w = self._get_df_trade_width(df, trade)
        if w is None:
            return -0.20

        entry = float(trade.open_rate)
        is_long = trade.is_long

        stop_dist = float(self.initial_stop_mult.value) * w

        move = (current_rate - entry) if is_long else (entry - current_rate)
        reached = int(np.floor(move / w)) if w > 0 else 0
        reached = max(0, reached)

        mode = str(self.exit_mode.value)

        if mode == "step_stop":
            if reached <= 0:
                stop_price = entry - stop_dist if is_long else entry + stop_dist
            else:
                proj_prev = (reached - 1) * w
                stop_price = (entry + proj_prev) if is_long else (entry - proj_prev)

        elif mode == "be_after_l1":
            stop_price = (
                entry if reached >= 1 else (entry - stop_dist if is_long else entry + stop_dist)
            )

        else:  # trail_atr
            trail_dist = max(0.5 * w, float(self.trail_atr_mult.value) * w)
            stop_price = (current_rate - trail_dist) if is_long else (current_rate + trail_dist)

        if is_long:
            sl_frac = (stop_price / current_rate) - 1.0
        else:
            sl_frac = 1.0 - (stop_price / current_rate)

        return float(np.clip(sl_frac, -0.50, -0.001))

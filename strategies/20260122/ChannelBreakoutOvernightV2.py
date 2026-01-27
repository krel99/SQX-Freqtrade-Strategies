# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy


class ChannelBreakoutOvernightV2(IStrategy):
    """
    Hyperopt-safe rewrite (matches backtest next time):

    Core rule for hyperopt parity:
      - populate_indicators() MUST NOT depend on hyperopt parameters.
      - Any computation that uses Int/Decimal/CategoricalParameter values must happen
        inside populate_entry_trend() and populate_exit_trend() (re-evaluated per epoch).

    This version:
      - populate_indicators(): only builds parameter-INDEPENDENT helpers:
          dt, tr, atrp_base, baseline_atrp_7d_base (7d rolling over atrp_base), candle body
      - populate_entry_trend(): computes parameter-dependent:
          channel windows, channel bounds, width filter, overnight regime, vol_ratio, trend_strength,
          entry_allowed, breakout confirmation, and entry signals.
      - custom_exit/custom_stoploss: uses per-trade cached context computed at entry time
        (channel width/mid/high/low and ATR at entry for stop scaling).

    Notes:
      - Works with RangeIndex because it uses dataframe['date'] -> df['dt'].
      - Compatible with 15m timeframe and 1m detail backtesting.
    """

    can_short = True
    timeframe = "5m"
    startup_candle_count = 900

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Defaults (may be overridden by params json)
    minimal_roi = {"0": 0.0}
    stoploss = -0.99

    # ===== Channel parameters (hyperopt) =====
    channel_start_hour = IntParameter(0, 23, default=8, space="buy", optimize=True)
    channel_start_minute = IntParameter(0, 55, default=0, space="buy", optimize=True)

    channel_min_width = DecimalParameter(
        0.0005, 0.02, default=0.0015, decimals=4, space="buy", optimize=True
    )
    channel_max_width = DecimalParameter(
        0.01, 0.15, default=0.06, decimals=4, space="buy", optimize=True
    )

    entry_expiry_minutes = IntParameter(30, 720, default=240, space="buy", optimize=True)

    # ===== Overnight regime (hyperopt) =====
    overnight_lookback_hours = IntParameter(3, 24, default=12, space="buy", optimize=True)
    atr_len = IntParameter(50, 400, default=200, space="buy", optimize=True)

    vol_ratio_max = DecimalParameter(0.6, 3.0, default=1.6, decimals=2, space="buy", optimize=True)
    directionality_threshold = DecimalParameter(
        0.15, 1.5, default=0.55, decimals=2, space="buy", optimize=True
    )

    allow_trade_flat = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    allow_trade_trend_with = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=True
    )
    allow_trade_trend_against = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=True
    )

    # ===== Entry confirmation (hyperopt) =====
    breakout_buffer_frac = DecimalParameter(
        0.0, 0.01, default=0.0005, decimals=4, space="buy", optimize=True
    )
    body_confirm_frac = DecimalParameter(
        0.0, 0.35, default=0.05, decimals=2, space="buy", optimize=True
    )
    confirm_bars = IntParameter(0, 1, default=0, space="buy", optimize=True)

    # ===== Shorts gating (hyperopt) =====
    short_mode_major = CategoricalParameter(
        ["never", "strong_trend", "always"], default="strong_trend", space="buy", optimize=True
    )
    short_mode_alt = CategoricalParameter(
        ["strong_trend", "always", "never"], default="strong_trend", space="buy", optimize=True
    )
    short_trend_threshold = DecimalParameter(
        0.4, 1.5, default=0.8, decimals=2, space="buy", optimize=True
    )

    # ===== Exits / risk (hyperopt) =====
    proj_takeprofit_mult = DecimalParameter(
        0.5, 5.0, default=2.0, decimals=2, space="sell", optimize=True
    )
    initial_stop_mult = DecimalParameter(
        0.2, 2.0, default=1.0, decimals=2, space="sell", optimize=True
    )
    atr_stop_mult = DecimalParameter(0.5, 6.0, default=2.5, decimals=2, space="sell", optimize=True)

    exit_mode = CategoricalParameter(
        ["step_stop", "be_after_l1", "trail_atr"], default="step_stop", space="sell", optimize=True
    )
    trail_atr_mult = DecimalParameter(
        0.5, 6.0, default=3.0, decimals=2, space="sell", optimize=True
    )

    early_fail_minutes = IntParameter(15, 240, default=90, space="sell", optimize=True)
    early_fail_on_mid_cross = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )
    early_fail_on_reenter = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )

    hard_max_hold_hours = IntParameter(6, 36, default=24, space="sell", optimize=True)
    cut_loser_hours = IntParameter(3, 18, default=8, space="sell", optimize=True)

    # ===== Helpers (parameter-independent) =====
    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns:
            raise ValueError("DataFrame has no 'date' column.")
        out = df.copy()
        out["dt"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        return out[out["dt"].notna()]

    @staticmethod
    def _infer_tf_minutes(df: pd.DataFrame) -> int:
        d = df["dt"].diff().dropna()
        if d.empty:
            return 5
        mins = int(round(d.median().total_seconds() / 60.0))
        return max(1, mins)

    @staticmethod
    def _day_start_utc(dt: pd.Series) -> pd.Series:
        return dt.dt.floor("D")

    @staticmethod
    def _true_range_from_ohlc(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        return pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        IMPORTANT: Must not use hyperopt parameters here.
        Build only parameter-independent columns used later.
        """
        df = self._ensure_dt(dataframe)

        # TR is parameter-independent
        df["tr"] = self._true_range_from_ohlc(df)

        # A fixed ATR% baseline (not the hyperopt atr_len) - used only to get a "normal vol" baseline.
        # This is intentionally fixed to avoid per-epoch recomputation.
        base_atr_len = 200
        df["atr_base"] = df["tr"].rolling(base_atr_len, min_periods=base_atr_len).mean()
        df["atrp_base"] = df["atr_base"] / df["close"]

        # 7d baseline computed on atrp_base (still parameter-independent)
        tfm = self._infer_tf_minutes(df)
        day_candles = max(1, int(round((24 * 60) / tfm)))
        win_7d = max(20, 7 * day_candles)
        df["baseline_atrp_7d_base"] = (
            df["atrp_base"].rolling(win_7d, min_periods=max(20, win_7d // 3)).mean()
        )

        # Candle body (parameter-independent)
        df["body"] = (df["close"] - df["open"]).abs()

        return df

    # ===== Parameter-dependent computations (must be in entry/exit trend) =====
    def _compute_channel_for_epoch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes channel-related columns using hyperopt parameters for this epoch.
        """
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

        tmp_high = df["high"].where(df["in_channel"])
        tmp_low = df["low"].where(df["in_channel"])

        ch_high = tmp_high.groupby(df["ch_start"]).transform("max")
        ch_low = tmp_low.groupby(df["ch_start"]).transform("min")

        # fill within each day's group
        df["ch_high"] = ch_high.groupby(df["ch_start"]).transform(lambda s: s.ffill().bfill())
        df["ch_low"] = ch_low.groupby(df["ch_start"]).transform(lambda s: s.ffill().bfill())

        df["ch_width"] = df["ch_high"] - df["ch_low"]
        df["ch_mid"] = (df["ch_high"] + df["ch_low"]) / 2.0
        df["ch_width_frac"] = df["ch_width"] / (df["ch_mid"] + 1e-9)

        expiry = pd.to_timedelta(int(self.entry_expiry_minutes.value), unit="m")
        df["entry_allowed"] = (df["dt"] >= df["ch_end"]) & (df["dt"] < (df["ch_end"] + expiry))

        # body relative to channel
        df["body_frac_of_ch"] = df["body"] / (df["ch_width"] + 1e-9)
        return df

    def _compute_regime_for_epoch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes regime using hyperopt parameters for this epoch.
        Uses ATR% derived from df['tr'] with atr_len (epoch-dependent).
        """
        length = int(self.atr_len.value)
        df["atr"] = df["tr"].rolling(length, min_periods=length).mean()
        df["atrp"] = df["atr"] / df["close"]

        tfm = self._infer_tf_minutes(df)
        lb_hours = int(self.overnight_lookback_hours.value)
        window_candles = max(1, int(round((lb_hours * 60) / tfm)))
        min_need = max(8, min(40, int(0.25 * window_candles)))

        overnight_atrp = np.full(len(df), np.nan, dtype=float)
        trend_strength = np.full(len(df), np.nan, dtype=float)
        trend_dir = np.zeros(len(df), dtype=int)

        unique_days = pd.Index(df["ch_start"].unique()).sort_values()
        for ch_start in unique_days:
            if pd.isna(ch_start):
                continue

            o_start = ch_start - pd.to_timedelta(lb_hours, unit="h")
            mask = (df["dt"] >= o_start) & (df["dt"] < ch_start)
            if int(mask.sum()) < min_need:
                continue

            atrp_vals = df.loc[mask, "atrp"].dropna()
            if atrp_vals.size < max(5, min_need // 2):
                continue

            vol = float(atrp_vals.mean())

            closes = df.loc[mask, "close"].dropna()
            if closes.size < 2:
                continue

            first_close = float(closes.iloc[0])
            last_close = float(closes.iloc[-1])
            if first_close <= 0:
                continue

            logret = float(np.log(last_close / first_close))
            strength = abs(logret) / (vol + 1e-9)

            daymask = df["ch_start"] == ch_start
            idxs = np.where(daymask.values)[0]
            overnight_atrp[idxs] = vol
            trend_strength[idxs] = strength
            trend_dir[idxs] = 1 if logret > 0 else (-1 if logret < 0 else 0)

        df["overnight_atrp"] = overnight_atrp
        df["trend_strength"] = pd.Series(trend_strength).fillna(0.0).values
        df["trend_dir"] = pd.Series(trend_dir).fillna(0).astype(int).values

        # vol_ratio based on parameter-independent 7d baseline; fail-open to 1.0
        baseline = df.get("baseline_atrp_7d_base", pd.Series(np.nan, index=df.index))
        df["vol_ratio"] = (df["overnight_atrp"] / (baseline + 1e-9)).fillna(1.0)

        return df

    def _pair_is_major(self, pair: str) -> bool:
        base = pair.split("/")[0].upper() if pair else ""
        return base in {"BTC", "ETH"}

    def _day_is_tradable(self, df: pd.DataFrame) -> pd.Series:
        width_ok = (df["ch_width_frac"] >= float(self.channel_min_width.value)) & (
            df["ch_width_frac"] <= float(self.channel_max_width.value)
        )
        vr_ok = df["vol_ratio"] <= float(self.vol_ratio_max.value)

        strength = df["trend_strength"]
        trending = strength >= float(self.directionality_threshold.value)
        flat_like = ~trending

        allow_flat = bool(self.allow_trade_flat.value)
        allow_with = bool(self.allow_trade_trend_with.value)
        allow_against = bool(self.allow_trade_trend_against.value)

        regime_ok = (flat_like & allow_flat) | (trending & (allow_with | allow_against))
        return width_ok & vr_ok & regime_ok

    def _confirm_breakout(self, df: pd.DataFrame, is_long: bool) -> pd.Series:
        buf = float(self.breakout_buffer_frac.value)
        body_req = float(self.body_confirm_frac.value)
        k = int(self.confirm_bars.value)

        if is_long:
            outside = df["close"] > (df["ch_high"] * (1.0 + buf))
        else:
            outside = df["close"] < (df["ch_low"] * (1.0 - buf))

        body_ok = df["body_frac_of_ch"] >= body_req

        if k <= 0:
            return outside & body_ok

        outside_k = outside.rolling(k + 1, min_periods=k + 1).apply(
            lambda x: 1.0 if x.all() else 0.0
        )
        return (outside_k == 1.0) & body_ok

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Parameter-dependent computations are done here so hyperopt recomputes per epoch.
        """
        df = dataframe.copy()

        # Ensure dt exists (it should from populate_indicators, but be defensive)
        if "dt" not in df.columns:
            df = self._ensure_dt(df)

        # (1) epoch-dependent channel + regime
        df = self._compute_channel_for_epoch(df)
        df = self._compute_regime_for_epoch(df)

        # (2) tradability filter
        tradable = self._day_is_tradable(df)
        allowed = tradable & df["entry_allowed"].fillna(False)

        # (3) regime direction permissions
        pair = metadata.get("pair", "")
        is_major = self._pair_is_major(pair)

        trending = df["trend_strength"] >= float(self.directionality_threshold.value)
        dirn = df["trend_dir"]

        allow_flat = bool(self.allow_trade_flat.value)
        allow_with = bool(self.allow_trade_trend_with.value)
        allow_against = bool(self.allow_trade_trend_against.value)

        long_ok = np.full(len(df), False, dtype=bool)
        short_ok = np.full(len(df), False, dtype=bool)

        flat_like = ~trending
        if allow_flat:
            long_ok |= flat_like
            short_ok |= flat_like

        if allow_with:
            long_ok |= trending & (dirn > 0)
            short_ok |= trending & (dirn < 0)
        if allow_against:
            long_ok |= trending & (dirn < 0)
            short_ok |= trending & (dirn > 0)

        # (4) short gating
        mode = str(self.short_mode_major.value) if is_major else str(self.short_mode_alt.value)
        strong_down = (dirn < 0) & (df["trend_strength"] >= float(self.short_trend_threshold.value))

        if mode == "never":
            short_ok &= False
        elif mode == "strong_trend":
            short_ok &= strong_down

        # (5) breakout confirmation
        long_trigger = self._confirm_breakout(df, True)
        short_trigger = self._confirm_breakout(df, False)

        df.loc[allowed & long_ok & long_trigger, "enter_long"] = 1
        df.loc[allowed & short_ok & short_trigger, "enter_short"] = 1

        df.loc[allowed & long_ok & long_trigger, "enter_tag"] = "V2_LONG"
        df.loc[allowed & short_ok & short_trigger, "enter_tag"] = "V2_SHORT"

        # Keep epoch-dependent columns for later custom_stoploss / custom_exit context
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Required when use_exit_signal=True; exits are managed by custom_exit/custom_stoploss
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df

    # ===== custom_* helpers =====
    @staticmethod
    def _trade_age_minutes(trade: Trade, now: datetime) -> float:
        open_dt = trade.open_date_utc
        if open_dt.tzinfo is None:
            open_dt = open_dt.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - open_dt).total_seconds() / 60.0

    @staticmethod
    def _trade_age_hours(trade: Trade, now: datetime) -> float:
        return ChannelBreakoutOvernightV2._trade_age_minutes(trade, now) / 60.0

    @staticmethod
    def _nearest_row(df: Optional[pd.DataFrame], trade: Trade) -> Optional[int]:
        if df is None or df.empty or "dt" not in df.columns:
            return None
        ts = pd.Timestamp(trade.open_date_utc, tz=timezone.utc)
        return int((df["dt"] - ts).abs().idxmin())

    def _entry_context(self, df: Optional[pd.DataFrame], trade: Trade) -> dict:
        """
        Pull channel and atr context near trade open from the dataframe (post-entry, with epoch columns present).
        Note: For best parity, use run with --timeframe-detail identical between hyperopt and backtest.
        """
        if df is None or df.empty:
            return {}
        idx = self._nearest_row(df, trade)
        if idx is None or idx not in df.index:
            return {}

        row = df.loc[idx]
        ctx = {}
        for k in ["ch_width", "ch_mid", "atr", "ch_high", "ch_low"]:
            v = row.get(k, np.nan)
            if np.isfinite(v):
                ctx[k] = float(v)
        return ctx

    def _profit_target_hit(self, trade: Trade, current_rate: float, ctx: dict) -> bool:
        w = ctx.get("ch_width", None)
        if not w or w <= 0:
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
        ctx = self._entry_context(df, trade)

        if self._profit_target_hit(trade, current_rate, ctx):
            return "proj_tp"

        age_min = self._trade_age_minutes(trade, current_time)
        if age_min <= float(self.early_fail_minutes.value):
            ch_mid = ctx.get("ch_mid", None)
            ch_high = ctx.get("ch_high", None)
            ch_low = ctx.get("ch_low", None)

            if bool(self.early_fail_on_mid_cross.value) and ch_mid is not None:
                if trade.is_long and current_rate < ch_mid:
                    return "early_fail_mid"
                if (not trade.is_long) and current_rate > ch_mid:
                    return "early_fail_mid"

            if (
                bool(self.early_fail_on_reenter.value)
                and ch_high is not None
                and ch_low is not None
            ):
                if trade.is_long and current_rate <= ch_high:
                    return "early_fail_reenter"
                if (not trade.is_long) and current_rate >= ch_low:
                    return "early_fail_reenter"

        age_h = self._trade_age_hours(trade, current_time)
        if age_h >= float(self.cut_loser_hours.value) and current_profit < 0:
            return "cut_loser_time"

        hard_max = max(6, min(36, int(self.hard_max_hold_hours.value)))
        if age_h >= hard_max:
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
        ctx = self._entry_context(df, trade)

        w = ctx.get("ch_width", None)
        atr = ctx.get("atr", None)
        if not w or w <= 0:
            return -0.20

        entry = float(trade.open_rate)
        is_long = trade.is_long

        stop_by_channel = float(self.initial_stop_mult.value) * w
        stop_by_atr = (
            (float(self.atr_stop_mult.value) * atr)
            if (atr is not None and atr > 0)
            else stop_by_channel
        )
        stop_dist = min(stop_by_channel, stop_by_atr)

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
            trail_by_channel = 0.5 * w
            trail_by_atr = (
                (float(self.trail_atr_mult.value) * atr)
                if (atr is not None and atr > 0)
                else trail_by_channel
            )
            trail_dist = max(trail_by_channel, trail_by_atr)
            stop_price = (current_rate - trail_dist) if is_long else (current_rate + trail_dist)

        if is_long:
            sl_frac = (stop_price / current_rate) - 1.0
        else:
            sl_frac = 1.0 - (stop_price / current_rate)

        return float(np.clip(sl_frac, -0.50, -0.001))

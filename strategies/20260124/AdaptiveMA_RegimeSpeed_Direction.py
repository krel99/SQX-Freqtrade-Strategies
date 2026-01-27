# --- Do not remove these imports ---
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


    class AdaptiveMA_RegimeSpeed_Direction(IStrategy):
    """
    Adaptive MA strategy (hyperopt-safe / epoch-safe):

    Core idea:
      - KAMA => direction filter (trend bias / slope)
      - VIDYA => speed filter (momentum / responsiveness)
      - FRAMA => regime filter (trend vs mean-revert / noise)

    Hyperopt parity rule:
      - populate_indicators() MUST NOT depend on hyperopt parameters.
      - Any computation using Parameter values is computed inside populate_entry_trend()
        and populate_exit_trend() (recomputed per epoch).

    "Adaptive period" without AI:
      - KAMA/VIDYA/FRAMA are already adaptive (their alpha changes per candle).
      - Additionally, this strategy can *optionally* scale each MA's effective alpha by a
        volatility ratio (ATR% / baseline ATR%), producing a dynamic "effective period"
        (faster in high vol, slower in low vol) while remaining deterministic.

    Notes:
      - This is written to be hyperopt-safe, not fastest-possible.
      - Uses iterative recursions (KAMA/VIDYA/FRAMA) so it is deterministic per epoch.
    """

    INTERFACE_VERSION = 3
    load_hyperopt_params = True

    timeframe = "15m"
    can_short = True

    process_only_new_candles = True
    startup_candle_count = 600

    # Default protections/ROI/SL can be overridden by hyperopt spaces below
    minimal_roi = {"0": 0.0}
    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ============================================================
    # Hyperopt spaces (ROI / stop / trailing) - optional
    # ============================================================
    @staticmethod
    def roi_space() -> List[Dimension]:
        return [
            Integer(10, 180, name="roi_t1"),
            Integer(10, 180, name="roi_t2"),
            Integer(10, 180, name="roi_t3"),
            Integer(10, 180, name="roi_t4"),
            SKDecimal(0.001, 0.08, decimals=3, name="roi_p1"),
            SKDecimal(0.001, 0.06, decimals=3, name="roi_p2"),
            SKDecimal(0.001, 0.05, decimals=3, name="roi_p3"),
            SKDecimal(0.001, 0.04, decimals=3, name="roi_p4"),
        ]

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        roi_table: Dict[int, float] = {}
        roi_table[0] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"] + params["roi_p4"]
        roi_table[params["roi_t4"]] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"]
        roi_table[params["roi_t4"] + params["roi_t3"]] = params["roi_p1"] + params["roi_p2"]
        roi_table[params["roi_t4"] + params["roi_t3"] + params["roi_t2"]] = params["roi_p1"]
        roi_table[params["roi_t4"] + params["roi_t3"] + params["roi_t2"] + params["roi_t1"]] = 0.0
        return roi_table

    @staticmethod
    def stoploss_space() -> List[Dimension]:
        return [SKDecimal(-0.50, -0.02, decimals=3, name="stoploss")]

    @staticmethod
    def trailing_space() -> List[Dimension]:
        return [
            Categorical([True, False], name="trailing_stop"),
            SKDecimal(0.001, 0.12, decimals=3, name="trailing_stop_positive"),
            SKDecimal(0.001, 0.08, decimals=3, name="trailing_stop_positive_offset"),
            Categorical([True, False], name="trailing_only_offset_is_reached"),
        ]

    # ============================================================
    # Parameters (aiming 20-40 optimizables)
    # ============================================================

    # --- KAMA (direction) ---
    kama_er_period = IntParameter(5, 40, default=10, space="buy", optimize=True)
    kama_fast = IntParameter(
        2, 10, default=2, space="buy", optimize=True
    )  # "fast period" in KAMA alpha
    kama_slow = IntParameter(
        20, 80, default=30, space="buy", optimize=True
    )  # "slow period" in KAMA alpha
    kama_slope_lookback = IntParameter(2, 30, default=6, space="buy", optimize=True)
    kama_slope_min = DecimalParameter(
        0.0, 0.03, default=0.002, decimals=4, space="buy", optimize=True
    )

    # --- VIDYA (speed) ---
    vidya_cmo_period = IntParameter(5, 40, default=14, space="buy", optimize=True)
    vidya_base_period = IntParameter(
        5, 80, default=20, space="buy", optimize=True
    )  # maps to base alpha
    vidya_speed_min = DecimalParameter(
        0.0, 0.04, default=0.002, decimals=4, space="buy", optimize=True
    )
    vidya_speed_lookback = IntParameter(1, 20, default=3, space="buy", optimize=True)

    # --- FRAMA (regime) ---
    frama_period = IntParameter(10, 80, default=16, space="buy", optimize=True)
    frama_fc = IntParameter(2, 20, default=4, space="buy", optimize=True)  # fast cap (period)
    frama_sc = IntParameter(30, 200, default=100, space="buy", optimize=True)  # slow cap (period)
    frama_regime_lookback = IntParameter(3, 30, default=8, space="buy", optimize=True)
    frama_regime_thresh = DecimalParameter(
        0.0, 0.05, default=0.006, decimals=4, space="buy", optimize=True
    )

    # --- Adaptive "period scaling" via vol ratio ---
    use_vol_adaptation = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=True
    )
    vol_ratio_len = IntParameter(50, 400, default=200, space="buy", optimize=True)
    vol_adapt_min = DecimalParameter(0.5, 1.0, default=0.8, decimals=2, space="buy", optimize=True)
    vol_adapt_max = DecimalParameter(1.0, 3.0, default=1.6, decimals=2, space="buy", optimize=True)
    vol_adapt_strength = DecimalParameter(
        0.0, 2.0, default=0.8, decimals=2, space="buy", optimize=True
    )

    # --- Entry filters / confirmations ---
    rsi_len = IntParameter(7, 30, default=14, space="buy", optimize=True)
    rsi_long_min = IntParameter(40, 65, default=48, space="buy", optimize=True)
    rsi_short_max = IntParameter(35, 60, default=52, space="buy", optimize=True)

    volume_ma_len = IntParameter(10, 80, default=30, space="buy", optimize=True)
    volume_ratio_min = DecimalParameter(
        0.5, 3.0, default=1.0, decimals=2, space="buy", optimize=True
    )

    confirm_bars = IntParameter(0, 2, default=1, space="buy", optimize=True)
    pullback_frac = DecimalParameter(
        0.0, 0.02, default=0.002, decimals=4, space="buy", optimize=True
    )

    # --- Exits / risk management ---
    exit_mode = CategoricalParameter(
        ["kama_flip", "vidya_slowdown", "frama_regime_flip", "hybrid"],
        default="hybrid",
        space="sell",
        optimize=True,
    )
    tp_atr_mult = DecimalParameter(0.5, 6.0, default=2.2, decimals=2, space="sell", optimize=True)
    sl_atr_mult = DecimalParameter(0.5, 6.0, default=2.6, decimals=2, space="sell", optimize=True)
    be_after_atr = DecimalParameter(0.0, 3.0, default=1.0, decimals=2, space="sell", optimize=True)
    time_stop_minutes = IntParameter(60, 24 * 60, default=8 * 60, space="sell", optimize=True)
    cut_loser_minutes = IntParameter(60, 18 * 60, default=6 * 60, space="sell", optimize=True)

    # --- Enable/disable blocks (keeps hyperopt flexible) ---
    use_rsi_filter = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_volume_filter = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=True
    )
    use_pullback_filter = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=True
    )

    # ============================================================
    # Protections (same pattern as your momentum strategy)
    # ============================================================
    protection_stoploss_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_stoploss_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_stoploss_lookback_period = IntParameter(10, 1440, default=120, space="protection")
    protection_stoploss_stop_duration = IntParameter(10, 360, default=60, space="protection")

    protection_cooldown_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_cooldown_period = IntParameter(1, 30, default=6, space="protection")

    protection_lowprofit_enabled = CategoricalParameter(
        [True, False], default=False, space="protection"
    )
    protection_lowprofit_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_lowprofit_lookback_period = IntParameter(10, 1440, default=360, space="protection")
    protection_lowprofit_stop_duration = IntParameter(10, 360, default=60, space="protection")
    protection_lowprofit_required_profit = DecimalParameter(
        -0.05, 0.05, default=0.0, decimals=3, space="protection"
    )

    protection_maxdrawdown_enabled = CategoricalParameter(
        [True, False], default=False, space="protection"
    )
    protection_maxdrawdown_trade_limit = IntParameter(3, 20, default=8, space="protection")
    protection_maxdrawdown_lookback_period = IntParameter(10, 1440, default=240, space="protection")
    protection_maxdrawdown_stop_duration = IntParameter(10, 360, default=90, space="protection")
    protection_maxdrawdown_allowed_drawdown = DecimalParameter(
        0.01, 0.30, default=0.10, decimals=2, space="protection"
    )

    # ============================================================
    # Utilities
    # ============================================================
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _p(self, param):
        """Get parameter value for hyperopt / non-hyperopt runs."""
        return param.value if hasattr(param, "value") else param

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

    # ============================================================
    # Adaptive MAs (deterministic, per-epoch)
    # ============================================================
    @staticmethod
    def _kama(close: pd.Series, er_period: int, fast: int, slow: int) -> pd.Series:
        """
        Kaufman Adaptive Moving Average.
        fast/slow are expressed as "periods" in standard KAMA formula.
        """
        c = close.values.astype(float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n == 0:
            return pd.Series(out, index=close.index)

        # ER
        change = np.abs(pd.Series(c).diff(er_period).values)
        volatility = pd.Series(np.abs(pd.Series(c).diff()).values).rolling(er_period).sum().values
        er = change / (volatility + 1e-12)

        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Seed with first non-nan
        first = np.where(~np.isnan(sc))[0]
        if first.size == 0:
            return pd.Series(out, index=close.index)
        i0 = int(first[0])
        out[i0] = c[i0]

        for i in range(i0 + 1, n):
            if np.isnan(sc[i]) or np.isnan(out[i - 1]):
                out[i] = out[i - 1] if not np.isnan(out[i - 1]) else c[i]
                continue
            out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])

        return pd.Series(out, index=close.index)

    @staticmethod
    def _vidya(close: pd.Series, cmo_period: int, base_period: int) -> pd.Series:
        """
        VIDYA using absolute CMO as adaptive alpha multiplier.
        alpha = (2/(base_period+1)) * abs(CMO)/100
        """
        c = close.values.astype(float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n == 0:
            return pd.Series(out, index=close.index)

        # CMO
        diff = pd.Series(c).diff()
        up = diff.clip(lower=0.0).rolling(cmo_period).sum()
        down = (-diff.clip(upper=0.0)).rolling(cmo_period).sum()
        cmo = 100.0 * (up - down) / (up + down + 1e-12)
        cmo_abs = np.abs(cmo.values) / 100.0

        base_alpha = 2.0 / (base_period + 1.0)
        alpha = base_alpha * cmo_abs

        first = np.where(~np.isnan(alpha))[0]
        if first.size == 0:
            return pd.Series(out, index=close.index)
        i0 = int(first[0])
        out[i0] = c[i0]
        for i in range(i0 + 1, n):
            a = alpha[i]
            if np.isnan(a) or np.isnan(out[i - 1]):
                out[i] = out[i - 1] if not np.isnan(out[i - 1]) else c[i]
                continue
            out[i] = out[i - 1] + a * (c[i] - out[i - 1])

        return pd.Series(out, index=close.index)

    @staticmethod
    def _frama(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int, fc: int, sc: int
    ) -> pd.Series:
        """
        FRAMA (Ehlers). Uses fractal dimension to adapt alpha.

        alpha_raw = exp(-4.6*(D-1))
        then clipped between alpha_fast and alpha_slow where:
          alpha_fast = 2/(fc+1), alpha_slow = 2/(sc+1)

        Requires period >= 2 and even-ish behavior; we'll handle odd by splitting floor/ceil.
        """
        h = high.values.astype(float)
        l = low.values.astype(float)
        c = close.values.astype(float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n == 0 or period < 2:
            return pd.Series(out, index=close.index)

        alpha_fast = 2.0 / (fc + 1.0)
        alpha_slow = 2.0 / (sc + 1.0)

        half = max(1, period // 2)

        for i in range(period - 1, n):
            # Window [i-period+1 .. i]
            i0 = i - period + 1
            mid = i0 + half

            h1 = np.nanmax(h[i0:mid])
            l1 = np.nanmin(l[i0:mid])
            n1 = (h1 - l1) / max(1, half)

            h2 = np.nanmax(h[mid : i + 1])
            l2 = np.nanmin(l[mid : i + 1])
            n2 = (h2 - l2) / max(1, period - half)

            h3 = np.nanmax(h[i0 : i + 1])
            l3 = np.nanmin(l[i0 : i + 1])
            n3 = (h3 - l3) / max(1, period)

            if n1 <= 0 or n2 <= 0 or n3 <= 0:
                d = 1.0
            else:
                d = (np.log(n1 + n2 + 1e-12) - np.log(n3 + 1e-12)) / np.log(2.0)

            alpha = float(np.exp(-4.6 * (d - 1.0)))
            alpha = float(np.clip(alpha, alpha_slow, alpha_fast))

            if i == period - 1:
                out[i] = c[i]
            else:
                out[i] = out[i - 1] + alpha * (c[i] - out[i - 1])

        return pd.Series(out, index=close.index)

    @staticmethod
    def _apply_vol_adaptation_alpha(
        alpha: np.ndarray, vol_ratio: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Modify alpha by vol_ratio in a smooth way:
          alpha' = clip(alpha * (vol_ratio ** strength), 0, 1)
        """
        vr = np.where(np.isfinite(vol_ratio), vol_ratio, 1.0)
        a2 = alpha * np.power(vr, strength)
        return np.clip(a2, 0.0, 1.0)

    # ============================================================
    # populate_indicators (parameter-INDEPENDENT only)
    # ============================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # Parameter-independent helpers
        df["tr"] = self._true_range_from_ohlc(df)

        # A fixed baseline ATR% (not hyperopt) for stable vol baseline
        base_atr_len = 200
        df["atr_base"] = df["tr"].rolling(base_atr_len, min_periods=base_atr_len).mean()
        df["atrp_base"] = df["atr_base"] / (df["close"] + 1e-12)

        # 7d-ish rolling baseline for atrp_base (timeframe-agnostic approximation)
        # 15m => ~96 candles/day. We'll use 7*96=672 with min periods.
        win_7d = 672
        df["atrp_baseline_7d"] = df["atrp_base"].rolling(win_7d, min_periods=win_7d // 3).mean()

        # A simple RSI (parameter-independent version not needed; but it's fine to keep minimal here)
        # NOTE: We keep RSI out of indicators to respect "no param usage" rule. We'll compute RSI in entry.
        return df

    # ============================================================
    # Epoch-dependent computations (MAs + filters)
    # ============================================================
    def _compute_epoch_columns(self, df: DataFrame) -> DataFrame:
        """
        Compute all parameter-dependent signals for this epoch.
        Returns a df with the necessary columns attached (no caching across epochs).
        """
        out = df.copy()

        # --- params ---
        er_period = int(self._p(self.kama_er_period))
        fast = int(self._p(self.kama_fast))
        slow = int(self._p(self.kama_slow))
        kama_lb = int(self._p(self.kama_slope_lookback))

        cmo_period = int(self._p(self.vidya_cmo_period))
        vid_base = int(self._p(self.vidya_base_period))
        vid_speed_lb = int(self._p(self.vidya_speed_lookback))

        fr_p = int(self._p(self.frama_period))
        fr_fc = int(self._p(self.frama_fc))
        fr_sc = int(self._p(self.frama_sc))
        fr_lb = int(self._p(self.frama_regime_lookback))

        # --- volatility ratio (epoch-dependent length) ---
        vr_len = int(self._p(self.vol_ratio_len))
        atr = out["tr"].rolling(vr_len, min_periods=vr_len).mean()
        atrp = atr / (out["close"] + 1e-12)
        baseline = out["atrp_baseline_7d"]
        vol_ratio = (atrp / (baseline + 1e-12)).clip(lower=0.1, upper=10.0).fillna(1.0)
        out["vol_ratio_epoch"] = vol_ratio

        use_vol = bool(self._p(self.use_vol_adaptation))
        vmin = float(self._p(self.vol_adapt_min))
        vmax = float(self._p(self.vol_adapt_max))
        vstr = float(self._p(self.vol_adapt_strength))

        # Normalize vol_ratio into [vmin, vmax] clamp (prevents extreme alpha jumps)
        vr_clamped = vol_ratio.clip(lower=vmin, upper=vmax).values.astype(float)

        # --- KAMA ---
        kama = self._kama(out["close"], er_period=er_period, fast=fast, slow=slow)
        out["kama"] = kama

        # KAMA slope as direction proxy (scaled by price)
        kama_slope = (kama - kama.shift(kama_lb)) / (out["close"] + 1e-12)
        out["kama_slope"] = kama_slope

        # --- VIDYA ---
        vidya = self._vidya(out["close"], cmo_period=cmo_period, base_period=vid_base)
        out["vidya"] = vidya

        # VIDYA "speed" proxy: absolute short-term delta / price
        vid_speed = (vidya - vidya.shift(vid_speed_lb)).abs() / (out["close"] + 1e-12)
        out["vidya_speed"] = vid_speed

        # --- FRAMA ---
        frama = self._frama(out["high"], out["low"], out["close"], period=fr_p, fc=fr_fc, sc=fr_sc)
        out["frama"] = frama

        # Regime proxy: FRAMA deviation / price (bigger deviation => trending/expanding)
        fr_regime = (out["close"] - frama).abs() / (out["close"] + 1e-12)
        out["frama_regime"] = fr_regime.rolling(fr_lb, min_periods=max(2, fr_lb // 2)).mean()

        # --- Optional: volatility-adapt the "speed" and "direction" proxies ---
        # (We DON'T retroactively change KAMA/VIDYA/FRAMA values here; they already adapt via alpha.
        #  This adaptation scales thresholds effectively by rescaling the signal magnitudes.)
        if use_vol and vstr > 0:
            scale = np.power(vr_clamped, vstr)
            out["kama_slope_adj"] = out["kama_slope"] * scale
            out["vidya_speed_adj"] = out["vidya_speed"] * scale
            out["frama_regime_adj"] = out["frama_regime"] * scale
        else:
            out["kama_slope_adj"] = out["kama_slope"]
            out["vidya_speed_adj"] = out["vidya_speed"]
            out["frama_regime_adj"] = out["frama_regime"]

        # --- RSI / volume filters (epoch-dependent lengths/thresholds) ---
        rsi_len = int(self._p(self.rsi_len))
        out["rsi"] = ta.RSI(out, timeperiod=rsi_len)

        vlen = int(self._p(self.volume_ma_len))
        out["vol_ma"] = out["volume"].rolling(vlen, min_periods=max(5, vlen // 3)).mean()
        out["vol_ratio"] = (
            (out["volume"] / (out["vol_ma"] + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        # Confirmation helpers
        out["kama_up"] = out["kama"] > out["kama"].shift(1)
        out["kama_down"] = out["kama"] < out["kama"].shift(1)
        out["price_above_kama"] = out["close"] > out["kama"]
        out["price_below_kama"] = out["close"] < out["kama"]

        return out

    def _confirm(self, s: pd.Series, k: int) -> pd.Series:
        """Require s to be true for k+1 bars (k=0 means immediate)."""
        if k <= 0:
            return s.fillna(False)
        v = s.rolling(k + 1, min_periods=k + 1).apply(lambda x: 1.0 if np.all(x) else 0.0)
        return (v == 1.0).fillna(False)

    # ============================================================
    # Entry / Exit trends (epoch-safe)
    # ============================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = self._compute_epoch_columns(dataframe)

        conditions_long = []
        conditions_short = []

        # --- thresholds ---
        kama_slope_min = float(self._p(self.kama_slope_min))
        vid_speed_min = float(self._p(self.vidya_speed_min))
        fr_thresh = float(self._p(self.frama_regime_thresh))
        k = int(self._p(self.confirm_bars))

        # Direction (KAMA) + Speed (VIDYA) + Regime (FRAMA)
        dir_long = df["kama_slope_adj"] > kama_slope_min
        dir_short = df["kama_slope_adj"] < -kama_slope_min

        speed_ok = df["vidya_speed_adj"] > vid_speed_min
        regime_ok = df["frama_regime_adj"] > fr_thresh

        # Base triggers: (a) price on correct side of KAMA (bias), (b) KAMA turning same direction
        base_long = df["price_above_kama"] & df["kama_up"]
        base_short = df["price_below_kama"] & df["kama_down"]

        # Optional pullback filter: require price to be within pullback_frac of KAMA (avoid chasing)
        if bool(self._p(self.use_pullback_filter)):
            pb = float(self._p(self.pullback_frac))
            near_kama = ((df["close"] - df["kama"]).abs() / (df["close"] + 1e-12)) <= pb
            base_long = base_long & near_kama
            base_short = base_short & near_kama

        # RSI filter
        if bool(self._p(self.use_rsi_filter)):
            rsi_long_min = int(self._p(self.rsi_long_min))
            rsi_short_max = int(self._p(self.rsi_short_max))
            conditions_long.append(df["rsi"] >= rsi_long_min)
            conditions_short.append(df["rsi"] <= rsi_short_max)

        # Volume filter
        if bool(self._p(self.use_volume_filter)):
            vrmin = float(self._p(self.volume_ratio_min))
            conditions_long.append(df["vol_ratio"] >= vrmin)
            conditions_short.append(df["vol_ratio"] >= vrmin)

        # Combine core
        conditions_long.append(self._confirm(base_long & dir_long & speed_ok & regime_ok, k))
        conditions_short.append(self._confirm(base_short & dir_short & speed_ok & regime_ok, k))

        if conditions_long:
            df.loc[reduce(lambda x, y: x & y, conditions_long), "enter_long"] = 1
            df.loc[reduce(lambda x, y: x & y, conditions_long), "enter_tag"] = (
                "KAMA_DIR+VIDYA_SPD+FRAMA_REG"
            )

        if conditions_short:
            df.loc[reduce(lambda x, y: x & y, conditions_short), "enter_short"] = 1
            df.loc[reduce(lambda x, y: x & y, conditions_short), "enter_tag"] = (
                "SHORT_KAMA_DIR+VIDYA_SPD+FRAMA_REG"
            )

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exits are handled by custom_exit/custom_stoploss for better management.
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df

    # ============================================================
    # Trade context helpers (for custom_exit/custom_stoploss)
    # ============================================================
    @staticmethod
    def _nearest_row(df: Optional[pd.DataFrame], trade: Trade) -> Optional[int]:
        if df is None or df.empty or "date" not in df.columns:
            return None
        ts = pd.Timestamp(trade.open_date_utc, tz="UTC")
        dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
        if dt.isna().all():
            return None
        return int((dt - ts).abs().idxmin())

    def _entry_ctx(self, df: Optional[pd.DataFrame], trade: Trade) -> dict:
        if df is None or df.empty:
            return {}
        # recompute epoch columns so context matches epoch parameters (important for hyperopt parity)
        df2 = self._compute_epoch_columns(df)
        idx = self._nearest_row(df2, trade)
        if idx is None or idx not in df2.index:
            return {}

        row = df2.loc[idx]
        ctx = {}
        for k in ["vol_ratio_epoch", "kama", "vidya", "frama", "frama_regime_adj"]:
            v = row.get(k, np.nan)
            if np.isfinite(v):
                ctx[k] = float(v)

        # entry ATR% for scaling stops/targets
        vr_len = int(self._p(self.vol_ratio_len))
        atr = df2["tr"].rolling(vr_len, min_periods=vr_len).mean()
        atr_val = float(atr.loc[idx]) if np.isfinite(atr.loc[idx]) else np.nan
        if np.isfinite(atr_val) and atr_val > 0:
            ctx["atr"] = atr_val
        return ctx

    @staticmethod
    def _trade_age_minutes(trade: Trade, now) -> float:
        open_dt = trade.open_date_utc
        if open_dt.tzinfo is None:
            open_dt = open_dt.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
        if now.tzinfo is None:
            now = now.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
        return (now - open_dt).total_seconds() / 60.0

    # ============================================================
    # custom_exit / custom_stoploss (management)
    # ============================================================
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        df = kwargs.get("dataframe", None)
        ctx = self._entry_ctx(df, trade)
        atr = ctx.get("atr", None)

        # Time-based exits
        age_min = self._trade_age_minutes(trade, current_time)
        if age_min >= float(self._p(self.time_stop_minutes)):
            return "time_stop"

        if age_min >= float(self._p(self.cut_loser_minutes)) and current_profit < 0:
            return "cut_loser_time"

        # ATR-based take profit
        if atr is not None and atr > 0:
            tp_mult = float(self._p(self.tp_atr_mult))
            entry = float(trade.open_rate)
            tp_price = (entry + tp_mult * atr) if trade.is_long else (entry - tp_mult * atr)
            if (current_rate >= tp_price) if trade.is_long else (current_rate <= tp_price):
                return "tp_atr"

        # Indicator-driven exits (epoch-safe: recompute now)
        if df is None or df.empty:
            return None

        dfx = self._compute_epoch_columns(df)
        idx = self._nearest_row(dfx, trade)
        if idx is None or idx not in dfx.index:
            return None

        mode = str(self._p(self.exit_mode))
        row = dfx.loc[idx]

        kama = float(row.get("kama", np.nan))
        vidya = float(row.get("vidya", np.nan))
        frama = float(row.get("frama", np.nan))
        regime = float(row.get("frama_regime_adj", np.nan))

        # Basic flips:
        kama_flip = (current_rate < kama) if trade.is_long else (current_rate > kama)
        frama_flip = regime < float(self._p(self.frama_regime_thresh))  # regime cooled off
        vid_slow = float(row.get("vidya_speed_adj", np.nan)) < float(self._p(self.vidya_speed_min))

        if mode == "kama_flip" and kama_flip:
            return "kama_flip"
        if mode == "frama_regime_flip" and frama_flip:
            return "frama_regime_flip"
        if mode == "vidya_slowdown" and vid_slow:
            return "vidya_slowdown"

        # Hybrid: prioritize take-profit/time; then exit on meaningful deterioration
        if mode == "hybrid":
            if kama_flip and frama_flip:
                return "hybrid_flip"
            if current_profit > 0 and vid_slow and kama_flip:
                return "hybrid_slowdown"
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        df = kwargs.get("dataframe", None)
        ctx = self._entry_ctx(df, trade)
        atr = ctx.get("atr", None)

        # Fallback if ATR isn't available
        if atr is None or atr <= 0:
            return -0.20

        entry = float(trade.open_rate)
        is_long = trade.is_long

        # Base SL distance
        sl_mult = float(self._p(self.sl_atr_mult))
        sl_dist = sl_mult * atr

        # Break-even logic after X * ATR move in favor
        be_after = float(self._p(self.be_after_atr))
        move = (current_rate - entry) if is_long else (entry - current_rate)
        if be_after > 0 and move >= be_after * atr:
            stop_price = entry  # break-even
        else:
            stop_price = (entry - sl_dist) if is_long else (entry + sl_dist)

        # Convert to stoploss fraction relative to current_rate (Freqtrade expects negative values)
        if is_long:
            sl_frac = (stop_price / current_rate) - 1.0
        else:
            sl_frac = 1.0 - (stop_price / current_rate)

        return float(np.clip(sl_frac, -0.50, -0.001))

    # ============================================================
    # Protections
    # ============================================================
    @property
    def protections(self):
        prot = []

        if bool(self._p(self.protection_stoploss_enabled)):
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": int(
                        self._p(self.protection_stoploss_lookback_period)
                    ),
                    "trade_limit": int(self._p(self.protection_stoploss_trade_limit)),
                    "stop_duration_candles": int(self._p(self.protection_stoploss_stop_duration)),
                    "only_per_pair": False,
                }
            )

        if bool(self._p(self.protection_cooldown_enabled)):
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": int(self._p(self.protection_cooldown_period)),
                }
            )

        if bool(self._p(self.protection_lowprofit_enabled)):
            prot.append(
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": int(
                        self._p(self.protection_lowprofit_lookback_period)
                    ),
                    "trade_limit": int(self._p(self.protection_lowprofit_trade_limit)),
                    "stop_duration_candles": int(self._p(self.protection_lowprofit_stop_duration)),
                    "required_profit": float(self._p(self.protection_lowprofit_required_profit)),
                    "only_per_pair": True,
                }
            )

        if bool(self._p(self.protection_maxdrawdown_enabled)):
            prot.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": int(
                        self._p(self.protection_maxdrawdown_lookback_period)
                    ),
                    "trade_limit": int(self._p(self.protection_maxdrawdown_trade_limit)),
                    "stop_duration_candles": int(
                        self._p(self.protection_maxdrawdown_stop_duration)
                    ),
                    "max_allowed_drawdown": float(
                        self._p(self.protection_maxdrawdown_allowed_drawdown)
                    ),
                }
            )

        return prot

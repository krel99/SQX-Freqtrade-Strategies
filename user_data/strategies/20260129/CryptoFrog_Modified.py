from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


# ======================================================================================
# Indicator helpers (no external deps; deterministic; candle-based)
# ======================================================================================


def typical_price(df: DataFrame) -> Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def stoch_rsi(
    df: DataFrame, rsi_len: int = 14, stoch_len: int = 14, k: int = 3, d: int = 3
) -> Tuple[Series, Series]:
    rsi = Series(ta.RSI(df, timeperiod=rsi_len), index=df.index)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    denom = (hi - lo).replace(0, np.nan)
    stoch = (rsi - lo) / denom
    k_line = (stoch.rolling(k).mean() * 100.0).fillna(0.0)
    d_line = k_line.rolling(d).mean().fillna(0.0)
    return k_line, d_line


def bb_width_expansion_signal(
    df: DataFrame, window: int = 20, stds: float = 1.0, lookback: int = 4, mult: float = 1.1
) -> Series:
    """
    Signal = 1 when current BB width exceeds (max BB width of previous lookback-1 bars)*mult.
    Vectorized equivalent of the original rolling-apply approach.
    """
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=window, stds=stds)
    bbw = (bb["upper"] - bb["lower"]) / bb["mid"]
    prior_max = bbw.shift(1).rolling(lookback - 1).max()
    return (bbw > (prior_max * mult)).astype(int).fillna(0)


def squeeze_on_ttm(
    df: DataFrame, length: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5
) -> Series:
    """
    TTM Squeeze-style: True when Bollinger Bands are inside the Keltner Channel.
    """
    tp = typical_price(df)

    bb_mid = tp.rolling(length).mean()
    bb_std = tp.rolling(length).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    atr = ta.ATR(df, timeperiod=length)
    kc_mid = tp.rolling(length).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    return ((bb_upper < kc_upper) & (bb_lower > kc_lower)).fillna(False)


def vfi_katsanos(
    df: DataFrame,
    period: int = 130,
    coef: float = 0.2,
    vcoef: float = 2.5,
    smooth: int = 3,
    vol_stdev_period: int = 30,
) -> Series:
    """
    Katsanos VFI approximation:
      - positive => bullish volume flow
      - negative => bearish volume flow
    """
    tp = typical_price(df)
    inter = np.log(tp.replace(0, np.nan)) - np.log(tp.shift(1).replace(0, np.nan))
    vinter = inter.rolling(vol_stdev_period).std()
    cutoff = coef * vinter * df["close"]

    vave = df["volume"].rolling(period).mean().shift(1)
    vmax = vave * vcoef
    vc = np.minimum(df["volume"], vmax)

    mf = tp - tp.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0))

    vfi_raw = Series(vcp, index=df.index).rolling(period).sum() / vave
    vfi_raw = vfi_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if smooth > 0:
        return Series(ta.EMA(vfi_raw, timeperiod=smooth), index=df.index).fillna(0.0)
    return vfi_raw


def rmi(df: DataFrame, length: int = 24, mom: int = 5) -> Series:
    diff = df["close"] - df["close"].shift(mom)
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    ema_up = ta.EMA(up, timeperiod=length)
    ema_down = ta.EMA(down, timeperiod=length)
    rmi_val = np.where(
        (ema_down == 0) | np.isnan(ema_down),
        0.0,
        100.0 - (100.0 / (1.0 + (ema_up / ema_down))),
    )
    return Series(rmi_val, index=df.index).fillna(0.0)


def ssl_channels_atr(df: DataFrame, length: int = 21, atr_len: int = 14) -> Tuple[Series, Series]:
    atr = ta.ATR(df, timeperiod=atr_len)
    sma_high = df["high"].rolling(length).mean() + atr
    sma_low = df["low"].rolling(length).mean() - atr

    hlv = np.where(df["close"] > sma_high, 1, np.where(df["close"] < sma_low, -1, np.nan))
    hlv = Series(hlv, index=df.index).ffill()

    ssl_down = np.where(hlv < 0, sma_high, sma_low)
    ssl_up = np.where(hlv < 0, sma_low, sma_high)
    return Series(ssl_down, index=df.index), Series(ssl_up, index=df.index)


def sroc(df: DataFrame, emalen: int = 13, smooth: int = 21) -> Series:
    ema_close = ta.EMA(df["close"], timeperiod=emalen)
    return Series(ta.ROC(ema_close, timeperiod=smooth), index=df.index).fillna(0.0)


# ======================================================================================
# Strategy
# ======================================================================================


class CryptoFrog_Modified(IStrategy):
    """
    Modified CryptoFrog:
    - Trailing stoploss ONLY for exits.
    - Weekend trade disable parameter.
    """

    INTERFACE_VERSION = 3
    can_short = True

    timeframe = "15m"
    informative_timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count = 300

    # -----------------------------
    # Trailing stoploss ONLY
    # -----------------------------
    minimal_roi = {"0": 100}
    stoploss = -0.085
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # -----------------------------
    # GLOBAL params
    # -----------------------------
    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)

    # -----------------------------
    # BUY space params (required for hyperopt buy space)
    # -----------------------------
    buy_mfi_low = IntParameter(10, 35, default=20, space="buy", optimize=True)
    buy_di_minus = IntParameter(20, 45, default=30, space="buy", optimize=True)
    buy_srsi_max = IntParameter(10, 40, default=30, space="buy", optimize=True)
    buy_fastd_max = IntParameter(10, 35, default=23, space="buy", optimize=True)
    buy_sqz_fallback_fastd = IntParameter(10, 30, default=20, space="buy", optimize=True)
    buy_require_bbw_expansion = BooleanParameter(default=True, space="buy", optimize=True)

    # -----------------------------
    # SELL space params
    # -----------------------------
    sell_mfi_high = IntParameter(65, 95, default=80, space="sell", optimize=True)
    sell_di_plus = IntParameter(20, 45, default=30, space="sell", optimize=True)
    sell_require_bbw_expansion = BooleanParameter(default=True, space="sell", optimize=True)

    # Dynamic ROI knobs (sell space)
    use_dynamic_roi = True
    droi_trend_type = CategoricalParameter(
        ["rmi", "ssl", "candle", "any"], default="any", space="sell", optimize=True
    )
    droi_pullback = BooleanParameter(default=True, space="sell", optimize=True)
    droi_pullback_amount = DecimalParameter(0.005, 0.03, default=0.005, space="sell", optimize=True)
    droi_pullback_respect_table = BooleanParameter(default=False, space="sell", optimize=True)

    # Custom trailing stoploss knobs (sell space)
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    trailing_atr_period = IntParameter(7, 30, default=14, space="sell", optimize=True)

    # Pair -> column -> (date-indexed DataFrame)
    _series_cache: Dict[str, Dict[str, DataFrame]] = {}

    def informative_pairs(self):
        if not self.dp:
            return []
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def _roi_from_table(self, trade_dur_min: int) -> float:
        """
        Stable ROI lookup regardless of internal Freqtrade signature changes.
        """
        steps = sorted((int(k), float(v)) for k, v in self.minimal_roi.items())
        roi = steps[0][1]
        for minute, val in steps:
            if trade_dur_min >= minute:
                roi = val
            else:
                break
        return roi

    def _asof(self, pair: str, col: str, t: datetime, fallback=None):
        frame = self._series_cache.get(pair, {}).get(col)
        if frame is None or frame.empty:
            return fallback
        try:
            sliced = frame.loc[:t]
            if sliced.empty:
                return fallback
            return sliced.iloc[-1, 0]
        except Exception:
            return fallback

    def _cache_series(self, pair: str, df: DataFrame) -> None:
        if "date" not in df.columns:
            return
        if pair not in self._series_cache:
            self._series_cache[pair] = {}
        for col in (
            "sroc",
            "ssl-dir",
            "rmi-up-trend",
            "rmi-down-trend",
            "candle-up-trend",
            "candle-down-trend",
        ):
            if col in df.columns:
                self._series_cache[pair][col] = df[["date", col]].copy().set_index("date")

    # -----------------------------
    # Informative indicators (1h)
    # -----------------------------
    def _add_smoothed_heiken_ashi(self, df: DataFrame, ema_smoothing: int = 4) -> DataFrame:
        out = df.copy()
        ha_close = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0

        ha_open = np.zeros(len(out), dtype=float)
        if len(out) > 0:
            ha_open[0] = (out["open"].iloc[0] + out["close"].iloc[0]) / 2.0
            for i in range(1, len(out)):
                ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

        ha_open = Series(ha_open, index=out.index)
        ha_high = DataFrame({"a": ha_open, "b": ha_close, "c": out["high"]}, index=out.index).max(
            axis=1
        )
        ha_low = DataFrame({"a": ha_open, "b": ha_close, "c": out["low"]}, index=out.index).min(
            axis=1
        )

        out["Smooth_HA_O"] = ta.EMA(ha_open, timeperiod=ema_smoothing)
        out["Smooth_HA_C"] = ta.EMA(ha_close, timeperiod=ema_smoothing)
        out["Smooth_HA_H"] = ta.EMA(ha_high, timeperiod=ema_smoothing)
        out["Smooth_HA_L"] = ta.EMA(ha_low, timeperiod=ema_smoothing)
        return out

    def _hansen_bias(self, df: DataFrame, sma_period: int = 6) -> Tuple[Series, Series]:
        hhclose = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        hhopen = (df["open"].shift(2) + df["close"].shift(2)) / 2.0
        emac = ta.SMA(hhclose, timeperiod=sma_period)
        emao = ta.SMA(hhopen, timeperiod=sma_period)
        return emac, emao

    def _compute_informative(self, df: DataFrame) -> DataFrame:
        out = df.copy()

        stoch_fast = ta.STOCHF(out)
        out["fastd"] = stoch_fast["fastd"]
        out["fastk"] = stoch_fast["fastk"]
        out["srsi_k"], out["srsi_d"] = stoch_rsi(out)

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(out), window=20, stds=1)
        out["bb_lowerband"] = bb["lower"]
        out["bb_middleband"] = bb["mid"]
        out["bb_upperband"] = bb["upper"]
        out["bbw_expansion"] = bb_width_expansion_signal(
            out, window=20, stds=1.0, lookback=4, mult=1.1
        )

        out["sar"] = ta.SAR(out)

        out = self._add_smoothed_heiken_ashi(out, ema_smoothing=4)
        out["emac"], out["emao"] = self._hansen_bias(out, sma_period=6)

        out["mfi"] = ta.MFI(out, timeperiod=14)
        out["sqzmi"] = squeeze_on_ttm(out, length=20, bb_mult=2.0, kc_mult=1.5)
        out["vfi"] = vfi_katsanos(out, period=130, coef=0.2, vcoef=2.5, smooth=3)

        out["dmi_plus"] = ta.PLUS_DI(out, timeperiod=14)
        out["dmi_minus"] = ta.MINUS_DI(out, timeperiod=14)
        out["adx"] = ta.ADX(out, timeperiod=14)

        out["rmi"] = rmi(out, length=24, mom=5)
        ssl_down, ssl_up = ssl_channels_atr(out, length=21, atr_len=14)
        out["ssl_down"] = ssl_down
        out["ssl_up"] = ssl_up
        out["ssl-dir"] = np.where(out["ssl_up"] > out["ssl_down"], "up", "down")

        out["sroc"] = sroc(out, emalen=13, smooth=21)

        rmi_up = (out["rmi"] >= out["rmi"].shift(1)).astype(int)
        out["rmi-up-trend"] = (rmi_up.rolling(5).sum() >= 3).astype(int)
        out["rmi-down-trend"] = (rmi_up.rolling(5).sum() <= 2).astype(int)

        candle_up = (out["close"] >= out["close"].shift(1)).astype(int)
        out["candle-up-trend"] = (candle_up.rolling(5).sum() >= 3).astype(int)
        out["candle-down-trend"] = (candle_up.rolling(5).sum() <= 2).astype(int)

        return out

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config["runmode"].value in ("backtest", "hyperopt"):
            assert timeframe_to_minutes(self.timeframe) <= 30, (
                "Backtest this timeframe in <= 30m timeframe."
            )

        if not self.dp:
            return dataframe

        pair = metadata["pair"]
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
        informative = self._compute_informative(informative.copy())

        merged = merge_informative_pair(
            dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True
        )

        # Keep OHLCV/date and emac/emao with suffix; strip suffix for other informative cols
        keep_suffix = {
            f"{c}_{self.informative_timeframe}"
            for c in ("date", "open", "high", "low", "close", "volume", "emac", "emao")
        }
        suffix = f"_{self.informative_timeframe}"

        def rename(col: str) -> str:
            if col.endswith(suffix) and col not in keep_suffix:
                return col[: -len(suffix)]
            return col

        merged.rename(columns=rename, inplace=True)

        self._cache_series(pair, merged)
        return merged

    # -----------------------------
    # Entry / Exit (both directions)
    # -----------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # Add ATR for trailing stop (dynamic for hyperopt)
        df["atr"] = ta.ATR(df, timeperiod=self.trailing_atr_period.value)

        # Weekend filter
        if self.disable_weekends.value:
            # 5 = Saturday, 6 = Sunday
            is_weekend = df["date"].dt.dayofweek >= 5
        else:
            is_weekend = Series([False] * len(df))

        # LONG gates
        long_gate_ha = df["close"] < df["Smooth_HA_L"]
        long_gate_bias = df["emac_1h"] < df["emao_1h"]

        long_setup_vol = (
            ((df["bbw_expansion"] == 1) | (~self.buy_require_bbw_expansion.value))
            & (df["sqzmi"] == False)
            & ((df["mfi"] < self.buy_mfi_low.value) | (df["dmi_minus"] > self.buy_di_minus.value))
        )
        long_setup_undersold = (
            (df["close"] < df["sar"])
            & ((df["srsi_d"] >= df["srsi_k"]) & (df["srsi_d"] < self.buy_srsi_max.value))
            & ((df["fastd"] > df["fastk"]) & (df["fastd"] < self.buy_fastd_max.value))
            & (df["mfi"] < (self.buy_mfi_low.value + 10))
        )
        long_setup_sideways = (
            (df["dmi_minus"] > self.buy_di_minus.value)
            & qtpylib.crossed_above(df["dmi_minus"], df["dmi_plus"])
            & (df["close"] < df["bb_lowerband"])
        )
        long_setup_sqz = (df["sqzmi"] == True) & (
            (df["fastd"] > df["fastk"]) & (df["fastd"] < self.buy_sqz_fallback_fastd.value)
        )

        long_any = long_setup_vol | long_setup_undersold | long_setup_sideways | long_setup_sqz
        long_sanity = (df["vfi"] < 0.0) & (df["volume"] > 0)
        df.loc[long_gate_ha & long_gate_bias & long_any & long_sanity & (~is_weekend), "enter_long"] = 1

        # SHORT mirrored
        short_gate_ha = df["close"] > df["Smooth_HA_H"]
        short_gate_bias = df["emac_1h"] > df["emao_1h"]

        mfi_high_like = 100 - int(self.buy_mfi_low.value)
        srsi_high_like = 100 - int(self.buy_srsi_max.value)
        fastd_high_like = 100 - int(self.buy_fastd_max.value)
        sqz_fastd_high_like = 100 - int(self.buy_sqz_fallback_fastd.value)

        short_setup_vol = (
            ((df["bbw_expansion"] == 1) | (~self.buy_require_bbw_expansion.value))
            & (df["sqzmi"] == False)
            & ((df["mfi"] > mfi_high_like) | (df["dmi_plus"] > self.buy_di_minus.value))
        )
        short_setup_overbought = (
            (df["close"] > df["sar"])
            & ((df["srsi_d"] <= df["srsi_k"]) & (df["srsi_d"] > srsi_high_like))
            & ((df["fastd"] < df["fastk"]) & (df["fastd"] > fastd_high_like))
            & (df["mfi"] > (mfi_high_like - 10))
        )
        short_setup_sideways = (
            (df["dmi_plus"] > self.buy_di_minus.value)
            & qtpylib.crossed_above(df["dmi_plus"], df["dmi_minus"])
            & (df["close"] > df["bb_upperband"])
        )
        short_setup_sqz = (df["sqzmi"] == True) & (
            (df["fastd"] < df["fastk"]) & (df["fastd"] > sqz_fastd_high_like)
        )

        short_any = (
            short_setup_vol | short_setup_overbought | short_setup_sideways | short_setup_sqz
        )
        short_sanity = (df["vfi"] > 0.0) & (df["volume"] > 0)
        df.loc[short_gate_ha & short_gate_bias & short_any & short_sanity & (~is_weekend), "enter_short"] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No exit signals. Trailing stop ONLY in custom_exit.
        """
        return dataframe

    # -----------------------------
    # Custom exit (Trailing Stop ONLY)
    # -----------------------------
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr", 0)

        if not trade.is_short:
            highest_rate = trade.max_rate
            trail_price = highest_rate - (atr * self.trailing_atr_k.value)
            if current_rate < trail_price:
                return "atr_trailing_exit"
        else:
            lowest_rate = trade.min_rate
            trail_price = lowest_rate + (atr * self.trailing_atr_k.value)
            if current_rate > trail_price:
                return "atr_trailing_exit"

        return None

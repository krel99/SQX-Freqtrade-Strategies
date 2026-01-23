# --- Do not remove these libs ---
# --- Strategy specific imports ---
from datetime import datetime

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class TuesdayStrategy(IStrategy):
    """
    Tuesday-only futures strategy (long + short) using Ehlers-style signals.

    Note:
    - Replaced `technical.indicators` Ehlers imports with local implementations:
      * Fisher Transform (Ehlers-style) + signal
      * Adaptive MA pair (MAMA/FAMA-like) using efficiency ratio (ER) to adapt alpha
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    timeframe = "15m"
    startup_candle_count = 900

    minimal_roi = {"0": 0.14, "60": 0.08, "180": 0.03}
    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Hyperparameters ---
    buy_day_of_week = 1  # Tuesday

    buy_hour_start = IntParameter(0, 12, default=6, space="buy")
    buy_hour_end = IntParameter(13, 23, default=23, space="buy")

    regime_ema_period = IntParameter(200, 800, default=500, space="buy")
    long_regime_buffer = DecimalParameter(0.985, 1.020, default=0.995, space="buy")
    short_regime_buffer = DecimalParameter(0.980, 1.015, default=1.005, space="buy")

    buy_eft_period = IntParameter(5, 30, default=10, space="buy")
    buy_eft_gate = DecimalParameter(-0.5, 0.8, default=-0.05, space="buy")

    buy_mama_fastlimit = DecimalParameter(0.1, 0.9, default=0.5, space="buy")
    buy_mama_slowlimit = DecimalParameter(0.01, 0.09, default=0.05, space="buy")

    buy_kc_period = IntParameter(10, 60, default=20, space="buy")
    buy_kc_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space="buy")

    buy_chop_period = IntParameter(10, 30, default=14, space="buy")
    buy_chop_min = IntParameter(45, 70, default=55, space="buy")

    buy_vol_sma_period = IntParameter(20, 80, default=30, space="buy")
    buy_vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space="buy")

    max_trade_minutes = IntParameter(120, 1440, default=600, space="sell")

    @property
    def protections(self):
        return [{"method": "CooldownPeriod", "stop_duration_candles": 80}]

    # -------------------------
    # Indicator implementations
    # -------------------------

    @staticmethod
    def ehlers_fisher_transform_df(
        df: DataFrame, period: int = 10, signal_period: int = 3
    ) -> DataFrame:
        """
        Ehlers Fisher Transform (classic approach):
        1) Normalize price to [-1, 1] via rolling min/max
        2) Smooth x
        3) Fisher = 0.5 * ln((1+x)/(1-x)) with recursion smoothing
        Returns DataFrame with columns: fisher_transform, fisher_signal
        """
        price = (df["high"] + df["low"]) / 2.0
        min_l = price.rolling(period).min()
        max_h = price.rolling(period).max()
        rng = (max_h - min_l).replace(0, np.nan)

        # Raw normalization to [-1, 1]
        v = 2.0 * ((price - min_l) / rng - 0.5)

        # Bound and smooth (Ehlers uses recursive smoothing)
        v = v.clip(-0.999, 0.999)

        x = np.zeros(len(df), dtype=float)
        fish = np.zeros(len(df), dtype=float)

        v_arr = v.to_numpy()

        for i in range(len(df)):
            if i == 0 or np.isnan(v_arr[i]):
                x[i] = 0.0
                fish[i] = 0.0
                continue

            # recursive smoothing of x
            x[i] = 0.33 * v_arr[i] + 0.67 * x[i - 1]
            x[i] = float(np.clip(x[i], -0.999, 0.999))

            # fisher transform + smoothing
            f = 0.5 * np.log((1.0 + x[i]) / (1.0 - x[i]))
            fish[i] = 0.5 * f + 0.5 * fish[i - 1]

        fisher_series = DataFrame({"fisher_transform": fish}, index=df.index)["fisher_transform"]
        fisher_signal = fisher_series.rolling(signal_period).mean()

        return DataFrame(
            {
                "fisher_transform": fisher_series,
                "fisher_signal": fisher_signal,
            },
            index=df.index,
        )

    @staticmethod
    def mama_fama_like_df(
        price: DataFrame,
        fastlimit: float = 0.5,
        slowlimit: float = 0.05,
        er_period: int = 10,
        fama_smooth: float = 0.5,
    ) -> DataFrame:
        """
        Practical MAMA/FAMA-like pair:
        - Uses Efficiency Ratio (ER) to adapt alpha between slowlimit..fastlimit
        - MAMA = adaptive EMA
        - FAMA = smoother EMA of MAMA (controlled by fama_smooth)
        This keeps the "adaptive fast/slow" behavior without external deps.
        """
        close = price["close"].to_numpy(dtype=float)

        # Efficiency Ratio
        change = np.abs(np.roll(close, -er_period) - close)
        change[-er_period:] = np.nan

        volatility = np.zeros_like(close)
        for i in range(len(close)):
            if i < er_period:
                volatility[i] = np.nan
            else:
                volatility[i] = np.sum(np.abs(np.diff(close[i - er_period : i + 1])))

        er = change / np.where(volatility == 0, np.nan, volatility)
        er = np.clip(er, 0.0, 1.0)

        alpha = slowlimit + (fastlimit - slowlimit) * er
        # Bound alpha reasonably
        alpha = np.clip(alpha, 0.001, 0.999)

        mama = np.zeros_like(close)
        fama = np.zeros_like(close)

        for i in range(len(close)):
            if i == 0 or np.isnan(alpha[i]) or np.isnan(close[i]):
                mama[i] = close[i] if not np.isnan(close[i]) else 0.0
                fama[i] = mama[i]
                continue

            mama[i] = alpha[i] * close[i] + (1.0 - alpha[i]) * mama[i - 1]
            # FAMA as slower smoother of MAMA
            fama_alpha = max(min(fama_smooth * alpha[i], 0.999), 0.001)
            fama[i] = fama_alpha * mama[i] + (1.0 - fama_alpha) * fama[i - 1]

        return DataFrame({"mama": mama, "fama": fama}, index=price.index)

    # -------------------------
    # Freqtrade hooks
    # -------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour"] = dataframe["date"].dt.hour

        # Pre-calculate EMA for all regime periods (200-800)
        for period in range(200, 801):
            dataframe[f"ema_regime_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate Fisher Transform for all periods (5-30)
        for period in range(5, 31):
            fisher = self.ehlers_fisher_transform_df(dataframe, period=period, signal_period=3)
            dataframe[f"fisher_transform_{period}"] = fisher["fisher_transform"]
            dataframe[f"fisher_signal_{period}"] = fisher["fisher_signal"]

        # Pre-calculate Keltner Channels EMA and ATR for all periods (10-60)
        for period in range(10, 61):
            dataframe[f"kc_ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
            dataframe[f"kc_atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Pre-calculate Choppiness for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"chop_{period}"] = self.choppiness(dataframe, period=period)

        # Pre-calculate Volume SMA for all periods (20-80)
        for period in range(20, 81):
            dataframe[f"vol_sma_{period}"] = dataframe["volume"].rolling(period).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        regime_ema_period = self.regime_ema_period.value
        eft_period = self.buy_eft_period.value
        kc_period = self.buy_kc_period.value
        kc_multiplier = self.buy_kc_multiplier.value
        chop_period = self.buy_chop_period.value
        vol_sma_period = self.buy_vol_sma_period.value

        # Select pre-calculated indicators
        ema_regime = dataframe[f"ema_regime_{regime_ema_period}"]
        fisher_transform = dataframe[f"fisher_transform_{eft_period}"]
        fisher_signal = dataframe[f"fisher_signal_{eft_period}"]
        kc_ema = dataframe[f"kc_ema_{kc_period}"]
        kc_atr = dataframe[f"kc_atr_{kc_period}"]
        chop = dataframe[f"chop_{chop_period}"]
        vol_sma = dataframe[f"vol_sma_{vol_sma_period}"]

        # Calculate Keltner bands dynamically (multiplier varies)
        kc_middle = kc_ema
        kc_upperband = kc_ema + (kc_atr * kc_multiplier)
        kc_lowerband = kc_ema - (kc_atr * kc_multiplier)

        # Calculate MAMA/FAMA dynamically (too many decimal combinations)
        mama_df = self.mama_fama_like_df(
            dataframe,
            fastlimit=float(self.buy_mama_fastlimit.value),
            slowlimit=float(self.buy_mama_slowlimit.value),
            er_period=max(5, eft_period),
            fama_smooth=0.5,
        )
        mama = mama_df["mama"]
        fama = mama_df["fama"]

        tuesday = dataframe["day_of_week"] == self.buy_day_of_week
        hours = (dataframe["hour"] >= self.buy_hour_start.value) & (
            dataframe["hour"] <= self.buy_hour_end.value
        )

        liquid = (dataframe["volume"] > 0) & (
            dataframe["volume"] >= vol_sma * float(self.buy_vol_mult.value)
        )

        setup = chop >= self.buy_chop_min.value

        fisher_up = qtpylib.crossed_above(fisher_transform, fisher_signal) | (
            fisher_transform > fisher_transform.shift(1)
        )
        fisher_dn = qtpylib.crossed_below(fisher_transform, fisher_signal) | (
            fisher_transform < fisher_transform.shift(1)
        )

        mama_bull = mama > fama
        mama_bear = mama < fama

        reclaim_mid = qtpylib.crossed_above(dataframe["close"], kc_middle)
        reject_mid = qtpylib.crossed_below(dataframe["close"], kc_middle)

        long_regime_ok = dataframe["close"] >= (ema_regime * float(self.long_regime_buffer.value))
        short_regime_ok = dataframe["close"] <= (ema_regime * float(self.short_regime_buffer.value))

        fisher_gate = float(self.buy_eft_gate.value)
        long_gate_ok = fisher_transform >= fisher_gate
        short_gate_ok = fisher_transform <= -fisher_gate

        dataframe.loc[
            tuesday
            & hours
            & liquid
            & setup
            & fisher_up
            & mama_bull
            & reclaim_mid
            & long_regime_ok
            & long_gate_ok,
            "enter_long",
        ] = 1

        dataframe.loc[
            tuesday
            & hours
            & liquid
            & setup
            & fisher_dn
            & mama_bear
            & reject_mid
            & short_regime_ok
            & short_gate_ok,
            "enter_short",
        ] = 1

        # Store for exit trend
        dataframe["fisher_transform"] = fisher_transform
        dataframe["fisher_signal"] = fisher_signal
        dataframe["kc_middle"] = kc_middle
        dataframe["kc_upperband"] = kc_upperband
        dataframe["kc_lowerband"] = kc_lowerband

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fisher_turn_down = qtpylib.crossed_below(
            dataframe["fisher_transform"], dataframe["fisher_signal"]
        )
        fisher_turn_up = qtpylib.crossed_above(
            dataframe["fisher_transform"], dataframe["fisher_signal"]
        )

        dataframe.loc[
            (dataframe["close"] >= dataframe["kc_upperband"]) | fisher_turn_down, "exit_long"
        ] = 1
        dataframe.loc[
            (dataframe["close"] <= dataframe["kc_lowerband"]) | fisher_turn_up, "exit_short"
        ] = 1
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        max_minutes = int(self.max_trade_minutes.value)
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes >= max_minutes:
            return "time_stop"
        return None

    def keltner_channels(self, dataframe: DataFrame, period: int, multiplier: float):
        ema = ta.EMA(dataframe, timeperiod=period)
        atr = ta.ATR(dataframe, timeperiod=period)
        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)
        return {"middle": ema, "upper": upper, "lower": lower}

    def choppiness(self, dataframe: DataFrame, period: int) -> DataFrame:
        tr = ta.TRANGE(dataframe)
        tr_sum = tr.rolling(window=period).sum()
        high_n = dataframe["high"].rolling(window=period).max()
        low_n = dataframe["low"].rolling(window=period).min()
        denom = (high_n - low_n).replace(0, float("nan"))
        chop = 100.0 * (np.log10(tr_sum / denom) / np.log10(period))
        return chop

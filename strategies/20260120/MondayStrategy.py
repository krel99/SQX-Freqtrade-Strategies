# --- Do not remove these libs ---
# --- Strategy specific imports ---
from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class MondayStrategy(IStrategy):
    """
    Monday-only futures strategy (long + short) designed to reliably take ~1 trade most Mondays.
    No informative timeframes; uses long-period EMA on 15m as a regime filter.

    Long idea: dip + recovery in non-bear regime.
    Short idea: pop + rejection in non-bull regime.

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    startup_candle_count = 700
    timeframe = "15m"

    minimal_roi = {"0": 0.12, "60": 0.07, "180": 0.03}
    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Hyperparameters ---

    buy_day_of_week = 0  # Monday (0 = Monday in Python's weekday convention)

    buy_hour_start = IntParameter(0, 12, default=6, space="buy")
    buy_hour_end = IntParameter(13, 23, default=23, space="buy")

    # Same-timeframe "HTF-like" regime EMA
    ema_regime_period = IntParameter(200, 600, default=400, space="buy")

    # Regime buffers:
    # Long allowed when close >= EMA * long_buffer
    # Short allowed when close <= EMA * short_buffer
    long_regime_buffer = DecimalParameter(0.985, 1.010, default=0.995, space="buy")
    short_regime_buffer = DecimalParameter(0.990, 1.020, default=1.005, space="buy")

    # RSI / Stoch thresholds
    rsi_period = IntParameter(10, 30, default=14, space="buy")
    long_rsi = IntParameter(25, 55, default=42, space="buy")
    short_rsi = IntParameter(45, 80, default=58, space="buy")

    stoch_k = IntParameter(5, 20, default=14, space="buy")
    stoch_d = IntParameter(1, 10, default=3, space="buy")
    long_stoch = IntParameter(10, 60, default=35, space="buy")
    short_stoch = IntParameter(40, 90, default=65, space="buy")

    # Bollinger
    bb_period = IntParameter(10, 40, default=20, space="buy")
    bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    # MACD (mild confirmation / turn)
    macd_fast = IntParameter(10, 20, default=12, space="buy")
    macd_slow = IntParameter(20, 40, default=26, space="buy")
    macd_signal = IntParameter(5, 15, default=9, space="buy")

    # Liquidity filter (mild)
    vol_sma_period = IntParameter(20, 80, default=30, space="buy")
    vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space="buy")

    # Trade duration hard stop
    max_trade_minutes = IntParameter(120, 1440, default=480, space="sell")  # 8 hours default

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 80}  # ~20 hours on 15m
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculate indicators for all possible hyperopt parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """
        # Time-based columns (fixed, no hyperopt)
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour"] = dataframe["date"].dt.hour

        # Pre-calculate EMA for all possible periods (200-600)
        for p in range(200, 601):
            dataframe[f"ema_regime_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # Pre-calculate RSI for all possible periods (10-30)
        for p in range(10, 31):
            dataframe[f"rsi_{p}"] = ta.RSI(dataframe, timeperiod=p)

        # Pre-calculate Stochastic for all possible parameter combinations
        # stoch_k: 5-20, stoch_d: 1-10
        for k in range(5, 21):
            for d in range(1, 11):
                st = ta.STOCH(
                    dataframe,
                    fastk_period=k,
                    slowk_period=3,
                    slowd_period=d,
                )
                dataframe[f"slowk_{k}_{d}"] = st["slowk"]
                dataframe[f"slowd_{k}_{d}"] = st["slowd"]

        # Pre-calculate MACD for all possible parameter combinations
        # macd_fast: 10-20, macd_slow: 20-40, macd_signal: 5-15
        for fast in range(10, 21):
            for slow in range(20, 41):
                for sig in range(5, 16):
                    if slow > fast:  # Ensure slow > fast
                        macd = ta.MACD(
                            dataframe,
                            fastperiod=fast,
                            slowperiod=slow,
                            signalperiod=sig,
                        )
                        dataframe[f"macd_{fast}_{slow}_{sig}"] = macd["macd"]
                        dataframe[f"macdsignal_{fast}_{slow}_{sig}"] = macd["macdsignal"]
                        dataframe[f"macdhist_{fast}_{slow}_{sig}"] = macd["macdhist"]

        # Pre-calculate Bollinger Bands for all possible parameter combinations
        # bb_period: 10-40, bb_stddev: 1.5-3.0 (step 0.1)
        for period in range(10, 41):
            for stddev in [round(x * 0.1, 1) for x in range(15, 31)]:
                bb = ta.BBANDS(
                    dataframe,
                    timeperiod=period,
                    nbdevup=stddev,
                    nbdevdn=stddev,
                )
                dataframe[f"bb_lowerband_{period}_{stddev}"] = bb["lowerband"]
                dataframe[f"bb_middleband_{period}_{stddev}"] = bb["middleband"]
                dataframe[f"bb_upperband_{period}_{stddev}"] = bb["upperband"]

        # Pre-calculate Volume SMA for all possible periods (20-80)
        for vsp in range(20, 81):
            dataframe[f"vol_sma_{vsp}"] = dataframe["volume"].rolling(vsp).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        ema_period = int(self.ema_regime_period.value)
        rsi_period = int(self.rsi_period.value)
        stoch_k = int(self.stoch_k.value)
        stoch_d = int(self.stoch_d.value)
        macd_fast = int(self.macd_fast.value)
        macd_slow = int(self.macd_slow.value)
        macd_sig = int(self.macd_signal.value)
        bb_period = int(self.bb_period.value)
        bb_std = round(float(self.bb_stddev.value), 1)
        vol_period = int(self.vol_sma_period.value)

        # Get pre-calculated indicators for current hyperopt values
        ema_regime = dataframe[f"ema_regime_{ema_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        slowk = dataframe[f"slowk_{stoch_k}_{stoch_d}"]
        slowd = dataframe[f"slowd_{stoch_k}_{stoch_d}"]
        macdhist = dataframe[f"macdhist_{macd_fast}_{macd_slow}_{macd_sig}"]
        bb_lowerband = dataframe[f"bb_lowerband_{bb_period}_{bb_std}"]
        bb_middleband = dataframe[f"bb_middleband_{bb_period}_{bb_std}"]
        bb_upperband = dataframe[f"bb_upperband_{bb_period}_{bb_std}"]
        vol_sma = dataframe[f"vol_sma_{vol_period}"]

        # Entry conditions
        monday = dataframe["day_of_week"] == self.buy_day_of_week
        hours = (dataframe["hour"] >= self.buy_hour_start.value) & (
            dataframe["hour"] <= self.buy_hour_end.value
        )

        liquid = (dataframe["volume"] > 0) & (
            dataframe["volume"] >= vol_sma * float(self.vol_mult.value)
        )

        # --- LONG ---
        long_regime_ok = dataframe["close"] >= (ema_regime * float(self.long_regime_buffer.value))

        long_dip = (
            (dataframe["close"] <= bb_lowerband)
            | (rsi <= self.long_rsi.value)
            | ((slowk <= self.long_stoch.value) & (slowd <= self.long_stoch.value))
        )

        long_reclaim = (
            qtpylib.crossed_above(dataframe["close"], bb_lowerband)
            | qtpylib.crossed_above(dataframe["close"], bb_middleband)
            | (macdhist > macdhist.shift(1))
        )

        dataframe.loc[
            monday & hours & liquid & long_regime_ok & long_dip & long_reclaim, "enter_long"
        ] = 1

        # --- SHORT ---
        short_regime_ok = dataframe["close"] <= (ema_regime * float(self.short_regime_buffer.value))

        short_pop = (
            (dataframe["close"] >= bb_upperband)
            | (rsi >= self.short_rsi.value)
            | ((slowk >= self.short_stoch.value) & (slowd >= self.short_stoch.value))
        )

        short_reject = (
            qtpylib.crossed_below(dataframe["close"], bb_upperband)
            | qtpylib.crossed_below(dataframe["close"], bb_middleband)
            | (macdhist < macdhist.shift(1))
        )

        dataframe.loc[
            monday & hours & liquid & short_regime_ok & short_pop & short_reject, "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        bb_period = int(self.bb_period.value)
        bb_std = round(float(self.bb_stddev.value), 1)

        bb_middleband = dataframe[f"bb_middleband_{bb_period}_{bb_std}"]

        # Mean-reversion exits
        dataframe.loc[dataframe["close"] > bb_middleband, "exit_long"] = 1
        dataframe.loc[dataframe["close"] < bb_middleband, "exit_short"] = 1
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
        """
        Hard time stop for BOTH long and short.
        """
        max_minutes = int(self.max_trade_minutes.value)
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes >= max_minutes:
            return "time_stop"
        return None

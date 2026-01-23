import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class FridayStrategy(IStrategy):
    """
    Friday-only volatility / momentum strategy.
    Goal:
    - Open 1 trade on (almost) every Friday (when market is tradable / liquid),
      ideally during the US-session / after-close volatility window.
    - Max 2 trades total (controlled via max_open_trades + cooldown protection).
    - No informative timeframes; use long-period EMA on the same timeframe as regime filter.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # --- Trade frequency control (global) ---
    # NOTE: This is strategy-side. Also set max_open_trades in your config for certainty.
    max_open_trades = 2
    process_only_new_candles = True

    # Ensure enough candles for the longest lookback (EMA + weekly-like ATR baseline)
    startup_candle_count = 600

    # --- Risk / exits ---
    minimal_roi = {"0": 0.12, "60": 0.07, "180": 0.03}

    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    timeframe = "15m"

    # --- Hyperparameters ---

    # Friday-only (constant, not a hyperopt parameter)
    buy_day_of_week = 4  # 4 = Friday

    # Trading hours (local exchange time = candle timestamps; set to your preference)
    # Default tuned toward "later Friday" activity, but still broad enough to catch most Fridays.
    buy_hour_start = IntParameter(12, 20, default=16, space="buy")
    buy_hour_end = IntParameter(20, 23, default=23, space="buy")

    # Regime filter via long EMA on same timeframe (15m).
    # 200 on 15m ~ 50 hours. 400 ~ 100 hours. This mimics HTF trend.
    buy_ema_period = IntParameter(200, 500, default=300, space="buy")
    long_ema_buffer = DecimalParameter(0.995, 1.010, default=1.000, space="buy")
    short_ema_buffer = DecimalParameter(0.990, 1.005, default=1.000, space="buy")
    # buffer lets you require close >= EMA * buffer (slightly above/near)

    # Donchian (used as a momentum confirmation, correctly shifted)
    buy_dc_period = IntParameter(20, 80, default=40, space="buy")

    # Volatility expansion: ATR% now vs ATR% baseline
    buy_atr_period = IntParameter(10, 30, default=14, space="buy")
    buy_atr_baseline = IntParameter(96, 800, default=480, space="buy")  # ~5 days on 15m is 480
    buy_atr_mult = DecimalParameter(1.05, 1.60, default=1.20, space="buy")

    # Optional band breakout confirmation (stddev bands)
    buy_sdb_period = IntParameter(20, 80, default=40, space="buy")
    buy_sdb_stddev = DecimalParameter(1.5, 3.5, default=2.2, space="buy")

    # Liquidity filter (kept mild so you still trade most Fridays)
    buy_vol_sma_period = IntParameter(20, 80, default=30, space="buy")
    buy_vol_mult = DecimalParameter(0.30, 1.20, default=0.60, space="buy")

    # Exit safety: avoid holding into weekend (optional)
    exit_hour_start = IntParameter(20, 23, default=23, space="sell")

    @property
    def protections(self):
        # Helps enforce "1 trade per pair per Friday window" and reduces churn.
        # This DOES NOT guarantee only 1 trade total across all pairs.
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 80,  # ~20 hours on 15m
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour"] = dataframe["date"].dt.hour

        # Pre-calculate EMA for all periods (200-500)
        for period in range(200, 501):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate Donchian channels for all periods (20-80)
        for period in range(20, 81):
            dataframe[f"dc_upper_{period}"] = dataframe["high"].rolling(period).max().shift(1)
            dataframe[f"dc_lower_{period}"] = dataframe["low"].rolling(period).min().shift(1)

        # Pre-calculate ATR for all periods (10-30)
        for period in range(10, 31):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)
            dataframe[f"atrp_{period}"] = dataframe[f"atr_{period}"] / dataframe["close"]

        # Pre-calculate ATR% baseline rolling means for key periods
        # (96-800 is too large, so we calculate a subset of common values)
        for baseline in [
            96,
            120,
            144,
            192,
            240,
            288,
            336,
            384,
            432,
            480,
            528,
            576,
            624,
            672,
            720,
            768,
            800,
        ]:
            for atr_period in range(10, 31):
                dataframe[f"atrp_base_{atr_period}_{baseline}"] = (
                    dataframe[f"atrp_{atr_period}"].rolling(baseline).mean()
                )

        # Pre-calculate SMA and STDDEV for all periods (20-80)
        for period in range(20, 81):
            dataframe[f"sma_{period}"] = ta.SMA(dataframe, timeperiod=period)
            dataframe[f"std_{period}"] = ta.STDDEV(dataframe, timeperiod=period)

        # Pre-calculate volume SMA for all periods (20-80)
        for period in range(20, 81):
            dataframe[f"vol_sma_{period}"] = dataframe["volume"].rolling(period).mean()

        # Candle range (simple expansion proxy)
        dataframe["range"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        ema_period = int(self.buy_ema_period.value)
        dc_period = int(self.buy_dc_period.value)
        atr_period = int(self.buy_atr_period.value)
        atr_baseline = int(self.buy_atr_baseline.value)
        sdb_period = int(self.buy_sdb_period.value)
        sdb_std = float(self.buy_sdb_stddev.value)
        vol_sma_period = int(self.buy_vol_sma_period.value)

        # Select pre-calculated indicators
        ema_long = dataframe[f"ema_{ema_period}"]
        dc_upper = dataframe[f"dc_upper_{dc_period}"]
        dc_lower = dataframe[f"dc_lower_{dc_period}"]
        atrp = dataframe[f"atrp_{atr_period}"]
        vol_sma = dataframe[f"vol_sma_{vol_sma_period}"]

        # Find nearest pre-calculated baseline
        baseline_options = [
            96,
            120,
            144,
            192,
            240,
            288,
            336,
            384,
            432,
            480,
            528,
            576,
            624,
            672,
            720,
            768,
            800,
        ]
        nearest_baseline = min(baseline_options, key=lambda x: abs(x - atr_baseline))
        atrp_base = dataframe[f"atrp_base_{atr_period}_{nearest_baseline}"]

        # Calculate stddev bands using pre-calculated SMA and STDDEV
        sma = dataframe[f"sma_{sdb_period}"]
        std = dataframe[f"std_{sdb_period}"]
        sdb_upper = sma + (std * sdb_std)
        sdb_lower = sma - (std * sdb_std)

        # Core Friday + hours gate
        friday = dataframe["day_of_week"] == self.buy_day_of_week
        hours = (dataframe["hour"] >= self.buy_hour_start.value) & (
            dataframe["hour"] <= self.buy_hour_end.value
        )

        # Regime filter (same timeframe, long period)
        # Slight buffer allows "near EMA" or "above EMA"
        long_buffer = float(self.long_ema_buffer.value)
        short_buffer = float(self.short_ema_buffer.value)
        long_regime_ok = dataframe["close"] >= (ema_long * long_buffer)
        short_regime_ok = dataframe["close"] <= (ema_long * short_buffer)

        # Mild liquidity filter
        vol_mult = float(self.buy_vol_mult.value)
        liquid = (dataframe["volume"] > 0) & (dataframe["volume"] >= vol_sma * vol_mult)

        # Volatility expansion: ATR% spikes vs its baseline
        atr_mult = float(self.buy_atr_mult.value)
        vol_event = atrp >= (atrp_base * atr_mult)

        # Momentum confirmation (NOT too strict):
        # - Either a Donchian breakout (cross above previous upper),
        # - Or a stddev-band push (close above upper band),
        # - Or simply a strong candle (range expansion) while vol_event is true.
        dc_break = qtpylib.crossed_above(dataframe["close"], dc_upper)
        band_push = dataframe["close"] > sdb_upper
        range_push = (
            dataframe["range"] > dataframe["range"].rolling(96).mean()
        )  # mild "bigger than usual today"

        long_trigger = vol_event & (dc_break | band_push | range_push)

        dataframe.loc[friday & hours & liquid & long_regime_ok & long_trigger, "enter_long"] = 1

        # --- SHORT ---
        # Donchian breakdown (cross below previous lower)
        dc_breakdown = qtpylib.crossed_below(dataframe["close"], dc_lower)
        # Stddev band push lower
        band_push_lower = dataframe["close"] < sdb_lower

        short_trigger = vol_event & (dc_breakdown | band_push_lower | range_push)

        dataframe.loc[friday & hours & liquid & short_regime_ok & short_trigger, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Optional: if you want to avoid holding into weekend, exit late Friday.
        # This is a "soft" exit signal; ROI/trailing/stoploss still apply.
        late_friday = (dataframe["day_of_week"] == 4) & (
            dataframe["hour"] >= self.exit_hour_start.value
        )
        dataframe.loc[late_friday, "exit_long"] = 1
        dataframe.loc[late_friday, "exit_short"] = 1
        return dataframe

# --- Do not remove these libs ---
# --- Strategy specific imports ---
from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class WednesdayStrategy(IStrategy):
    """
    Wednesday-only futures strategy (long + short) using "less common" indicators:
    ZLEMA, HMA, Ultimate Oscillator, Awesome Oscillator, and ATR.

    Goal:
    - Take ~1 trade on most Wednesdays (max 2 open trades).
    - No informative timeframes; use long EMA regime filter on 15m.
    - Use ATR as volatility/impulse context (ATR% vs baseline).
    - Hard time stop included.
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    timeframe = "15m"
    startup_candle_count = 900

    minimal_roi = {"0": 0.13, "60": 0.08, "180": 0.03}
    stoploss = -0.11

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Hyperparameters ---

    buy_day_of_week = 2  # Wednesday (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)

    buy_hour_start = IntParameter(0, 12, default=6, space="buy")
    buy_hour_end = IntParameter(13, 23, default=23, space="buy")

    # Regime filter (same timeframe "HTF-like")
    regime_ema_period = IntParameter(200, 800, default=500, space="buy")
    long_regime_buffer = DecimalParameter(0.985, 1.020, default=0.995, space="buy")
    short_regime_buffer = DecimalParameter(0.980, 1.015, default=1.005, space="buy")

    # ZLEMA / HMA
    buy_zlema_period = IntParameter(10, 50, default=20, space="buy")
    buy_hma_period = IntParameter(10, 50, default=20, space="buy")

    # Ultimate Oscillator thresholds
    # (Make them sensible for "dip then turn" rather than choking trade frequency.)
    buy_uo_long = IntParameter(25, 55, default=45, space="buy")
    buy_uo_short = IntParameter(45, 75, default=55, space="buy")

    # Awesome Oscillator: use sign / turn rather than huge threshold
    buy_ao_gate = DecimalParameter(-1.0, 1.0, default=0.0, space="buy")

    # ATR context as ATR% spike vs baseline (more robust than single-candle jump)
    buy_atr_period = IntParameter(10, 30, default=14, space="buy")
    buy_atr_baseline = IntParameter(96, 800, default=480, space="buy")  # ~5 days on 15m
    buy_atr_mult = DecimalParameter(1.00, 1.60, default=1.15, space="buy")

    # Mild liquidity filter (keep mild to trade most Wednesdays)
    buy_vol_sma_period = IntParameter(20, 80, default=30, space="buy")
    buy_vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space="buy")

    # Hard trade duration limit
    max_trade_minutes = IntParameter(120, 1440, default=600, space="sell")  # 10 hours

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 80}  # ~20 hours on 15m
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour"] = dataframe["date"].dt.hour

        # Regime EMA
        dataframe["ema_regime"] = ta.EMA(dataframe, timeperiod=int(self.regime_ema_period.value))

        # ZLEMA / HMA
        dataframe["zlema"] = self.zlema(dataframe, period=int(self.buy_zlema_period.value))
        dataframe["hma"] = qtpylib.hma(dataframe["close"], window=int(self.buy_hma_period.value))

        # Ultimate Oscillator
        # (Using classic defaults is fine; keeping your "less common" vibe but ensuring stability)
        dataframe["uo"] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Awesome Oscillator
        dataframe["ao"] = qtpylib.awesome_oscillator(dataframe)

        # ATR and ATR% baseline
        atr_p = int(self.buy_atr_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_p)
        dataframe["atrp"] = dataframe["atr"] / dataframe["close"]
        dataframe["atrp_base"] = dataframe["atrp"].rolling(int(self.buy_atr_baseline.value)).mean()

        # Mild liquidity baseline
        vsp = int(self.buy_vol_sma_period.value)
        dataframe["vol_sma"] = dataframe["volume"].rolling(vsp).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        wed = dataframe["day_of_week"] == self.buy_day_of_week
        hours = (dataframe["hour"] >= self.buy_hour_start.value) & (
            dataframe["hour"] <= self.buy_hour_end.value
        )

        liquid = (dataframe["volume"] > 0) & (
            dataframe["volume"] >= dataframe["vol_sma"] * float(self.buy_vol_mult.value)
        )

        # ATR expansion context
        atr_mult = float(self.buy_atr_mult.value)
        atr_event = dataframe["atrp"] >= (dataframe["atrp_base"] * atr_mult)

        # Moving average direction signal (ZLEMA vs HMA)
        bull_cross = qtpylib.crossed_above(dataframe["zlema"], dataframe["hma"])
        bear_cross = qtpylib.crossed_below(dataframe["zlema"], dataframe["hma"])

        # Oscillator context
        ao_gate = float(self.buy_ao_gate.value)
        ao_bull = dataframe["ao"] >= ao_gate
        ao_bear = dataframe["ao"] <= -ao_gate

        uo_long_ok = dataframe["uo"] <= self.buy_uo_long.value
        uo_short_ok = dataframe["uo"] >= self.buy_uo_short.value

        # Regime filters (same timeframe)
        long_regime_ok = dataframe["close"] >= (
            dataframe["ema_regime"] * float(self.long_regime_buffer.value)
        )
        short_regime_ok = dataframe["close"] <= (
            dataframe["ema_regime"] * float(self.short_regime_buffer.value)
        )

        # --- LONG ---
        # Wednesday + time + liquidity + (ATR context OR MA cross) + UO low + AO supportive
        # (The OR here is intentional to keep trade frequency high on Wednesdays.)
        long_trigger = (bull_cross | atr_event) & uo_long_ok & ao_bull
        dataframe.loc[wed & hours & liquid & long_regime_ok & long_trigger, "enter_long"] = 1

        # --- SHORT ---
        short_trigger = (bear_cross | atr_event) & uo_short_ok & ao_bear
        dataframe.loc[wed & hours & liquid & short_regime_ok & short_trigger, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Simple structured exits aligned with the indicator theme:
        # - Exit long when ZLEMA crosses below HMA OR AO turns negative-ish.
        # - Exit short when ZLEMA crosses above HMA OR AO turns positive-ish.
        ao_gate = float(self.buy_ao_gate.value)

        dataframe.loc[
            qtpylib.crossed_below(dataframe["zlema"], dataframe["hma"])
            | (dataframe["ao"] < -ao_gate),
            "exit_long",
        ] = 1

        dataframe.loc[
            qtpylib.crossed_above(dataframe["zlema"], dataframe["hma"])
            | (dataframe["ao"] > ao_gate),
            "exit_short",
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
        # Hard time stop for both directions
        max_minutes = int(self.max_trade_minutes.value)
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes >= max_minutes:
            return "time_stop"
        return None

    def zlema(self, dataframe: DataFrame, period: int) -> DataFrame:
        """
        Zero Lag Exponential Moving Average (ZLEMA).
        Using the classic EMA-of-EMA lag reduction method you already used.
        """
        ema = ta.EMA(dataframe["close"], timeperiod=period)
        ema_ema = ta.EMA(ema, timeperiod=period)
        zlema = ema + (ema - ema_ema)
        return zlema

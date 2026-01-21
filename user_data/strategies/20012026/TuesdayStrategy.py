# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import ehlers_fisher_transform, ehlers_mama
import numpy as np
# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce


class TuesdayStrategy(IStrategy):
    """
    Tuesday-only futures strategy (long + short) using Ehlers indicators.

    Concept:
    - Use Choppiness as a context filter: look for "range -> move" transition.
    - Trigger on Ehlers Fisher turn + MAMA/FAMA direction,
      confirmed by reclaim / reject of Keltner midline.
    - Same timeframe only; long EMA regime filter to avoid fighting the larger drift.
    - Designed to take ~1 trade on most Tuesdays (max 2 open trades).
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    timeframe = '15m'
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

    # Tuesday
    buy_day_of_week = CategoricalParameter([1], space='buy', default=1)  # 1 = Tuesday

    # Trading hours (make it broad enough to trade most Tuesdays; tune to your exchange session)
    buy_hour_start = IntParameter(0, 12, default=6, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Same-timeframe "HTF-like" regime EMA
    regime_ema_period = IntParameter(200, 800, default=500, space='buy')
    long_regime_buffer = DecimalParameter(0.985, 1.020, default=0.995, space='buy')
    short_regime_buffer = DecimalParameter(0.980, 1.015, default=1.005, space='buy')

    # Ehlers Fisher Transform
    buy_eft_period = IntParameter(5, 30, default=10, space='buy')
    # Use Fisher crossing / slope instead of static threshold to increase hit-rate & reduce overfit
    buy_eft_gate = DecimalParameter(-0.5, 0.8, default=-0.05, space='buy')

    # Ehlers MAMA
    buy_mama_fastlimit = DecimalParameter(0.1, 0.9, default=0.5, space='buy')
    buy_mama_slowlimit = DecimalParameter(0.01, 0.09, default=0.05, space='buy')

    # Keltner Channels
    buy_kc_period = IntParameter(10, 60, default=20, space='buy')
    buy_kc_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='buy')

    # Choppiness Index (standardized)
    buy_chop_period = IntParameter(10, 30, default=14, space='buy')
    # Higher = choppier. We'll use it as "setup context" and then trigger on turns.
    buy_chop_min = IntParameter(45, 70, default=55, space='buy')

    # Mild liquidity filter (keep mild so you trade most Tuesdays)
    buy_vol_sma_period = IntParameter(20, 80, default=30, space='buy')
    buy_vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space='buy')

    # Hard trade duration limit
    max_trade_minutes = IntParameter(120, 1440, default=600, space='sell')  # default 10 hours

    @property
    def protections(self):
        # Helps enforce “one attempt per pair” during Tuesday window
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 80}  # ~20 hours on 15m
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['hour'] = dataframe['date'].dt.hour

        # Regime EMA (same timeframe)
        dataframe['ema_regime'] = ta.EMA(dataframe, timeperiod=int(self.regime_ema_period.value))

        # Ehlers Fisher Transform
        fisher = ehlers_fisher_transform(dataframe, period=int(self.buy_eft_period.value))
        dataframe['fisher_transform'] = fisher['fisher_transform']
        dataframe['fisher_signal'] = fisher['fisher_signal']

        # Ehlers MAMA
        mama = ehlers_mama(
            dataframe,
            fastlimit=float(self.buy_mama_fastlimit.value),
            slowlimit=float(self.buy_mama_slowlimit.value)
        )
        dataframe['mama'] = mama['mama']
        dataframe['fama'] = mama['fama']

        # Keltner Channels (+ midline)
        kc = self.keltner_channels(
            dataframe,
            period=int(self.buy_kc_period.value),
            multiplier=float(self.buy_kc_multiplier.value)
        )
        dataframe['kc_middle'] = kc['middle']
        dataframe['kc_upperband'] = kc['upper']
        dataframe['kc_lowerband'] = kc['lower']

        # Choppiness Index (standard)
        dataframe['chop'] = self.choppiness(dataframe, period=int(self.buy_chop_period.value))

        # Mild liquidity baseline
        vsp = int(self.buy_vol_sma_period.value)
        dataframe['vol_sma'] = dataframe['volume'].rolling(vsp).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tuesday = dataframe['day_of_week'] == self.buy_day_of_week.value
        hours = (dataframe['hour'] >= self.buy_hour_start.value) & (dataframe['hour'] <= self.buy_hour_end.value)

        liquid = (dataframe['volume'] > 0) & (dataframe['volume'] >= dataframe['vol_sma'] * float(self.buy_vol_mult.value))

        # Context: we want choppy conditions (range) before the move triggers
        setup = dataframe['chop'] >= self.buy_chop_min.value

        # Ehlers turning signals (higher hit-rate than static thresholds):
        fisher_up = qtpylib.crossed_above(dataframe['fisher_transform'], dataframe['fisher_signal']) | (
            dataframe['fisher_transform'] > dataframe['fisher_transform'].shift(1)
        )
        fisher_dn = qtpylib.crossed_below(dataframe['fisher_transform'], dataframe['fisher_signal']) | (
            dataframe['fisher_transform'] < dataframe['fisher_transform'].shift(1)
        )

        # MAMA direction
        mama_bull = dataframe['mama'] > dataframe['fama']
        mama_bear = dataframe['mama'] < dataframe['fama']

        # Keltner midline reclaim/reject = simple, robust confirmation
        reclaim_mid = qtpylib.crossed_above(dataframe['close'], dataframe['kc_middle'])
        reject_mid = qtpylib.crossed_below(dataframe['close'], dataframe['kc_middle'])

        # Regime filters (same TF)
        long_regime_ok = dataframe['close'] >= (dataframe['ema_regime'] * float(self.long_regime_buffer.value))
        short_regime_ok = dataframe['close'] <= (dataframe['ema_regime'] * float(self.short_regime_buffer.value))

        # Gates (keeps Fisher from triggering too early; also helps reduce noise)
        fisher_gate = float(self.buy_eft_gate.value)
        long_gate_ok = dataframe['fisher_transform'] >= fisher_gate
        short_gate_ok = dataframe['fisher_transform'] <= -fisher_gate

        # --- LONG ENTRY ---
        # Choppy setup + Fisher turning up + MAMA bullish + reclaim midline + regime ok
        dataframe.loc[
            tuesday & hours & liquid & setup & fisher_up & mama_bull & reclaim_mid & long_regime_ok & long_gate_ok,
            'enter_long'
        ] = 1

        # --- SHORT ENTRY ---
        # Choppy setup + Fisher turning down + MAMA bearish + reject midline + regime ok
        dataframe.loc[
            tuesday & hours & liquid & setup & fisher_dn & mama_bear & reject_mid & short_regime_ok & short_gate_ok,
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Mean-reversion / structure exits:
        # - If long: take exit on reach toward upper band or Fisher turning down
        # - If short: take exit on reach toward lower band or Fisher turning up
        fisher_turn_down = qtpylib.crossed_below(dataframe['fisher_transform'], dataframe['fisher_signal'])
        fisher_turn_up = qtpylib.crossed_above(dataframe['fisher_transform'], dataframe['fisher_signal'])

        dataframe.loc[(dataframe['close'] >= dataframe['kc_upperband']) | fisher_turn_down, 'exit_long'] = 1
        dataframe.loc[(dataframe['close'] <= dataframe['kc_lowerband']) | fisher_turn_up, 'exit_short'] = 1
        return dataframe

    # --- Hard time stop for both directions ---
    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
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
        return {'middle': ema, 'upper': upper, 'lower': lower}

    def choppiness(self, dataframe: DataFrame, period: int) -> DataFrame:
        """
        Standard Choppiness Index:
        CHOP = 100 * log10( sum(TR, n) / (max(high,n)-min(low,n)) ) / log10(n)
        """
        tr = ta.TRANGE(dataframe)
        tr_sum = tr.rolling(window=period).sum()
        high_n = dataframe['high'].rolling(window=period).max()
        low_n = dataframe['low'].rolling(window=period).min()
        denom = (high_n - low_n).replace(0, float('nan'))

        chop = 100.0 * (np.log10(tr_sum / denom) / np.log10(period))
        return chop

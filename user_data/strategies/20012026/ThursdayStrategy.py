# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import heikin_ashi

# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce


class ThursdayStrategy(IStrategy):
    """
    Thursday-only futures strategy (long + short) focusing on price/volume.

    Concept:
    - Use Supertrend direction + Heikin-Ashi confirmation as structure.
    - Use MFI as "pressure" (oversold/overbought) signal.
    - Use OBV slope / confirmation (more robust than strict divergence gating).
    - Same timeframe only; long EMA as regime filter.
    - Hard time stop included.
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    timeframe = '15m'
    startup_candle_count = 900

    minimal_roi = {"0": 0.14, "60": 0.08, "180": 0.03}
    stoploss = -0.13

    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.028
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Hyperparameters ---

    buy_day_of_week = CategoricalParameter([3], space='buy', default=3)  # 3 = Thursday

    buy_hour_start = IntParameter(0, 12, default=6, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Regime EMA (same timeframe)
    regime_ema_period = IntParameter(200, 800, default=500, space='buy')
    long_regime_buffer = DecimalParameter(0.985, 1.020, default=0.995, space='buy')
    short_regime_buffer = DecimalParameter(0.980, 1.015, default=1.005, space='buy')

    # Supertrend
    buy_st_period = IntParameter(7, 30, default=10, space='buy')
    buy_st_multiplier = IntParameter(2, 6, default=3, space='buy')

    # Money Flow Index (MFI)
    buy_mfi_period = IntParameter(10, 30, default=14, space='buy')
    buy_mfi_long = IntParameter(15, 55, default=35, space='buy')   # loosened to improve Thursday hit-rate
    buy_mfi_short = IntParameter(45, 85, default=65, space='buy')

    # OBV confirmation (use slope vs baseline rather than strict divergence)
    buy_obv_period = IntParameter(10, 60, default=20, space='buy')

    # Mild liquidity filter
    buy_vol_sma_period = IntParameter(20, 80, default=30, space='buy')
    buy_vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space='buy')

    # Hard trade duration limit
    max_trade_minutes = IntParameter(120, 1440, default=720, space='sell')  # default 12 hours

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 80}  # ~20 hours on 15m
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['hour'] = dataframe['date'].dt.hour

        # Regime EMA
        dataframe['ema_regime'] = ta.EMA(dataframe, timeperiod=int(self.regime_ema_period.value))

        # Supertrend (keep your implementation, but also expose direction flag)
        st = self.supertrend(dataframe, int(self.buy_st_period.value), int(self.buy_st_multiplier.value))
        dataframe['supertrend'] = st['ST']
        dataframe['st_uptrend'] = st['in_uptrend'].astype('int')

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=int(self.buy_mfi_period.value))

        # OBV + slope/confirmation
        dataframe['obv'] = ta.OBV(dataframe)
        p = int(self.buy_obv_period.value)
        dataframe['obv_slope'] = dataframe['obv'] - dataframe['obv'].shift(p)

        # Heikin-Ashi candles
        ha = heikin_ashi(dataframe)
        dataframe['ha_close'] = ha['close']
        dataframe['ha_open'] = ha['open']

        # Liquidity baseline
        vsp = int(self.buy_vol_sma_period.value)
        dataframe['vol_sma'] = dataframe['volume'].rolling(vsp).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        thu = dataframe['day_of_week'] == self.buy_day_of_week.value
        hours = (dataframe['hour'] >= self.buy_hour_start.value) & (dataframe['hour'] <= self.buy_hour_end.value)

        liquid = (dataframe['volume'] > 0) & (dataframe['volume'] >= dataframe['vol_sma'] * float(self.buy_vol_mult.value))

        # Regime filters (same timeframe)
        long_regime_ok = dataframe['close'] >= (dataframe['ema_regime'] * float(self.long_regime_buffer.value))
        short_regime_ok = dataframe['close'] <= (dataframe['ema_regime'] * float(self.short_regime_buffer.value))

        # Heikin-Ashi direction
        ha_bull = dataframe['ha_close'] > dataframe['ha_open']
        ha_bear = dataframe['ha_close'] < dataframe['ha_open']

        # Supertrend direction (and simple reclaim/reject of supertrend line)
        st_bull = dataframe['st_uptrend'] == 1
        st_bear = dataframe['st_uptrend'] == 0
        reclaim_st = qtpylib.crossed_above(dataframe['close'], dataframe['supertrend'])
        reject_st = qtpylib.crossed_below(dataframe['close'], dataframe['supertrend'])

        # MFI pressure
        mfi_long_ok = dataframe['mfi'] <= self.buy_mfi_long.value
        mfi_short_ok = dataframe['mfi'] >= self.buy_mfi_short.value

        # OBV confirmation: prefer OBV slope aligned with direction
        obv_up = dataframe['obv_slope'] > 0
        obv_dn = dataframe['obv_slope'] < 0

        # --- LONG ---
        # Structure (ST up OR reclaim) + HA bullish + MFI low-ish + OBV supportive + regime ok
        long_trigger = (st_bull | reclaim_st) & ha_bull & mfi_long_ok & obv_up
        dataframe.loc[thu & hours & liquid & long_regime_ok & long_trigger, 'enter_long'] = 1

        # --- SHORT ---
        # Structure (ST down OR reject) + HA bearish + MFI high-ish + OBV supportive + regime ok
        short_trigger = (st_bear | reject_st) & ha_bear & mfi_short_ok & obv_dn
        dataframe.loc[thu & hours & liquid & short_regime_ok & short_trigger, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Structured exits:
        # - Long exits when price falls back below supertrend OR HA flips bearish.
        # - Short exits when price rises above supertrend OR HA flips bullish.
        dataframe.loc[
            qtpylib.crossed_below(dataframe['close'], dataframe['supertrend']) | (dataframe['ha_close'] < dataframe['ha_open']),
            'exit_long'
        ] = 1

        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe['supertrend']) | (dataframe['ha_close'] > dataframe['ha_open']),
            'exit_short'
        ] = 1

        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # Hard time stop (both directions)
        max_minutes = int(self.max_trade_minutes.value)
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes >= max_minutes:
            return "time_stop"
        return None

    def supertrend(self, dataframe: DataFrame, period, multiplier):
        """
        Keeps your original supertrend logic, but returns both ST line and direction flag.
        Note: This is Python-loop based (slower). If you later want performance,
        we can replace it with a vectorized/indicator-library supertrend.
        """
        df = dataframe.copy()

        df['atr'] = ta.ATR(df, timeperiod=period)
        df['hl2'] = (df['high'] + df['low']) / 2

        df['upperband'] = df['hl2'] + (multiplier * df['atr'])
        df['lowerband'] = df['hl2'] - (multiplier * df['atr'])

        df['in_uptrend'] = True

        for current in range(1, len(df.index)):
            previous = current - 1

            if df['close'][current] > df['upperband'][previous]:
                df.loc[current, 'in_uptrend'] = True
            elif df['close'][current] < df['lowerband'][previous]:
                df.loc[current, 'in_uptrend'] = False
            else:
                df.loc[current, 'in_uptrend'] = df['in_uptrend'][previous]

                if df['in_uptrend'][current] and df['lowerband'][current] < df['lowerband'][previous]:
                    df.loc[current, 'lowerband'] = df['lowerband'][previous]
                if (not df['in_uptrend'][current]) and df['upperband'][current] > df['upperband'][previous]:
                    df.loc[current, 'upperband'] = df['upperband'][previous]

        st = df.apply(lambda row: row['lowerband'] if row['in_uptrend'] else row['upperband'], axis=1)

        return DataFrame({'ST': st, 'in_uptrend': df['in_uptrend']}, index=df.index)

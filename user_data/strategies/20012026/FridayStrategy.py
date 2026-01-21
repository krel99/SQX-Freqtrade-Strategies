from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


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

    # --- Trade frequency control (global) ---
    # NOTE: This is strategy-side. Also set max_open_trades in your config for certainty.
    max_open_trades = 2
    process_only_new_candles = True

    # Ensure enough candles for the longest lookback (EMA + weekly-like ATR baseline)
    startup_candle_count = 600

    # --- Risk / exits ---
    minimal_roi = {
        "0": 0.12,
        "60": 0.07,
        "180": 0.03
    }

    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    timeframe = '15m'

    # --- Hyperparameters ---

    # Friday-only (keep as a constant; no real hyperopt value, but keeping your structure)
    buy_day_of_week = CategoricalParameter([4], space='buy', default=4)  # 4 = Friday

    # Trading hours (local exchange time = candle timestamps; set to your preference)
    # Default tuned toward "later Friday" activity, but still broad enough to catch most Fridays.
    buy_hour_start = IntParameter(12, 20, default=16, space='buy')
    buy_hour_end = IntParameter(20, 23, default=23, space='buy')

    # Regime filter via long EMA on same timeframe (15m).
    # 200 on 15m ~ 50 hours. 400 ~ 100 hours. This mimics HTF trend.
    buy_ema_period = IntParameter(200, 500, default=300, space='buy')
    buy_ema_buffer = DecimalParameter(0.995, 1.010, default=1.000, space='buy')
    # buffer lets you require close >= EMA * buffer (slightly above/near)

    # Donchian (used as a momentum confirmation, correctly shifted)
    buy_dc_period = IntParameter(20, 80, default=40, space='buy')

    # Volatility expansion: ATR% now vs ATR% baseline
    buy_atr_period = IntParameter(10, 30, default=14, space='buy')
    buy_atr_baseline = IntParameter(96, 800, default=480, space='buy')  # ~5 days on 15m is 480
    buy_atr_mult = DecimalParameter(1.05, 1.60, default=1.20, space='buy')

    # Optional band breakout confirmation (stddev bands)
    buy_sdb_period = IntParameter(20, 80, default=40, space='buy')
    buy_sdb_stddev = DecimalParameter(1.5, 3.5, default=2.2, space='buy')

    # Liquidity filter (kept mild so you still trade most Fridays)
    buy_vol_sma_period = IntParameter(20, 80, default=30, space='buy')
    buy_vol_mult = DecimalParameter(0.30, 1.20, default=0.60, space='buy')

    # Exit safety: avoid holding into weekend (optional)
    exit_hour_start = IntParameter(20, 23, default=23, space='sell')

    @property
    def protections(self):
        # Helps enforce "1 trade per pair per Friday window" and reduces churn.
        # This DOES NOT guarantee only 1 trade total across all pairs—that's max_open_trades + pairlist.
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 80  # ~20 hours on 15m
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['hour'] = dataframe['date'].dt.hour

        # Long EMA regime filter (same timeframe)
        ema_period = int(self.buy_ema_period.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=ema_period)

        # Donchian - IMPORTANT: shift(1) so it's based on previous candles (no "impossible breakout")
        dc_period = int(self.buy_dc_period.value)
        dataframe['dc_upper'] = dataframe['high'].rolling(dc_period).max().shift(1)
        dataframe['dc_lower'] = dataframe['low'].rolling(dc_period).min().shift(1)

        # ATR% and ATR% baseline
        atr_period = int(self.buy_atr_period.value)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)
        dataframe['atrp'] = dataframe['atr'] / dataframe['close']

        baseline = int(self.buy_atr_baseline.value)
        dataframe['atrp_base'] = dataframe['atrp'].rolling(baseline).mean()

        # Stddev bands for “event candle” / expansion confirmation
        sdb_period = int(self.buy_sdb_period.value)
        sdb_std = float(self.buy_sdb_stddev.value)
        sma = ta.SMA(dataframe, timeperiod=sdb_period)
        std = ta.STDDEV(dataframe, timeperiod=sdb_period)
        dataframe['sdb_upper'] = sma + (std * sdb_std)
        dataframe['sdb_lower'] = sma - (std * sdb_std)

        # Mild liquidity sanity (don’t make it too strict if you want “almost every Friday”)
        volp = int(self.buy_vol_sma_period.value)
        dataframe['vol_sma'] = dataframe['volume'].rolling(volp).mean()

        # Candle range (simple expansion proxy)
        dataframe['range'] = (dataframe['high'] - dataframe['low']) / dataframe['close']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core Friday + hours gate
        friday = dataframe['day_of_week'] == self.buy_day_of_week.value
        hours = (dataframe['hour'] >= self.buy_hour_start.value) & (dataframe['hour'] <= self.buy_hour_end.value)

        # Regime filter (same timeframe, long period)
        # Slight buffer allows “near EMA” or “above EMA”
        buffer = float(self.buy_ema_buffer.value)
        regime_ok = dataframe['close'] >= (dataframe['ema_long'] * buffer)

        # Mild liquidity filter
        vol_mult = float(self.buy_vol_mult.value)
        liquid = (dataframe['volume'] > 0) & (dataframe['volume'] >= dataframe['vol_sma'] * vol_mult)

        # Volatility expansion: ATR% spikes vs its baseline
        atr_mult = float(self.buy_atr_mult.value)
        vol_event = dataframe['atrp'] >= (dataframe['atrp_base'] * atr_mult)

        # Momentum confirmation (NOT too strict):
        # - Either a Donchian breakout (cross above previous upper),
        # - Or a stddev-band push (close above upper band),
        # - Or simply a strong candle (range expansion) while vol_event is true.
        dc_break = qtpylib.crossed_above(dataframe['close'], dataframe['dc_upper'])
        band_push = dataframe['close'] > dataframe['sdb_upper']
        range_push = dataframe['range'] > dataframe['range'].rolling(96).mean()  # mild "bigger than usual today"

        trigger = vol_event & (dc_break | band_push | range_push)

        dataframe.loc[friday & hours & liquid & regime_ok & trigger, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Optional: if you want to avoid holding into weekend, exit late Friday.
        # This is a "soft" exit signal; ROI/trailing/stoploss still apply.
        late_friday = (dataframe['day_of_week'] == 4) & (dataframe['hour'] >= self.exit_hour_start.value)
        dataframe.loc[late_friday, 'exit_long'] = 1
        return dataframe

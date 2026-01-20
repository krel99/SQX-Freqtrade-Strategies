# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce


class MondayStrategy(IStrategy):
    """
    Monday-only futures strategy (long + short) designed to reliably take ~1 trade most Mondays.
    No informative timeframes; uses long-period EMA on 15m as a regime filter.

    Long idea: dip + recovery in non-bear regime.
    Short idea: pop + rejection in non-bull regime.
    """

    INTERFACE_VERSION = 3
    can_short = True

    max_open_trades = 2
    process_only_new_candles = True

    startup_candle_count = 700
    timeframe = '15m'

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

    buy_day_of_week = CategoricalParameter([0], space='buy', default=0)  # Monday

    buy_hour_start = IntParameter(0, 12, default=6, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Same-timeframe "HTF-like" regime EMA
    ema_regime_period = IntParameter(200, 600, default=400, space='buy')

    # Regime buffers:
    # Long allowed when close >= EMA * long_buffer
    # Short allowed when close <= EMA * short_buffer
    long_regime_buffer = DecimalParameter(0.985, 1.010, default=0.995, space='buy')
    short_regime_buffer = DecimalParameter(0.990, 1.020, default=1.005, space='buy')

    # RSI / Stoch thresholds
    rsi_period = IntParameter(10, 30, default=14, space='buy')
    long_rsi = IntParameter(25, 55, default=42, space='buy')
    short_rsi = IntParameter(45, 80, default=58, space='buy')

    stoch_k = IntParameter(5, 20, default=14, space='buy')
    stoch_d = IntParameter(1, 10, default=3, space='buy')
    long_stoch = IntParameter(10, 60, default=35, space='buy')
    short_stoch = IntParameter(40, 90, default=65, space='buy')

    # Bollinger
    bb_period = IntParameter(10, 40, default=20, space='buy')
    bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space='buy')

    # MACD (mild confirmation / turn)
    macd_fast = IntParameter(10, 20, default=12, space='buy')
    macd_slow = IntParameter(20, 40, default=26, space='buy')
    macd_signal = IntParameter(5, 15, default=9, space='buy')

    # Liquidity filter (mild)
    vol_sma_period = IntParameter(20, 80, default=30, space='buy')
    vol_mult = DecimalParameter(0.20, 1.20, default=0.50, space='buy')

    # Trade duration hard stop
    max_trade_minutes = IntParameter(120, 1440, default=480, space='sell')  # 8 hours default

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 80}  # ~20 hours on 15m
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['hour'] = dataframe['date'].dt.hour

        # Regime EMA
        p = int(self.ema_regime_period.value)
        dataframe['ema_regime'] = ta.EMA(dataframe, timeperiod=p)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))

        # Stoch
        st = ta.STOCH(
            dataframe,
            fastk_period=int(self.stoch_k.value),
            slowk_period=3,
            slowd_period=int(self.stoch_d.value)
        )
        dataframe['slowk'] = st['slowk']
        dataframe['slowd'] = st['slowd']

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=int(self.macd_fast.value),
            slowperiod=int(self.macd_slow.value),
            signalperiod=int(self.macd_signal.value)
        )
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger
        bb = ta.BBANDS(
            dataframe,
            timeperiod=int(self.bb_period.value),
            nbdevup=float(self.bb_stddev.value),
            nbdevdn=float(self.bb_stddev.value)
        )
        dataframe['bb_lowerband'] = bb['lowerband']
        dataframe['bb_middleband'] = bb['middleband']
        dataframe['bb_upperband'] = bb['upperband']

        # Liquidity baseline
        vsp = int(self.vol_sma_period.value)
        dataframe['vol_sma'] = dataframe['volume'].rolling(vsp).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        monday = dataframe['day_of_week'] == self.buy_day_of_week.value
        hours = (dataframe['hour'] >= self.buy_hour_start.value) & (dataframe['hour'] <= self.buy_hour_end.value)

        liquid = (dataframe['volume'] > 0) & (dataframe['volume'] >= dataframe['vol_sma'] * float(self.vol_mult.value))

        # --- LONG ---
        long_regime_ok = dataframe['close'] >= (dataframe['ema_regime'] * float(self.long_regime_buffer.value))

        long_dip = (
            (dataframe['close'] <= dataframe['bb_lowerband']) |
            (dataframe['rsi'] <= self.long_rsi.value) |
            ((dataframe['slowk'] <= self.long_stoch.value) & (dataframe['slowd'] <= self.long_stoch.value))
        )

        long_reclaim = (
            qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband']) |
            qtpylib.crossed_above(dataframe['close'], dataframe['bb_middleband']) |
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1))
        )

        dataframe.loc[monday & hours & liquid & long_regime_ok & long_dip & long_reclaim, 'enter_long'] = 1

        # --- SHORT ---
        short_regime_ok = dataframe['close'] <= (dataframe['ema_regime'] * float(self.short_regime_buffer.value))

        short_pop = (
            (dataframe['close'] >= dataframe['bb_upperband']) |
            (dataframe['rsi'] >= self.short_rsi.value) |
            ((dataframe['slowk'] >= self.short_stoch.value) & (dataframe['slowd'] >= self.short_stoch.value))
        )

        short_reject = (
            qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband']) |
            qtpylib.crossed_below(dataframe['close'], dataframe['bb_middleband']) |
            (dataframe['macdhist'] < dataframe['macdhist'].shift(1))
        )

        dataframe.loc[monday & hours & liquid & short_regime_ok & short_pop & short_reject, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Mean-reversion exits
        dataframe.loc[dataframe['close'] > dataframe['bb_middleband'], 'exit_long'] = 1
        dataframe.loc[dataframe['close'] < dataframe['bb_middleband'], 'exit_short'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        Hard time stop for BOTH long and short.
        """
        max_minutes = int(self.max_trade_minutes.value)
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes >= max_minutes:
            return "time_stop"
        return None

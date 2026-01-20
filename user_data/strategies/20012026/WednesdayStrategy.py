# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce

class WednesdayStrategy(IStrategy):
    """
    This strategy is designed to trade only on Wednesdays, using less common indicators.
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.16,
        "30": 0.11,
        "60": 0.07
    }

    # Stoploss:
    stoploss = -0.11

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # --- Hyperparameters ---

    # Day of week
    buy_day_of_week = CategoricalParameter([2], space='buy', default=2) # 2 = Wednesday

    # Trading hours
    buy_hour_start = IntParameter(0, 12, default=0, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # ZLEMA
    buy_zlema_period = IntParameter(10, 30, default=20, space='buy')

    # HMA
    buy_hma_period = IntParameter(10, 30, default=20, space='buy')

    # Ultimate Oscillator
    buy_uo_period1 = IntParameter(5, 15, default=7, space='buy')
    buy_uo_period2 = IntParameter(10, 20, default=14, space='buy')
    buy_uo_period3 = IntParameter(20, 40, default=28, space='buy')
    buy_uo_threshold = IntParameter(30, 60, default=50, space='buy')

    # Awesome Oscillator
    buy_ao_threshold = DecimalParameter(0.0, 5.0, default=0.0, space='buy')

    # ATR
    buy_atr_period = IntParameter(10, 20, default=14, space='buy')
    buy_atr_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Day of week
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek

        # ZLEMA
        dataframe['zlema'] = self.zlema(dataframe, period=self.buy_zlema_period.value)

        # HMA
        dataframe['hma'] = qtpylib.hma(dataframe, period=self.buy_hma_period.value)

        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe,
                                    timeperiod1=self.buy_uo_period1.value,
                                    timeperiod2=self.buy_uo_period2.value,
                                    timeperiod3=self.buy_uo_period3.value)

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.buy_atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Day of week filter
        conditions.append(dataframe['day_of_week'] == self.buy_day_of_week.value)

        # Trading hours filter
        conditions.append(dataframe['date'].dt.hour >= self.buy_hour_start.value)
        conditions.append(dataframe['date'].dt.hour <= self.buy_hour_end.value)

        # ZLEMA and HMA
        conditions.append(qtpylib.crossed_above(dataframe['zlema'], dataframe['hma']))

        # Ultimate Oscillator
        conditions.append(dataframe['uo'] < self.buy_uo_threshold.value)

        # Awesome Oscillator
        conditions.append(dataframe['ao'] > self.buy_ao_threshold.value)

        # ATR
        conditions.append(dataframe['close'] > dataframe['close'].shift(1) + (dataframe['atr'] * self.buy_atr_multiplier.value))

        if conditions:
            dataframe.loc[
                reduce(lambda a, b: a & b, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def zlema(self, dataframe: DataFrame, period: int) -> DataFrame:
        """
        Calculates the Zero Lag Exponential Moving Average (ZLEMA).
        """
        lag = (period - 1) // 2
        ema = ta.EMA(dataframe['close'], timeperiod=period)
        ema_ema = ta.EMA(ema, timeperiod=period)
        zlema = ema + (ema - ema_ema)
        return zlema

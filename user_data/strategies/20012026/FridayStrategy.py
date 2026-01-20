# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce

class FridayStrategy(IStrategy):
    """
    This strategy is designed to trade only on Fridays, using a mix of channels and oscillators.
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.19,
        "30": 0.14,
        "60": 0.10
    }

    # Stoploss:
    stoploss = -0.14

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # --- Hyperparameters ---

    # Day of week
    buy_day_of_week = CategoricalParameter([4], space='buy', default=4) # 4 = Friday

    # Trading hours
    buy_hour_start = IntParameter(0, 12, default=0, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Donchian Channels
    buy_dc_period = IntParameter(10, 30, default=20, space='buy')

    # Williams %R
    buy_willr_period = IntParameter(10, 20, default=14, space='buy')
    buy_willr_threshold = IntParameter(-90, -70, default=-80, space='buy')

    # Commodity Channel Index (CCI)
    buy_cci_period = IntParameter(10, 20, default=14, space='buy')
    buy_cci_threshold = IntParameter(-120, -80, default=-100, space='buy')

    # Envelopes
    buy_env_period = IntParameter(10, 30, default=20, space='buy')
    buy_env_width = DecimalParameter(0.01, 0.05, default=0.02, space='buy')

    # Standard Deviation Bands
    buy_sdb_period = IntParameter(10, 30, default=20, space='buy')
    buy_sdb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Day of week
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek

        # Donchian Channels
        dc = self.donchian(dataframe, period=self.buy_dc_period.value)
        dataframe['dc_upper'] = dc['upper']
        dataframe['dc_lower'] = dc['lower']

        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=self.buy_willr_period.value)

        # Commodity Channel Index (CCI)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=self.buy_cci_period.value)

        # Envelopes
        env = self.envelope(dataframe, period=self.buy_env_period.value, width=self.buy_env_width.value)
        dataframe['env_upper'] = env['upper']
        dataframe['env_lower'] = env['lower']

        # Standard Deviation Bands
        sdb = self.stddev_bands(dataframe, period=self.buy_sdb_period.value, stddev=self.buy_sdb_stddev.value)
        dataframe['sdb_upper'] = sdb['upper']
        dataframe['sdb_lower'] = sdb['lower']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Day of week filter
        conditions.append(dataframe['day_of_week'] == self.buy_day_of_week.value)

        # Trading hours filter
        conditions.append(dataframe['date'].dt.hour >= self.buy_hour_start.value)
        conditions.append(dataframe['date'].dt.hour <= self.buy_hour_end.value)

        # Donchian Channels
        conditions.append(dataframe['close'] > dataframe['dc_upper'])

        # Williams %R
        conditions.append(dataframe['willr'] < self.buy_willr_threshold.value)

        # Commodity Channel Index (CCI)
        conditions.append(dataframe['cci'] < self.buy_cci_threshold.value)

        if conditions:
            dataframe.loc[
                reduce(lambda a, b: a & b, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def donchian(self, dataframe: DataFrame, period: int):
        upper = dataframe['high'].rolling(window=period).max()
        lower = dataframe['low'].rolling(window=period).min()
        return {'upper': upper, 'lower': lower}

    def envelope(self, dataframe: DataFrame, period: int, width: float):
        sma = ta.SMA(dataframe, timeperiod=period)
        upper = sma * (1 + width)
        lower = sma * (1 - width)
        return {'upper': upper, 'lower': lower}

    def stddev_bands(self, dataframe: DataFrame, period: int, stddev: float):
        sma = ta.SMA(dataframe, timeperiod=period)
        std = ta.STDDEV(dataframe, timeperiod=period)
        upper = sma + (std * stddev)
        lower = sma - (std * stddev)
        return {'upper': upper, 'lower': lower}

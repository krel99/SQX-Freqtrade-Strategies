# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import ehlers_fisher_transform, ehlers_mama

# --- Strategy specific imports ---
from datetime import datetime
from functools import reduce

class TuesdayStrategy(IStrategy):
    """
    This strategy is designed to trade only on Tuesdays, using Ehlers indicators.
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.18,
        "30": 0.12,
        "60": 0.08
    }

    # Stoploss:
    stoploss = -0.12

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # --- Hyperparameters ---

    # Day of week
    buy_day_of_week = CategoricalParameter([1], space='buy', default=1) # 1 = Tuesday

    # Trading hours
    buy_hour_start = IntParameter(0, 12, default=0, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Ehlers Fisher Transform
    buy_eft_period = IntParameter(5, 20, default=10, space='buy')
    buy_eft_threshold = DecimalParameter(-1.0, 1.0, default=0.0, space='buy')

    # Ehlers MAMA
    buy_mama_fastlimit = DecimalParameter(0.1, 0.9, default=0.5, space='buy')
    buy_mama_slowlimit = DecimalParameter(0.01, 0.09, default=0.05, space='buy')

    # Keltner Channels
    buy_kc_period = IntParameter(10, 30, default=20, space='buy')
    buy_kc_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='buy')

    # Choppiness Index
    buy_chop_period = IntParameter(10, 20, default=14, space='buy')
    buy_chop_threshold = IntParameter(30, 70, default=50, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Day of week
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek

        # Ehlers Fisher Transform
        fisher = ehlers_fisher_transform(dataframe, period=self.buy_eft_period.value)
        dataframe['fisher_transform'] = fisher['fisher_transform']
        dataframe['fisher_signal'] = fisher['fisher_signal']

        # Ehlers MAMA
        mama = ehlers_mama(dataframe, fastlimit=self.buy_mama_fastlimit.value, slowlimit=self.buy_mama_slowlimit.value)
        dataframe['mama'] = mama['mama']
        dataframe['fama'] = mama['fama']

        # Keltner Channels
        keltner = self.keltner_channels(dataframe, period=self.buy_kc_period.value, multiplier=self.buy_kc_multiplier.value)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_lowerband'] = keltner['lower']

        # Choppiness Index
        dataframe['chop'] = self.choppiness(dataframe, period=self.buy_chop_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Day of week filter
        conditions.append(dataframe['day_of_week'] == self.buy_day_of_week.value)

        # Trading hours filter
        conditions.append(dataframe['date'].dt.hour >= self.buy_hour_start.value)
        conditions.append(dataframe['date'].dt.hour <= self.buy_hour_end.value)

        # Ehlers Fisher Transform
        conditions.append(dataframe['fisher_transform'] > self.buy_eft_threshold.value)

        # Ehlers MAMA
        conditions.append(qtpylib.crossed_above(dataframe['mama'], dataframe['fama']))

        # Keltner Channels
        conditions.append(dataframe['close'] < dataframe['kc_lowerband'])

        # Choppiness Index
        conditions.append(dataframe['chop'] > self.buy_chop_threshold.value)

        if conditions:
            dataframe.loc[
                reduce(lambda a, b: a & b, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def keltner_channels(self, dataframe: DataFrame, period: int, multiplier: float):
        ema = ta.EMA(dataframe, timeperiod=period)
        atr = ta.ATR(dataframe, timeperiod=period)

        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)

        return {'upper': upper, 'lower': lower}

    def choppiness(self, dataframe: DataFrame, period: int) -> DataFrame:
        """
        Calculates the Choppiness Index.
        """
        atr_sum = ta.ATR(dataframe, timeperiod=1).rolling(window=period).sum()
        high = dataframe['high'].rolling(window=period).max()
        low = dataframe['low'].rolling(window=period).min()

        chop = 100 * (atr_sum / (high - low))
        return chop

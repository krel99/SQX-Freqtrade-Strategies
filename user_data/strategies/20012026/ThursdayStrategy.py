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
    This strategy is designed to trade only on Thursdays, focusing on price and volume analysis.
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.17,
        "30": 0.13,
        "60": 0.09
    }

    # Stoploss:
    stoploss = -0.13

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.028
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # --- Hyperparameters ---

    # Day of week
    buy_day_of_week = CategoricalParameter([3], space='buy', default=3) # 3 = Thursday

    # Trading hours
    buy_hour_start = IntParameter(0, 12, default=0, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # Supertrend
    buy_st_period = IntParameter(7, 20, default=10, space='buy')
    buy_st_multiplier = IntParameter(2, 5, default=3, space='buy')

    # Money Flow Index (MFI)
    buy_mfi_period = IntParameter(10, 20, default=14, space='buy')
    buy_mfi_threshold = IntParameter(20, 40, default=30, space='buy')

    # On-Balance Volume (OBV)
    buy_obv_divergence_period = IntParameter(10, 30, default=20, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Day of week
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek

        # Supertrend
        st = self.supertrend(dataframe, self.buy_st_period.value, self.buy_st_multiplier.value)
        dataframe['supertrend'] = st['ST']

        # Money Flow Index (MFI)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=self.buy_mfi_period.value)

        # On-Balance Volume (OBV)
        dataframe['obv'] = ta.OBV(dataframe)

        # Heikin-Ashi
        heikin_ashi_df = heikin_ashi(dataframe)
        dataframe['ha_close'] = heikin_ashi_df['close']
        dataframe['ha_open'] = heikin_ashi_df['open']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Day of week filter
        conditions.append(dataframe['day_of_week'] == self.buy_day_of_week.value)

        # Trading hours filter
        conditions.append(dataframe['date'].dt.hour >= self.buy_hour_start.value)
        conditions.append(dataframe['date'].dt.hour <= self.buy_hour_end.value)

        # Supertrend
        conditions.append(dataframe['close'] > dataframe['supertrend'])

        # Money Flow Index (MFI)
        conditions.append(dataframe['mfi'] < self.buy_mfi_threshold.value)

        # On-Balance Volume (OBV)
        obv_divergence = (dataframe['obv'] > dataframe['obv'].shift(self.buy_obv_divergence_period.value)) & \
                         (dataframe['close'] < dataframe['close'].shift(self.buy_obv_divergence_period.value))
        conditions.append(obv_divergence)

        # Heikin-Ashi
        conditions.append(dataframe['ha_close'] > dataframe['ha_open'])

        if conditions:
            dataframe.loc[
                reduce(lambda a, b: a & b, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def supertrend(self, dataframe: DataFrame, period, multiplier):

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
                if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                    df.loc[current, 'upperband'] = df['upperband'][previous]

        st = df.apply(lambda row: row['lowerband'] if row['in_uptrend'] else row['upperband'], axis=1)

        return DataFrame({'ST': st}, index=df.index)

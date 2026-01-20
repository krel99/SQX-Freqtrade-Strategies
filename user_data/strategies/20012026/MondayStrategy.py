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
    This strategy is designed to trade only on Mondays, using a combination of indicators.
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.15,
        "30": 0.1,
        "60": 0.05
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # --- Hyperparameters ---

    # Day of week
    buy_day_of_week = CategoricalParameter([0], space='buy', default=0) # 0 = Monday

    # Trading hours
    buy_hour_start = IntParameter(0, 12, default=0, space='buy')
    buy_hour_end = IntParameter(13, 23, default=23, space='buy')

    # EMA Cross
    buy_ema_short_period = IntParameter(5, 20, default=10, space='buy')
    buy_ema_long_period = IntParameter(20, 50, default=25, space='buy')

    # Ichimoku Cloud
    buy_ichimoku_span_a_period = IntParameter(20, 50, default=26, space='buy')
    buy_ichimoku_span_b_period = IntParameter(50, 100, default=52, space='buy')
    buy_ichimoku_kijun_sen_period = IntParameter(20, 50, default=26, space='buy')

    # RSI
    buy_rsi_period = IntParameter(10, 30, default=14, space='buy')
    buy_rsi_value = IntParameter(20, 40, default=30, space='buy')

    # Stochastic
    buy_stoch_k = IntParameter(5, 20, default=14, space='buy')
    buy_stoch_d = IntParameter(1, 10, default=3, space='buy')
    buy_stoch_value = IntParameter(10, 40, default=20, space='buy')

    # MACD
    buy_macd_fast = IntParameter(10, 20, default=12, space='buy')
    buy_macd_slow = IntParameter(20, 40, default=26, space='buy')
    buy_macd_signal = IntParameter(5, 15, default=9, space='buy')

    # Bollinger Bands
    buy_bb_period = IntParameter(10, 30, default=20, space='buy')
    buy_bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Day of week
        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek

        # EMA Cross
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short_period.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long_period.value)

        # Ichimoku Cloud
        ichimoku = self.ichimoku(dataframe,
                                conversion_line_period=9,
                                base_line_periods=self.buy_ichimoku_kijun_sen_period.value,
                                lagging_span_2_periods=self.buy_ichimoku_span_b_period.value,
                                displacement=self.buy_ichimoku_span_a_period.value)
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_span_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_span_b'] = ichimoku['senkou_span_b']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)

        # Stochastic
        stoch = ta.STOCH(dataframe,
                         fastk_period=self.buy_stoch_k.value,
                         slowk_period=3,
                         slowd_period=self.buy_stoch_d.value)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # MACD
        macd = ta.MACD(dataframe,
                       fastperiod=self.buy_macd_fast.value,
                       slowperiod=self.buy_macd_slow.value,
                       signalperiod=self.buy_macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe,
                              timeperiod=self.buy_bb_period.value,
                              nbdevup=self.buy_bb_stddev.value,
                              nbdevdn=self.buy_bb_stddev.value)
        dataframe['bb_lowerband'] = bollinger['lowerband']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Day of week filter
        conditions.append(dataframe['day_of_week'] == self.buy_day_of_week.value)

        # Trading hours filter
        conditions.append(dataframe['date'].dt.hour >= self.buy_hour_start.value)
        conditions.append(dataframe['date'].dt.hour <= self.buy_hour_end.value)

        # EMA Cross
        conditions.append(qtpylib.crossed_above(dataframe['ema_short'], dataframe['ema_long']))

        # Ichimoku Cloud
        conditions.append(dataframe['close'] > dataframe['senkou_span_a'])
        conditions.append(dataframe['close'] > dataframe['senkou_span_b'])

        # RSI
        conditions.append(dataframe['rsi'] < self.buy_rsi_value.value)

        # Stochastic
        conditions.append(dataframe['slowk'] < self.buy_stoch_value.value)
        conditions.append(dataframe['slowd'] < self.buy_stoch_value.value)

        # MACD
        conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        # Bollinger Bands
        conditions.append(dataframe['close'] < dataframe['bb_lowerband'])

        if conditions:
            dataframe.loc[
                reduce(lambda a, b: a & b, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def ichimoku(self, dataframe: DataFrame, conversion_line_period: int, base_line_periods: int, lagging_span_2_periods: int, displacement: int):

        # Tenkan-sen (Conversion Line)
        period9_high = dataframe['high'].rolling(window=conversion_line_period).max()
        period9_low = dataframe['low'].rolling(window=conversion_line_period).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line)
        period26_high = dataframe['high'].rolling(window=base_line_periods).max()
        period26_low = dataframe['low'].rolling(window=base_line_periods).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Senkou Span B (Leading Span B)
        period52_high = dataframe['high'].rolling(window=lagging_span_2_periods).max()
        period52_low = dataframe['low'].rolling(window=lagging_span_2_periods).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(displacement)

        return {'tenkan_sen': tenkan_sen, 'kijun_sen': kijun_sen, 'senkou_span_a': senkou_span_a, 'senkou_span_b': senkou_span_b}

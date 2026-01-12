# --- Do not remove these libs ---
from typing import Dict, List
from pandas import DataFrame
import numpy as np
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
# --------------------------------

class FractalStrategy(IStrategy):
    """
    This is a strategy based on 3-candle and 5-candle fractals.
    """
    # Strategy interface version - attribute needed by Freqtrade
    INTERFACE_VERSION = 3
    can_short: bool = True

    fractal_window = CategoricalParameter([3, 5], default=3, space='buy')
    breakout_threshold = DecimalParameter(0.01, 0.05, default=0.02, space='buy')
    ma_period = IntParameter(50, 200, default=100, space='buy')

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.04
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # Timeframe
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fractals
        window_size = self.fractal_window.value
        mid_point = (window_size - 1) // 2

        # High Fractals (Tops)
        is_fractal_top = (
            dataframe['high'].rolling(window=window_size, center=True).max() == dataframe['high']
        )
        dataframe['fractal_top'] = np.where(is_fractal_top, dataframe['high'], np.nan)
        dataframe['fractal_top'].fillna(method='ffill', inplace=True)

        # Low Fractals (Bottoms)
        is_fractal_bottom = (
            dataframe['low'].rolling(window=window_size, center=True).min() == dataframe['low']
        )
        dataframe['fractal_bottom'] = np.where(is_fractal_bottom, dataframe['low'], np.nan)
        dataframe['fractal_bottom'].fillna(method='ffill', inplace=True)

        # Moving Average
        dataframe['ema_long'] = dataframe['close'].ewm(span=self.ma_period.value, adjust=False).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry
        dataframe['enter_long_signal'] = (
            (dataframe['close'] > dataframe['fractal_top'] * (1 + self.breakout_threshold.value)) &
            (dataframe['close'] > dataframe['ema_long'])
        )
        dataframe.loc[dataframe['enter_long_signal'], 'enter_long'] = 1
        dataframe.loc[dataframe['enter_long_signal'], 'fractal_top_entry'] = dataframe['fractal_top']
        dataframe.loc[dataframe['enter_long_signal'], 'fractal_bottom_entry'] = dataframe['fractal_bottom']

        # Short entry
        dataframe['enter_short_signal'] = (
            (dataframe['close'] < dataframe['fractal_bottom'] * (1 - self.breakout_threshold.value)) &
            (dataframe['close'] < dataframe['ema_long'])
        )
        dataframe.loc[dataframe['enter_short_signal'], 'enter_short'] = 1
        dataframe.loc[dataframe['enter_short_signal'], 'fractal_top_entry'] = dataframe['fractal_top']
        dataframe.loc[dataframe['enter_short_signal'], 'fractal_bottom_entry'] = dataframe['fractal_bottom']

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No specific exit trend logic, exits are handled by custom_exit and custom_stoploss
        return dataframe
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_entry_candle = dataframe.iloc[trade.open_tick_index]

        fractal_top_entry = trade_entry_candle['fractal_top_entry']
        fractal_bottom_entry = trade_entry_candle['fractal_bottom_entry']

        corridor_width = fractal_top_entry - fractal_bottom_entry

        if trade.is_long:
            take_profit_price = trade.open_rate + (corridor_width * 4)
            if current_rate >= take_profit_price:
                return 'take_profit'
        else:
            take_profit_price = trade.open_rate - (corridor_width * 4)
            if current_rate <= take_profit_price:
                return 'take_profit'

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_entry_candle = dataframe.iloc[trade.open_tick_index]

        fractal_top_entry = trade_entry_candle['fractal_top_entry']
        fractal_bottom_entry = trade_entry_candle['fractal_bottom_entry']

        if trade.is_long:
            stoploss_price = fractal_bottom_entry
            return (trade.open_rate - stoploss_price) / trade.open_rate
        else:
            stoploss_price = fractal_top_entry
            return (stoploss_price - trade.open_rate) / trade.open_rate

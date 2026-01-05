# --- Do not remove these imports ---
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy

class Strategy05012026(IStrategy):
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = '15m'

    # Can short
    can_short = True

    # Minimal ROI
    minimal_roi = {
        "0": 0.02
    }

    # Stoploss
    stoploss = -0.005

    # Trailing stop
    trailing_stop = False

    # Disable exit signal
    use_exit_signal = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=30)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Long entry
        dataframe.loc[
            (
                (dataframe['rsi'] > 47) &
                (dataframe['rsi'].shift(1) < dataframe['rsi']) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) &
                (dataframe['macdhist'] > 0) &
                (dataframe['close'] > dataframe['ema9'])
            ),
            'enter_long'] = 1

        # Short entry
        dataframe.loc[
            (
                (dataframe['rsi'] < 50) &
                (dataframe['rsi'].shift(1) > dataframe['rsi']) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
                (dataframe['macdhist'] < 0) &
                (dataframe['close'] < dataframe['ema9'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # This strategy does not use exit signals
        return dataframe

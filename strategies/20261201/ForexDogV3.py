# --- Do not remove these libs ---
from freqtrade.strategy.hyper import IntParameter
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------


class ForexDogBase(IStrategy):
    # Base strategy for ForexDog variations

    # Hyperparameters
    buy_params = {}
    sell_params = {}

    # EMA periods
    ema_p1 = IntParameter(3, 12, default=5, space='buy')
    ema_p2 = IntParameter(13, 27, default=20, space='buy')
    ema_p3 = IntParameter(28, 45, default=40, space='buy')
    ema_p4 = IntParameter(46, 65, default=50, space='buy')
    ema_p5 = IntParameter(66, 90, default=80, space='buy')
    ema_p6 = IntParameter(91, 140, default=100, space='buy')
    ema_p7 = IntParameter(141, 300, default=200, space='buy')
    ema_p8 = IntParameter(301, 520, default=400, space='buy')
    ema_p9 = IntParameter(521, 1120, default=640, space='buy')
    ema_p10 = IntParameter(1121, 1760, default=1600, space='buy')
    ema_p11 = IntParameter(1761, 2560, default=1920, space='buy')
    ema_p12 = IntParameter(2561, 4000, default=3200, space='buy')

    # ATR period for stoploss
    atr_period = IntParameter(10, 20, default=14, space='buy')

    # Stoploss ATR multiplier
    atr_multiplier = IntParameter(1, 5, default=2, space='buy')

    # Time-based exit
    max_trade_duration = IntParameter(100, 400, default=200, space='buy')

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # Trailing stoploss
    trailing_stop = False

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 10
    }

    # Stoploss
    stoploss = -0.99

    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 4000

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate all 12 EMAs
        for i in range(1, 13):
            p_val = getattr(self, f"ema_p{i}").value
            dataframe[f'ema_{i}'] = ta.EMA(dataframe, timeperiod=p_val)

        # ATR for stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # RSI for V2
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Volume MA for V3
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)

        return dataframe


class ForexDogV3(ForexDogBase):
    # ForexDog Variation 3: Volatility-Adapted Entry

    # Use custom stoploss
    use_custom_stoploss = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        conditions = [
            (df['close'] > df['ema_1']) &
            (df['close'] > df['ema_2']) &
            (df['close'] > df['ema_3']) &
            (df['close'] > df['ema_4']) &
            (df['close'] > df['ema_5']) &
            (df['close'] > df['ema_6']) &
            (df['close'] > df['ema_7']) &
            (df['close'] > df['ema_8']) &
            (df['close'] > df['ema_9']) &
            (df['close'] > df['ema_10']),

            qtpylib.crossed_above(df['close'], df['ema_5']),

            (df['ema_11'] - df['close']) / df['close'] > 0.02,

            df['atr'] > (df['atr'].rolling(20).mean() * 1.2),

            df['volume'] > df['volume_ma'],
        ]

        df.loc[
            (
                (conditions[0]) &
                (conditions[1]) &
                (conditions[2]) &
                (conditions[3]) &
                (conditions[4])
            ),
            'enter_long'] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # Take profit when price touches the next slow EMA
        df.loc[
            (qtpylib.crossed_above(df['close'], df['ema_11'])),
            'exit_long'] = 1

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # ATR-based stoploss
        stoploss_price = last_candle['ema_6'] - (last_candle['atr'] * self.atr_multiplier.value)

        # Calculate stoploss percentage
        stoploss_pct = (stoploss_price - current_rate) / current_rate
        return stoploss_pct

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Time-based exit for losing trades
        timeframe_minutes = int(self.timeframe[:-1])
        if (current_time - trade.open_date_utc).total_seconds() / 60 > (self.max_trade_duration.value * timeframe_minutes) and current_profit < 0:
            return 'time_based_exit'

        return None

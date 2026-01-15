# --- Do not remove these libs ---
from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, IStrategy


# --------------------------------


class ForexDogBase(IStrategy):
    """
    Base strategy for ForexDog variations
    Contains all common parameters and indicator calculations
    """

    # Hyperparameters
    buy_params = {}
    sell_params = {}

    # EMA periods (buy space)
    ema_p1 = IntParameter(3, 12, default=5, space="buy")
    ema_p2 = IntParameter(13, 27, default=20, space="buy")
    ema_p3 = IntParameter(28, 45, default=40, space="buy")
    ema_p4 = IntParameter(46, 65, default=50, space="buy")
    ema_p5 = IntParameter(66, 90, default=80, space="buy")
    ema_p6 = IntParameter(91, 140, default=100, space="buy")
    ema_p7 = IntParameter(141, 300, default=200, space="buy")
    ema_p8 = IntParameter(301, 520, default=400, space="buy")
    ema_p9 = IntParameter(521, 1120, default=640, space="buy")
    ema_p10 = IntParameter(1121, 1760, default=1600, space="buy")
    ema_p11 = IntParameter(1761, 2560, default=1920, space="buy")
    ema_p12 = IntParameter(2561, 4000, default=3200, space="buy")

    # ATR period for stoploss
    atr_period = IntParameter(10, 20, default=14, space="buy")

    # Stoploss ATR multiplier
    atr_multiplier = IntParameter(1, 5, default=2, space="buy")

    # Time-based exit (buy space)
    max_trade_duration = IntParameter(100, 400, default=200, space="buy")

    # Exit/Sell space parameters for hyperopt
    exit_ema_num = IntParameter(1, 12, default=7, space="sell")  # Which EMA to use for exit
    exit_profit_threshold = IntParameter(
        0, 50, default=10, space="sell"
    )  # Min profit % to exit (in 0.1% increments)
    exit_atr_multiplier = IntParameter(
        1, 5, default=3, space="sell"
    )  # ATR multiplier for trailing exit
    exit_rsi_threshold = IntParameter(
        50, 80, default=70, space="sell"
    )  # RSI threshold for overbought exit
    exit_time_multiplier = IntParameter(
        50, 300, default=150, space="sell"
    )  # Time-based exit multiplier for losing trades

    # Optimal timeframe for the strategy
    timeframe = "15m"

    # Trailing stoploss
    trailing_stop = False

    # Minimal ROI designed for the strategy.
    minimal_roi = {"0": 10}

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
        """
        Populate indicators that will be used in the strategy.
        This method is called for each pair with each timeframe.
        """
        # Populate all 12 EMAs
        for i in range(1, 13):
            p_val = getattr(self, f"ema_p{i}").value
            dataframe[f"ema_{i}"] = ta.EMA(dataframe, timeperiod=p_val)

        # ATR for stoploss
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # RSI for V2
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Volume MA for V3
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        This method should be overridden in child classes
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        This method should be overridden in child classes
        """
        return dataframe

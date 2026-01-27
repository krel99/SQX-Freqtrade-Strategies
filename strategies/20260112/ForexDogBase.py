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

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
    Pre-calculates EMAs for all possible parameter values.
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
        Pre-calculate EMAs for all possible parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """
        # Pre-calculate EMAs for all possible periods across all parameter ranges
        # EMA p1: 3-12
        for p in range(3, 13):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p2: 13-27
        for p in range(13, 28):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p3: 28-45
        for p in range(28, 46):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p4: 46-65
        for p in range(46, 66):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p5: 66-90
        for p in range(66, 91):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p6: 91-140
        for p in range(91, 141):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p7: 141-300
        for p in range(141, 301):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p8: 301-520
        for p in range(301, 521):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p9: 521-1120
        for p in range(521, 1121):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p10: 1121-1760
        for p in range(1121, 1761):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p11: 1761-2560
        for p in range(1761, 2561):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # EMA p12: 2561-4000
        for p in range(2561, 4001):
            dataframe[f"ema_period_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # Pre-calculate ATR for all possible periods (10-20)
        for p in range(10, 21):
            dataframe[f"atr_{p}"] = ta.ATR(dataframe, timeperiod=p)

        # RSI for V2 (fixed period, not hyperoptimized in base)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Volume MA for V3 (fixed period, not hyperoptimized in base)
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def get_ema_columns(self, dataframe: DataFrame) -> dict:
        """
        Get the EMA columns based on current hyperopt parameter values.
        Returns a dict with ema_1 through ema_12 mapped to actual column names.
        """
        return {
            "ema_1": dataframe[f"ema_period_{self.ema_p1.value}"],
            "ema_2": dataframe[f"ema_period_{self.ema_p2.value}"],
            "ema_3": dataframe[f"ema_period_{self.ema_p3.value}"],
            "ema_4": dataframe[f"ema_period_{self.ema_p4.value}"],
            "ema_5": dataframe[f"ema_period_{self.ema_p5.value}"],
            "ema_6": dataframe[f"ema_period_{self.ema_p6.value}"],
            "ema_7": dataframe[f"ema_period_{self.ema_p7.value}"],
            "ema_8": dataframe[f"ema_period_{self.ema_p8.value}"],
            "ema_9": dataframe[f"ema_period_{self.ema_p9.value}"],
            "ema_10": dataframe[f"ema_period_{self.ema_p10.value}"],
            "ema_11": dataframe[f"ema_period_{self.ema_p11.value}"],
            "ema_12": dataframe[f"ema_period_{self.ema_p12.value}"],
        }

    def get_atr(self, dataframe: DataFrame):
        """Get ATR column based on current hyperopt parameter value."""
        return dataframe[f"atr_{self.atr_period.value}"]

    def get_ema_by_number(self, dataframe: DataFrame, ema_num: int):
        """Get specific EMA by number (1-12) based on current hyperopt values."""
        ema_params = [
            self.ema_p1,
            self.ema_p2,
            self.ema_p3,
            self.ema_p4,
            self.ema_p5,
            self.ema_p6,
            self.ema_p7,
            self.ema_p8,
            self.ema_p9,
            self.ema_p10,
            self.ema_p11,
            self.ema_p12,
        ]
        period = ema_params[ema_num - 1].value
        return dataframe[f"ema_period_{period}"]

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

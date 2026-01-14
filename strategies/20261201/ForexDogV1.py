# --- Do not remove these libs ---
from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, IStrategy


# Import the base strategy with fallback for different import methods
try:
    from .ForexDogBase import ForexDogBase
except ImportError:
    from ForexDogBase import ForexDogBase


# --------------------------------


class ForexDogV1(ForexDogBase):
    """
    ForexDog Variation 1: Basic Crossover

    This variation uses basic EMA crossovers with ATR-based stoploss.
    It enters when price crosses above EMA2 while being above the first 6 EMAs,
    and exits when price touches EMA7.
    """

    # Use custom stoploss
    use_custom_stoploss = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        df = dataframe

        # Simplified entry conditions for V1:
        # 1. Price is above EMA1 and EMA2 (basic uptrend)
        # 2. EMA1 > EMA2 (trend confirmation)
        # 3. Price recently pulled back near EMA2 (entry opportunity)
        df.loc[
            (
                # Price is above fast EMAs
                (df["close"] > df["ema_1"])
                & (df["close"] > df["ema_2"])
                # EMAs are aligned for uptrend
                & (df["ema_1"] > df["ema_2"])
                # Price is within 2% of EMA2 (pullback entry)
                & ((df["close"] - df["ema_2"]) / df["ema_2"] < 0.02)
                # Not too far above EMA1 (not overextended)
                & ((df["close"] - df["ema_1"]) / df["ema_1"] < 0.01)
            ),
            "enter_long",
        ] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        df = dataframe

        # Exit when price is sufficiently above EMA3 or crosses below EMA1
        df.loc[
            (
                # Take profit when price is 3% above EMA3
                ((df["close"] - df["ema_3"]) / df["ema_3"] > 0.03)
                # Or exit if price crosses below EMA1 (stop loss)
                | (qtpylib.crossed_below(df["close"], df["ema_1"]))
            ),
            "exit_long",
        ] = 1

        return df

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Custom stoploss logic - uses ATR-based dynamic stoploss
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # ATR-based stoploss
        stoploss_price = last_candle["ema_3"] - (last_candle["atr"] * self.atr_multiplier.value)

        # Calculate stoploss percentage from absolute
        return (stoploss_price - current_rate) / current_rate

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Custom exit logic - implements time-based exit for losing trades
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Time-based exit for losing trades using exit_time_multiplier
        timeframe_minutes = int(self.timeframe[:-1])
        max_minutes = self.exit_time_multiplier.value * timeframe_minutes
        if (
            current_time - trade.open_date_utc
        ).total_seconds() / 60 > max_minutes and current_profit < 0:
            return "time_based_exit"

        # Exit if minimum profit threshold reached
        profit_threshold = self.exit_profit_threshold.value * 0.001  # Convert to decimal
        if current_profit > profit_threshold:
            return "profit_threshold_reached"

        return None

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

    FIXED: Now uses base class helper methods for EMAs to ensure
    hyperopt parameters are properly evaluated each epoch.
    """

    # Use custom stoploss
    use_custom_stoploss = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe.
        Uses base class helper methods to get EMAs based on current hyperopt values.
        """
        df = dataframe

        # Get EMAs based on current hyperopt parameter values
        ema_1 = self.get_ema_by_number(df, 1)
        ema_2 = self.get_ema_by_number(df, 2)

        # Simplified entry conditions for V1:
        # 1. Price is above EMA1 and EMA2 (basic uptrend)
        # 2. EMA1 > EMA2 (trend confirmation)
        # 3. Price recently pulled back near EMA2 (entry opportunity)
        df.loc[
            (
                # Price is above fast EMAs
                (df["close"] > ema_1)
                & (df["close"] > ema_2)
                # EMAs are aligned for uptrend
                & (ema_1 > ema_2)
                # Price is within 2% of EMA2 (pullback entry)
                & ((df["close"] - ema_2) / ema_2 < 0.02)
                # Not too far above EMA1 (not overextended)
                & ((df["close"] - ema_1) / ema_1 < 0.01)
            ),
            "enter_long",
        ] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe.
        Uses base class helper methods to get EMAs based on current hyperopt values.
        """
        df = dataframe

        # Get EMAs based on current hyperopt parameter values
        ema_1 = self.get_ema_by_number(df, 1)
        ema_3 = self.get_ema_by_number(df, 3)

        # Exit when price is sufficiently above EMA3 or crosses below EMA1
        df.loc[
            (
                # Take profit when price is 3% above EMA3
                ((df["close"] - ema_3) / ema_3 > 0.03)
                # Or exit if price crosses below EMA1 (stop loss)
                | (qtpylib.crossed_below(df["close"], ema_1))
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

        # Get EMA and ATR based on current hyperopt parameter values
        ema_3_period = self.ema_p3.value
        atr_period = self.atr_period.value

        ema_3_value = last_candle[f"ema_period_{ema_3_period}"]
        atr_value = last_candle[f"atr_{atr_period}"]

        # ATR-based stoploss
        stoploss_price = ema_3_value - (atr_value * self.atr_multiplier.value)

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

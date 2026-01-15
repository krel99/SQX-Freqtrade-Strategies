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


class ForexDogV2(ForexDogBase):
    """
    ForexDog Variation 2: Momentum Confirmation

    This variation uses momentum indicators (RSI) along with EMA alignments
    for stronger trend confirmation. It requires price to be above 8 EMAs
    and RSI > 60 for entry, exiting when price touches EMA9.
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        Uses momentum confirmation with RSI and EMA alignment
        """
        df = dataframe

        # Simplified entry conditions for V2 with RSI momentum:
        # 1. Price is above EMA1, EMA2, and EMA3 (basic uptrend)
        # 2. RSI shows momentum but not overbought
        # 3. EMAs are aligned for uptrend
        # 4. Price pulled back near EMA3 (entry opportunity)
        df.loc[
            (
                # Price is above key EMAs
                (df["close"] > df["ema_1"])
                & (df["close"] > df["ema_2"])
                & (df["close"] > df["ema_3"])
                # RSI momentum confirmation (not too low, not overbought)
                & (df["rsi"] > 55)
                & (df["rsi"] < 75)
                # Basic EMA alignment
                & (df["ema_1"] > df["ema_2"])
                & (df["ema_2"] > df["ema_3"])
                # Price is within 3% of EMA3 (pullback entry)
                & ((df["close"] - df["ema_3"]) / df["ema_3"] < 0.03)
            ),
            "enter_long",
        ] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        df = dataframe

        # Exit when RSI is overbought or price extends too far from EMAs
        df.loc[
            (
                # RSI overbought exit
                (df["rsi"] > 75)
                # Or price is too extended above EMA4
                | ((df["close"] - df["ema_4"]) / df["ema_4"] > 0.05)
                # Or price crosses below EMA2 (stop loss)
                | (qtpylib.crossed_below(df["close"], df["ema_2"]))
            ),
            "exit_long",
        ] = 1

        return df

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

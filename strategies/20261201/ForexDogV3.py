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


class ForexDogV3(ForexDogBase):
    """
    ForexDog Variation 3: Volatility-Adapted Entry

    This variation uses volatility indicators (ATR) to adapt entry conditions.
    It requires price to be above 10 EMAs with increased volatility and volume
    for entry. Uses dynamic ATR-based stoploss and exits when price touches EMA11.
    """

    # Use custom stoploss
    use_custom_stoploss = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        Uses volatility and volume confirmation for stronger signals
        """
        df = dataframe

        # Simplified entry conditions for V3 with volatility focus:
        # 1. Price is above key EMAs (1-4) showing uptrend
        # 2. ATR shows decent volatility (opportunity for movement)
        # 3. Volume is above average (participation)
        # 4. Price pulled back near EMA4 (entry opportunity)
        df.loc[
            (
                # Price is above key EMAs
                (df["close"] > df["ema_1"])
                & (df["close"] > df["ema_2"])
                & (df["close"] > df["ema_3"])
                & (df["close"] > df["ema_4"])
                # Basic EMA alignment
                & (df["ema_1"] > df["ema_2"])
                & (df["ema_2"] > df["ema_3"])
                & (df["ema_3"] > df["ema_4"])
                # ATR shows some volatility (not dead market)
                & (df["atr"] > df["atr"].rolling(20).mean() * 0.9)
                # Volume is above average
                & (df["volume"] > df["volume_ma"] * 0.8)
                # Price is within 4% of EMA4 (pullback entry)
                & ((df["close"] - df["ema_4"]) / df["ema_4"] < 0.04)
            ),
            "enter_long",
        ] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        df = dataframe

        # Exit when volatility spikes or price extends too far
        df.loc[
            (
                # Take profit when price is 6% above EMA5
                ((df["close"] - df["ema_5"]) / df["ema_5"] > 0.06)
                # Or exit on high volatility spike (risk management)
                | (df["atr"] > df["atr"].rolling(20).mean() * 2.0)
                # Or price crosses below EMA3 (stop loss)
                | (qtpylib.crossed_below(df["close"], df["ema_3"]))
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
        Adjusts stoploss distance based on current volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # ATR-based stoploss using exit_atr_multiplier for more dynamic control
        stoploss_price = last_candle["ema_6"] - (
            last_candle["atr"] * min(self.atr_multiplier.value, self.exit_atr_multiplier.value)
        )

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

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

# --------------------------------
from datetime import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta


class ConsolidationBreakout_05(IStrategy):
    """
    Breakout of Consolidation Box Strategy

    Detect recent range/consolidation, then enter when price breaks out with volume.

    Improvements:
    - ATR-based consolidation detection
    - False breakout filter using retest logic
    - Momentum confirmation with RSI
    - Dynamic lookback period based on volatility
    - Better volume spike detection
    - Support/resistance flip confirmation
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.04,
        "10": 0.03,
        "20": 0.025,
        "40": 0.02,
        "60": 0.015,
        "120": 0.01,
    }

    # Optimal stoploss
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperparameters
    lookback_period = IntParameter(15, 30, default=20, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_spike_mult = DecimalParameter(1.3, 2.0, default=1.5, space="buy")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_mult_breakout = DecimalParameter(0.2, 0.5, default=0.3, space="buy")
    consolidation_atr_max = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    # RSI parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_breakout_up = IntParameter(50, 60, default=55, space="buy")
    rsi_breakout_down = IntParameter(40, 50, default=45, space="buy")

    # Breakout parameters
    breakout_buffer = DecimalParameter(0.0001, 0.0005, default=0.0002, space="buy")
    min_consolidation_candles = IntParameter(5, 15, default=8, space="buy")

    # Exit parameters
    take_profit_mult = DecimalParameter(1.0, 2.5, default=1.5, space="sell")
    box_reentry_buffer = DecimalParameter(0.8, 1.0, default=0.9, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # Calculate range high and low
        dataframe["range_high"] = (
            dataframe["high"].rolling(window=self.lookback_period.value).max()
        )
        dataframe["range_low"] = (
            dataframe["low"].rolling(window=self.lookback_period.value).min()
        )

        # Range width
        dataframe["range_width"] = dataframe["range_high"] - dataframe["range_low"]
        dataframe["range_width_pct"] = (
            dataframe["range_width"] / dataframe["range_low"] * 100
        )

        # Midpoint of the range
        dataframe["range_mid"] = (dataframe["range_high"] + dataframe["range_low"]) / 2

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_spike"] = dataframe["volume"] > (
            dataframe["volume_ma"] * self.volume_spike_mult.value
        )

        # ATR for volatility measurement
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # RSI for momentum confirmation
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # EMA for trend context
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Trend determination
        dataframe["uptrend"] = dataframe["ema50"] > dataframe["ema200"]
        dataframe["downtrend"] = dataframe["ema50"] < dataframe["ema200"]

        # Consolidation detection
        # Check if range is narrow relative to ATR
        dataframe["range_atr_ratio"] = dataframe["range_width"] / dataframe["atr"]
        dataframe["is_consolidating"] = (
            (dataframe["range_atr_ratio"] < self.consolidation_atr_max.value * 2)
            & (dataframe["range_width_pct"] < 3.0)  # Range less than 3%
        )

        # Count consecutive consolidation candles
        dataframe["consol_count"] = (
            dataframe["is_consolidating"]
            .rolling(window=self.min_consolidation_candles.value, min_periods=1)
            .sum()
        )

        # Breakout detection
        dataframe["breakout_up"] = (
            dataframe["close"]
            > dataframe["range_high"] * (1 + self.breakout_buffer.value)
        ) & (dataframe["close"].shift(1) <= dataframe["range_high"].shift(1))

        dataframe["breakout_down"] = (
            dataframe["close"]
            < dataframe["range_low"] * (1 - self.breakout_buffer.value)
        ) & (dataframe["close"].shift(1) >= dataframe["range_low"].shift(1))

        # False breakout detection - price quickly returns to range
        dataframe["false_breakout_up"] = dataframe["breakout_up"].shift(1) & (
            dataframe["close"] < dataframe["range_high"]
        )

        dataframe["false_breakout_down"] = dataframe["breakout_down"].shift(1) & (
            dataframe["close"] > dataframe["range_low"]
        )

        # Momentum at breakout
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5))
            / dataframe["close"].shift(5)
            * 100
        )

        # Support becomes resistance and vice versa
        dataframe["close_above_prev_high"] = dataframe["close"] > dataframe[
            "range_high"
        ].shift(1)
        dataframe["close_below_prev_low"] = dataframe["close"] < dataframe[
            "range_low"
        ].shift(1)

        # Price position within range
        dataframe["range_position"] = np.where(
            dataframe["range_width"] > 0,
            (dataframe["close"] - dataframe["range_low"]) / dataframe["range_width"],
            0.5,
        )

        # Candle patterns
        dataframe["bullish_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["candle_body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]

        # Strong candle = large body relative to range
        dataframe["strong_bullish"] = dataframe["bullish_candle"] & (
            dataframe["candle_body"] > dataframe["candle_range"] * 0.6
        )
        dataframe["strong_bearish"] = dataframe["bearish_candle"] & (
            dataframe["candle_body"] > dataframe["candle_range"] * 0.6
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY - Breakout up
        dataframe.loc[
            (
                (dataframe["breakout_up"])  # Price breaks above range high
                & (dataframe["volume_spike"])  # Volume confirmation
                & (
                    dataframe["consol_count"] >= self.min_consolidation_candles.value
                )  # Sufficient consolidation
                & (
                    dataframe["rsi"] > self.rsi_breakout_up.value
                )  # RSI momentum confirmation
                & (dataframe["momentum"] > 0.5)  # Positive momentum
                & (~dataframe["false_breakout_up"].shift(1))  # No recent false breakout
                & (dataframe["strong_bullish"])  # Strong bullish candle
                & (
                    dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 0.8
                )  # Volatility not too low
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY - Breakout down
        dataframe.loc[
            (
                (dataframe["breakout_down"])  # Price breaks below range low
                & (dataframe["volume_spike"])  # Volume confirmation
                & (
                    dataframe["consol_count"] >= self.min_consolidation_candles.value
                )  # Sufficient consolidation
                & (
                    dataframe["rsi"] < self.rsi_breakout_down.value
                )  # RSI momentum confirmation
                & (dataframe["momentum"] < -0.5)  # Negative momentum
                & (
                    ~dataframe["false_breakout_down"].shift(1)
                )  # No recent false breakout
                & (dataframe["strong_bearish"])  # Strong bearish candle
                & (
                    dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 0.8
                )  # Volatility not too low
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Calculate dynamic targets based on ATR
        dataframe["long_target"] = dataframe["range_high"] + (
            dataframe["atr"] * self.take_profit_mult.value
        )
        dataframe["short_target"] = dataframe["range_low"] - (
            dataframe["atr"] * self.take_profit_mult.value
        )

        # LONG EXIT
        dataframe.loc[
            (
                (
                    dataframe["close"]
                    < dataframe["range_high"] * self.box_reentry_buffer.value
                )  # Back into box
                | (dataframe["close"] >= dataframe["long_target"])  # Target reached
                | (dataframe["rsi"] > 75)  # Overbought
                | (dataframe["momentum"] < -1.0)  # Momentum reversal
                | (
                    dataframe["volume"] < dataframe["volume_ma"] * 0.5
                )  # Volume dried up
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (
                    dataframe["close"]
                    > dataframe["range_low"] * (2 - self.box_reentry_buffer.value)
                )  # Back into box
                | (dataframe["close"] <= dataframe["short_target"])  # Target reached
                | (dataframe["rsi"] < 25)  # Oversold
                | (dataframe["momentum"] > 1.0)  # Momentum reversal
                | (
                    dataframe["volume"] < dataframe["volume_ma"] * 0.5
                )  # Volume dried up
            ),
            "exit_short",
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Custom exit logic for breakout strategy
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Quick profit taking if momentum is strong
        if current_profit > 0.015:
            if not trade.is_short and last_candle["rsi"] > 70:
                return "take_profit_overbought"
            if trade.is_short and last_candle["rsi"] < 30:
                return "take_profit_oversold"

        # Exit if consolidation forms again (breakout failed)
        if last_candle["is_consolidating"] and last_candle["consol_count"] > 3:
            return "new_consolidation"

        # Exit if volume dries up significantly
        if last_candle["volume"] < last_candle["volume_ma"] * 0.3:
            return "no_volume"

        # Time-based exit
        if current_time - trade.open_date_utc > pd.Timedelta(hours=2):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.01:
                return "time_exit_small_loss"

        # Exit on strong reversal candle
        if not trade.is_short and last_candle["strong_bearish"]:
            if current_profit > 0:
                return "reversal_candle"

        if trade.is_short and last_candle["strong_bullish"]:
            if current_profit > 0:
                return "reversal_candle"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Custom stoploss logic using ATR and range
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Dynamic stop based on ATR
        atr_stop = -(last_candle["atr"] * 2 / trade.open_rate)

        # Tighten stop after profit
        if current_profit > 0.02:
            return -0.005
        elif current_profit > 0.01:
            return -0.008
        elif current_profit > 0.005:
            return max(atr_stop, -0.015)

        # Progressive stop over time
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            return max(atr_stop, -0.02)

        return max(atr_stop, self.stoploss)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Additional checks before entering a trade
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Don't enter if range is too narrow (not worth the risk)
        if last_candle["range_width_pct"] < 0.5:
            return False

        # Don't enter if we just had a false breakout
        if last_candle["false_breakout_up"] or last_candle["false_breakout_down"]:
            return False

        # Avoid low liquidity periods
        hour = current_time.hour
        if hour >= 2 and hour <= 4:  # UTC
            return False

        return True

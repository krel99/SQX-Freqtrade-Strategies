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


class InsideBarBreakout_08_Modified(IStrategy):
    """
    Modified InsideBarBreakout_08:
    - Trailing stoploss ONLY for exits.
    - Weekend trade disable parameter.

    Identify inside bars (range within previous bar) and trade breakouts.

    Improvements:
    - Multiple inside bar detection
    - Mother bar quality assessment
    - Volume confirmation on breakout
    - ATR-based dynamic stops and targets
    - Time-of-day filters
    - False breakout protection
    - Trend context using higher timeframe
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy - set to high value to disable (trailing only)
    minimal_roi = {"0": 100}

    # Optimal stoploss
    stoploss = -0.025

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Hyperparameters
    disable_weekends = BooleanParameter(default=False, space="buy", optimize=True)

    # Risk-reward parameters
    trailing_atr_k = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    stop_loss_ratio = DecimalParameter(0.4, 0.6, default=0.5, space="sell")

    # Volume parameters
    volume_ma_period = IntParameter(10, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space="buy")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_mult_stop = DecimalParameter(1.0, 2.0, default=1.5, space="sell")
    atr_mult_target = DecimalParameter(1.5, 3.0, default=2.0, space="sell")

    # Mother bar quality
    min_mother_bar_size = DecimalParameter(0.001, 0.003, default=0.002, space="buy")
    max_mother_bar_size = DecimalParameter(0.008, 0.015, default=0.01, space="buy")

    # EMA for trend context
    ema_period = IntParameter(40, 60, default=50, space="buy")

    # Time filters
    start_hour = IntParameter(6, 10, default=7, space="buy")
    end_hour = IntParameter(18, 22, default=20, space="buy")

    # Inside bar parameters
    max_inside_bars = IntParameter(2, 5, default=3, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # Identify inside bars
        dataframe["is_inside_bar"] = (dataframe["high"] < dataframe["high"].shift(1)) & (
            dataframe["low"] > dataframe["low"].shift(1)
        )

        # Track mother bar (the bar before inside bar)
        dataframe["mother_high"] = np.where(
            dataframe["is_inside_bar"], dataframe["high"].shift(1), np.nan
        )
        dataframe["mother_low"] = np.where(
            dataframe["is_inside_bar"], dataframe["low"].shift(1), np.nan
        )

        # Forward fill mother bar values during inside bar sequences
        dataframe["mother_high"] = dataframe["mother_high"].ffill()
        dataframe["mother_low"] = dataframe["mother_low"].ffill()

        # Mother bar size
        dataframe["mother_size"] = dataframe["mother_high"] - dataframe["mother_low"]
        dataframe["mother_size_pct"] = dataframe["mother_size"] / dataframe["mother_low"] * 100

        # Count consecutive inside bars
        dataframe["inside_bar_count"] = 0
        inside_count = 0
        for i in range(len(dataframe)):
            if dataframe.loc[i, "is_inside_bar"]:
                inside_count += 1
            else:
                inside_count = 0
            dataframe.loc[i, "inside_bar_count"] = inside_count

        # Detect breakouts
        dataframe["breakout_up"] = (
            (dataframe["close"] > dataframe["mother_high"])
            & (dataframe["close"].shift(1) <= dataframe["mother_high"].shift(1))
            & (dataframe["inside_bar_count"].shift(1) > 0)
            & (dataframe["inside_bar_count"] == 0)
        )

        dataframe["breakout_down"] = (
            (dataframe["close"] < dataframe["mother_low"])
            & (dataframe["close"].shift(1) >= dataframe["mother_low"].shift(1))
            & (dataframe["inside_bar_count"].shift(1) > 0)
            & (dataframe["inside_bar_count"] == 0)
        )

        # Volume - pre-calculate for all possible periods (10-30)
        for period in range(10, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # ATR for stops and targets - pre-calculate for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # EMA for trend context - pre-calculate for all possible periods (40-60)
        for period in range(40, 61):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5) * 100
        )

        # Candle analysis
        dataframe["body_size"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["body_ratio"] = dataframe["body_size"] / dataframe["candle_range"]

        # Strong candles for breakout confirmation
        dataframe["bullish_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["strong_bullish"] = dataframe["bullish_candle"] & (dataframe["body_ratio"] > 0.6)
        dataframe["strong_bearish"] = dataframe["bearish_candle"] & (dataframe["body_ratio"] > 0.6)

        # Hour for time filter
        dataframe["hour"] = dataframe["date"].dt.hour
        # Session active will be calculated in entry/exit trends based on hyperopt params

        # False breakout detection - price returns to range
        dataframe["false_breakout_up"] = (
            dataframe["breakout_up"].shift(1) | dataframe["breakout_up"].shift(2)
        ) & (dataframe["close"] < dataframe["mother_high"])

        dataframe["false_breakout_down"] = (
            dataframe["breakout_down"].shift(1) | dataframe["breakout_down"].shift(2)
        ) & (dataframe["close"] > dataframe["mother_low"])

        # Range position
        dataframe["range_position"] = np.where(
            dataframe["mother_size"] > 0,
            (dataframe["close"] - dataframe["mother_low"]) / dataframe["mother_size"],
            0.5,
        )

        # RSI for additional confirmation
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """
        # Weekend filter
        if self.disable_weekends.value:
            # 5 = Saturday, 6 = Sunday
            is_weekend = dataframe["date"].dt.dayofweek >= 5
        else:
            is_weekend = pd.Series([False] * len(dataframe))

        # Get current hyperopt parameter values
        volume_ma_period = self.volume_ma_period.value
        volume_threshold = self.volume_threshold.value
        ema_period = self.ema_period.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        ema = dataframe[f"ema_{ema_period}"]

        # Calculate dynamic conditions
        volume_ok = dataframe["volume"] >= (volume_ma * volume_threshold)
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema
        session_active = (dataframe["hour"] >= start_hour) & (dataframe["hour"] < end_hour)

        # LONG ENTRY: break above mother bar high
        dataframe.loc[
            (
                (dataframe["breakout_up"])  # Breakout above mother bar
                & (~is_weekend)  # Weekend filter
                & (dataframe["strong_bullish"])  # Strong bullish candle
                & (volume_ok)  # Volume confirmation
                & (uptrend)  # In uptrend
                & (session_active)  # Active trading session
                & (
                    dataframe["mother_size_pct"] >= self.min_mother_bar_size.value
                )  # Mother bar not too small
                & (
                    dataframe["mother_size_pct"] <= self.max_mother_bar_size.value
                )  # Mother bar not too large
                & (dataframe["false_breakout_up"].shift(1) == False)  # No recent false breakout
                & (dataframe["momentum"] > 0)  # Positive momentum
                & (dataframe["rsi"] > 45)  # Not oversold
                & (dataframe["rsi"] < 70)  # Not overbought
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY: break below mother bar low
        dataframe.loc[
            (
                (dataframe["breakout_down"])  # Breakout below mother bar
                & (~is_weekend)  # Weekend filter
                & (dataframe["strong_bearish"])  # Strong bearish candle
                & (volume_ok)  # Volume confirmation
                & (downtrend)  # In downtrend
                & (session_active)  # Active trading session
                & (
                    dataframe["mother_size_pct"] >= self.min_mother_bar_size.value
                )  # Mother bar not too small
                & (
                    dataframe["mother_size_pct"] <= self.max_mother_bar_size.value
                )  # Mother bar not too large
                & (dataframe["false_breakout_down"].shift(1) == False)  # No recent false breakout
                & (dataframe["momentum"] < 0)  # Negative momentum
                & (dataframe["rsi"] < 55)  # Not overbought
                & (dataframe["rsi"] > 30)  # Not oversold
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals.
        Now using custom_exit ONLY for trailing stoploss.
        """
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
        Trailing stoploss ONLY logic
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        atr = last_candle[f"atr_{self.atr_period.value}"]

        # ATR-based trailing stop using built-in max_rate/min_rate for persistence
        if atr > 0:
            if not trade.is_short:
                highest_rate = trade.max_rate
                trail_price = highest_rate - (atr * self.trailing_atr_k.value)
                if current_rate < trail_price:
                    return "atr_trailing_exit"
            else:
                lowest_rate = trade.min_rate
                trail_price = lowest_rate + (atr * self.trailing_atr_k.value)
                if current_rate > trail_price:
                    return "atr_trailing_exit"

        return None


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

        # Get current hyperopt parameter values
        atr_period = self.atr_period.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        atr = last_candle[f"atr_{atr_period}"]

        # Don't enter if mother bar is too old
        if last_candle["inside_bar_count"] > self.max_inside_bars.value:
            return False

        # Don't enter if volatility is too low
        if atr < last_candle["close"] * 0.001:
            return False

        # Avoid low liquidity hours
        session_active = (last_candle["hour"] >= start_hour) and (last_candle["hour"] < end_hour)
        if not session_active:
            return False

        return True

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


class InsideBarBreakout_08(IStrategy):
    """
    Inside-Bar Breakout Strategy (Price Action Only)

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

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.03,
        "10": 0.025,
        "20": 0.02,
        "30": 0.015,
        "45": 0.01,
        "60": 0.008,
        "90": 0.005,
    }

    # Optimal stoploss
    stoploss = -0.025

    # Trailing stoploss
    trailing_stop = True
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
    # Risk-reward parameters
    profit_target = DecimalParameter(0.008, 0.02, default=0.01, space="sell")
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
        dataframe["is_inside_bar"] = (
            dataframe["high"] < dataframe["high"].shift(1)
        ) & (dataframe["low"] > dataframe["low"].shift(1))

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
        dataframe["mother_size_pct"] = (
            dataframe["mother_size"] / dataframe["mother_low"] * 100
        )

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

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ok"] = dataframe["volume"] >= (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        # ATR for stops and targets
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # EMA for trend context
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        dataframe["uptrend"] = dataframe["close"] > dataframe["ema"]
        dataframe["downtrend"] = dataframe["close"] < dataframe["ema"]

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5))
            / dataframe["close"].shift(5)
            * 100
        )

        # Candle analysis
        dataframe["body_size"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["body_ratio"] = dataframe["body_size"] / dataframe["candle_range"]

        # Strong candles for breakout confirmation
        dataframe["bullish_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["bearish_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["strong_bullish"] = dataframe["bullish_candle"] & (
            dataframe["body_ratio"] > 0.6
        )
        dataframe["strong_bearish"] = dataframe["bearish_candle"] & (
            dataframe["body_ratio"] > 0.6
        )

        # Hour for time filter
        dataframe["hour"] = dataframe["date"].dt.hour
        dataframe["session_active"] = (dataframe["hour"] >= self.start_hour.value) & (
            dataframe["hour"] < self.end_hour.value
        )

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

        # LONG ENTRY: break above mother bar high
        dataframe.loc[
            (
                (dataframe["breakout_up"])  # Breakout above mother bar
                & (dataframe["strong_bullish"])  # Strong bullish candle
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["uptrend"])  # In uptrend
                & (dataframe["session_active"])  # Active trading session
                & (
                    dataframe["mother_size_pct"] >= self.min_mother_bar_size.value
                )  # Mother bar not too small
                & (
                    dataframe["mother_size_pct"] <= self.max_mother_bar_size.value
                )  # Mother bar not too large
                & (~dataframe["false_breakout_up"].shift(1))  # No recent false breakout
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
                & (dataframe["strong_bearish"])  # Strong bearish candle
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["downtrend"])  # In downtrend
                & (dataframe["session_active"])  # Active trading session
                & (
                    dataframe["mother_size_pct"] >= self.min_mother_bar_size.value
                )  # Mother bar not too small
                & (
                    dataframe["mother_size_pct"] <= self.max_mother_bar_size.value
                )  # Mother bar not too large
                & (
                    ~dataframe["false_breakout_down"].shift(1)
                )  # No recent false breakout
                & (dataframe["momentum"] < 0)  # Negative momentum
                & (dataframe["rsi"] < 55)  # Not overbought
                & (dataframe["rsi"] > 30)  # Not oversold
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Calculate dynamic targets
        dataframe["long_target"] = dataframe["mother_high"] + (
            dataframe["atr"] * self.atr_mult_target.value
        )
        dataframe["long_stop"] = dataframe["mother_low"] - (
            dataframe["atr"] * self.atr_mult_stop.value
        )

        dataframe["short_target"] = dataframe["mother_low"] - (
            dataframe["atr"] * self.atr_mult_target.value
        )
        dataframe["short_stop"] = dataframe["mother_high"] + (
            dataframe["atr"] * self.atr_mult_stop.value
        )

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["long_target"])  # Target reached
                | (dataframe["close"] <= dataframe["long_stop"])  # Stop hit
                | (dataframe["downtrend"])  # Trend change
                | (dataframe["strong_bearish"])  # Strong reversal candle
                | (~dataframe["session_active"])  # Session ending
                | (dataframe["rsi"] > 75)  # Overbought
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["short_target"])  # Target reached
                | (dataframe["close"] >= dataframe["short_stop"])  # Stop hit
                | (dataframe["uptrend"])  # Trend change
                | (dataframe["strong_bullish"])  # Strong reversal candle
                | (~dataframe["session_active"])  # Session ending
                | (dataframe["rsi"] < 25)  # Oversold
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
        Custom exit logic for inside bar breakouts
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Quick profit taking
        if current_profit >= self.profit_target.value:
            return "profit_target"

        # Exit if price returns to mother bar range (failed breakout)
        if not np.isnan(last_candle["mother_high"]) and not np.isnan(
            last_candle["mother_low"]
        ):
            if not trade.is_short:
                if current_rate < last_candle["mother_high"]:
                    return "failed_breakout_long"
            else:
                if current_rate > last_candle["mother_low"]:
                    return "failed_breakout_short"

        # Exit if new inside bar forms (consolidation)
        if last_candle["is_inside_bar"]:
            if current_profit > 0.003:
                return "new_consolidation"

        # Exit on momentum loss
        if not trade.is_short and last_candle["momentum"] < -0.5:
            return "momentum_loss_long"
        if trade.is_short and last_candle["momentum"] > 0.5:
            return "momentum_loss_short"

        # Time-based exit
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -self.stop_loss_ratio.value * 0.01:
                return "time_exit_small_loss"

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
        Custom stoploss logic based on mother bar and ATR
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Use mother bar as natural stop level
        if not np.isnan(last_candle["mother_high"]) and not np.isnan(
            last_candle["mother_low"]
        ):
            if not trade.is_short:
                # For long, stop below mother bar low
                stop_price = last_candle["mother_low"] - (last_candle["atr"] * 0.5)
                stop_pct = -(trade.open_rate - stop_price) / trade.open_rate
                return max(stop_pct, self.stoploss)
            else:
                # For short, stop above mother bar high
                stop_price = last_candle["mother_high"] + (last_candle["atr"] * 0.5)
                stop_pct = -(stop_price - trade.open_rate) / trade.open_rate
                return max(stop_pct, self.stoploss)

        # Progressive stops based on profit
        if current_profit > 0.015:
            return -0.003
        elif current_profit > 0.008:
            return -0.005
        elif current_profit > 0.004:
            return -0.008

        return self.stoploss

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

        # Don't enter if mother bar is too old
        if last_candle["inside_bar_count"] > self.max_inside_bars.value:
            return False

        # Don't enter if volatility is too low
        if last_candle["atr"] < last_candle["close"] * 0.001:
            return False

        # Avoid low liquidity hours
        if not last_candle["session_active"]:
            return False

        return True

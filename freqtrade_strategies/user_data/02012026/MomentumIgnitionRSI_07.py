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


class MomentumIgnitionRSI_07(IStrategy):
    """
    Momentum Ignition with RSI Confirmation (1m / 5m)

    Buy strong momentum bursts in direction of recent move,
    confirming that the move is not overextended (RSI not too high for long, not too low for short).

    Improvements:
    - Volume surge detection for momentum confirmation
    - ATR-based volatility filter
    - MACD histogram for additional momentum confirmation
    - Dynamic ROC thresholds based on market volatility
    - Better exit timing with momentum exhaustion detection
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "1m"

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy - aggressive for 1m
    minimal_roi = {
        "0": 0.015,
        "3": 0.01,
        "6": 0.008,
        "12": 0.005,
        "20": 0.003,
    }

    # Optimal stoploss
    stoploss = -0.025

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.01
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
    # ROC parameters
    roc_period = IntParameter(3, 8, default=5, space="buy")
    roc_threshold_long = DecimalParameter(0.5, 1.5, default=1.0, space="buy")
    roc_threshold_short = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    # RSI parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_min_long = IntParameter(45, 55, default=50, space="buy")
    rsi_max_long = IntParameter(65, 75, default=70, space="buy")
    rsi_min_short = IntParameter(25, 35, default=30, space="buy")
    rsi_max_short = IntParameter(45, 55, default=50, space="buy")

    # Exit parameters
    profit_target = DecimalParameter(0.005, 0.015, default=0.008, space="sell")
    rsi_exit_long = IntParameter(70, 80, default=75, space="sell")
    rsi_exit_short = IntParameter(20, 30, default=25, space="sell")

    # Volume parameters
    volume_ma_period = IntParameter(10, 30, default=20, space="buy")
    volume_surge_mult = DecimalParameter(1.2, 2.0, default=1.5, space="buy")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_min_mult = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    # MACD parameters
    macd_fast = IntParameter(8, 15, default=12, space="buy")
    macd_slow = IntParameter(20, 30, default=26, space="buy")
    macd_signal = IntParameter(7, 11, default=9, space="buy")

    # Momentum exhaustion detection
    momentum_lookback = IntParameter(3, 8, default=5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # Rate of Change
        dataframe["roc"] = (
            dataframe["close"] / dataframe["close"].shift(self.roc_period.value) - 1
        ) * 100

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_surge"] = dataframe["volume"] > (
            dataframe["volume_ma"] * self.volume_surge_mult.value
        )

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe["atr_ma"] = ta.SMA(dataframe["atr"], timeperiod=20)

        # MACD for additional momentum confirmation
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Candle color
        dataframe["green_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["red_candle"] = dataframe["close"] < dataframe["open"]

        # Consecutive green/red candles
        dataframe["consecutive_green"] = (
            dataframe["green_candle"].rolling(window=3, min_periods=1).sum()
        )
        dataframe["consecutive_red"] = (
            dataframe["red_candle"].rolling(window=3, min_periods=1).sum()
        )

        # Price momentum
        dataframe["momentum_1"] = (
            (dataframe["close"] - dataframe["close"].shift(1))
            / dataframe["close"].shift(1)
            * 100
        )
        dataframe["momentum_3"] = (
            (dataframe["close"] - dataframe["close"].shift(3))
            / dataframe["close"].shift(3)
            * 100
        )

        # Dynamic ROC thresholds based on volatility
        dataframe["roc_threshold_dynamic"] = (
            dataframe["atr"] / dataframe["close"] * 100 * 2
        )

        # Momentum acceleration
        dataframe["roc_increasing"] = dataframe["roc"] > dataframe["roc"].shift(1)
        dataframe["roc_decreasing"] = dataframe["roc"] < dataframe["roc"].shift(1)

        # RSI momentum
        dataframe["rsi_increasing"] = dataframe["rsi"] > dataframe["rsi"].shift(1)
        dataframe["rsi_decreasing"] = dataframe["rsi"] < dataframe["rsi"].shift(1)

        # EMA for trend context
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        # Price position relative to EMAs
        dataframe["above_ema20"] = dataframe["close"] > dataframe["ema20"]
        dataframe["below_ema20"] = dataframe["close"] < dataframe["ema20"]

        # Momentum exhaustion detection
        dataframe["high_recent"] = (
            dataframe["high"].rolling(window=self.momentum_lookback.value).max()
        )
        dataframe["low_recent"] = (
            dataframe["low"].rolling(window=self.momentum_lookback.value).min()
        )

        # Volume profile
        dataframe["volume_increasing"] = dataframe["volume"] > dataframe[
            "volume"
        ].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY
        dataframe.loc[
            (
                (
                    dataframe["roc"] > self.roc_threshold_long.value
                )  # Strong positive ROC
                & (
                    dataframe["roc"] > dataframe["roc_threshold_dynamic"] * 0.5
                )  # Dynamic check
                & (dataframe["rsi"] >= self.rsi_min_long.value)  # RSI not oversold
                & (dataframe["rsi"] <= self.rsi_max_long.value)  # RSI not overbought
                & (dataframe["green_candle"])  # Last candle green
                & (dataframe["volume_surge"])  # Volume surge
                & (
                    dataframe["atr"] > dataframe["atr_ma"] * self.atr_min_mult.value
                )  # Volatility check
                & (dataframe["macd_hist"] > 0)  # MACD histogram positive
                & (
                    dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)
                )  # Increasing momentum
                & (dataframe["above_ema20"])  # Above short-term EMA
                & (dataframe["consecutive_green"] >= 2)  # At least 2 green candles
                & (dataframe["rsi_increasing"])  # RSI momentum up
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (
                    dataframe["roc"] < -self.roc_threshold_short.value
                )  # Strong negative ROC
                & (
                    dataframe["roc"] < -dataframe["roc_threshold_dynamic"] * 0.5
                )  # Dynamic check
                & (dataframe["rsi"] >= self.rsi_min_short.value)  # RSI not oversold
                & (dataframe["rsi"] <= self.rsi_max_short.value)  # RSI not overbought
                & (dataframe["red_candle"])  # Last candle red
                & (dataframe["volume_surge"])  # Volume surge
                & (
                    dataframe["atr"] > dataframe["atr_ma"] * self.atr_min_mult.value
                )  # Volatility check
                & (dataframe["macd_hist"] < 0)  # MACD histogram negative
                & (
                    dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)
                )  # Decreasing momentum
                & (dataframe["below_ema20"])  # Below short-term EMA
                & (dataframe["consecutive_red"] >= 2)  # At least 2 red candles
                & (dataframe["rsi_decreasing"])  # RSI momentum down
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["rsi"] > self.rsi_exit_long.value)  # RSI overbought
                | (
                    dataframe["red_candle"] & (dataframe["momentum_1"] < -0.5)
                )  # Momentum stall
                | (dataframe["roc"] < -0.5)  # ROC reversal
                | (
                    dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)
                )  # MACD momentum loss
                & (
                    dataframe["macd_hist"].shift(1) < dataframe["macd_hist"].shift(2)
                )  # Consistent loss
                | (dataframe["close"] < dataframe["ema20"])  # Below EMA20
                | (dataframe["consecutive_red"] >= 2)  # Multiple red candles
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_exit_short.value)  # RSI oversold
                | (
                    dataframe["green_candle"] & (dataframe["momentum_1"] > 0.5)
                )  # Momentum stall
                | (dataframe["roc"] > 0.5)  # ROC reversal
                | (
                    dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)
                )  # MACD momentum loss
                & (
                    dataframe["macd_hist"].shift(1) > dataframe["macd_hist"].shift(2)
                )  # Consistent loss
                | (dataframe["close"] > dataframe["ema20"])  # Above EMA20
                | (dataframe["consecutive_green"] >= 2)  # Multiple green candles
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
        Custom exit logic for momentum trades
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Quick profit taking
        if current_profit >= self.profit_target.value:
            return "profit_target_reached"

        # Exit on momentum exhaustion
        if not trade.is_short:
            # Long: exit if price can't make new highs
            if current_rate < last_candle["high_recent"] * 0.999:
                if current_profit > 0.003:
                    return "momentum_exhausted_long"
        else:
            # Short: exit if price can't make new lows
            if current_rate > last_candle["low_recent"] * 1.001:
                if current_profit > 0.003:
                    return "momentum_exhausted_short"

        # Exit if ROC reverses strongly
        if not trade.is_short and last_candle["roc"] < -1.5:
            return "roc_reversal_down"
        if trade.is_short and last_candle["roc"] > 1.5:
            return "roc_reversal_up"

        # Exit if volume dries up
        if last_candle["volume"] < last_candle["volume_ma"] * 0.5:
            if current_profit > 0:
                return "volume_dried_up"

        # Time-based exit for 1m timeframe (quick trades)
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=15):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.005:
                return "time_exit_small_loss"

        # Exit if MACD histogram shows strong reversal
        if not trade.is_short and last_candle["macd_hist"] < -0.0001:
            return "macd_reversal_bearish"
        if trade.is_short and last_candle["macd_hist"] > 0.0001:
            return "macd_reversal_bullish"

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
        Custom stoploss logic for momentum trades
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Dynamic stop based on ATR
        atr_stop = -(last_candle["atr"] * 2 / trade.open_rate)

        # Tighten stop as profit increases
        if current_profit > 0.01:
            return -0.002  # Very tight stop
        elif current_profit > 0.006:
            return -0.003  # Tight stop
        elif current_profit > 0.003:
            return -0.005  # Moderate stop

        # Progressive stop over time for 1m timeframe
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=10):
            return max(atr_stop, -0.015)
        elif current_time - trade.open_date_utc > pd.Timedelta(minutes=5):
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

        # Don't enter if ATR is too low (no momentum opportunity)
        if last_candle["atr"] < last_candle["atr_ma"] * 0.5:
            return False

        # Don't enter if we just had a strong move (avoid chasing)
        if abs(last_candle["momentum_3"]) > 3:
            return False

        # Avoid low liquidity hours
        hour = current_time.hour
        if hour >= 2 and hour <= 5:  # UTC
            return False

        return True

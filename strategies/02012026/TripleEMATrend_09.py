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


class TripleEMATrend_09(IStrategy):
    """
    Triple EMA Trend-Rider (TEMA 9/21/55)

    Use three EMAs to define trend and momentum; enter on pullback to middle EMA while all EMAs aligned.

    Improvements:
    - Volume-weighted EMA touches for better entry precision
    - ADX for trend strength confirmation
    - RSI divergence detection
    - Dynamic position sizing based on trend strength
    - Better exit with partial profit taking
    - Candle pattern confirmation
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.035,
        "10": 0.025,
        "20": 0.018,
        "30": 0.015,
        "45": 0.012,
        "60": 0.01,
        "90": 0.008,
        "120": 0.005,
    }

    # Optimal stoploss
    stoploss = -0.035

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.018
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Hyperparameters
    # EMA periods
    ema_fast_period = IntParameter(7, 12, default=9, space="buy")
    ema_mid_period = IntParameter(18, 25, default=21, space="buy")
    ema_slow_period = IntParameter(50, 60, default=55, space="buy")

    # Touch detection
    touch_threshold = DecimalParameter(0.001, 0.003, default=0.002, space="buy")

    # Trend strength
    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_min = IntParameter(20, 35, default=25, space="buy")

    # RSI parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_min_long = IntParameter(35, 45, default=40, space="buy")
    rsi_max_short = IntParameter(55, 65, default=60, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space="buy")

    # Exit parameters
    profit_target = DecimalParameter(0.012, 0.02, default=0.015, space="sell")
    ema_fast_exit_distance = DecimalParameter(0.002, 0.005, default=0.003, space="sell")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_mult = DecimalParameter(1.0, 2.0, default=1.5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # Three EMAs
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=self.ema_mid_period.value)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)

        # EMA alignment
        dataframe["bullish_alignment"] = (dataframe["ema9"] > dataframe["ema21"]) & (
            dataframe["ema21"] > dataframe["ema55"]
        )
        dataframe["bearish_alignment"] = (dataframe["ema9"] < dataframe["ema21"]) & (
            dataframe["ema21"] < dataframe["ema55"]
        )

        # EMA slopes for momentum
        for ema in ["ema9", "ema21", "ema55"]:
            dataframe[f"{ema}_slope"] = (
                (dataframe[ema] - dataframe[ema].shift(5))
                / dataframe[ema].shift(5)
                * 100
            )

        # Distance between EMAs (trend strength)
        dataframe["ema_spread_fast_mid"] = (
            (dataframe["ema9"] - dataframe["ema21"]) / dataframe["ema21"] * 100
        )
        dataframe["ema_spread_mid_slow"] = (
            (dataframe["ema21"] - dataframe["ema55"]) / dataframe["ema55"] * 100
        )

        # EMA21 touch detection with wicks
        dataframe["ema21_touch"] = (
            (
                abs(dataframe["close"] - dataframe["ema21"]) / dataframe["ema21"]
                < self.touch_threshold.value
            )
            | (
                abs(dataframe["high"] - dataframe["ema21"]) / dataframe["ema21"]
                < self.touch_threshold.value
            )
            | (
                abs(dataframe["low"] - dataframe["ema21"]) / dataframe["ema21"]
                < self.touch_threshold.value
            )
        )

        # Bullish/Bearish rejection from EMA21
        dataframe["bullish_rejection"] = (
            (dataframe["low"] <= dataframe["ema21"])  # Low touches or pierces EMA21
            & (dataframe["close"] > dataframe["ema21"])  # Close above EMA21
            & (dataframe["close"] > dataframe["open"])  # Bullish candle
        )

        dataframe["bearish_rejection"] = (
            (dataframe["high"] >= dataframe["ema21"])  # High touches or pierces EMA21
            & (dataframe["close"] < dataframe["ema21"])  # Close below EMA21
            & (dataframe["close"] < dataframe["open"])  # Bearish candle
        )

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # RSI divergence (simplified)
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["rsi_higher"] = dataframe["rsi"] > dataframe["rsi"].shift(lookback)
        dataframe["bullish_div"] = dataframe["price_lower"] & dataframe["rsi_higher"]

        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(
            lookback
        )
        dataframe["rsi_lower"] = dataframe["rsi"] < dataframe["rsi"].shift(lookback)
        dataframe["bearish_div"] = dataframe["price_higher"] & dataframe["rsi_lower"]

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ok"] = dataframe["volume"] >= (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Candle patterns
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["shooting_star"] = ta.CDLSHOOTINGSTAR(dataframe)
        dataframe["bullish_engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["bearish_engulfing"] = ta.CDLENGULFING(dataframe) * -1

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5))
            / dataframe["close"].shift(5)
            * 100
        )

        # Position relative to EMAs
        dataframe["above_ema9"] = dataframe["close"] > dataframe["ema9"]
        dataframe["below_ema9"] = dataframe["close"] < dataframe["ema9"]
        dataframe["above_ema21"] = dataframe["close"] > dataframe["ema21"]
        dataframe["below_ema21"] = dataframe["close"] < dataframe["ema21"]

        # Count bars since EMA cross
        dataframe["bars_since_cross"] = 0
        cross_up = (dataframe["ema9"] > dataframe["ema21"]) & (
            dataframe["ema9"].shift(1) <= dataframe["ema21"].shift(1)
        )
        cross_down = (dataframe["ema9"] < dataframe["ema21"]) & (
            dataframe["ema9"].shift(1) >= dataframe["ema21"].shift(1)
        )

        bars_count = 0
        for i in range(len(dataframe)):
            if cross_up.iloc[i] or cross_down.iloc[i]:
                bars_count = 0
            else:
                bars_count += 1
            dataframe.loc[i, "bars_since_cross"] = bars_count

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG: strong uptrend alignment and pullback to EMA21
        dataframe.loc[
            (
                (dataframe["bullish_alignment"])  # EMAs aligned bullishly
                & (
                    (dataframe["bullish_rejection"])  # Price rejects from EMA21
                    | (
                        dataframe["ema21_touch"] & dataframe["above_ema21"]
                    )  # Or touches and holds
                )
                & (dataframe["adx"] > self.adx_min.value)  # Strong trend
                & (dataframe["rsi"] > self.rsi_min_long.value)  # Not oversold
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["ema9_slope"] > 0.1)  # Fast EMA rising
                & (dataframe["ema21_slope"] > 0)  # Mid EMA rising
                & (dataframe["ema_spread_fast_mid"] > 0.1)  # Good separation
                & (dataframe["bars_since_cross"] > 3)  # Not right after cross
                & (
                    (dataframe["hammer"] > 0)  # Candle pattern confirmation
                    | (dataframe["bullish_engulfing"] > 0)
                    | (dataframe["bullish_div"])  # Or divergence
                    | (dataframe["momentum"] > 0)  # Or positive momentum
                )
            ),
            "enter_long",
        ] = 1

        # SHORT: strong downtrend alignment and pullback to EMA21
        dataframe.loc[
            (
                (dataframe["bearish_alignment"])  # EMAs aligned bearishly
                & (
                    (dataframe["bearish_rejection"])  # Price rejects from EMA21
                    | (
                        dataframe["ema21_touch"] & dataframe["below_ema21"]
                    )  # Or touches and holds
                )
                & (dataframe["adx"] > self.adx_min.value)  # Strong trend
                & (dataframe["rsi"] < self.rsi_max_short.value)  # Not overbought
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["ema9_slope"] < -0.1)  # Fast EMA falling
                & (dataframe["ema21_slope"] < 0)  # Mid EMA falling
                & (dataframe["ema_spread_fast_mid"] < -0.1)  # Good separation
                & (dataframe["bars_since_cross"] > 3)  # Not right after cross
                & (
                    (dataframe["shooting_star"] < 0)  # Candle pattern confirmation
                    | (dataframe["bearish_engulfing"] > 0)
                    | (dataframe["bearish_div"])  # Or divergence
                    | (dataframe["momentum"] < 0)  # Or negative momentum
                )
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
                (dataframe["below_ema9"])  # Price breaks below fast EMA
                | (~dataframe["bullish_alignment"])  # Lost alignment
                | (dataframe["ema9_slope"] < -0.2)  # Fast EMA turning down strongly
                | (dataframe["rsi"] > 75)  # Overbought
                | (
                    abs(dataframe["close"] - dataframe["ema9"])
                    > dataframe["close"] * self.ema_fast_exit_distance.value
                )  # Too far from fast EMA
                | (dataframe["bearish_engulfing"] > 0)  # Strong reversal pattern
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["above_ema9"])  # Price breaks above fast EMA
                | (~dataframe["bearish_alignment"])  # Lost alignment
                | (dataframe["ema9_slope"] > 0.2)  # Fast EMA turning up strongly
                | (dataframe["rsi"] < 25)  # Oversold
                | (
                    abs(dataframe["close"] - dataframe["ema9"])
                    > dataframe["close"] * self.ema_fast_exit_distance.value
                )  # Too far from fast EMA
                | (dataframe["bullish_engulfing"] > 0)  # Strong reversal pattern
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
        Custom exit logic for trend riding
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Take profit at target
        if current_profit >= self.profit_target.value:
            return "profit_target_reached"

        # Exit if alignment is lost
        if not trade.is_short and not last_candle["bullish_alignment"]:
            if current_profit > 0.005:
                return "alignment_lost_profit"
            elif current_profit > -0.005:
                return "alignment_lost_small"

        if trade.is_short and not last_candle["bearish_alignment"]:
            if current_profit > 0.005:
                return "alignment_lost_profit"
            elif current_profit > -0.005:
                return "alignment_lost_small"

        # Exit if trend weakens (ADX drops)
        if last_candle["adx"] < 20:
            if current_profit > 0:
                return "trend_weakening"

        # Exit on EMA cross
        if not trade.is_short and last_candle["ema9"] < last_candle["ema21"]:
            return "ema_bearish_cross"
        if trade.is_short and last_candle["ema9"] > last_candle["ema21"]:
            return "ema_bullish_cross"

        # Time-based exit
        if current_time - trade.open_date_utc > pd.Timedelta(hours=2):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.01:
                return "time_exit_small_loss"

        # Protect profits if momentum reverses
        if current_profit > 0.01:
            if not trade.is_short and last_candle["momentum"] < -1:
                return "momentum_reversal_protect"
            if trade.is_short and last_candle["momentum"] > 1:
                return "momentum_reversal_protect"

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
        Custom stoploss logic using EMAs and ATR
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Use EMA55 as ultimate stop
        if not trade.is_short:
            ema_stop = -(trade.open_rate - last_candle["ema55"]) / trade.open_rate
            ema_stop = max(ema_stop, self.stoploss)
        else:
            ema_stop = -(last_candle["ema55"] - trade.open_rate) / trade.open_rate
            ema_stop = max(ema_stop, self.stoploss)

        # ATR-based stop
        atr_stop = -(last_candle["atr"] * self.atr_mult.value / trade.open_rate)

        # Use the tighter of the two
        dynamic_stop = max(ema_stop, atr_stop, self.stoploss)

        # Progressive stops based on profit
        if current_profit > 0.02:
            return -0.005
        elif current_profit > 0.015:
            return -0.008
        elif current_profit > 0.01:
            return -0.01
        elif current_profit > 0.005:
            return max(dynamic_stop, -0.015)

        return dynamic_stop

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

        # Don't enter if EMAs are too close (weak trend)
        if abs(last_candle["ema_spread_fast_mid"]) < 0.05:
            return False

        # Don't enter if we just had a cross (wait for stability)
        if last_candle["bars_since_cross"] < 2:
            return False

        # Avoid low liquidity hours
        hour = current_time.hour
        if hour >= 2 and hour <= 5:  # UTC
            return False

        return True

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
        Pre-calculates all indicator variants for hyperopt compatibility.
        """

        # Pre-calculate EMAs for all possible periods
        # ema_fast_period: 7-12, ema_mid_period: 18-25, ema_slow_period: 50-60
        for period in range(7, 13):
            dataframe[f"ema_fast_{period}"] = ta.EMA(dataframe, timeperiod=period)
        for period in range(18, 26):
            dataframe[f"ema_mid_{period}"] = ta.EMA(dataframe, timeperiod=period)
        for period in range(50, 61):
            dataframe[f"ema_slow_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate ADX for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Pre-calculate RSI for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all possible periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate ATR for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # RSI divergence lookback (fixed)
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(lookback)

        # Candle patterns
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["shooting_star"] = ta.CDLSHOOTINGSTAR(dataframe)
        dataframe["bullish_engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["bearish_engulfing"] = ta.CDLENGULFING(dataframe) * -1

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5) * 100
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # Get current hyperopt parameter values
        ema_fast_period = self.ema_fast_period.value
        ema_mid_period = self.ema_mid_period.value
        ema_slow_period = self.ema_slow_period.value
        touch_threshold = self.touch_threshold.value
        adx_period = self.adx_period.value
        adx_min = self.adx_min.value
        rsi_period = self.rsi_period.value
        rsi_min_long = self.rsi_min_long.value
        rsi_max_short = self.rsi_max_short.value
        volume_ma_period = self.volume_ma_period.value
        volume_threshold = self.volume_threshold.value

        # Select pre-calculated indicators
        ema_fast = dataframe[f"ema_fast_{ema_fast_period}"]
        ema_mid = dataframe[f"ema_mid_{ema_mid_period}"]
        ema_slow = dataframe[f"ema_slow_{ema_slow_period}"]
        adx = dataframe[f"adx_{adx_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]

        # Calculate derived indicators
        bullish_alignment = (ema_fast > ema_mid) & (ema_mid > ema_slow)
        bearish_alignment = (ema_fast < ema_mid) & (ema_mid < ema_slow)

        # EMA mid touch detection with wicks
        ema_mid_touch = (
            (abs(dataframe["close"] - ema_mid) / ema_mid < touch_threshold)
            | (abs(dataframe["high"] - ema_mid) / ema_mid < touch_threshold)
            | (abs(dataframe["low"] - ema_mid) / ema_mid < touch_threshold)
        )

        # Bullish/Bearish rejection from EMA mid
        bullish_rejection = (
            (dataframe["low"] <= ema_mid)  # Low touches or pierces EMA
            & (dataframe["close"] > ema_mid)  # Close above EMA
            & (dataframe["close"] > dataframe["open"])  # Bullish candle
        )

        bearish_rejection = (
            (dataframe["high"] >= ema_mid)  # High touches or pierces EMA
            & (dataframe["close"] < ema_mid)  # Close below EMA
            & (dataframe["close"] < dataframe["open"])  # Bearish candle
        )

        # Volume check
        volume_ok = dataframe["volume"] >= (volume_ma * volume_threshold)

        # Position relative to EMAs
        above_ema_fast = dataframe["close"] > ema_fast
        below_ema_fast = dataframe["close"] < ema_fast
        above_ema_mid = dataframe["close"] > ema_mid
        below_ema_mid = dataframe["close"] < ema_mid

        # EMA slopes
        ema_fast_slope = (ema_fast - ema_fast.shift(5)) / ema_fast.shift(5) * 100
        ema_mid_slope = (ema_mid - ema_mid.shift(5)) / ema_mid.shift(5) * 100

        # RSI divergence
        lookback = 10
        rsi_higher = rsi > rsi.shift(lookback)
        rsi_lower = rsi < rsi.shift(lookback)
        bullish_div = dataframe["price_lower"] & rsi_higher
        bearish_div = dataframe["price_higher"] & rsi_lower

        # EMA spread (trend strength)
        ema_spread_fast_mid = (ema_fast - ema_mid) / ema_mid * 100

        # LONG: strong uptrend alignment and pullback to EMA mid
        dataframe.loc[
            (
                (bullish_alignment)  # EMAs aligned bullishly
                & (
                    (bullish_rejection)  # Price rejects from EMA mid
                    | (ema_mid_touch & above_ema_mid)  # Or touches and holds
                )
                & (adx > adx_min)  # Strong trend
                & (rsi > rsi_min_long)  # Not oversold
                & (volume_ok)  # Volume confirmation
                & (ema_fast_slope > 0.1)  # Fast EMA rising
                & (ema_mid_slope > 0)  # Mid EMA rising
                & (ema_spread_fast_mid > 0.1)  # Good separation
                & (
                    (dataframe["hammer"] > 0)  # Candle pattern confirmation
                    | (dataframe["bullish_engulfing"] > 0)
                    | (bullish_div)  # Or divergence
                    | (dataframe["momentum"] > 0)  # Or positive momentum
                )
            ),
            "enter_long",
        ] = 1

        # SHORT: strong downtrend alignment and pullback to EMA mid
        dataframe.loc[
            (
                (bearish_alignment)  # EMAs aligned bearishly
                & (
                    (bearish_rejection)  # Price rejects from EMA mid
                    | (ema_mid_touch & below_ema_mid)  # Or touches and holds
                )
                & (adx > adx_min)  # Strong trend
                & (rsi < rsi_max_short)  # Not overbought
                & (volume_ok)  # Volume confirmation
                & (ema_fast_slope < -0.1)  # Fast EMA falling
                & (ema_mid_slope < 0)  # Mid EMA falling
                & (ema_spread_fast_mid < -0.1)  # Good separation
                & (
                    (dataframe["shooting_star"] < 0)  # Candle pattern confirmation
                    | (dataframe["bearish_engulfing"] > 0)
                    | (bearish_div)  # Or divergence
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

        # Get current hyperopt parameter values
        ema_fast_period = self.ema_fast_period.value
        ema_mid_period = self.ema_mid_period.value
        ema_slow_period = self.ema_slow_period.value
        rsi_period = self.rsi_period.value
        ema_fast_exit_distance = self.ema_fast_exit_distance.value

        # Select pre-calculated indicators
        ema_fast = dataframe[f"ema_fast_{ema_fast_period}"]
        ema_mid = dataframe[f"ema_mid_{ema_mid_period}"]
        ema_slow = dataframe[f"ema_slow_{ema_slow_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate derived indicators
        bullish_alignment = (ema_fast > ema_mid) & (ema_mid > ema_slow)
        bearish_alignment = (ema_fast < ema_mid) & (ema_mid < ema_slow)

        # Position relative to EMAs
        above_ema_fast = dataframe["close"] > ema_fast
        below_ema_fast = dataframe["close"] < ema_fast

        # EMA slope
        ema_fast_slope = (ema_fast - ema_fast.shift(5)) / ema_fast.shift(5) * 100

        # LONG EXIT
        dataframe.loc[
            (
                (below_ema_fast)  # Price breaks below fast EMA
                | (~bullish_alignment)  # Lost alignment
                | (ema_fast_slope < -0.2)  # Fast EMA turning down strongly
                | (rsi > 75)  # Overbought
                | (
                    abs(dataframe["close"] - ema_fast) > dataframe["close"] * ema_fast_exit_distance
                )  # Too far from fast EMA
                | (dataframe["bearish_engulfing"] > 0)  # Strong reversal pattern
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (above_ema_fast)  # Price breaks above fast EMA
                | (~bearish_alignment)  # Lost alignment
                | (ema_fast_slope > 0.2)  # Fast EMA turning up strongly
                | (rsi < 25)  # Oversold
                | (
                    abs(dataframe["close"] - ema_fast) > dataframe["close"] * ema_fast_exit_distance
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

        # Get current hyperopt parameter values
        ema_fast_period = self.ema_fast_period.value
        ema_mid_period = self.ema_mid_period.value
        ema_slow_period = self.ema_slow_period.value

        # Get pre-calculated indicators
        ema_fast = last_candle[f"ema_fast_{ema_fast_period}"]
        ema_mid = last_candle[f"ema_mid_{ema_mid_period}"]
        ema_slow = last_candle[f"ema_slow_{ema_slow_period}"]

        # Calculate alignment
        bullish_alignment = (ema_fast > ema_mid) and (ema_mid > ema_slow)
        bearish_alignment = (ema_fast < ema_mid) and (ema_mid < ema_slow)

        # Take profit at target
        if current_profit >= self.profit_target.value:
            return "profit_target_reached"

        # Exit if alignment is lost
        if not trade.is_short and not bullish_alignment:
            if current_profit > 0.005:
                return "alignment_lost_profit"
            elif current_profit > -0.005:
                return "alignment_lost_small"

        if trade.is_short and not bearish_alignment:
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

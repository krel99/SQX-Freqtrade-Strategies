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
from datetime import datetime, time
from freqtrade.persistence import Trade
import talib.abstract as ta


class VWAPReversion_06(IStrategy):
    """
    VWAP Reversion Strategy (Intraday)

    Use session VWAP; fade deviations and revert to VWAP.
    Works best for ranging/mean-reverting markets.

    Improvements:
    - Custom rolling VWAP implementation
    - Dynamic deviation bands based on volatility
    - Volume-weighted momentum indicator
    - Time-of-day session filters
    - RSI divergence confirmation
    - Better exit strategies with partial targets
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.025,
        "10": 0.02,
        "20": 0.015,
        "30": 0.01,
        "60": 0.008,
        "120": 0.005,
    }

    # Optimal stoploss
    stoploss = -0.04

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
    startup_candle_count: int = 200

    # Hyperparameters
    # VWAP parameters
    vwap_period = IntParameter(20, 60, default=40, space="buy")  # Rolling window for VWAP
    dev_period = IntParameter(15, 30, default=20, space="buy")  # Period for deviation calculation

    # Deviation multipliers
    dev_mult_lower = DecimalParameter(1.5, 2.5, default=2.0, space="buy")
    dev_mult_upper = DecimalParameter(1.5, 2.5, default=2.0, space="buy")

    # Exit parameters
    vwap_touch_exit = BooleanParameter(default=True, space="sell")
    partial_exit_ratio = DecimalParameter(0.3, 0.7, default=0.5, space="sell")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space="buy")

    # RSI parameters for confirmation
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(25, 35, default=30, space="buy")
    rsi_overbought = IntParameter(65, 75, default=70, space="buy")

    # Time filters
    start_hour = IntParameter(7, 10, default=8, space="buy")
    end_hour = IntParameter(18, 22, default=20, space="buy")

    # ATR for volatility adjustment
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_mult = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    def calculate_vwap(self, dataframe: DataFrame, period: int) -> pd.Series:
        """
        Calculate rolling VWAP (Volume Weighted Average Price)
        """
        # Calculate typical price
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3

        # Calculate rolling VWAP
        volume_sum = dataframe["volume"].rolling(window=period).sum()
        tpv_sum = (typical_price * dataframe["volume"]).rolling(window=period).sum()

        vwap = tpv_sum / volume_sum

        # Handle division by zero
        vwap = vwap.fillna(typical_price)

        return vwap

    def calculate_deviation(self, dataframe: DataFrame, vwap: pd.Series) -> pd.Series:
        """Calculate deviation from VWAP"""
        return dataframe["close"] - vwap

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Pre-calculates all indicator variants for hyperopt compatibility.
        """

        # Pre-calculate VWAP for all possible periods (20-60)
        for period in range(20, 61):
            dataframe[f"vwap_{period}"] = self.calculate_vwap(dataframe, period)

        # Pre-calculate deviation std for all dev_periods (15-30)
        for vwap_period in range(20, 61):
            vwap = dataframe[f"vwap_{vwap_period}"]
            deviation = dataframe["close"] - vwap
            for dev_period in range(15, 31):
                dataframe[f"dev_std_{vwap_period}_{dev_period}"] = deviation.rolling(
                    window=dev_period
                ).std()

        # Pre-calculate ATR for all periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Pre-calculate RSI for all periods (10-20)
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Volume-weighted momentum (no hyperopt params)
        dataframe["vwm"] = (
            ((dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5))
            * dataframe["volume"]
        ).rolling(window=10).sum() / dataframe["volume"].rolling(window=10).sum()

        # RSI divergence (simplified) - uses fixed lookback
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(lookback)

        # Add hour for time filtering
        dataframe["hour"] = dataframe["date"].dt.hour

        # Cumulative volume for the session (reset daily)
        dataframe["day"] = dataframe["date"].dt.date
        dataframe["cum_volume"] = dataframe.groupby("day")["volume"].cumsum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # Get current hyperopt parameter values
        vwap_period = self.vwap_period.value
        dev_period = self.dev_period.value
        dev_mult_upper = self.dev_mult_upper.value
        dev_mult_lower = self.dev_mult_lower.value
        atr_period = self.atr_period.value
        atr_mult = self.atr_mult.value
        rsi_period = self.rsi_period.value
        volume_ma_period = self.volume_ma_period.value
        volume_threshold = self.volume_threshold.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        vwap = dataframe[f"vwap_{vwap_period}"]
        dev_std = dataframe[f"dev_std_{vwap_period}_{dev_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]

        # Calculate derived values
        deviation = dataframe["close"] - vwap
        upper_dev = vwap + (dev_mult_upper * dev_std)
        lower_dev = vwap - (dev_mult_lower * dev_std)
        upper_dev_atr = vwap + (dev_mult_upper * atr * atr_mult)
        lower_dev_atr = vwap - (dev_mult_lower * atr * atr_mult)

        # Use the wider of the two bands
        upper_band = pd.concat([upper_dev, upper_dev_atr], axis=1).max(axis=1)
        lower_band = pd.concat([lower_dev, lower_dev_atr], axis=1).min(axis=1)

        # Volume check
        volume_ok = dataframe["volume"] >= (volume_ma * volume_threshold)

        # Session filter
        session_active = (dataframe["hour"] >= start_hour) & (dataframe["hour"] < end_hour)

        # VWAP slope
        vwap_slope = (vwap - vwap.shift(5)) / vwap.shift(5) * 100

        # Distance from VWAP in ATR units
        dist_from_vwap_atr = deviation / atr

        # RSI divergence
        lookback = 10
        rsi_higher = rsi > rsi.shift(lookback)
        rsi_lower = rsi < rsi.shift(lookback)
        bullish_div = dataframe["price_lower"] & rsi_higher
        bearish_div = dataframe["price_higher"] & rsi_lower

        # LONG ENTRY: price too far below VWAP
        dataframe.loc[
            (
                (dataframe["close"] < lower_band)  # Below lower deviation
                & (rsi < self.rsi_oversold.value)  # RSI confirms oversold
                & (volume_ok)  # Volume confirmation
                & (session_active)  # Within trading session
                & (abs(vwap_slope) < 1.0)  # VWAP not trending strongly
                & (
                    (bullish_div)  # Bullish divergence
                    | (dataframe["vwm"] > -0.01)  # Or not too negative momentum
                )
                & (dist_from_vwap_atr < -1.5)  # Significant deviation
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY: price too far above VWAP
        dataframe.loc[
            (
                (dataframe["close"] > upper_band)  # Above upper deviation
                & (rsi > self.rsi_overbought.value)  # RSI confirms overbought
                & (volume_ok)  # Volume confirmation
                & (session_active)  # Within trading session
                & (abs(vwap_slope) < 1.0)  # VWAP not trending strongly
                & (
                    (bearish_div)  # Bearish divergence
                    | (dataframe["vwm"] < 0.01)  # Or not too positive momentum
                )
                & (dist_from_vwap_atr > 1.5)  # Significant deviation
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Get current hyperopt parameter values
        vwap_period = self.vwap_period.value
        dev_period = self.dev_period.value
        dev_mult_upper = self.dev_mult_upper.value
        dev_mult_lower = self.dev_mult_lower.value
        atr_period = self.atr_period.value
        atr_mult = self.atr_mult.value
        rsi_period = self.rsi_period.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value
        partial_exit_ratio = self.partial_exit_ratio.value

        # Select pre-calculated indicators
        vwap = dataframe[f"vwap_{vwap_period}"]
        dev_std = dataframe[f"dev_std_{vwap_period}_{dev_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate derived values
        upper_dev = vwap + (dev_mult_upper * dev_std)
        lower_dev = vwap - (dev_mult_lower * dev_std)
        upper_dev_atr = vwap + (dev_mult_upper * atr * atr_mult)
        lower_dev_atr = vwap - (dev_mult_lower * atr * atr_mult)

        # Use the wider of the two bands
        upper_band = pd.concat([upper_dev, upper_dev_atr], axis=1).max(axis=1)
        lower_band = pd.concat([lower_dev, lower_dev_atr], axis=1).min(axis=1)

        # Session filter
        session_active = (dataframe["hour"] >= start_hour) & (dataframe["hour"] < end_hour)

        # VWAP slope
        vwap_slope = (vwap - vwap.shift(5)) / vwap.shift(5) * 100

        # Calculate partial exit targets
        long_partial_target = lower_band + ((vwap - lower_band) * partial_exit_ratio)
        short_partial_target = upper_band - ((upper_band - vwap) * partial_exit_ratio)

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] >= vwap)
                if self.vwap_touch_exit.value
                else (dataframe["close"] >= long_partial_target)
            )
            | (rsi > 65)  # No longer oversold
            | (vwap_slope < -0.5)  # VWAP turning down
            | (~session_active)  # Session ending
            | (dataframe["close"] > upper_band),  # Extreme reversal
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] <= vwap)
                if self.vwap_touch_exit.value
                else (dataframe["close"] <= short_partial_target)
            )
            | (rsi < 35)  # No longer overbought
            | (vwap_slope > 0.5)  # VWAP turning up
            | (~session_active)  # Session ending
            | (dataframe["close"] < lower_band),  # Extreme reversal
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
        Custom exit logic for VWAP reversion
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get current hyperopt parameter values
        vwap_period = self.vwap_period.value
        atr_period = self.atr_period.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        vwap = last_candle[f"vwap_{vwap_period}"]
        atr = last_candle[f"atr_{atr_period}"]

        # Calculate VWAP slope from dataframe
        vwap_col = dataframe[f"vwap_{vwap_period}"]
        vwap_slope = (
            ((vwap_col.iloc[-1] - vwap_col.iloc[-6]) / vwap_col.iloc[-6] * 100)
            if len(dataframe) > 5
            else 0
        )

        # Session check
        session_active = (last_candle["hour"] >= start_hour) and (last_candle["hour"] < end_hour)

        # Distance from VWAP in ATR units
        deviation = last_candle["close"] - vwap
        dist_from_vwap_atr = deviation / atr if atr > 0 else 0

        # Exit if we're past VWAP with profit
        if not trade.is_short:
            if last_candle["close"] > vwap and current_profit > 0.005:
                return "vwap_crossed_profit"
        else:
            if last_candle["close"] < vwap and current_profit > 0.005:
                return "vwap_crossed_profit"

        # Exit if session is ending
        if not session_active and current_profit > -0.005:
            return "session_end"

        # Exit if VWAP starts trending strongly against position
        if not trade.is_short and vwap_slope < -1.0:
            return "vwap_trending_down"
        if trade.is_short and vwap_slope > 1.0:
            return "vwap_trending_up"

        # Quick profit if deviation narrows
        if current_profit > 0.01:
            if abs(dist_from_vwap_atr) < 0.5:
                return "deviation_narrowed"

        # Time-based exit
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.01:
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
        Custom stoploss logic based on ATR and VWAP distance
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get current hyperopt parameter values
        vwap_period = self.vwap_period.value
        atr_period = self.atr_period.value
        start_hour = self.start_hour.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        vwap = last_candle[f"vwap_{vwap_period}"]
        atr = last_candle[f"atr_{atr_period}"]

        # Session check
        session_active = (last_candle["hour"] >= start_hour) and (last_candle["hour"] < end_hour)

        # Dynamic stop based on ATR
        atr_stop = -(atr * 2.5 / trade.open_rate)

        # Tighten stop when approaching VWAP
        if not trade.is_short:
            if current_rate > vwap * 0.998:
                return -0.003  # Very tight stop near target
        else:
            if current_rate < vwap * 1.002:
                return -0.003  # Very tight stop near target

        # Progressive stops based on profit
        if current_profit > 0.015:
            return -0.004
        elif current_profit > 0.008:
            return -0.006
        elif current_profit > 0.003:
            return -0.01

        # Use tighter stop if session is ending
        if not session_active:
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

        # Get current hyperopt parameter values
        vwap_period = self.vwap_period.value
        atr_period = self.atr_period.value
        volume_ma_period = self.volume_ma_period.value
        end_hour = self.end_hour.value

        # Select pre-calculated indicators
        vwap = last_candle[f"vwap_{vwap_period}"]
        atr = last_candle[f"atr_{atr_period}"]
        volume_ma = last_candle[f"volume_ma_{volume_ma_period}"]

        # Calculate VWAP slope from dataframe
        vwap_col = dataframe[f"vwap_{vwap_period}"]
        vwap_slope = (
            ((vwap_col.iloc[-1] - vwap_col.iloc[-6]) / vwap_col.iloc[-6] * 100)
            if len(dataframe) > 5
            else 0
        )

        # Distance from VWAP in ATR units
        deviation = last_candle["close"] - vwap
        dist_from_vwap_atr = deviation / atr if atr > 0 else 0

        # Don't enter if deviation is too extreme (likely to continue)
        if abs(dist_from_vwap_atr) > 3:
            return False

        # Don't enter if volume is too low
        if last_candle["cum_volume"] < volume_ma * 10:
            return False

        # Avoid entries near session end
        if last_candle["hour"] >= end_hour - 1:
            return False

        # Don't enter if VWAP is trending too strongly
        if abs(vwap_slope) > 1.5:
            return False

        return True

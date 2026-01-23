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


class BBMeanReversion_04(IStrategy):
    """
    5-Min Bollinger Band Mean Reversion Strategy

    Fade extremes relative to Bollinger Bands in a ranging market.
    Uses ADX to avoid strong trends.

    Improvements:
    - RSI divergence detection for better entries
    - Bollinger Band squeeze detection
    - Volume profile analysis
    - Dynamic band multiplier based on volatility
    - Support/resistance levels
    - Better exit logic with scaled exits

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
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
        "30": 0.012,
        "60": 0.008,
        "120": 0.005,
    }

    # Optimal stoploss
    stoploss = -0.045

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
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperparameters
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space="buy")

    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_max = IntParameter(15, 25, default=20, space="buy")

    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(25, 35, default=30, space="buy")
    rsi_overbought = IntParameter(65, 75, default=70, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.1, space="buy")

    # Keltner Channel for squeeze detection
    kc_period = IntParameter(15, 25, default=20, space="buy")
    kc_mult = DecimalParameter(1.0, 2.0, default=1.5, space="buy")

    # Exit parameters
    bb_exit_ratio = DecimalParameter(0.3, 0.7, default=0.5, space="sell")
    take_profit_pct = DecimalParameter(0.015, 0.03, default=0.02, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculate indicators for all possible hyperopt parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """

        # Pre-calculate Bollinger Bands for all possible period/std combinations
        for period in range(15, 26):
            for std in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
                bollinger = ta.BBANDS(
                    dataframe,
                    timeperiod=period,
                    nbdevup=std,
                    nbdevdn=std,
                    matype=0,
                )
                dataframe[f"bb_upper_{period}_{std}"] = bollinger["upperband"]
                dataframe[f"bb_mid_{period}_{std}"] = bollinger["middleband"]
                dataframe[f"bb_lower_{period}_{std}"] = bollinger["lowerband"]

        # Pre-calculate ADX for all possible periods
        for period in range(10, 21):
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Pre-calculate RSI for all possible periods
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all possible periods
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate Keltner Channel components for all possible periods
        for period in range(15, 26):
            dataframe[f"kc_ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
            dataframe[f"kc_atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Fixed ATR for general use
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # RSI divergence detection (simplified) - using fixed lookback
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(lookback)

        # Support/Resistance levels (using rolling min/max)
        support_period = 50
        resistance_period = 50
        dataframe["support"] = dataframe["low"].rolling(window=support_period).min()
        dataframe["resistance"] = dataframe["high"].rolling(window=resistance_period).max()

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5) * 100
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        adx_period = self.adx_period.value
        rsi_period = self.rsi_period.value
        volume_ma_period = self.volume_ma_period.value
        kc_period = self.kc_period.value
        kc_mult = self.kc_mult.value

        # Round bb_std to 1 decimal place to match pre-calculated columns
        bb_std_rounded = round(bb_std, 1)

        # Get the pre-calculated indicators for current hyperopt values
        bb_upper = dataframe[f"bb_upper_{bb_period}_{bb_std_rounded}"]
        bb_mid = dataframe[f"bb_mid_{bb_period}_{bb_std_rounded}"]
        bb_lower = dataframe[f"bb_lower_{bb_period}_{bb_std_rounded}"]
        adx = dataframe[f"adx_{adx_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        kc_ema = dataframe[f"kc_ema_{kc_period}"]
        kc_atr = dataframe[f"kc_atr_{kc_period}"]

        # Calculate derived values using hyperopt parameters
        bb_width = (bb_upper - bb_lower) / bb_mid

        volume_ok = dataframe["volume"] > (volume_ma * self.volume_threshold.value)

        kc_upper = kc_ema + (kc_atr * kc_mult)
        kc_lower = kc_ema - (kc_atr * kc_mult)

        # Bollinger Band Squeeze (BB inside KC = low volatility)
        bb_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Close position relative to bands
        close_below_lower = dataframe["close"] < bb_lower
        close_above_upper = dataframe["close"] > bb_upper

        # RSI divergence using pre-calculated price comparisons
        rsi_higher = rsi > rsi.shift(10)
        rsi_lower = rsi < rsi.shift(10)
        bullish_div = dataframe["price_lower"] & rsi_higher
        bearish_div = dataframe["price_higher"] & rsi_lower

        # Distance from support/resistance
        dist_from_support = (dataframe["close"] - dataframe["support"]) / dataframe["close"]
        dist_from_resistance = (dataframe["resistance"] - dataframe["close"]) / dataframe["close"]

        # LONG ENTRY (buy low)
        dataframe.loc[
            (
                (adx < self.adx_max.value)  # Avoid strong trend
                & (close_below_lower)  # Price pierces lower band
                & (rsi < self.rsi_oversold.value)  # RSI confirms oversold
                & (volume_ok)  # Volume confirmation
                & (bb_width > 0.01)  # BB wide enough (not squeezed)
                & (
                    (bullish_div)  # Bullish divergence
                    | (dataframe["momentum"] < -1.0)  # Or strong downward momentum (oversold)
                )
                & (dist_from_support < 0.02)  # Near support
                & (bb_squeeze.shift(1).fillna(True) == False)  # Not coming out of squeeze
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY (sell high)
        dataframe.loc[
            (
                (adx < self.adx_max.value)  # Avoid strong trend
                & (close_above_upper)  # Price pierces upper band
                & (rsi > self.rsi_overbought.value)  # RSI confirms overbought
                & (volume_ok)  # Volume confirmation
                & (bb_width > 0.01)  # BB wide enough (not squeezed)
                & (
                    (bearish_div)  # Bearish divergence
                    | (dataframe["momentum"] > 1.0)  # Or strong upward momentum (overbought)
                )
                & (dist_from_resistance < 0.02)  # Near resistance
                & (bb_squeeze.shift(1).fillna(True) == False)  # Not coming out of squeeze
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        adx_period = self.adx_period.value
        rsi_period = self.rsi_period.value

        # Round bb_std to 1 decimal place to match pre-calculated columns
        bb_std_rounded = round(bb_std, 1)

        # Get the pre-calculated indicators for current hyperopt values
        bb_upper = dataframe[f"bb_upper_{bb_period}_{bb_std_rounded}"]
        bb_mid = dataframe[f"bb_mid_{bb_period}_{bb_std_rounded}"]
        bb_lower = dataframe[f"bb_lower_{bb_period}_{bb_std_rounded}"]
        adx = dataframe[f"adx_{adx_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]

        # Calculate target exit points
        long_target = bb_lower + ((bb_mid - bb_lower) * self.bb_exit_ratio.value)
        short_target = bb_upper - ((bb_upper - bb_mid) * self.bb_exit_ratio.value)

        close_below_lower = dataframe["close"] < bb_lower
        close_above_upper = dataframe["close"] > bb_upper

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] >= long_target)  # Reached target
                | (dataframe["close"] >= bb_mid)  # Reached middle band
                | (rsi > 65)  # RSI no longer oversold
                | (adx > 30)  # Trend becoming too strong
                | (close_above_upper)  # Extreme reversal
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] <= short_target)  # Reached target
                | (dataframe["close"] <= bb_mid)  # Reached middle band
                | (rsi < 35)  # RSI no longer overbought
                | (adx > 30)  # Trend becoming too strong
                | (close_below_lower)  # Extreme reversal
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
        Custom exit logic for mean reversion
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        bb_period = self.bb_period.value
        bb_std = round(self.bb_std.value, 1)
        adx_period = self.adx_period.value
        kc_period = self.kc_period.value
        kc_mult = self.kc_mult.value

        bb_upper = last_candle[f"bb_upper_{bb_period}_{bb_std}"]
        bb_lower = last_candle[f"bb_lower_{bb_period}_{bb_std}"]
        bb_mid = last_candle[f"bb_mid_{bb_period}_{bb_std}"]
        adx = last_candle[f"adx_{adx_period}"]
        kc_ema = last_candle[f"kc_ema_{kc_period}"]
        kc_atr = last_candle[f"kc_atr_{kc_period}"]

        bb_width = (bb_upper - bb_lower) / bb_mid
        kc_upper = kc_ema + (kc_atr * kc_mult)
        kc_lower = kc_ema - (kc_atr * kc_mult)
        bb_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        # Quick profit taking for mean reversion
        if current_profit > self.take_profit_pct.value:
            return "take_profit"

        # Exit if ADX shows trending market developing
        if adx > 35:
            return "trend_developing"

        # Exit if BB squeeze is starting (volatility contraction)
        if bb_squeeze and bb_width < 0.008:
            return "bb_squeeze_exit"

        # Exit long if we hit upper band (full reversion)
        if not trade.is_short and last_candle["close"] > bb_upper:
            return "hit_upper_band"

        # Exit short if we hit lower band (full reversion)
        if trade.is_short and last_candle["close"] < bb_lower:
            return "hit_lower_band"

        # Time-based exit for stuck trades
        if current_time - trade.open_date_utc > pd.Timedelta(hours=2):
            if current_profit > -0.005:
                return "time_exit"

        # Exit if momentum shifts strongly against position
        if not trade.is_short and last_candle["momentum"] < -2.0:
            return "momentum_shift_down"

        if trade.is_short and last_candle["momentum"] > 2.0:
            return "momentum_shift_up"

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
        Custom stoploss logic for mean reversion
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        bb_period = self.bb_period.value
        bb_std = round(self.bb_std.value, 1)

        bb_upper = last_candle[f"bb_upper_{bb_period}_{bb_std}"]
        bb_lower = last_candle[f"bb_lower_{bb_period}_{bb_std}"]
        bb_mid = last_candle[f"bb_mid_{bb_period}_{bb_std}"]

        bb_width = (bb_upper - bb_lower) / bb_mid

        # Dynamic stop based on BB width
        if bb_width > 0.03:  # Wide bands = high volatility
            dynamic_stop = -0.06
        elif bb_width > 0.02:
            dynamic_stop = -0.045
        else:
            dynamic_stop = -0.035

        # Progressive stops based on profit
        if current_profit > 0.015:
            return -0.003  # Very tight stop
        elif current_profit > 0.008:
            return -0.005  # Tight stop
        elif current_profit > 0.003:
            return -0.008  # Moderate stop

        # Time-based tightening
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            return max(dynamic_stop, -0.025)

        return dynamic_stop

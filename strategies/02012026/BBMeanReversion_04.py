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
        Adds several different TA indicators to the given DataFrame
        """

        # Bollinger Bands
        bollinger = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_period.value,
            nbdevup=self.bb_std.value,
            nbdevdn=self.bb_std.value,
            matype=0,
        )
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_mid"] = bollinger["middleband"]
        dataframe["bb_lower"] = bollinger["lowerband"]

        # BB width and position
        dataframe["bb_width"] = (
            dataframe["bb_upper"] - dataframe["bb_lower"]
        ) / dataframe["bb_mid"]
        dataframe["bb_position"] = (dataframe["close"] - dataframe["bb_lower"]) / (
            dataframe["bb_upper"] - dataframe["bb_lower"]
        )

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ok"] = dataframe["volume"] > (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Keltner Channels for squeeze detection
        kc_ema = ta.EMA(dataframe, timeperiod=self.kc_period.value)
        kc_atr = ta.ATR(dataframe, timeperiod=self.kc_period.value)
        dataframe["kc_upper"] = kc_ema + (kc_atr * self.kc_mult.value)
        dataframe["kc_lower"] = kc_ema - (kc_atr * self.kc_mult.value)

        # Bollinger Band Squeeze (BB inside KC = low volatility)
        dataframe["bb_squeeze"] = (dataframe["bb_upper"] < dataframe["kc_upper"]) & (
            dataframe["bb_lower"] > dataframe["kc_lower"]
        )

        # Price touches/pierces bands
        dataframe["touches_lower"] = dataframe["low"] <= dataframe["bb_lower"]
        dataframe["touches_upper"] = dataframe["high"] >= dataframe["bb_upper"]

        # Close position relative to bands
        dataframe["close_below_lower"] = dataframe["close"] < dataframe["bb_lower"]
        dataframe["close_above_upper"] = dataframe["close"] > dataframe["bb_upper"]

        # RSI divergence detection (simplified)
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["rsi_higher"] = dataframe["rsi"] > dataframe["rsi"].shift(lookback)
        dataframe["bullish_div"] = dataframe["price_lower"] & dataframe["rsi_higher"]

        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(
            lookback
        )
        dataframe["rsi_lower"] = dataframe["rsi"] < dataframe["rsi"].shift(lookback)
        dataframe["bearish_div"] = dataframe["price_higher"] & dataframe["rsi_lower"]

        # Support/Resistance levels (using rolling min/max)
        support_period = 50
        resistance_period = 50
        dataframe["support"] = dataframe["low"].rolling(window=support_period).min()
        dataframe["resistance"] = (
            dataframe["high"].rolling(window=resistance_period).max()
        )

        # Distance from support/resistance
        dataframe["dist_from_support"] = (
            dataframe["close"] - dataframe["support"]
        ) / dataframe["close"]
        dataframe["dist_from_resistance"] = (
            dataframe["resistance"] - dataframe["close"]
        ) / dataframe["close"]

        # BB band slope for momentum
        dataframe["bb_upper_slope"] = (
            (dataframe["bb_upper"] - dataframe["bb_upper"].shift(5))
            / dataframe["bb_upper"].shift(5)
            * 100
        )
        dataframe["bb_lower_slope"] = (
            (dataframe["bb_lower"] - dataframe["bb_lower"].shift(5))
            / dataframe["bb_lower"].shift(5)
            * 100
        )

        # Price momentum
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(5))
            / dataframe["close"].shift(5)
            * 100
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY (buy low)
        dataframe.loc[
            (
                (dataframe["adx"] < self.adx_max.value)  # Avoid strong trend
                & (dataframe["close_below_lower"])  # Price pierces lower band
                & (dataframe["rsi"] < self.rsi_oversold.value)  # RSI confirms oversold
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["bb_width"] > 0.01)  # BB wide enough (not squeezed)
                & (
                    (dataframe["bullish_div"])  # Bullish divergence
                    | (
                        dataframe["momentum"] < -1.0
                    )  # Or strong downward momentum (oversold)
                )
                & (dataframe["dist_from_support"] < 0.02)  # Near support
                & (
                    dataframe["bb_squeeze"].shift(1).fillna(True) == False
                )  # Not coming out of squeeze
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY (sell high)
        dataframe.loc[
            (
                (dataframe["adx"] < self.adx_max.value)  # Avoid strong trend
                & (dataframe["close_above_upper"])  # Price pierces upper band
                & (
                    dataframe["rsi"] > self.rsi_overbought.value
                )  # RSI confirms overbought
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["bb_width"] > 0.01)  # BB wide enough (not squeezed)
                & (
                    (dataframe["bearish_div"])  # Bearish divergence
                    | (
                        dataframe["momentum"] > 1.0
                    )  # Or strong upward momentum (overbought)
                )
                & (dataframe["dist_from_resistance"] < 0.02)  # Near resistance
                & (
                    dataframe["bb_squeeze"].shift(1).fillna(True) == False
                )  # Not coming out of squeeze
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Calculate target exit points
        dataframe["long_target"] = dataframe["bb_lower"] + (
            (dataframe["bb_mid"] - dataframe["bb_lower"]) * self.bb_exit_ratio.value
        )
        dataframe["short_target"] = dataframe["bb_upper"] - (
            (dataframe["bb_upper"] - dataframe["bb_mid"]) * self.bb_exit_ratio.value
        )

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["long_target"])  # Reached target
                | (dataframe["close"] >= dataframe["bb_mid"])  # Reached middle band
                | (dataframe["rsi"] > 65)  # RSI no longer oversold
                | (dataframe["adx"] > 30)  # Trend becoming too strong
                | (dataframe["close_above_upper"])  # Extreme reversal
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["short_target"])  # Reached target
                | (dataframe["close"] <= dataframe["bb_mid"])  # Reached middle band
                | (dataframe["rsi"] < 35)  # RSI no longer overbought
                | (dataframe["adx"] > 30)  # Trend becoming too strong
                | (dataframe["close_below_lower"])  # Extreme reversal
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

        # Quick profit taking for mean reversion
        if current_profit > self.take_profit_pct.value:
            return "take_profit"

        # Exit if ADX shows trending market developing
        if last_candle["adx"] > 35:
            return "trend_developing"

        # Exit if BB squeeze is starting (volatility contraction)
        if last_candle["bb_squeeze"] and last_candle["bb_width"] < 0.008:
            return "bb_squeeze_exit"

        # Exit long if we hit upper band (full reversion)
        if not trade.is_short and last_candle["close"] > last_candle["bb_upper"]:
            return "hit_upper_band"

        # Exit short if we hit lower band (full reversion)
        if trade.is_short and last_candle["close"] < last_candle["bb_lower"]:
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

        # Dynamic stop based on BB width
        if last_candle["bb_width"] > 0.03:  # Wide bands = high volatility
            dynamic_stop = -0.06
        elif last_candle["bb_width"] > 0.02:
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

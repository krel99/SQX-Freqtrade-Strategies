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


class StochEMABounce_03(IStrategy):
    """
    Stochastic + EMA Bounce Strategy

    Trend filter with EMA; enter when Stoch is oversold/overbought and price rejects EMA.

    Improvements:
    - Enhanced EMA bounce detection with wick analysis
    - Volume profile for confirmation
    - ATR-based dynamic stops and targets
    - Stochastic divergence detection
    - Better exit timing
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
        "15": 0.025,
        "30": 0.02,
        "45": 0.015,
        "60": 0.01,
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
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Hyperparameters
    ema_period = IntParameter(40, 60, default=50, space="buy")

    stoch_k_period = IntParameter(10, 20, default=14, space="buy")
    stoch_smooth_k = IntParameter(2, 5, default=3, space="buy")
    stoch_smooth_d = IntParameter(2, 5, default=3, space="buy")

    # Stochastic thresholds
    stoch_oversold = IntParameter(15, 25, default=20, space="buy")
    stoch_overbought = IntParameter(75, 85, default=80, space="buy")

    stoch_exit_long = IntParameter(65, 75, default=70, space="sell")
    stoch_exit_short = IntParameter(25, 35, default=30, space="sell")

    # EMA bounce parameters
    ema_touch_threshold = DecimalParameter(0.001, 0.003, default=0.002, space="buy")
    wick_ratio_min = DecimalParameter(0.3, 0.6, default=0.4, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space="buy")

    # ATR parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # EMA
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Stochastic
        stoch = ta.STOCH(
            dataframe,
            fastk_period=self.stoch_k_period.value,
            slowk_period=self.stoch_smooth_k.value,
            slowd_period=self.stoch_smooth_d.value,
        )
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ok"] = dataframe["volume"] > (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        # ATR for dynamic stops
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Price position relative to EMA
        dataframe["close_pct_from_ema"] = (
            dataframe["close"] - dataframe["ema50"]
        ) / dataframe["ema50"]
        dataframe["high_pct_from_ema"] = (
            dataframe["high"] - dataframe["ema50"]
        ) / dataframe["ema50"]
        dataframe["low_pct_from_ema"] = (
            dataframe["low"] - dataframe["ema50"]
        ) / dataframe["ema50"]

        # EMA touch detection
        dataframe["ema_touch"] = (
            (abs(dataframe["close_pct_from_ema"]) < self.ema_touch_threshold.value)
            | (abs(dataframe["high_pct_from_ema"]) < self.ema_touch_threshold.value)
            | (abs(dataframe["low_pct_from_ema"]) < self.ema_touch_threshold.value)
        )

        # Wick analysis for rejection
        dataframe["body_size"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["upper_wick"] = dataframe["high"] - dataframe[["close", "open"]].max(
            axis=1
        )
        dataframe["lower_wick"] = (
            dataframe[["close", "open"]].min(axis=1) - dataframe["low"]
        )

        # Bullish rejection (lower wick through EMA, close above)
        dataframe["bullish_rejection"] = (
            (dataframe["low"] <= dataframe["ema50"])  # Low pierces EMA
            & (dataframe["close"] > dataframe["ema50"])  # Close above EMA
            & (
                dataframe["lower_wick"]
                > (dataframe["candle_range"] * self.wick_ratio_min.value)
            )  # Significant lower wick
        )

        # Bearish rejection (upper wick through EMA, close below)
        dataframe["bearish_rejection"] = (
            (dataframe["high"] >= dataframe["ema50"])  # High pierces EMA
            & (dataframe["close"] < dataframe["ema50"])  # Close below EMA
            & (
                dataframe["upper_wick"]
                > (dataframe["candle_range"] * self.wick_ratio_min.value)
            )  # Significant upper wick
        )

        # Stochastic conditions
        dataframe["stoch_oversold"] = (
            dataframe["stoch_k"] < self.stoch_oversold.value
        ) & (dataframe["stoch_d"] < self.stoch_oversold.value)

        dataframe["stoch_overbought"] = (
            dataframe["stoch_k"] > self.stoch_overbought.value
        ) & (dataframe["stoch_d"] > self.stoch_overbought.value)

        # Stochastic divergence detection (simple version)
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["stoch_higher"] = dataframe["stoch_k"] > dataframe["stoch_k"].shift(
            lookback
        )
        dataframe["bullish_div"] = dataframe["price_lower"] & dataframe["stoch_higher"]

        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(
            lookback
        )
        dataframe["stoch_lower"] = dataframe["stoch_k"] < dataframe["stoch_k"].shift(
            lookback
        )
        dataframe["bearish_div"] = dataframe["price_higher"] & dataframe["stoch_lower"]

        # Trend conditions
        dataframe["uptrend"] = dataframe["close"] > dataframe["ema50"]
        dataframe["downtrend"] = dataframe["close"] < dataframe["ema50"]

        # EMA slope for trend strength
        dataframe["ema_slope"] = (
            (dataframe["ema50"] - dataframe["ema50"].shift(5))
            / dataframe["ema50"].shift(5)
            * 100
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY
        dataframe.loc[
            (
                (dataframe["uptrend"])  # Uptrend
                & (dataframe["stoch_oversold"])  # Oversold
                & (
                    dataframe["bullish_rejection"] | dataframe["ema_touch"]
                )  # EMA bounce
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["ema_slope"] > 0)  # EMA trending up
                & (
                    dataframe["bullish_div"]  # Bonus: bullish divergence
                    | (
                        dataframe["stoch_k"] > dataframe["stoch_k"].shift(1)
                    )  # Or stoch turning up
                )
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (dataframe["downtrend"])  # Downtrend
                & (dataframe["stoch_overbought"])  # Overbought
                & (
                    dataframe["bearish_rejection"] | dataframe["ema_touch"]
                )  # EMA bounce
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["ema_slope"] < 0)  # EMA trending down
                & (
                    dataframe["bearish_div"]  # Bonus: bearish divergence
                    | (
                        dataframe["stoch_k"] < dataframe["stoch_k"].shift(1)
                    )  # Or stoch turning down
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
                (
                    dataframe["stoch_k"] > self.stoch_exit_long.value
                )  # Stochastic exit level
                | (dataframe["downtrend"])  # Trend change
                | (
                    (dataframe["stoch_k"] < dataframe["stoch_d"])  # Stoch bearish cross
                    & (dataframe["stoch_k"].shift(1) >= dataframe["stoch_d"].shift(1))
                )
                | (dataframe["bearish_rejection"])  # Strong bearish rejection
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (
                    dataframe["stoch_k"] < self.stoch_exit_short.value
                )  # Stochastic exit level
                | (dataframe["uptrend"])  # Trend change
                | (
                    (dataframe["stoch_k"] > dataframe["stoch_d"])  # Stoch bullish cross
                    & (dataframe["stoch_k"].shift(1) <= dataframe["stoch_d"].shift(1))
                )
                | (dataframe["bullish_rejection"])  # Strong bullish rejection
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
        Custom exit logic for better profit protection
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Exit if stochastic reaches extreme levels
        if not trade.is_short and last_candle["stoch_k"] > 85:
            return "stoch_extreme_overbought"

        if trade.is_short and last_candle["stoch_k"] < 15:
            return "stoch_extreme_oversold"

        # Exit on strong momentum shift
        if not trade.is_short and last_candle["ema_slope"] < -0.2:
            return "ema_momentum_down"

        if trade.is_short and last_candle["ema_slope"] > 0.2:
            return "ema_momentum_up"

        # Time-based exit if trade is not performing
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            if current_profit < 0.003:
                return "time_exit_underperforming"

        # Protect profits after significant move
        if current_profit > 0.02:
            if not trade.is_short and last_candle["stoch_k"] < last_candle["stoch_d"]:
                return "profit_protection_long"
            if trade.is_short and last_candle["stoch_k"] > last_candle["stoch_d"]:
                return "profit_protection_short"

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
        ATR-based dynamic stoploss
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Dynamic stop based on ATR
        atr_stop = -(last_candle["atr"] * self.atr_multiplier.value / trade.open_rate)

        # Use the tighter of ATR stop or default stop
        dynamic_stop = max(atr_stop, self.stoploss)

        # Progressive stops based on profit
        if current_profit > 0.02:
            return -0.005  # Very tight stop at 2% profit
        elif current_profit > 0.01:
            return -0.01  # Tight stop at 1% profit
        elif current_profit > 0.005:
            return max(dynamic_stop, -0.015)  # Moderate stop at 0.5% profit

        return dynamic_stop

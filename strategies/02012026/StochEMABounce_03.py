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

    timeframe = "15m"

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
        Pre-calculates all indicator variants for hyperopt compatibility.
        """

        # Pre-calculate EMA for all possible periods (40-60)
        for period in range(40, 61):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate Stochastic for all combinations
        # stoch_k_period: 10-20, stoch_smooth_k: 2-5, stoch_smooth_d: 2-5
        for k_period in range(10, 21):
            for smooth_k in range(2, 6):
                for smooth_d in range(2, 6):
                    stoch = ta.STOCH(
                        dataframe,
                        fastk_period=k_period,
                        slowk_period=smooth_k,
                        slowd_period=smooth_d,
                    )
                    dataframe[f"stoch_k_{k_period}_{smooth_k}_{smooth_d}"] = stoch["slowk"]
                    dataframe[f"stoch_d_{k_period}_{smooth_k}_{smooth_d}"] = stoch["slowd"]

        # Pre-calculate Volume MA for all periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate ATR for all periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Wick analysis for rejection (no hyperopt params here)
        dataframe["body_size"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["upper_wick"] = dataframe["high"] - dataframe[["close", "open"]].max(axis=1)
        dataframe["lower_wick"] = dataframe[["close", "open"]].min(axis=1) - dataframe["low"]

        # Stochastic divergence detection (simple version) - uses fixed lookback
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(lookback)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # Get current hyperopt parameter values
        ema_period = self.ema_period.value
        stoch_k_period = self.stoch_k_period.value
        stoch_smooth_k = self.stoch_smooth_k.value
        stoch_smooth_d = self.stoch_smooth_d.value
        volume_ma_period = self.volume_ma_period.value
        volume_threshold = self.volume_threshold.value
        ema_touch_threshold = self.ema_touch_threshold.value
        wick_ratio_min = self.wick_ratio_min.value
        stoch_oversold = self.stoch_oversold.value
        stoch_overbought = self.stoch_overbought.value

        # Select pre-calculated indicators
        ema = dataframe[f"ema_{ema_period}"]
        stoch_k = dataframe[f"stoch_k_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]
        stoch_d = dataframe[f"stoch_d_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]

        # Calculate derived indicators using selected values
        close_pct_from_ema = (dataframe["close"] - ema) / ema
        high_pct_from_ema = (dataframe["high"] - ema) / ema
        low_pct_from_ema = (dataframe["low"] - ema) / ema

        # EMA touch detection
        ema_touch = (
            (abs(close_pct_from_ema) < ema_touch_threshold)
            | (abs(high_pct_from_ema) < ema_touch_threshold)
            | (abs(low_pct_from_ema) < ema_touch_threshold)
        )

        # Volume check
        volume_ok = dataframe["volume"] > (volume_ma * volume_threshold)

        # Bullish rejection (lower wick through EMA, close above)
        bullish_rejection = (
            (dataframe["low"] <= ema)  # Low pierces EMA
            & (dataframe["close"] > ema)  # Close above EMA
            & (dataframe["lower_wick"] > (dataframe["candle_range"] * wick_ratio_min))
        )

        # Bearish rejection (upper wick through EMA, close below)
        bearish_rejection = (
            (dataframe["high"] >= ema)  # High pierces EMA
            & (dataframe["close"] < ema)  # Close below EMA
            & (dataframe["upper_wick"] > (dataframe["candle_range"] * wick_ratio_min))
        )

        # Stochastic conditions
        stoch_oversold_cond = (stoch_k < stoch_oversold) & (stoch_d < stoch_oversold)
        stoch_overbought_cond = (stoch_k > stoch_overbought) & (stoch_d > stoch_overbought)

        # Trend conditions
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema

        # EMA slope for trend strength
        ema_slope = (ema - ema.shift(5)) / ema.shift(5) * 100

        # Stochastic divergence
        lookback = 10
        stoch_higher = stoch_k > stoch_k.shift(lookback)
        stoch_lower = stoch_k < stoch_k.shift(lookback)
        bullish_div = dataframe["price_lower"] & stoch_higher
        bearish_div = dataframe["price_higher"] & stoch_lower

        # LONG ENTRY
        dataframe.loc[
            (
                (uptrend)  # Uptrend
                & (stoch_oversold_cond)  # Oversold
                & (bullish_rejection | ema_touch)  # EMA bounce
                & (volume_ok)  # Volume confirmation
                & (ema_slope > 0)  # EMA trending up
                & (
                    bullish_div  # Bonus: bullish divergence
                    | (stoch_k > stoch_k.shift(1))  # Or stoch turning up
                )
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (downtrend)  # Downtrend
                & (stoch_overbought_cond)  # Overbought
                & (bearish_rejection | ema_touch)  # EMA bounce
                & (volume_ok)  # Volume confirmation
                & (ema_slope < 0)  # EMA trending down
                & (
                    bearish_div  # Bonus: bearish divergence
                    | (stoch_k < stoch_k.shift(1))  # Or stoch turning down
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
        ema_period = self.ema_period.value
        stoch_k_period = self.stoch_k_period.value
        stoch_smooth_k = self.stoch_smooth_k.value
        stoch_smooth_d = self.stoch_smooth_d.value
        stoch_exit_long = self.stoch_exit_long.value
        stoch_exit_short = self.stoch_exit_short.value
        wick_ratio_min = self.wick_ratio_min.value

        # Select pre-calculated indicators
        ema = dataframe[f"ema_{ema_period}"]
        stoch_k = dataframe[f"stoch_k_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]
        stoch_d = dataframe[f"stoch_d_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]

        # Calculate derived indicators
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema

        # Bullish rejection (lower wick through EMA, close above)
        bullish_rejection = (
            (dataframe["low"] <= ema)
            & (dataframe["close"] > ema)
            & (dataframe["lower_wick"] > (dataframe["candle_range"] * wick_ratio_min))
        )

        # Bearish rejection (upper wick through EMA, close below)
        bearish_rejection = (
            (dataframe["high"] >= ema)
            & (dataframe["close"] < ema)
            & (dataframe["upper_wick"] > (dataframe["candle_range"] * wick_ratio_min))
        )

        # LONG EXIT
        dataframe.loc[
            (
                (stoch_k > stoch_exit_long)  # Stochastic exit level
                | (downtrend)  # Trend change
                | (
                    (stoch_k < stoch_d)  # Stoch bearish cross
                    & (stoch_k.shift(1) >= stoch_d.shift(1))
                )
                | (bearish_rejection)  # Strong bearish rejection
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (stoch_k < stoch_exit_short)  # Stochastic exit level
                | (uptrend)  # Trend change
                | (
                    (stoch_k > stoch_d)  # Stoch bullish cross
                    & (stoch_k.shift(1) <= stoch_d.shift(1))
                )
                | (bullish_rejection)  # Strong bullish rejection
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

        # Get current hyperopt parameter values
        ema_period = self.ema_period.value
        stoch_k_period = self.stoch_k_period.value
        stoch_smooth_k = self.stoch_smooth_k.value
        stoch_smooth_d = self.stoch_smooth_d.value

        # Select pre-calculated indicators
        stoch_k = last_candle[f"stoch_k_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]
        stoch_d = last_candle[f"stoch_d_{stoch_k_period}_{stoch_smooth_k}_{stoch_smooth_d}"]
        ema = last_candle[f"ema_{ema_period}"]

        # Calculate EMA slope from dataframe
        ema_col = dataframe[f"ema_{ema_period}"]
        ema_slope = (
            ((ema_col.iloc[-1] - ema_col.iloc[-6]) / ema_col.iloc[-6] * 100)
            if len(dataframe) > 5
            else 0
        )

        # Exit if stochastic reaches extreme levels
        if not trade.is_short and stoch_k > 85:
            return "stoch_extreme_overbought"

        if trade.is_short and stoch_k < 15:
            return "stoch_extreme_oversold"

        # Exit on strong momentum shift
        if not trade.is_short and ema_slope < -0.2:
            return "ema_momentum_down"

        if trade.is_short and ema_slope > 0.2:
            return "ema_momentum_up"

        # Time-based exit if trade is not performing
        if current_time - trade.open_date_utc > pd.Timedelta(hours=1):
            if current_profit < 0.003:
                return "time_exit_underperforming"

        # Protect profits after significant move
        if current_profit > 0.02:
            if not trade.is_short and stoch_k < stoch_d:
                return "profit_protection_long"
            if trade.is_short and stoch_k > stoch_d:
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

        # Get current hyperopt parameter values
        atr_period = self.atr_period.value
        atr_multiplier = self.atr_multiplier.value

        # Select pre-calculated ATR
        atr = last_candle[f"atr_{atr_period}"]

        # Dynamic stop based on ATR
        atr_stop = -(atr * atr_multiplier / trade.open_rate)

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

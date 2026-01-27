# --- Do not remove these imports ---
from functools import reduce
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


class MarketMicrostructureFlow(IStrategy):
    """
    Simplified Market Microstructure & Order Flow Strategy

    Core concepts:
    - Volume delta analysis (simplified)
    - ATR-based adaptive bands
    - Trend detection with EMAs
    - RSI and MACD for momentum

    Optimized for 15m futures trading
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "15m"

    # Can short - futures compatible
    can_short = True

    # ROI table - conservative for futures
    minimal_roi = {"0": 0.08, "30": 0.04, "90": 0.02, "180": 0.01}

    # Stoploss
    stoploss = -0.05  # Tighter stop for futures

    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order types mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    # Hyperparameters

    # Volume Delta Parameters
    delta_period = IntParameter(5, 20, default=10, space="buy")
    delta_threshold = DecimalParameter(0.1, 0.5, default=0.25, decimals=2, space="buy")

    # ATR Bands Parameters
    atr_period = IntParameter(10, 30, default=14, space="buy")
    band_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")

    # Trend Parameters
    ema_fast = IntParameter(5, 15, default=9, space="buy")
    ema_slow = IntParameter(20, 50, default=21, space="buy")
    ema_baseline = IntParameter(50, 100, default=50, space="buy")

    # Momentum Parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_buy = IntParameter(25, 45, default=35, space="buy")
    rsi_sell = IntParameter(55, 75, default=65, space="buy")

    # MACD Parameters
    macd_fast = IntParameter(8, 16, default=12, space="buy")
    macd_slow = IntParameter(20, 30, default=26, space="buy")
    macd_signal = IntParameter(7, 12, default=9, space="buy")

    # Entry Thresholds
    min_trend_strength = DecimalParameter(0.0, 1.0, default=0.3, decimals=1, space="buy")
    volume_factor = DecimalParameter(0.8, 2.0, default=1.2, decimals=1, space="buy")

    # Exit Parameters
    take_profit = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="sell")
    atr_exit_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators - simplified version
        """

        # Volume Delta - Simple approximation
        # Buy pressure when close > (high + low) / 2
        mid_point = (dataframe["high"] + dataframe["low"]) / 2

        # Calculate buy and sell volume estimates
        buy_ratio = (dataframe["close"] - dataframe["low"]) / (
            dataframe["high"] - dataframe["low"] + 0.0001
        )
        buy_ratio = buy_ratio.clip(0, 1)

        dataframe["buy_volume"] = dataframe["volume"] * buy_ratio
        dataframe["sell_volume"] = dataframe["volume"] * (1 - buy_ratio)
        dataframe["volume_delta"] = dataframe["buy_volume"] - dataframe["sell_volume"]

        # Smooth volume delta
        dataframe["delta_ma"] = (
            dataframe["volume_delta"].rolling(window=self.delta_period.value).mean()
        )
        dataframe["delta_std"] = (
            dataframe["volume_delta"].rolling(window=self.delta_period.value).std()
        )

        # Delta normalized (z-score)
        dataframe["delta_zscore"] = (dataframe["volume_delta"] - dataframe["delta_ma"]) / (
            dataframe["delta_std"] + 0.0001
        )

        # ATR and Adaptive Bands
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Use EMA as middle line
        dataframe["middle_band"] = ta.EMA(dataframe, timeperiod=self.atr_period.value)
        dataframe["upper_band"] = dataframe["middle_band"] + (
            dataframe["atr"] * self.band_mult.value
        )
        dataframe["lower_band"] = dataframe["middle_band"] - (
            dataframe["atr"] * self.band_mult.value
        )

        # Band position (0 = at lower, 1 = at upper)
        band_range = dataframe["upper_band"] - dataframe["lower_band"]
        dataframe["band_position"] = (dataframe["close"] - dataframe["lower_band"]) / (
            band_range + 0.0001
        )
        dataframe["band_position"] = dataframe["band_position"].clip(0, 1)

        # EMAs for trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["ema_baseline"] = ta.EMA(dataframe, timeperiod=self.ema_baseline.value)

        # Trend strength (-1 to 1)
        dataframe["trend"] = 0
        dataframe.loc[dataframe["ema_fast"] > dataframe["ema_slow"], "trend"] = 1
        dataframe.loc[dataframe["ema_fast"] < dataframe["ema_slow"], "trend"] = -1

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Volume metrics
        dataframe["volume_ma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        # OBV (On Balance Volume)
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_ma"] = dataframe["obv"].rolling(window=20).mean()
        dataframe["obv_trend"] = dataframe["obv"] > dataframe["obv_ma"]

        # CMF (Chaikin Money Flow)
        mfm = (
            (dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"] + 0.0001)
        mfv = mfm * dataframe["volume"]
        dataframe["cmf"] = mfv.rolling(window=20).sum() / (
            dataframe["volume"].rolling(window=20).sum() + 0.0001
        )

        # VWAP
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        cum_volume = dataframe["volume"].cumsum()
        cum_pv = (typical_price * dataframe["volume"]).cumsum()
        dataframe["vwap"] = cum_pv / (cum_volume + 0.0001)

        # Distance from VWAP
        dataframe["vwap_distance"] = (dataframe["close"] - dataframe["vwap"]) / dataframe["close"]

        # Bollinger Bands for additional reference
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals - simplified logic
        """

        # Long Entry Conditions
        long_conditions = []

        # 1. Price at or near lower band
        long_conditions.append(
            (dataframe["close"] <= dataframe["lower_band"] * 1.01)
            | (dataframe["band_position"] < 0.2)
        )

        # 2. Positive volume delta or recovering
        long_conditions.append(
            (dataframe["delta_ma"] > 0) | (dataframe["delta_zscore"] > self.delta_threshold.value)
        )

        # 3. RSI not oversold but not overbought
        long_conditions.append((dataframe["rsi"] > self.rsi_buy.value) & (dataframe["rsi"] < 60))

        # 4. Trend is up or neutral
        long_conditions.append(dataframe["trend"] >= 0)

        # 5. MACD histogram rising
        long_conditions.append(dataframe["macdhist"] > dataframe["macdhist"].shift(1))

        # 6. Volume above threshold
        long_conditions.append(dataframe["volume_ratio"] > self.volume_factor.value)

        # Combine conditions (need at least 4 of 6)
        if long_conditions:
            long_count = reduce(lambda x, y: x.astype(int) + y.astype(int), long_conditions)
            dataframe.loc[long_count >= 4, "enter_long"] = 1

        # Short Entry Conditions
        short_conditions = []

        # 1. Price at or near upper band
        short_conditions.append(
            (dataframe["close"] >= dataframe["upper_band"] * 0.99)
            | (dataframe["band_position"] > 0.8)
        )

        # 2. Negative volume delta
        short_conditions.append(
            (dataframe["delta_ma"] < 0) | (dataframe["delta_zscore"] < -self.delta_threshold.value)
        )

        # 3. RSI not overbought but not oversold
        short_conditions.append((dataframe["rsi"] < self.rsi_sell.value) & (dataframe["rsi"] > 40))

        # 4. Trend is down or neutral
        short_conditions.append(dataframe["trend"] <= 0)

        # 5. MACD histogram falling
        short_conditions.append(dataframe["macdhist"] < dataframe["macdhist"].shift(1))

        # 6. Volume above threshold
        short_conditions.append(dataframe["volume_ratio"] > self.volume_factor.value)

        # Combine conditions (need at least 4 of 6)
        if short_conditions:
            short_count = reduce(lambda x, y: x.astype(int) + y.astype(int), short_conditions)
            dataframe.loc[short_count >= 4, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals - simplified logic
        """

        # Long Exit Conditions
        long_exit = (
            # Price reached upper band
            (dataframe["close"] >= dataframe["upper_band"] * self.atr_exit_mult.value)
            |
            # RSI overbought
            (dataframe["rsi"] > 70)
            |
            # Strong negative delta
            (dataframe["delta_zscore"] < -1.5)
            |
            # MACD bearish cross
            (
                (dataframe["macd"] < dataframe["macdsignal"])
                & (dataframe["macd"].shift(1) >= dataframe["macdsignal"].shift(1))
            )
        )

        dataframe.loc[long_exit, "exit_long"] = 1

        # Short Exit Conditions
        short_exit = (
            # Price reached lower band
            (dataframe["close"] <= dataframe["lower_band"] / self.atr_exit_mult.value)
            |
            # RSI oversold
            (dataframe["rsi"] < 30)
            |
            # Strong positive delta
            (dataframe["delta_zscore"] > 1.5)
            |
            # MACD bullish cross
            (
                (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["macd"].shift(1) <= dataframe["macdsignal"].shift(1))
            )
        )

        dataframe.loc[short_exit, "exit_short"] = 1

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
        Custom exit logic
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # Quick profit take
        if current_profit > self.take_profit.value:
            if last_candle["volume_ratio"] > 2.0:
                return "high_volume_profit"

        # Exit on strong delta reversal
        if trade.is_short:
            if last_candle["delta_zscore"] > 2.0:
                return "delta_reversal"
        else:  # Long trade
            if last_candle["delta_zscore"] < -2.0:
                return "delta_reversal"

        # Exit if too far from VWAP
        if abs(last_candle["vwap_distance"]) > 0.03:
            if current_profit > 0:
                return "vwap_deviation"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Custom stoploss using ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return -0.05  # Default stoploss

        last_candle = dataframe.iloc[-1]

        # ATR-based stop
        atr_stop = (last_candle["atr"] * 2.0) / current_rate

        # Tighten stop when in profit
        if current_profit > 0.02:
            return -min(atr_stop, 0.01)
        elif current_profit > 0.01:
            return -min(atr_stop, 0.02)
        else:
            return -min(atr_stop, 0.05)

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Adjust leverage based on volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return 1.0

        last_candle = dataframe.iloc[-1]

        # Use band width as volatility measure
        band_width = (last_candle["upper_band"] - last_candle["lower_band"]) / last_candle[
            "middle_band"
        ]

        if band_width < 0.02:  # Low volatility
            return min(3.0, max_leverage)
        elif band_width < 0.04:  # Medium volatility
            return min(2.0, max_leverage)
        else:  # High volatility
            return min(1.0, max_leverage)

# --- Do not remove these imports ---
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


class VolumeProfileAdaptiveStrategy(IStrategy):
    """
    Volume Profile Adaptive Strategy

    This strategy uses volume-based indicators combined with trend and momentum
    indicators that are distinct from typical MA/BB/RSI combinations. It employs:
    - Volume Profile indicators (VWAP, OBV, CMF, MFI)
    - Ichimoku Cloud for trend identification
    - Parabolic SAR for trend reversals
    - Williams %R for oversold/overbought conditions
    - ADX for trend strength
    - CCI for momentum

    All parameters are hyperoptimizable for finding the optimal configuration.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = "15m"

    # Can short
    can_short = True

    # ROI table
    minimal_roi = {"0": 0.12, "30": 0.06, "60": 0.03, "120": 0.01}

    # Stoploss
    stoploss = -0.08

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # Exit signal
    use_exit_signal = CategoricalParameter([True, False], default=True, space="sell")

    # -------------------------------------------------------------------------
    # Hyperoptimizable parameters for indicators
    # -------------------------------------------------------------------------

    # Ichimoku Cloud parameters
    ichi_conversion = IntParameter(7, 12, default=9, space="buy")
    ichi_base = IntParameter(20, 35, default=26, space="buy")
    ichi_span_b = IntParameter(40, 65, default=52, space="buy")
    ichi_displacement = IntParameter(20, 35, default=26, space="buy")

    # Parabolic SAR parameters
    sar_acceleration = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="buy")
    sar_maximum = DecimalParameter(0.1, 0.3, default=0.2, decimals=2, space="buy")

    # Williams %R parameters
    willr_period = IntParameter(7, 21, default=14, space="buy")
    willr_buy_threshold = IntParameter(-95, -70, default=-80, space="buy")
    willr_sell_threshold = IntParameter(-30, -5, default=-20, space="sell")

    # ADX parameters
    adx_period = IntParameter(10, 30, default=14, space="buy")
    adx_strength = IntParameter(15, 40, default=25, space="buy")

    # CCI parameters
    cci_period = IntParameter(10, 30, default=20, space="buy")
    cci_buy_threshold = IntParameter(-150, -80, default=-100, space="buy")
    cci_sell_threshold = IntParameter(80, 150, default=100, space="sell")

    # On-Balance Volume (OBV) parameters
    obv_ema_short = IntParameter(3, 15, default=5, space="buy")
    obv_ema_long = IntParameter(15, 50, default=21, space="buy")

    # Chaikin Money Flow parameters
    cmf_period = IntParameter(10, 30, default=20, space="buy")
    cmf_buy_threshold = DecimalParameter(-0.3, 0.1, default=-0.05, decimals=2, space="buy")
    cmf_sell_threshold = DecimalParameter(-0.1, 0.3, default=0.05, decimals=2, space="sell")

    # Money Flow Index parameters
    mfi_period = IntParameter(10, 25, default=14, space="buy")
    mfi_buy_threshold = IntParameter(10, 35, default=20, space="buy")
    mfi_sell_threshold = IntParameter(65, 90, default=80, space="sell")

    # Volume parameters
    volume_ma_period = IntParameter(10, 50, default=20, space="buy")
    volume_threshold = DecimalParameter(0.8, 2.5, default=1.2, decimals=1, space="buy")

    # -------------------------------------------------------------------------
    # Entry/Exit logic parameters
    # -------------------------------------------------------------------------

    # Minimum conditions required for entry
    min_conditions_long = IntParameter(2, 5, default=3, space="buy")
    min_conditions_short = IntParameter(2, 5, default=3, space="sell")

    # Ichimoku position requirements
    require_price_above_cloud = CategoricalParameter([True, False], default=True, space="buy")
    require_price_below_cloud = CategoricalParameter([True, False], default=True, space="sell")

    # Trend strength requirement
    trend_strength_weight = DecimalParameter(0.0, 1.0, default=0.6, decimals=1, space="buy")

    def calculate_ichimoku(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate Ichimoku Cloud indicators manually
        """
        # Get parameter values
        conversion_period = self.ichi_conversion.value
        base_period = self.ichi_base.value
        span_b_period = self.ichi_span_b.value
        displacement = self.ichi_displacement.value

        # Tenkan-sen (Conversion Line)
        high_conversion = dataframe["high"].rolling(window=conversion_period).max()
        low_conversion = dataframe["low"].rolling(window=conversion_period).min()
        dataframe["ichi_tenkan"] = (high_conversion + low_conversion) / 2

        # Kijun-sen (Base Line)
        high_base = dataframe["high"].rolling(window=base_period).max()
        low_base = dataframe["low"].rolling(window=base_period).min()
        dataframe["ichi_kijun"] = (high_base + low_base) / 2

        # Senkou Span A (Leading Span A) - plotted displacement periods ahead
        dataframe["ichi_senkou_a"] = (
            (dataframe["ichi_tenkan"] + dataframe["ichi_kijun"]) / 2
        ).shift(displacement)

        # Senkou Span B (Leading Span B) - plotted displacement periods ahead
        high_span_b = dataframe["high"].rolling(window=span_b_period).max()
        low_span_b = dataframe["low"].rolling(window=span_b_period).min()
        dataframe["ichi_senkou_b"] = ((high_span_b + low_span_b) / 2).shift(displacement)

        # Chikou Span (Lagging Span) - close plotted displacement periods behind
        dataframe["ichi_chikou"] = dataframe["close"].shift(-displacement)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds custom indicators to the dataframe
        """

        # Ichimoku Cloud (manual implementation)
        dataframe = self.calculate_ichimoku(dataframe)

        # Cloud thickness (volatility indicator)
        dataframe["cloud_thickness"] = abs(dataframe["ichi_senkou_a"] - dataframe["ichi_senkou_b"])
        dataframe["cloud_direction"] = np.where(
            dataframe["ichi_senkou_a"] > dataframe["ichi_senkou_b"], 1, -1
        )

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(
            dataframe, acceleration=self.sar_acceleration.value, maximum=self.sar_maximum.value
        )
        dataframe["sar_position"] = np.where(dataframe["close"] > dataframe["sar"], 1, -1)

        # Williams %R
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=self.willr_period.value)

        # ADX (Average Directional Index)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period.value)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period.value)
        dataframe["di_diff"] = dataframe["plus_di"] - dataframe["minus_di"]

        # CCI (Commodity Channel Index)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.cci_period.value)

        # On-Balance Volume and EMAs
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_ema_short"] = ta.EMA(dataframe["obv"], timeperiod=self.obv_ema_short.value)
        dataframe["obv_ema_long"] = ta.EMA(dataframe["obv"], timeperiod=self.obv_ema_long.value)
        dataframe["obv_signal"] = np.where(
            dataframe["obv_ema_short"] > dataframe["obv_ema_long"], 1, -1
        )

        # Chaikin Money Flow
        dataframe["cmf"] = self.calculate_cmf(dataframe, self.cmf_period.value)

        # Money Flow Index
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=self.mfi_period.value)

        # VWAP (Volume Weighted Average Price)
        dataframe["vwap"] = self.calculate_vwap(dataframe)
        dataframe["price_vs_vwap"] = (dataframe["close"] - dataframe["vwap"]) / dataframe["vwap"]

        # Volume analysis
        dataframe["volume_mean"] = (
            dataframe["volume"].rolling(window=self.volume_ma_period.value).mean()
        )
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean"]

        # Price position relative to Ichimoku cloud
        dataframe["price_above_cloud"] = np.where(
            dataframe["close"] > dataframe[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1), 1, 0
        )
        dataframe["price_below_cloud"] = np.where(
            dataframe["close"] < dataframe[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1), 1, 0
        )
        dataframe["price_in_cloud"] = np.where(
            (dataframe["price_above_cloud"] == 0) & (dataframe["price_below_cloud"] == 0), 1, 0
        )

        # Trend strength indicator combining ADX and cloud
        dataframe["trend_strength"] = (dataframe["adx"] / 100) * 0.5 + abs(
            dataframe["cloud_direction"]
        ) * abs(dataframe["cloud_thickness"] / dataframe["close"]) * 0.5

        # Handle NaN values by forward filling
        dataframe.ffill(inplace=True)

        return dataframe

    def calculate_cmf(self, dataframe: DataFrame, period: int) -> pd.Series:
        """
        Calculate Chaikin Money Flow
        """
        # Calculate money flow multiplier
        mf_multiplier = (
            (dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"])

        # Handle division by zero
        mf_multiplier = mf_multiplier.fillna(0)

        # Calculate money flow volume
        mf_volume = mf_multiplier * dataframe["volume"]

        # Calculate CMF
        cmf = (
            mf_volume.rolling(window=period).sum()
            / dataframe["volume"].rolling(window=period).sum()
        )

        return cmf

    def calculate_vwap(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)
        """
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3

        # Calculate cumulative values
        cumulative_volume = dataframe["volume"].cumsum()
        cumulative_pv = (typical_price * dataframe["volume"]).cumsum()

        # Initialize VWAP
        vwap = cumulative_pv / cumulative_volume

        # Reset VWAP daily (approximate for 15m timeframe)
        periods_per_day = 96  # 24 hours * 4 (15-minute periods)

        for i in range(0, len(dataframe), periods_per_day):
            end_idx = min(i + periods_per_day, len(dataframe))
            if i > 0:
                # Reset cumulative calculations for each day
                day_volume = dataframe["volume"].iloc[i:end_idx]
                day_typical_price = typical_price.iloc[i:end_idx]

                day_cumulative_volume = day_volume.cumsum()
                day_cumulative_pv = (day_typical_price * day_volume).cumsum()

                vwap.iloc[i:end_idx] = day_cumulative_pv / day_cumulative_volume

        return vwap

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals
        """
        # Long entry conditions
        conditions_long = []

        # 1. Price above cloud (bullish)
        if self.require_price_above_cloud:
            conditions_long.append(dataframe["price_above_cloud"] == 1)

        # 2. SAR below price (bullish trend)
        conditions_long.append(dataframe["sar_position"] == 1)

        # 3. Williams %R oversold
        conditions_long.append(dataframe["willr"] <= self.willr_buy_threshold.value)

        # 4. CCI oversold
        conditions_long.append(dataframe["cci"] <= self.cci_buy_threshold.value)

        # 5. OBV trending up
        conditions_long.append(dataframe["obv_signal"] == 1)

        # 6. CMF positive or recovering
        conditions_long.append(dataframe["cmf"] >= self.cmf_buy_threshold.value)

        # 7. MFI oversold
        conditions_long.append(dataframe["mfi"] <= self.mfi_buy_threshold.value)

        # 8. ADX showing trend strength
        conditions_long.append(dataframe["adx"] >= self.adx_strength.value)

        # 9. DI+ > DI- (bullish momentum)
        conditions_long.append(dataframe["di_diff"] > 0)

        # 10. Price near or below VWAP (value entry)
        conditions_long.append(dataframe["price_vs_vwap"] <= 0.01)

        # 11. Volume confirmation
        conditions_long.append(dataframe["volume_ratio"] >= self.volume_threshold.value)

        # 12. Cloud bullish
        conditions_long.append(dataframe["cloud_direction"] == 1)

        # Count satisfied conditions
        long_conditions_count = reduce(lambda x, y: x + y.astype(int), conditions_long, 0)

        # Apply trend strength weight
        weighted_conditions = long_conditions_count + (
            dataframe["trend_strength"] * self.trend_strength_weight.value
        )

        # Handle NaN values - set them to 0 to prevent entry
        weighted_conditions = weighted_conditions.fillna(0)

        dataframe.loc[
            (weighted_conditions >= self.min_conditions_long.value)
            & (dataframe["volume"] > 0)
            & (weighted_conditions.notna()),
            "enter_long",
        ] = 1

        # Short entry conditions
        conditions_short = []

        # 1. Price below cloud (bearish)
        if self.require_price_below_cloud:
            conditions_short.append(dataframe["price_below_cloud"] == 1)

        # 2. SAR above price (bearish trend)
        conditions_short.append(dataframe["sar_position"] == -1)

        # 3. Williams %R overbought
        conditions_short.append(dataframe["willr"] >= self.willr_sell_threshold.value)

        # 4. CCI overbought
        conditions_short.append(dataframe["cci"] >= self.cci_sell_threshold.value)

        # 5. OBV trending down
        conditions_short.append(dataframe["obv_signal"] == -1)

        # 6. CMF negative
        conditions_short.append(dataframe["cmf"] <= self.cmf_sell_threshold.value)

        # 7. MFI overbought
        conditions_short.append(dataframe["mfi"] >= self.mfi_sell_threshold.value)

        # 8. ADX showing trend strength
        conditions_short.append(dataframe["adx"] >= self.adx_strength.value)

        # 9. DI- > DI+ (bearish momentum)
        conditions_short.append(dataframe["di_diff"] < 0)

        # 10. Price above VWAP (overvalued)
        conditions_short.append(dataframe["price_vs_vwap"] >= 0.01)

        # 11. Volume confirmation
        conditions_short.append(dataframe["volume_ratio"] >= self.volume_threshold.value)

        # 12. Cloud bearish
        conditions_short.append(dataframe["cloud_direction"] == -1)

        # Count satisfied conditions
        short_conditions_count = reduce(lambda x, y: x + y.astype(int), conditions_short, 0)

        # Apply trend strength weight
        weighted_conditions_short = short_conditions_count + (
            dataframe["trend_strength"] * self.trend_strength_weight.value
        )

        # Handle NaN values - set them to 0 to prevent entry
        weighted_conditions_short = weighted_conditions_short.fillna(0)

        dataframe.loc[
            (weighted_conditions_short >= self.min_conditions_short.value)
            & (dataframe["volume"] > 0)
            & (weighted_conditions_short.notna()),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals
        """
        # Exit long conditions (any of these trigger exit)
        exit_long_conditions = []

        # 1. SAR flipped to bearish
        exit_long_conditions.append(dataframe["sar_position"] == -1)

        # 2. Williams %R overbought
        exit_long_conditions.append(dataframe["willr"] >= self.willr_sell_threshold.value)

        # 3. CCI overbought
        exit_long_conditions.append(dataframe["cci"] >= self.cci_sell_threshold.value)

        # 4. MFI overbought
        exit_long_conditions.append(dataframe["mfi"] >= self.mfi_sell_threshold.value)

        # 5. Price entered cloud from above
        exit_long_conditions.append(
            (dataframe["price_in_cloud"] == 1) & (dataframe["price_above_cloud"].shift(1) == 1)
        )

        # 6. CMF turned negative
        exit_long_conditions.append((dataframe["cmf"] < 0) & (dataframe["cmf"].shift(1) >= 0))

        # Any condition triggers exit
        if self.use_exit_signal:
            exit_long_signal = reduce(lambda x, y: x | y, exit_long_conditions)
            # Only set exit where signal is True and not NaN
            dataframe.loc[(exit_long_signal == True) & (exit_long_signal.notna()), "exit_long"] = 1

        # Exit short conditions (any of these trigger exit)
        exit_short_conditions = []

        # 1. SAR flipped to bullish
        exit_short_conditions.append(dataframe["sar_position"] == 1)

        # 2. Williams %R oversold
        exit_short_conditions.append(dataframe["willr"] <= self.willr_buy_threshold.value)

        # 3. CCI oversold
        exit_short_conditions.append(dataframe["cci"] <= self.cci_buy_threshold.value)

        # 4. MFI oversold
        exit_short_conditions.append(dataframe["mfi"] <= self.mfi_buy_threshold.value)

        # 5. Price entered cloud from below
        exit_short_conditions.append(
            (dataframe["price_in_cloud"] == 1) & (dataframe["price_below_cloud"].shift(1) == 1)
        )

        # 6. CMF turned positive
        exit_short_conditions.append((dataframe["cmf"] > 0) & (dataframe["cmf"].shift(1) <= 0))

        # Any condition triggers exit
        if self.use_exit_signal:
            exit_short_signal = reduce(lambda x, y: x | y, exit_short_conditions)
            # Only set exit where signal is True and not NaN
            dataframe.loc[
                (exit_short_signal == True) & (exit_short_signal.notna()), "exit_short"
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
    ) -> Optional[str]:
        """
        Custom exit logic based on trade performance
        """
        # Dynamic exit based on profit and trade duration
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60

        # Quick profit taking
        if current_profit > 0.02 and trade_duration < 30:
            return "quick_profit"

        # Extended trade management
        if trade_duration > 180:  # 3 hours
            if current_profit > 0.005:
                return "time_profit"
            elif current_profit < -0.05:
                return "time_loss"

        return None

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
        Customize leverage for futures trading based on market conditions
        """
        # Conservative leverage approach
        return min(3.0, max_leverage)

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Dynamic stoploss based on trade performance
        """
        # Tighten stoploss as profit increases
        if current_profit > 0.05:
            return -0.02  # 2% trailing stop when 5% profit
        elif current_profit > 0.03:
            return -0.03  # 3% trailing stop when 3% profit
        elif current_profit > 0.01:
            return -0.05  # 5% trailing stop when 1% profit

        # Regular stoploss
        return self.stoploss

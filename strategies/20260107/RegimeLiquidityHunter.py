# --- Do not remove these imports ---
from functools import reduce
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from scipy import stats

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


class RegimeLiquidityHunter(IStrategy):
    """
    Regime-Switching Liquidity Hunter Strategy

    This strategy identifies market regimes (trending vs mean-reverting) and hunts
    for liquidity pools where stop losses typically cluster. It uses statistical
    analysis to identify high-probability entry/exit points based on the current
    market regime.

    Key Features:
    - Market regime detection (trending/ranging/volatile)
    - Liquidity pool identification
    - Statistical z-score and percentile rank analysis
    - Adaptive behavior based on volatility
    - Different logic for different market conditions
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "15m"

    # Can short - futures compatible
    can_short = True

    # ROI table - adaptive based on regime
    minimal_roi = {"0": 0.06, "45": 0.03, "90": 0.015, "180": 0.01}

    # Stoploss
    stoploss = -0.04  # Tighter stop for liquidity hunting

    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 120

    # Trade tracking for protection
    recent_trades_tracker = {}
    consecutive_losses = {}
    last_loss_candle = {}  # Track when last loss occurred for cooldowns

    # Hyperparameters

    # Regime Detection Parameters
    regime_lookback = IntParameter(20, 60, default=40, space="buy")
    trend_threshold = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="buy")
    volatility_lookback = IntParameter(10, 30, default=20, space="buy")

    # Liquidity Detection Parameters
    liquidity_lookback = IntParameter(30, 100, default=50, space="buy")
    liquidity_sensitivity = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")
    volume_spike_threshold = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")

    # Statistical Parameters
    zscore_period = IntParameter(10, 30, default=20, space="buy")
    zscore_entry_long = DecimalParameter(-3.0, -1.0, default=-2.0, decimals=1, space="buy")
    zscore_entry_short = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # Mean Reversion Parameters
    mean_period = IntParameter(20, 50, default=30, space="buy")
    reversion_strength = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="buy")

    # Trend Following Parameters
    trend_fast = IntParameter(5, 20, default=10, space="buy")
    trend_slow = IntParameter(20, 60, default=40, space="buy")
    momentum_period = IntParameter(10, 30, default=20, space="buy")

    # Risk Parameters
    volatility_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="sell")
    regime_switch_delay = IntParameter(2, 10, default=5, space="buy")

    # Exit Parameters
    take_profit_multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell")
    stop_loss_multiplier = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space="sell")

    # Protection Parameters - Temporary cooldowns, not permanent stops
    cooldown_after_losses = IntParameter(
        5, 30, default=15, space="buy"
    )  # Temporary cooldown after loss streak
    cooldown_after_stoploss = IntParameter(0, 20, default=10, space="buy")  # candles
    min_time_between_trades = IntParameter(0, 10, default=3, space="buy")  # candles
    max_trades_per_day = IntParameter(3, 10, default=5, space="buy")
    loss_streak_threshold = IntParameter(
        2, 5, default=3, space="buy"
    )  # Losses before cooldown kicks in

    # Volatility Protection
    max_volatility_enter = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="buy")
    min_liquidity_score = DecimalParameter(0.2, 0.6, default=0.3, decimals=1, space="buy")

    # Drawdown Protection
    max_drawdown_percent = DecimalParameter(5.0, 15.0, default=10.0, decimals=1, space="sell")
    reduce_position_after_losses = IntParameter(1, 3, default=2, space="sell")

    # Time Filters
    trade_during_high_volume_only = CategoricalParameter([True, False], default=True, space="buy")
    avoid_trading_hours = CategoricalParameter([True, False], default=False, space="buy")

    # Regime Confidence
    min_regime_confidence = DecimalParameter(0.5, 0.9, default=0.7, decimals=1, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators for regime detection and liquidity hunting
        """

        # === Regime Detection Indicators ===

        # Calculate trend strength using ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.regime_lookback.value)

        # Calculate volatility using ATR and standard deviation
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.volatility_lookback.value)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volatility"] = (
            dataframe["close"].pct_change().rolling(window=self.volatility_lookback.value).std()
        )

        # Identify market regime
        # 0 = ranging, 1 = trending up, -1 = trending down
        dataframe["regime"] = 0

        # Use ADX for trend strength
        trending = dataframe["adx"] > (25 * self.trend_threshold.value)

        # Determine trend direction
        ema_fast = ta.EMA(dataframe, timeperiod=self.trend_fast.value)
        ema_slow = ta.EMA(dataframe, timeperiod=self.trend_slow.value)

        dataframe.loc[trending & (ema_fast > ema_slow), "regime"] = 1
        dataframe.loc[trending & (ema_fast < ema_slow), "regime"] = -1

        # Smooth regime to avoid whipsaws
        dataframe["regime_smooth"] = (
            dataframe["regime"].rolling(window=self.regime_switch_delay.value).median()
        )

        # === Liquidity Pool Detection ===

        # Identify potential liquidity pools using volume and price levels
        dataframe["volume_ma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_spike"] = dataframe["volume"] / dataframe["volume_ma"]

        # Calculate price levels where volume accumulated (potential liquidity)
        dataframe["high_volume_level"] = dataframe.loc[
            dataframe["volume_spike"] > self.volume_spike_threshold.value, "close"
        ]
        dataframe["high_volume_level"] = dataframe["high_volume_level"].ffill(
            limit=self.liquidity_lookback.value
        )

        # Distance from high volume levels
        dataframe["liquidity_distance"] = (
            abs(dataframe["close"] - dataframe["high_volume_level"]) / dataframe["close"]
        )

        # Identify support/resistance levels using pivots
        dataframe["pivot"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["resistance1"] = 2 * dataframe["pivot"] - dataframe["low"]
        dataframe["support1"] = 2 * dataframe["pivot"] - dataframe["high"]

        # === Statistical Indicators ===

        # Z-score of price
        price_mean = dataframe["close"].rolling(window=self.zscore_period.value).mean()
        price_std = dataframe["close"].rolling(window=self.zscore_period.value).std()
        dataframe["price_zscore"] = (dataframe["close"] - price_mean) / (price_std + 1e-10)

        # Percentile rank of current price
        dataframe["price_percentile"] = (
            dataframe["close"]
            .rolling(window=self.liquidity_lookback.value)
            .apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0)
        )

        # Volume percentile
        dataframe["volume_percentile"] = (
            dataframe["volume"]
            .rolling(window=self.liquidity_lookback.value)
            .apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0)
        )

        # === Mean Reversion Indicators ===

        # Calculate various moving averages for mean reversion
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.mean_period.value)
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.mean_period.value)
        dataframe["kama"] = ta.KAMA(dataframe, timeperiod=self.mean_period.value)

        # Distance from mean
        dataframe["mean_distance"] = (dataframe["close"] - dataframe["sma"]) / dataframe["sma"]
        dataframe["mean_distance_ema"] = (dataframe["close"] - dataframe["ema"]) / dataframe["ema"]

        # === Momentum Indicators ===

        # RSI with dynamic period
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.momentum_period.value)

        # Stochastic oscillator
        stoch = ta.STOCH(
            dataframe,
            fastk_period=self.momentum_period.value,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        # Williams %R
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=self.momentum_period.value)

        # === Volatility Bands ===

        # Keltner Channels for volatility-based bands
        keltner_ema = ta.EMA(dataframe, timeperiod=20)
        keltner_atr = dataframe["atr"]
        dataframe["keltner_upper"] = keltner_ema + (keltner_atr * self.liquidity_sensitivity.value)
        dataframe["keltner_lower"] = keltner_ema - (keltner_atr * self.liquidity_sensitivity.value)
        dataframe["keltner_mid"] = keltner_ema

        # Position within Keltner Channel
        keltner_range = dataframe["keltner_upper"] - dataframe["keltner_lower"]
        dataframe["keltner_position"] = (dataframe["close"] - dataframe["keltner_lower"]) / (
            keltner_range + 1e-10
        )

        # === Liquidity Hunt Indicators ===

        # Identify potential stop loss clusters
        # Assume stops are placed just below recent lows or above recent highs
        recent_high = dataframe["high"].rolling(window=self.liquidity_lookback.value).max()
        recent_low = dataframe["low"].rolling(window=self.liquidity_lookback.value).min()

        dataframe["stop_hunt_long"] = (dataframe["low"] < recent_low * 1.002) & (
            dataframe["close"] > recent_low
        )
        dataframe["stop_hunt_short"] = (dataframe["high"] > recent_high * 0.998) & (
            dataframe["close"] < recent_high
        )

        # === Composite Indicators ===

        # Regime-adjusted momentum
        dataframe["regime_momentum"] = dataframe["rsi"].copy()
        dataframe.loc[dataframe["regime_smooth"] == 0, "regime_momentum"] = (
            50 - (50 - dataframe["rsi"]) * self.reversion_strength.value
        )

        # Liquidity score (higher = more likely to have liquidity)
        dataframe["liquidity_score"] = (
            dataframe["volume_percentile"] * 0.3
            + (1 - dataframe["liquidity_distance"]) * 0.3
            + dataframe["volume_spike"].clip(0, 3) / 3 * 0.4
        )

        # === Protection Indicators ===

        # Track recent price movements for volatility protection
        dataframe["price_change_pct"] = dataframe["close"].pct_change(periods=5) * 100
        dataframe["max_recent_volatility"] = dataframe["volatility"].rolling(window=20).max()

        # Track volume patterns for time filter
        dataframe["hour"] = (
            pd.to_datetime(dataframe.index).hour
            if isinstance(dataframe.index, pd.DatetimeIndex)
            else 0
        )
        dataframe["is_high_volume_hour"] = dataframe["volume"] > dataframe["volume_ma"] * 0.8

        # Calculate regime confidence (how stable is the regime)
        dataframe["regime_changes"] = dataframe["regime"].diff().abs()
        dataframe["regime_stability"] = 1 - (
            dataframe["regime_changes"].rolling(window=20).sum() / 20
        )

        # Track recent highs/lows for drawdown calculation
        dataframe["recent_high"] = dataframe["high"].rolling(window=50).max()
        dataframe["recent_low"] = dataframe["low"].rolling(window=50).min()
        dataframe["current_drawdown"] = (
            (dataframe["recent_high"] - dataframe["close"]) / dataframe["recent_high"] * 100
        )

        # Add trade quality score
        dataframe["trade_quality"] = (
            (dataframe["liquidity_score"] * 0.3)
            + (dataframe["regime_stability"] * 0.3)
            + ((1 - dataframe["volatility"] / dataframe["max_recent_volatility"]) * 0.4)
        ).clip(0, 1)

        return dataframe

    def check_protection_conditions(
        self, dataframe: DataFrame, pair: str, current_index: int
    ) -> bool:
        """
        Check if protection conditions allow trading
        Returns True if trading is allowed, False otherwise
        """
        # Initialize tracking for this pair if needed
        if pair not in self.recent_trades_tracker:
            self.recent_trades_tracker[pair] = []
            self.consecutive_losses[pair] = 0
            self.last_loss_candle[pair] = 0

        # Apply cooldown after consecutive losses (not a hard stop)
        if self.consecutive_losses.get(pair, 0) >= self.loss_streak_threshold.value:
            candles_since_loss = current_index - self.last_loss_candle.get(pair, 0)
            if candles_since_loss < self.cooldown_after_losses.value:
                return False  # Still in cooldown period
            else:
                # Cooldown expired, reset counter and allow trading to resume
                self.consecutive_losses[pair] = 0
                self.last_loss_candle[pair] = 0

        # Check cooldown after recent trades
        if self.recent_trades_tracker.get(pair):
            last_trade_index = self.recent_trades_tracker[pair][-1]
            if current_index - last_trade_index < self.min_time_between_trades.value:
                return False

        # Check daily trade limit
        recent_24h = [
            t for t in self.recent_trades_tracker.get(pair, []) if current_index - t < 96
        ]  # 96 * 15min = 24 hours
        if len(recent_24h) >= self.max_trades_per_day.value:
            return False

        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals based on regime and liquidity conditions
        """
        pair = metadata.get("pair", "")

        # === Protection Filters ===

        # Volatility protection - don't trade in extreme volatility
        volatility_filter = (
            dataframe["volatility"] < dataframe["atr_pct"] * self.max_volatility_enter.value
        )

        # Liquidity protection - ensure minimum liquidity
        liquidity_filter = dataframe["liquidity_score"] > self.min_liquidity_score.value

        # Regime confidence filter
        regime_confidence_filter = dataframe["regime_stability"] > self.min_regime_confidence.value

        # Drawdown protection
        drawdown_filter = dataframe["current_drawdown"] < self.max_drawdown_percent.value

        # Time filter (optional)
        time_filter = True
        if self.trade_during_high_volume_only.value:
            time_filter = dataframe["is_high_volume_hour"]

        if self.avoid_trading_hours.value:
            # Avoid typically volatile hours (market open/close)
            time_filter = time_filter & ~dataframe["hour"].isin([0, 1, 8, 9, 15, 16])

        # Combine all protection filters
        protection_passed = (
            volatility_filter
            & liquidity_filter
            & regime_confidence_filter
            & drawdown_filter
            & time_filter
        )

        # === Long Entry Conditions ===

        # Ranging Market Long Entry (Mean Reversion)
        ranging_long = (
            protection_passed  # Apply protection filters
            & (dataframe["regime_smooth"] == 0)  # Ranging market
            & (dataframe["price_zscore"] < self.zscore_entry_long.value)  # Oversold statistically
            & (dataframe["price_percentile"] < 0.2)  # In bottom 20% of range
            & (dataframe["rsi"] < 35)  # Momentum oversold
            & (dataframe["mean_distance"] < -0.02 * self.reversion_strength.value)  # Below mean
            & (dataframe["liquidity_score"] > 0.5)  # Good liquidity
        )

        # Trending Market Long Entry (Trend Following)
        trending_long = (
            protection_passed  # Apply protection filters
            & (dataframe["regime_smooth"] == 1)  # Uptrend
            & (dataframe["price_zscore"] > -0.5)
            & (dataframe["price_zscore"] < 1.0)  # Not overextended
            & (dataframe["rsi"] > 40)
            & (dataframe["rsi"] < 70)  # Momentum not extreme
            & (dataframe["close"] > dataframe["keltner_lower"])  # Above lower band
            & (dataframe["keltner_position"] < 0.7)  # Not at top of channel
            & (dataframe["volume_percentile"] > 0.4)  # Decent volume
        )

        # Liquidity Hunt Long Entry (Stop Hunt)
        liquidity_hunt_long = (
            protection_passed  # Apply protection filters
            & (dataframe["stop_hunt_long"] == True)  # Stop hunt detected
            & (dataframe["volume_spike"] > self.volume_spike_threshold.value)  # High volume
            & (dataframe["close"] > dataframe["open"])  # Bullish recovery
            & (dataframe["trade_quality"] > 0.5)  # Minimum trade quality
        )

        # Combine all long conditions
        dataframe.loc[ranging_long | trending_long | liquidity_hunt_long, "enter_long"] = 1

        # === Short Entry Conditions ===

        # Ranging Market Short Entry (Mean Reversion)
        ranging_short = (
            protection_passed  # Apply protection filters
            & (dataframe["regime_smooth"] == 0)  # Ranging market
            & (
                dataframe["price_zscore"] > self.zscore_entry_short.value
            )  # Overbought statistically
            & (dataframe["price_percentile"] > 0.8)  # In top 20% of range
            & (dataframe["rsi"] > 65)  # Momentum overbought
            & (dataframe["mean_distance"] > 0.02 * self.reversion_strength.value)  # Above mean
            & (dataframe["liquidity_score"] > 0.5)  # Good liquidity
        )

        # Trending Market Short Entry (Trend Following)
        trending_short = (
            protection_passed  # Apply protection filters
            & (dataframe["regime_smooth"] == -1)  # Downtrend
            & (dataframe["price_zscore"] < 0.5)
            & (dataframe["price_zscore"] > -1.0)  # Not overextended
            & (dataframe["rsi"] < 60)
            & (dataframe["rsi"] > 30)  # Momentum not extreme
            & (dataframe["close"] < dataframe["keltner_upper"])  # Below upper band
            & (dataframe["keltner_position"] > 0.3)  # Not at bottom of channel
            & (dataframe["volume_percentile"] > 0.4)  # Decent volume
        )

        # Liquidity Hunt Short Entry (Stop Hunt)
        liquidity_hunt_short = (
            protection_passed  # Apply protection filters
            & (dataframe["stop_hunt_short"] == True)  # Stop hunt detected
            & (dataframe["volume_spike"] > self.volume_spike_threshold.value)  # High volume
            & (dataframe["close"] < dataframe["open"])  # Bearish rejection
            & (dataframe["trade_quality"] > 0.5)  # Minimum trade quality
        )

        # Combine all short conditions
        dataframe.loc[ranging_short | trending_short | liquidity_hunt_short, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals based on regime and statistical conditions
        """

        # === Long Exit Conditions ===

        # Statistical exit (price returned to mean or overextended)
        statistical_exit_long = (
            (dataframe["price_zscore"] > 1.5)  # Statistically overbought
            | (dataframe["price_percentile"] > 0.9)  # At top of range
        )

        # Regime change exit
        regime_exit_long = (
            (dataframe["regime_smooth"] == -1)  # Changed to downtrend
            & (dataframe["regime_smooth"].shift(5) >= 0)  # Was not in downtrend recently
        )

        # Momentum exit
        momentum_exit_long = (
            (dataframe["rsi"] > 75)  # Overbought
            | (dataframe["stoch_k"] > 90)  # Stochastic overbought
        )

        # Target reached (using ATR)
        # Note: This is simplified, in reality would track entry price
        target_exit_long = dataframe["mean_distance"] > 0.02 * self.take_profit_multiplier.value

        # Combine exits
        dataframe.loc[
            statistical_exit_long | regime_exit_long | momentum_exit_long | target_exit_long,
            "exit_long",
        ] = 1

        # === Short Exit Conditions ===

        # Statistical exit
        statistical_exit_short = (
            (dataframe["price_zscore"] < -1.5)  # Statistically oversold
            | (dataframe["price_percentile"] < 0.1)  # At bottom of range
        )

        # Regime change exit
        regime_exit_short = (
            (dataframe["regime_smooth"] == 1)  # Changed to uptrend
            & (dataframe["regime_smooth"].shift(5) <= 0)  # Was not in uptrend recently
        )

        # Momentum exit
        momentum_exit_short = (
            (dataframe["rsi"] < 25)  # Oversold
            | (dataframe["stoch_k"] < 10)  # Stochastic oversold
        )

        # Target reached
        target_exit_short = dataframe["mean_distance"] < -0.02 * self.take_profit_multiplier.value

        # Combine exits
        dataframe.loc[
            statistical_exit_short | regime_exit_short | momentum_exit_short | target_exit_short,
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
        Custom exit based on regime and liquidity conditions
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Track trade outcome for protection logic
        if current_profit < -0.02:  # Loss threshold
            self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1
            self.last_loss_candle[pair] = len(dataframe) - 1  # Record when loss occurred
        elif current_profit > 0.005:  # Small profit is enough to break the streak
            # Reset counters on any profitable trade to allow recovery
            self.consecutive_losses[pair] = 0
            self.last_loss_candle[pair] = 0

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # Exit if regime changed significantly
        if trade.is_short:
            if last_candle["regime_smooth"] > 0.5:  # Strong uptrend started
                return "regime_change_exit"
        else:  # Long trade
            if last_candle["regime_smooth"] < -0.5:  # Strong downtrend started
                return "regime_change_exit"

        # Exit on extreme statistical values
        if abs(last_candle["price_zscore"]) > 3.0:
            if current_profit > 0:
                return "statistical_extreme_exit"

        # Exit if liquidity dried up
        if last_candle["liquidity_score"] < 0.2 and current_profit > 0.005:
            return "low_liquidity_exit"

        # Liquidity grab exit - if stop hunt in opposite direction
        if trade.is_short and last_candle["stop_hunt_long"]:
            return "liquidity_grab_exit"
        elif not trade.is_short and last_candle["stop_hunt_short"]:
            return "liquidity_grab_exit"

        # Emergency exit on extreme volatility
        if last_candle["volatility"] > last_candle["atr_pct"] * 4:
            if current_profit > -0.01:  # Accept small loss to avoid larger one
                return "extreme_volatility_exit"

        # Exit if drawdown is getting too large
        if last_candle["current_drawdown"] > self.max_drawdown_percent.value * 1.5:
            return "max_drawdown_exit"

        # Quality deterioration exit
        if last_candle["trade_quality"] < 0.2 and current_profit > -0.005:
            return "low_quality_exit"

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
        Dynamic stoploss based on volatility and regime
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return -0.04  # Default stoploss

        last_candle = dataframe.iloc[-1]

        # Tighter stop during loss streak (but not too tight to allow recovery)
        loss_multiplier = 1.0
        consecutive_losses = self.consecutive_losses.get(pair, 0)
        if consecutive_losses >= 2:
            # Progressive but reasonable tightening: 0.85x after 2 losses, 0.7x after 3
            loss_multiplier = max(0.7, 1.0 - (consecutive_losses * 0.15))

        # Base stop on ATR
        atr_stop = (
            last_candle["atr"] * 2 * self.stop_loss_multiplier.value * loss_multiplier
        ) / current_rate

        # Adjust based on regime
        if abs(last_candle["regime_smooth"]) > 0.5:  # Trending market
            # Wider stops in trends
            atr_stop = atr_stop * 1.5
        else:  # Ranging market
            # Tighter stops in ranges
            atr_stop = atr_stop * 0.75

        # Tighten stop when in profit
        if current_profit > 0.03:
            return -min(atr_stop, 0.01)
        elif current_profit > 0.015:
            return -min(atr_stop, 0.02)
        else:
            return -min(atr_stop, 0.04)

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
        Adjust leverage based on regime and volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return 1.0

        last_candle = dataframe.iloc[-1]

        # Base leverage on volatility
        volatility_pct = last_candle["volatility"] * 100

        if volatility_pct < 1.0:  # Very low volatility
            base_leverage = 3.0
        elif volatility_pct < 2.0:  # Low volatility
            base_leverage = 2.0
        elif volatility_pct < 3.0:  # Medium volatility
            base_leverage = 1.5
        else:  # High volatility
            base_leverage = 1.0

        # Adjust based on regime
        if abs(last_candle["regime_smooth"]) < 0.5:  # Ranging market
            # Higher leverage in ranging markets (mean reversion)
            base_leverage *= 1.2
        else:  # Trending market
            # Lower leverage in trending markets (more risk)
            base_leverage *= 0.8

        # Apply volatility multiplier parameter
        final_leverage = base_leverage * self.volatility_multiplier.value

        return min(final_leverage, max_leverage)

    def protections(self):
        """
        Define protections for the strategy
        More details: https://www.freqtrade.io/en/stable/plugins/#protections
        """
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,  # 12 hours on 15m
                "trade_limit": 4,  # After 4 stoplosses in lookback period
                "stop_duration_candles": self.cooldown_after_stoploss.value,
                "only_per_pair": True,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 200,  # ~50 hours
                "trade_limit": 5,
                "stop_duration_candles": 20,
                "max_allowed_drawdown": self.max_drawdown_percent.value / 100,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,  # 90 hours
                "trade_limit": 2,
                "stop_duration_candles": 20,
                "required_profit": -0.05,  # -5% minimum
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.min_time_between_trades.value,
            },
        ]

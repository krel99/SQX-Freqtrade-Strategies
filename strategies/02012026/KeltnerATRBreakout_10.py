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


class KeltnerATRBreakout_10(IStrategy):
    """
    Keltner Channel Break + ATR Trail Strategy

    Use Keltner Channels to catch volatility expansions; trail the position with ATR.

    Improvements:
    - Dynamic multiplier adjustment based on market conditions
    - Volume surge detection for better breakouts
    - RSI filter to avoid overbought/oversold extremes
    - Momentum confirmation with ROC
    - False breakout protection
    - Better trailing stop implementation
    - Volatility-based position sizing
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "15m"

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.05,
        "20": 0.04,
        "40": 0.03,
        "60": 0.025,
        "90": 0.02,
        "120": 0.015,
        "180": 0.01,
    }

    # Optimal stoploss
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperparameters
    # Keltner Channel parameters
    ema_period = IntParameter(15, 25, default=20, space="buy")
    atr_period = IntParameter(10, 20, default=14, space="buy")
    kc_mult_base = DecimalParameter(1.0, 2.0, default=1.5, space="buy")

    # ATR trailing parameters
    atr_trail_mult = DecimalParameter(1.0, 2.5, default=1.5, space="sell")
    atr_trail_period = IntParameter(10, 20, default=14, space="sell")

    # Breakout confirmation
    breakout_candles = IntParameter(1, 3, default=1, space="buy")
    min_breakout_strength = DecimalParameter(0.001, 0.003, default=0.002, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(15, 30, default=20, space="buy")
    volume_surge_mult = DecimalParameter(1.2, 2.0, default=1.5, space="buy")

    # RSI filter
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_min_long = IntParameter(35, 45, default=40, space="buy")
    rsi_max_long = IntParameter(65, 75, default=70, space="buy")
    rsi_min_short = IntParameter(25, 35, default=30, space="buy")
    rsi_max_short = IntParameter(55, 65, default=60, space="buy")

    # ROC momentum
    roc_period = IntParameter(5, 15, default=10, space="buy")
    roc_threshold = DecimalParameter(0.5, 2.0, default=1.0, space="buy")

    # ADX for trend strength
    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_min = IntParameter(15, 30, default=20, space="buy")

    # Exit parameters
    profit_target_atr_mult = DecimalParameter(2.0, 4.0, default=3.0, space="sell")
    time_exit_bars = IntParameter(20, 40, default=30, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Pre-calculates all indicator variants for hyperopt compatibility.
        """

        # Pre-calculate EMA for all possible periods (15-25)
        for period in range(15, 26):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate ATR for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # ATR MA (using default ATR period 14 for base calculation)
        dataframe["atr_ma"] = ta.SMA(dataframe["atr_14"], timeperiod=20)

        # Pre-calculate RSI for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate ROC for all possible periods (5-15)
        for period in range(5, 16):
            dataframe[f"roc_{period}"] = (
                dataframe["close"] / dataframe["close"].shift(period) - 1
            ) * 100

        # Pre-calculate Volume MA for all possible periods (15-30)
        for period in range(15, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate ADX for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Bollinger Bands for volatility comparison
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = dataframe["bb_upper"] - dataframe["bb_lower"]

        # Candle strength
        dataframe["candle_body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["strong_candle"] = dataframe["candle_body"] > dataframe["candle_range"] * 0.6

        # Calculate highest/lowest since entry (for trailing stop simulation)
        dataframe["highest_20"] = dataframe["high"].rolling(window=20).max()
        dataframe["lowest_20"] = dataframe["low"].rolling(window=20).min()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # Get current hyperopt parameter values
        ema_period = self.ema_period.value
        atr_period = self.atr_period.value
        kc_mult_base = self.kc_mult_base.value
        breakout_candles = self.breakout_candles.value
        rsi_period = self.rsi_period.value
        roc_period = self.roc_period.value
        volume_ma_period = self.volume_ma_period.value
        volume_surge_mult = self.volume_surge_mult.value
        adx_period = self.adx_period.value

        # Select pre-calculated indicators
        ema = dataframe[f"ema_{ema_period}"]
        atr = dataframe[f"atr_{atr_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        roc = dataframe[f"roc_{roc_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        adx = dataframe[f"adx_{adx_period}"]

        # Calculate dynamic multiplier based on volatility
        volatility_ratio = atr / dataframe["atr_ma"]
        kc_mult_dynamic = kc_mult_base * volatility_ratio

        # Keltner Channels
        kc_upper = ema + (kc_mult_dynamic * atr)
        kc_lower = ema - (kc_mult_dynamic * atr)

        # Channel width
        kc_width = kc_upper - kc_lower

        # Detect breakouts
        close_above_upper = dataframe["close"] > kc_upper
        close_below_lower = dataframe["close"] < kc_lower

        # First close above/below band
        breakout_up = close_above_upper & (close_above_upper.shift(1) == False)
        breakout_down = close_below_lower & (close_below_lower.shift(1) == False)

        # Sustained breakout (multiple candles)
        sustained_breakout_up = close_above_upper.rolling(window=breakout_candles).min() == 1
        sustained_breakout_down = close_below_lower.rolling(window=breakout_candles).min() == 1

        # Volume surge
        volume_surge = dataframe["volume"] > (volume_ma * volume_surge_mult)

        # Volatility squeeze detection (KC inside BB = low volatility)
        squeeze = (kc_upper < dataframe["bb_upper"]) & (kc_lower > dataframe["bb_lower"])
        squeeze_release = (~squeeze) & squeeze.shift(1)

        # Trend determination
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema

        # EMA slope for momentum
        ema_slope = (ema - ema.shift(5)) / ema.shift(5) * 100

        # False breakout detection
        lookback = 3
        false_breakout_up = (breakout_up.rolling(window=lookback).max() == 1) & (
            dataframe["close"] < kc_upper
        )
        false_breakout_down = (breakout_down.rolling(window=lookback).max() == 1) & (
            dataframe["close"] > kc_lower
        )

        # LONG breakout
        dataframe.loc[
            (
                (dataframe["breakout_up"])  # Initial breakout
                & (dataframe["sustained_breakout_up"])  # Sustained breakout
                & (dataframe["volume_surge"])  # Volume confirmation
                & (dataframe["rsi"] > self.rsi_min_long.value)  # Not oversold
                & (dataframe["rsi"] < self.rsi_max_long.value)  # Not overbought
                & (dataframe["roc"] > self.roc_threshold.value)  # Positive momentum
                & (dataframe["adx"] > self.adx_min.value)  # Trending market
                & (dataframe["false_breakout_up"].shift(1) == False)  # No recent false breakout
                & (dataframe["strong_candle"])  # Strong breakout candle
                & (dataframe["ema_slope"] > 0)  # EMA trending up
                & (dataframe["kc_width_pct"] > 1.0)  # Channel wide enough (not compressed)
                & (
                    dataframe["squeeze_release"]
                    | ~dataframe["squeeze"]
                    | (dataframe["volatility_ratio"] > 1.0)
                )  # Not in squeeze or just released
            ),
            "enter_long",
        ] = 1

        # SHORT breakout
        dataframe.loc[
            (
                (dataframe["breakout_down"])  # Initial breakout
                & (dataframe["sustained_breakout_down"])  # Sustained breakout
                & (dataframe["volume_surge"])  # Volume confirmation
                & (dataframe["rsi"] < self.rsi_max_short.value)  # Not overbought
                & (dataframe["rsi"] > self.rsi_min_short.value)  # Not oversold
                & (dataframe["roc"] < -self.roc_threshold.value)  # Negative momentum
                & (dataframe["adx"] > self.adx_min.value)  # Trending market
                & (dataframe["false_breakout_down"].shift(1) == False)  # No recent false breakout
                & (dataframe["strong_candle"])  # Strong breakout candle
                & (dataframe["ema_slope"] < 0)  # EMA trending down
                & (dataframe["kc_width_pct"] > 1.0)  # Channel wide enough (not compressed)
                & (
                    dataframe["squeeze_release"]
                    | ~dataframe["squeeze"]
                    | (dataframe["volatility_ratio"] > 1.0)
                )  # Not in squeeze or just released
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Calculate ATR-based trailing stops
        dataframe["long_trail_stop"] = dataframe["highest_20"] - (
            dataframe["atr"] * self.atr_trail_mult.value
        )
        dataframe["short_trail_stop"] = dataframe["lowest_20"] + (
            dataframe["atr"] * self.atr_trail_mult.value
        )

        # Calculate profit targets
        dataframe["long_target"] = dataframe["kc_upper"] + (
            dataframe["atr"] * self.profit_target_atr_mult.value
        )
        dataframe["short_target"] = dataframe["kc_lower"] - (
            dataframe["atr"] * self.profit_target_atr_mult.value
        )

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["long_trail_stop"])  # Trail stop hit
                | (dataframe["close"] < dataframe["ema"])  # Below midline
                | (dataframe["close"] >= dataframe["long_target"])  # Target reached
                | (dataframe["rsi"] > 80)  # Extreme overbought
                | (dataframe["roc"] < -2.0)  # Strong momentum reversal
                | (dataframe["squeeze"])  # Entering squeeze (volatility contraction)
                | (dataframe["false_breakout_up"])  # False breakout detected
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["short_trail_stop"])  # Trail stop hit
                | (dataframe["close"] > dataframe["ema"])  # Above midline
                | (dataframe["close"] <= dataframe["short_target"])  # Target reached
                | (dataframe["rsi"] < 20)  # Extreme oversold
                | (dataframe["roc"] > 2.0)  # Strong momentum reversal
                | (dataframe["squeeze"])  # Entering squeeze (volatility contraction)
                | (dataframe["false_breakout_down"])  # False breakout detected
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
        Custom exit logic with ATR trailing
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Time-based exit
        trade_duration_bars = (current_time - trade.open_date_utc).total_seconds() / (
            15 * 60
        )  # 15m candles
        if trade_duration_bars > self.time_exit_bars.value:
            if current_profit > 0:
                return "time_exit_profit"
            elif current_profit > -0.01:
                return "time_exit_small_loss"

        # Exit if channel narrows significantly (consolidation)
        if last_candle["kc_width_pct"] < 0.5:
            return "channel_too_narrow"

        # Exit if we're back inside the channel after breakout
        if not trade.is_short:
            if current_rate < last_candle["kc_upper"]:
                if current_profit > 0:
                    return "back_inside_channel_profit"
                elif current_profit > -0.005:
                    return "back_inside_channel_small_loss"
        else:
            if current_rate > last_candle["kc_lower"]:
                if current_profit > 0:
                    return "back_inside_channel_profit"
                elif current_profit > -0.005:
                    return "back_inside_channel_small_loss"

        # Exit on volatility explosion (protect profits)
        if last_candle["volatility_ratio"] > 2.0 and current_profit > 0.02:
            return "volatility_spike_protect"

        # Exit if ADX drops (trend weakening)
        if last_candle["adx"] < 15 and current_profit > 0:
            return "trend_weakening"

        # Dynamic ATR trailing stop
        if not trade.is_short:
            trail_stop = last_candle["highest_20"] - (
                last_candle["atr"] * self.atr_trail_mult.value
            )
            if current_rate < trail_stop and current_profit > 0.01:
                return "atr_trail_stop"
        else:
            trail_stop = last_candle["lowest_20"] + (last_candle["atr"] * self.atr_trail_mult.value)
            if current_rate > trail_stop and current_profit > 0.01:
                return "atr_trail_stop"

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
        Custom stoploss logic using ATR
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Dynamic stop based on ATR
        atr_stop = -(last_candle["atr"] * 2.5 / trade.open_rate)

        # Use Keltner Channel as stop level
        if not trade.is_short:
            kc_stop = -(trade.open_rate - last_candle["kc_lower"]) / trade.open_rate
        else:
            kc_stop = -(last_candle["kc_upper"] - trade.open_rate) / trade.open_rate

        # Use the tighter of the two
        dynamic_stop = max(atr_stop, kc_stop, self.stoploss)

        # Progressive stops based on profit
        if current_profit > 0.03:
            return -0.008
        elif current_profit > 0.02:
            return -0.012
        elif current_profit > 0.015:
            return -0.015
        elif current_profit > 0.01:
            return max(dynamic_stop, -0.02)

        # Tighten stop over time
        if current_time - trade.open_date_utc > pd.Timedelta(hours=3):
            return max(dynamic_stop, -0.03)

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

        # Don't enter if channel is too narrow
        if last_candle["kc_width_pct"] < 0.8:
            return False

        # Don't enter if ATR is too low (no volatility)
        if last_candle["atr"] < last_candle["close"] * 0.002:
            return False

        # Don't enter during squeeze unless it's a release
        if last_candle["squeeze"] and not last_candle["squeeze_release"]:
            return False

        # Avoid low liquidity hours
        hour = current_time.hour
        if hour >= 2 and hour <= 4:  # UTC
            return False

        return True

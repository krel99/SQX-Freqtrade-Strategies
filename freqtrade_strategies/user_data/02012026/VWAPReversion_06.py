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

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Can this strategy go short?
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
    vwap_period = IntParameter(
        20, 60, default=40, space="buy"
    )  # Rolling window for VWAP
    dev_period = IntParameter(
        15, 30, default=20, space="buy"
    )  # Period for deviation calculation

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # Calculate VWAP
        dataframe["vwap"] = self.calculate_vwap(dataframe, self.vwap_period.value)

        # Calculate deviation from VWAP
        dataframe["deviation"] = dataframe["close"] - dataframe["vwap"]
        dataframe["deviation_pct"] = (dataframe["deviation"] / dataframe["vwap"]) * 100

        # Calculate standard deviation of the deviation
        dataframe["dev_std"] = (
            dataframe["deviation"].rolling(window=self.dev_period.value).std()
        )

        # Calculate dynamic bands
        dataframe["upper_dev"] = dataframe["vwap"] + (
            self.dev_mult_upper.value * dataframe["dev_std"]
        )
        dataframe["lower_dev"] = dataframe["vwap"] - (
            self.dev_mult_lower.value * dataframe["dev_std"]
        )

        # ATR for volatility adjustment
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Adjust bands with ATR
        dataframe["upper_dev_atr"] = dataframe["vwap"] + (
            self.dev_mult_upper.value * dataframe["atr"] * self.atr_mult.value
        )
        dataframe["lower_dev_atr"] = dataframe["vwap"] - (
            self.dev_mult_lower.value * dataframe["atr"] * self.atr_mult.value
        )

        # Use the wider of the two bands
        dataframe["upper_band"] = dataframe[["upper_dev", "upper_dev_atr"]].max(axis=1)
        dataframe["lower_band"] = dataframe[["lower_dev", "lower_dev_atr"]].min(axis=1)

        # RSI for confirmation
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Volume analysis
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_ok"] = dataframe["volume"] >= (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        # Volume-weighted momentum
        dataframe["vwm"] = (
            (
                (dataframe["close"] - dataframe["close"].shift(5))
                / dataframe["close"].shift(5)
            )
            * dataframe["volume"]
        ).rolling(window=10).sum() / dataframe["volume"].rolling(window=10).sum()

        # Price position relative to VWAP
        dataframe["price_above_vwap"] = dataframe["close"] > dataframe["vwap"]
        dataframe["price_below_vwap"] = dataframe["close"] < dataframe["vwap"]

        # Band touches
        dataframe["touches_upper"] = dataframe["high"] >= dataframe["upper_band"]
        dataframe["touches_lower"] = dataframe["low"] <= dataframe["lower_band"]

        # VWAP slope for trend
        dataframe["vwap_slope"] = (
            (dataframe["vwap"] - dataframe["vwap"].shift(5))
            / dataframe["vwap"].shift(5)
            * 100
        )

        # RSI divergence (simplified)
        lookback = 10
        dataframe["price_lower"] = dataframe["low"] < dataframe["low"].shift(lookback)
        dataframe["rsi_higher"] = dataframe["rsi"] > dataframe["rsi"].shift(lookback)
        dataframe["bullish_div"] = dataframe["price_lower"] & dataframe["rsi_higher"]

        dataframe["price_higher"] = dataframe["high"] > dataframe["high"].shift(
            lookback
        )
        dataframe["rsi_lower"] = dataframe["rsi"] < dataframe["rsi"].shift(lookback)
        dataframe["bearish_div"] = dataframe["price_higher"] & dataframe["rsi_lower"]

        # Add hour for time filtering
        dataframe["hour"] = dataframe["date"].dt.hour

        # Session filter
        dataframe["session_active"] = (dataframe["hour"] >= self.start_hour.value) & (
            dataframe["hour"] < self.end_hour.value
        )

        # Cumulative volume for the session (reset daily)
        dataframe["day"] = dataframe["date"].dt.date
        dataframe["cum_volume"] = dataframe.groupby("day")["volume"].cumsum()

        # Distance from VWAP in ATR units
        dataframe["dist_from_vwap_atr"] = dataframe["deviation"] / dataframe["atr"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY: price too far below VWAP
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["lower_band"])  # Below lower deviation
                & (dataframe["rsi"] < self.rsi_oversold.value)  # RSI confirms oversold
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["session_active"])  # Within trading session
                & (abs(dataframe["vwap_slope"]) < 1.0)  # VWAP not trending strongly
                & (
                    (dataframe["bullish_div"])  # Bullish divergence
                    | (dataframe["vwm"] > -0.01)  # Or not too negative momentum
                )
                & (dataframe["dist_from_vwap_atr"] < -1.5)  # Significant deviation
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY: price too far above VWAP
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["upper_band"])  # Above upper deviation
                & (
                    dataframe["rsi"] > self.rsi_overbought.value
                )  # RSI confirms overbought
                & (dataframe["volume_ok"])  # Volume confirmation
                & (dataframe["session_active"])  # Within trading session
                & (abs(dataframe["vwap_slope"]) < 1.0)  # VWAP not trending strongly
                & (
                    (dataframe["bearish_div"])  # Bearish divergence
                    | (dataframe["vwm"] < 0.01)  # Or not too positive momentum
                )
                & (dataframe["dist_from_vwap_atr"] > 1.5)  # Significant deviation
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """

        # Calculate partial exit targets
        dataframe["long_partial_target"] = dataframe["lower_band"] + (
            (dataframe["vwap"] - dataframe["lower_band"])
            * self.partial_exit_ratio.value
        )
        dataframe["short_partial_target"] = dataframe["upper_band"] - (
            (dataframe["upper_band"] - dataframe["vwap"])
            * self.partial_exit_ratio.value
        )

        # LONG EXIT
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["vwap"])
                if self.vwap_touch_exit.value
                else (dataframe["close"] >= dataframe["long_partial_target"])
            )
            | (dataframe["rsi"] > 65)  # No longer oversold
            | (dataframe["vwap_slope"] < -0.5)  # VWAP turning down
            | (~dataframe["session_active"])  # Session ending
            | (dataframe["close"] > dataframe["upper_band"]),  # Extreme reversal
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["vwap"])
                if self.vwap_touch_exit.value
                else (dataframe["close"] <= dataframe["short_partial_target"])
            )
            | (dataframe["rsi"] < 35)  # No longer overbought
            | (dataframe["vwap_slope"] > 0.5)  # VWAP turning up
            | (~dataframe["session_active"])  # Session ending
            | (dataframe["close"] < dataframe["lower_band"]),  # Extreme reversal
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

        # Exit if we're past VWAP with profit
        if not trade.is_short:
            if last_candle["close"] > last_candle["vwap"] and current_profit > 0.005:
                return "vwap_crossed_profit"
        else:
            if last_candle["close"] < last_candle["vwap"] and current_profit > 0.005:
                return "vwap_crossed_profit"

        # Exit if session is ending
        if not last_candle["session_active"] and current_profit > -0.005:
            return "session_end"

        # Exit if VWAP starts trending strongly against position
        if not trade.is_short and last_candle["vwap_slope"] < -1.0:
            return "vwap_trending_down"
        if trade.is_short and last_candle["vwap_slope"] > 1.0:
            return "vwap_trending_up"

        # Quick profit if deviation narrows
        if current_profit > 0.01:
            if abs(last_candle["dist_from_vwap_atr"]) < 0.5:
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

        # Dynamic stop based on ATR
        atr_stop = -(last_candle["atr"] * 2.5 / trade.open_rate)

        # Tighten stop when approaching VWAP
        if not trade.is_short:
            if current_rate > last_candle["vwap"] * 0.998:
                return -0.003  # Very tight stop near target
        else:
            if current_rate < last_candle["vwap"] * 1.002:
                return -0.003  # Very tight stop near target

        # Progressive stops based on profit
        if current_profit > 0.015:
            return -0.004
        elif current_profit > 0.008:
            return -0.006
        elif current_profit > 0.003:
            return -0.01

        # Use tighter stop if session is ending
        if not last_candle["session_active"]:
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

        # Don't enter if deviation is too extreme (likely to continue)
        if abs(last_candle["dist_from_vwap_atr"]) > 3:
            return False

        # Don't enter if volume is too low
        if last_candle["cum_volume"] < last_candle["volume_ma"] * 10:
            return False

        # Avoid entries near session end
        if last_candle["hour"] >= self.end_hour.value - 1:
            return False

        # Don't enter if VWAP is trending too strongly
        if abs(last_candle["vwap_slope"]) > 1.5:
            return False

        return True

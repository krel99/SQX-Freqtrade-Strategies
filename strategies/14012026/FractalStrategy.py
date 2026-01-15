# --- Do not remove these libs ---
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)

if TYPE_CHECKING:
    from freqtrade.persistence import Trade

# --------------------------------


class FractalStrategy(IStrategy):
    """
    Futures strategy based on Williams Fractals with long and short positions.

    Williams Fractals:
    - 5-candle pattern: Classic Williams Fractals (2 lower highs on each side for top, 2 higher lows for bottom)
    - 3-candle pattern: Simplified version (1 lower high on each side for top, 1 higher low for bottom)

    The strategy enters positions on fractal breakouts with additional filters for trend and momentum.
    """

    # Strategy interface version - attribute needed by Freqtrade
    INTERFACE_VERSION = 3
    can_short: bool = True

    # === LONG ENTRY PARAMETERS (buy space) ===
    # Fractal settings for long entries
    long_fractal_window = CategoricalParameter([3, 5], default=5, space="buy", optimize=True)
    long_breakout_threshold = DecimalParameter(
        0.001, 0.05, default=0.01, decimals=3, space="buy", optimize=True
    )

    # MA filter for long entries
    long_ma_period = IntParameter(20, 500, default=100, space="buy", optimize=True)
    long_ma_type = CategoricalParameter(
        ["EMA", "SMA", "WMA"], default="EMA", space="buy", optimize=True
    )
    long_use_ma_filter = BooleanParameter(default=True, space="buy", optimize=True)

    # RSI filter for long entries
    long_rsi_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    long_rsi_min = IntParameter(20, 50, default=30, space="buy", optimize=True)
    long_use_rsi_filter = BooleanParameter(default=True, space="buy", optimize=True)

    # Volume filter for long entries
    long_volume_ma_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    long_volume_threshold = DecimalParameter(
        0.5, 3.0, default=1.2, decimals=1, space="buy", optimize=True
    )
    long_use_volume_filter = BooleanParameter(default=True, space="buy", optimize=True)

    # Trend strength filter for long entries
    long_adx_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    long_adx_min = IntParameter(15, 40, default=25, space="buy", optimize=True)
    long_use_adx_filter = BooleanParameter(default=False, space="buy", optimize=True)

    # === SHORT ENTRY PARAMETERS (sell space) ===
    # Fractal settings for short entries
    short_fractal_window = CategoricalParameter([3, 5], default=5, space="sell", optimize=True)
    short_breakout_threshold = DecimalParameter(
        0.001, 0.05, default=0.01, decimals=3, space="sell", optimize=True
    )

    # MA filter for short entries
    short_ma_period = IntParameter(20, 500, default=100, space="sell", optimize=True)
    short_ma_type = CategoricalParameter(
        ["EMA", "SMA", "WMA"], default="EMA", space="sell", optimize=True
    )
    short_use_ma_filter = BooleanParameter(default=True, space="sell", optimize=True)

    # RSI filter for short entries
    short_rsi_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_rsi_max = IntParameter(50, 80, default=70, space="sell", optimize=True)
    short_use_rsi_filter = BooleanParameter(default=True, space="sell", optimize=True)

    # Volume filter for short entries
    short_volume_ma_period = IntParameter(10, 50, default=20, space="sell", optimize=True)
    short_volume_threshold = DecimalParameter(
        0.5, 3.0, default=1.2, decimals=1, space="sell", optimize=True
    )
    short_use_volume_filter = BooleanParameter(default=True, space="sell", optimize=True)

    # Trend strength filter for short entries
    short_adx_period = IntParameter(7, 30, default=14, space="sell", optimize=True)
    short_adx_min = IntParameter(15, 40, default=25, space="sell", optimize=True)
    short_use_adx_filter = BooleanParameter(default=False, space="sell", optimize=True)

    # === EXIT PARAMETERS (profit space) ===
    # Volatility-normalized take profit
    volatility_tp_X = DecimalParameter(
        2.0, 3.0, default=2.5, decimals=1, space="profit", optimize=True
    )

    # Price-based ATR trailing stop
    atr_trailing_k = DecimalParameter(
        1.0, 2.0, default=1.5, decimals=1, space="profit", optimize=True
    )

    # === STRATEGY SETTINGS ===
    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.03, "120": 0.01}

    # Stoploss
    stoploss = -0.08

    # Trailing stop (basic settings, enhanced by custom logic)
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = "15m"

    # Custom info storage for trade-specific data
    custom_info = {}

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    def _calculate_fractals(self, dataframe: DataFrame, window_size: int) -> tuple:
        """
        Calculate Williams Fractals for the given window size.

        Args:
            dataframe: Price dataframe
            window_size: 3 or 5 candles for fractal pattern

        Returns:
            Tuple of (fractal_tops, fractal_bottoms) as numpy arrays
        """
        mid_point = window_size // 2

        # Initialize arrays
        fractal_tops = np.full(len(dataframe), np.nan)
        fractal_bottoms = np.full(len(dataframe), np.nan)

        for i in range(mid_point, len(dataframe) - mid_point):
            # Check for fractal top (swing high)
            is_top = True
            for j in range(i - mid_point, i):
                if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                    is_top = False
                    break
            if is_top:
                for j in range(i + 1, i + mid_point + 1):
                    if dataframe["high"].iloc[j] >= dataframe["high"].iloc[i]:
                        is_top = False
                        break
            if is_top:
                fractal_tops[i] = dataframe["high"].iloc[i]

            # Check for fractal bottom (swing low)
            is_bottom = True
            for j in range(i - mid_point, i):
                if dataframe["low"].iloc[j] <= dataframe["low"].iloc[i]:
                    is_bottom = False
                    break
            if is_bottom:
                for j in range(i + 1, i + mid_point + 1):
                    if dataframe["low"].iloc[j] <= dataframe["low"].iloc[i]:
                        is_bottom = False
                        break
            if is_bottom:
                fractal_bottoms[i] = dataframe["low"].iloc[i]

        return fractal_tops, fractal_bottoms

    def _calculate_ma(self, dataframe: DataFrame, period: int, ma_type: str) -> np.ndarray:
        """
        Calculate moving average based on type.
        """
        if ma_type == "EMA":
            return ta.EMA(dataframe["close"], timeperiod=period)
        elif ma_type == "SMA":
            return ta.SMA(dataframe["close"], timeperiod=period)
        elif ma_type == "WMA":
            return ta.WMA(dataframe["close"], timeperiod=period)
        else:
            return ta.EMA(dataframe["close"], timeperiod=period)  # Default to EMA

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate fractals for both window sizes
        for window_size in [3, 5]:
            tops, bottoms = self._calculate_fractals(dataframe, window_size)
            dataframe[f"fractal_top_{window_size}"] = tops
            dataframe[f"fractal_bottom_{window_size}"] = bottoms

            # Forward fill to have the last valid fractal value
            dataframe[f"fractal_top_{window_size}"] = dataframe[
                f"fractal_top_{window_size}"
            ].ffill()
            dataframe[f"fractal_bottom_{window_size}"] = dataframe[
                f"fractal_bottom_{window_size}"
            ].ffill()

        # Calculate MAs based on actual parameter values
        # For long positions
        if hasattr(self, "long_ma_period"):
            ma_period = self.long_ma_period.value
            ma_type = self.long_ma_type.value
            dataframe[f"{ma_type}_{ma_period}"] = self._calculate_ma(dataframe, ma_period, ma_type)

        # For short positions
        if hasattr(self, "short_ma_period"):
            ma_period = self.short_ma_period.value
            ma_type = self.short_ma_type.value
            dataframe[f"{ma_type}_{ma_period}"] = self._calculate_ma(dataframe, ma_period, ma_type)

        # Calculate RSI based on actual parameter values
        if hasattr(self, "long_rsi_period"):
            rsi_period = self.long_rsi_period.value
            dataframe[f"rsi_{rsi_period}"] = ta.RSI(dataframe["close"], timeperiod=rsi_period)

        if hasattr(self, "short_rsi_period"):
            rsi_period = self.short_rsi_period.value
            dataframe[f"rsi_{rsi_period}"] = ta.RSI(dataframe["close"], timeperiod=rsi_period)

        # Calculate Volume MA based on actual parameter values
        if hasattr(self, "long_volume_ma_period"):
            vol_period = self.long_volume_ma_period.value
            dataframe[f"volume_ma_{vol_period}"] = ta.SMA(
                dataframe["volume"], timeperiod=vol_period
            )

        if hasattr(self, "short_volume_ma_period"):
            vol_period = self.short_volume_ma_period.value
            dataframe[f"volume_ma_{vol_period}"] = ta.SMA(
                dataframe["volume"], timeperiod=vol_period
            )

        # Calculate ADX based on actual parameter values
        if hasattr(self, "long_adx_period"):
            adx_period = self.long_adx_period.value
            dataframe[f"adx_{adx_period}"] = ta.ADX(dataframe, timeperiod=adx_period)

        if hasattr(self, "short_adx_period"):
            adx_period = self.short_adx_period.value
            dataframe[f"adx_{adx_period}"] = ta.ADX(dataframe, timeperiod=adx_period)

        # Additional indicators for market structure
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["bb_upper"], dataframe["bb_middle"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
        )

        # Market phase detection
        dataframe["hl2"] = (dataframe["high"] + dataframe["low"]) / 2
        dataframe["hlc3"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["ohlc4"] = (
            dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]
        ) / 4

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === LONG ENTRY CONDITIONS ===
        # Get fractal columns based on parameter
        long_fractal_col_top = f"fractal_top_{self.long_fractal_window.value}"
        long_fractal_col_bottom = f"fractal_bottom_{self.long_fractal_window.value}"

        # Base condition: Fractal breakout
        long_conditions = dataframe["close"] > dataframe[long_fractal_col_top] * (
            1 + self.long_breakout_threshold.value
        )

        # MA filter
        if self.long_use_ma_filter.value:
            ma_col = f"{self.long_ma_type.value}_{self.long_ma_period.value}"
            # Calculate MA if it doesn't exist
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.long_ma_period.value, self.long_ma_type.value
                )
            long_conditions = long_conditions & (dataframe["close"] > dataframe[ma_col])

        # RSI filter
        if self.long_use_rsi_filter.value:
            rsi_col = f"rsi_{self.long_rsi_period.value}"
            # Calculate RSI if it doesn't exist
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.long_rsi_period.value
                )
            long_conditions = long_conditions & (dataframe[rsi_col] > self.long_rsi_min.value)

        # Volume filter
        if self.long_use_volume_filter.value:
            vol_ma_col = f"volume_ma_{self.long_volume_ma_period.value}"
            # Calculate Volume MA if it doesn't exist
            if vol_ma_col not in dataframe.columns:
                dataframe[vol_ma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.long_volume_ma_period.value
                )
            long_conditions = long_conditions & (
                dataframe["volume"] > dataframe[vol_ma_col] * self.long_volume_threshold.value
            )

        # ADX filter
        if self.long_use_adx_filter.value:
            adx_col = f"adx_{self.long_adx_period.value}"
            # Calculate ADX if it doesn't exist
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.long_adx_period.value)
            long_conditions = long_conditions & (dataframe[adx_col] > self.long_adx_min.value)

        dataframe.loc[long_conditions, "enter_long"] = 1

        # Store fractal values for long entries
        dataframe.loc[long_conditions, "fractal_top_entry"] = dataframe[long_fractal_col_top]
        dataframe.loc[long_conditions, "fractal_bottom_entry"] = dataframe[long_fractal_col_bottom]

        # === SHORT ENTRY CONDITIONS ===
        # Get fractal columns based on parameter
        short_fractal_col_top = f"fractal_top_{self.short_fractal_window.value}"
        short_fractal_col_bottom = f"fractal_bottom_{self.short_fractal_window.value}"

        # Base condition: Fractal breakout
        short_conditions = dataframe["close"] < dataframe[short_fractal_col_bottom] * (
            1 - self.short_breakout_threshold.value
        )

        # MA filter
        if self.short_use_ma_filter.value:
            ma_col = f"{self.short_ma_type.value}_{self.short_ma_period.value}"
            # Calculate MA if it doesn't exist
            if ma_col not in dataframe.columns:
                dataframe[ma_col] = self._calculate_ma(
                    dataframe, self.short_ma_period.value, self.short_ma_type.value
                )
            short_conditions = short_conditions & (dataframe["close"] < dataframe[ma_col])

        # RSI filter
        if self.short_use_rsi_filter.value:
            rsi_col = f"rsi_{self.short_rsi_period.value}"
            # Calculate RSI if it doesn't exist
            if rsi_col not in dataframe.columns:
                dataframe[rsi_col] = ta.RSI(
                    dataframe["close"], timeperiod=self.short_rsi_period.value
                )
            short_conditions = short_conditions & (dataframe[rsi_col] < self.short_rsi_max.value)

        # Volume filter
        if self.short_use_volume_filter.value:
            vol_ma_col = f"volume_ma_{self.short_volume_ma_period.value}"
            # Calculate Volume MA if it doesn't exist
            if vol_ma_col not in dataframe.columns:
                dataframe[vol_ma_col] = ta.SMA(
                    dataframe["volume"], timeperiod=self.short_volume_ma_period.value
                )
            short_conditions = short_conditions & (
                dataframe["volume"] > dataframe[vol_ma_col] * self.short_volume_threshold.value
            )

        # ADX filter
        if self.short_use_adx_filter.value:
            adx_col = f"adx_{self.short_adx_period.value}"
            # Calculate ADX if it doesn't exist
            if adx_col not in dataframe.columns:
                dataframe[adx_col] = ta.ADX(dataframe, timeperiod=self.short_adx_period.value)
            short_conditions = short_conditions & (dataframe[adx_col] > self.short_adx_min.value)

        dataframe.loc[short_conditions, "enter_short"] = 1

        # Store fractal values for short entries
        dataframe.loc[short_conditions, "fractal_top_entry"] = dataframe[short_fractal_col_top]
        dataframe.loc[short_conditions, "fractal_bottom_entry"] = dataframe[
            short_fractal_col_bottom
        ]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit signals can be added here if needed
        # Currently using custom_exit for dynamic exits
        return dataframe

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
        Store fractal values and trade metadata when entering a trade.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # Initialize custom_info if not exists
        if not hasattr(self, "custom_info"):
            self.custom_info = {}

        # Store trade-specific information
        if "fractal_top_entry" in last_candle and "fractal_bottom_entry" in last_candle:
            self.custom_info[pair] = {
                "fractal_top_entry": last_candle["fractal_top_entry"],
                "fractal_bottom_entry": last_candle["fractal_bottom_entry"],
                "atr_entry": last_candle.get("atr", 0),
                "entry_time": current_time,
                "side": side,
                "highest_price": rate,
                "lowest_price": rate,
            }

        return True

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Dynamic exit logic based on ATR trailing stop and other conditions.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        atr_now = last_candle.get("atr", 0)

        if not hasattr(self, "custom_info") or pair not in self.custom_info:
            return None

        trade_info = self.custom_info[pair]
        entry_atr = trade_info.get("atr_entry", atr_now) or atr_now

        # Update highest/lowest prices
        trade_info["highest_price"] = max(trade_info.get("highest_price", 0), current_rate)
        trade_info["lowest_price"] = min(
            trade_info.get("lowest_price", float("inf")), current_rate
        )

        # --- Trailing-first exit philosophy ---

        # ATR-based trailing stop
        if not trade.is_short:
            trail_price = trade_info["highest_price"] - (atr_now * self.atr_trailing_k.value)
            if current_rate < trail_price:
                return "atr_trailing_exit"
        else:
            trail_price = trade_info["lowest_price"] + (atr_now * self.atr_trailing_k.value)
            if current_rate > trail_price:
                return "atr_trailing_exit"

        # --- Other exit conditions ---

        # Volatility-normalized take profit
        if atr_now > 0:
            normalized_profit = abs(current_rate - trade.open_rate) / atr_now
            if normalized_profit >= self.volatility_tp_X.value:
                return "volatility_tp"

        # Time-decay exit for losing trades
        trade_age_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if trade_age_hours > 3 and current_profit < -0.01:
            return "time_decay_loss"

        # Volatility collapse emergency exit
        if entry_atr > 0 and (atr_now / entry_atr) < 0.6 and current_profit < -0.02:
            return "volatility_collapse_exit"

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
        Dynamic stoploss based on fractal levels and ATR.
        """
        # Get stored fractal values
        if not hasattr(self, "custom_info"):
            return -1  # Return max stoploss if no info available

        if pair not in self.custom_info:
            return -1  # Return max stoploss if no info available

        trade_info = self.custom_info[pair]
        fractal_top_entry = trade_info.get("fractal_top_entry")
        fractal_bottom_entry = trade_info.get("fractal_bottom_entry")
        atr_entry = trade_info.get("atr_entry", 0)

        if fractal_top_entry is None or fractal_bottom_entry is None:
            return -1  # Return max stoploss if values not available

        # Calculate stoploss based on fractal levels
        if not trade.is_short:
            # Long position: stop below fractal bottom
            stoploss_price = fractal_bottom_entry - (atr_entry * 0.5)  # Add ATR buffer
            stoploss_pct = (trade.open_rate - stoploss_price) / trade.open_rate

            # Move stoploss to breakeven after certain profit
            if current_profit > 0.015:
                breakeven_stop = -0.002  # Small buffer below breakeven
                return max(breakeven_stop, -stoploss_pct)
        else:
            # Short position: stop above fractal top
            stoploss_price = fractal_top_entry + (atr_entry * 0.5)  # Add ATR buffer
            stoploss_pct = (stoploss_price - trade.open_rate) / trade.open_rate

            # Move stoploss to breakeven after certain profit
            if current_profit > 0.015:
                breakeven_stop = -0.002  # Small buffer below breakeven
                return max(breakeven_stop, -stoploss_pct)

        # Ensure stoploss is not worse than the configured maximum
        return max(-stoploss_pct, self.stoploss)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage based on market conditions.
        """
        # Conservative leverage approach
        return min(3.0, max_leverage)

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


class MACDSARScalper_02(IStrategy):
    """
    1-Min MACD + Parabolic SAR Trend-Follow Scalper

    Trade when MACD crosses in trend direction and Parabolic SAR confirms.
    Uses EMA 50 as trend filter for better accuracy.

    Improvements:
    - Added histogram momentum confirmation
    - Volume spike detection for better entries
    - ATR-based volatility filter
    - Time-based trading sessions

    FIXED: Hyperopt parameters now used in populate_entry_trend/populate_exit_trend
    instead of populate_indicators for proper hyperopt compatibility.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short = True

    # Minimal ROI designed for the strategy - aggressive for 1m scalping
    minimal_roi = {"0": 0.012, "2": 0.008, "5": 0.005, "10": 0.003}

    # Optimal stoploss
    stoploss = -0.02  # 2% stop for quick scalping

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Hyperparameters
    macd_fast = IntParameter(10, 15, default=12, space="buy")
    macd_slow = IntParameter(20, 30, default=26, space="buy")
    macd_signal = IntParameter(7, 11, default=9, space="buy")

    sar_af = DecimalParameter(0.01, 0.03, default=0.02, space="buy")
    sar_max = DecimalParameter(0.15, 0.25, default=0.2, space="buy")

    ema_period = IntParameter(40, 60, default=50, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(10, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(1.2, 2.0, default=1.5, space="buy")

    # ATR filter
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_min_threshold = DecimalParameter(0.0001, 0.0005, default=0.0002, space="buy")

    # Histogram threshold for momentum
    hist_threshold = DecimalParameter(0.0001, 0.0005, default=0.0002, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculate indicators for all possible hyperopt parameter values.
        This ensures hyperopt works correctly by having all variants available.
        """

        # Pre-calculate MACD for all possible parameter combinations
        for fast in range(10, 16):
            for slow in range(20, 31):
                for signal in range(7, 12):
                    if slow > fast:  # Ensure slow > fast
                        macd = ta.MACD(
                            dataframe,
                            fastperiod=fast,
                            slowperiod=slow,
                            signalperiod=signal,
                        )
                        dataframe[f"macd_{fast}_{slow}_{signal}"] = macd["macd"]
                        dataframe[f"macd_signal_{fast}_{slow}_{signal}"] = macd["macdsignal"]
                        dataframe[f"macd_hist_{fast}_{slow}_{signal}"] = macd["macdhist"]

        # Pre-calculate Parabolic SAR for all possible parameter combinations
        # sar_af: 0.01-0.03 (step 0.01), sar_max: 0.15-0.25 (step 0.01)
        for af in [0.01, 0.02, 0.03]:
            for mx in [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
                dataframe[f"sar_{af}_{mx}"] = ta.SAR(dataframe, acceleration=af, maximum=mx)

        # Pre-calculate EMA for all possible periods (40-60)
        for period in range(40, 61):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all possible periods (10-30)
        for period in range(10, 31):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate ATR for all possible periods (10-20)
        for period in range(10, 21):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Price action (fixed, no hyperopt parameters)
        dataframe["price_increase"] = dataframe["close"] > dataframe["close"].shift(1)
        dataframe["price_decrease"] = dataframe["close"] < dataframe["close"].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals.
        Hyperopt parameters are used here so they're evaluated each epoch.
        """
        # Get hyperopt parameter values
        fast = self.macd_fast.value
        slow = self.macd_slow.value
        signal = self.macd_signal.value
        af = round(self.sar_af.value, 2)
        mx = round(self.sar_max.value, 2)
        ema_period = self.ema_period.value
        volume_ma_period = self.volume_ma_period.value
        atr_period = self.atr_period.value

        # Get pre-calculated indicators for current hyperopt values
        macd = dataframe[f"macd_{fast}_{slow}_{signal}"]
        macd_signal_line = dataframe[f"macd_signal_{fast}_{slow}_{signal}"]
        macd_hist = dataframe[f"macd_hist_{fast}_{slow}_{signal}"]
        sar = dataframe[f"sar_{af}_{mx}"]
        ema = dataframe[f"ema_{ema_period}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        atr = dataframe[f"atr_{atr_period}"]

        # Calculate derived values using hyperopt parameters
        # Trend conditions
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema

        # MACD crossovers
        macd_cross_up = (macd > macd_signal_line) & (macd.shift(1) <= macd_signal_line.shift(1))
        macd_cross_down = (macd < macd_signal_line) & (macd.shift(1) >= macd_signal_line.shift(1))

        # SAR conditions
        sar_below = sar < dataframe["close"]
        sar_above = sar > dataframe["close"]

        # Histogram momentum using hyperopt threshold
        hist_positive_momentum = macd_hist > self.hist_threshold.value
        hist_negative_momentum = macd_hist < -self.hist_threshold.value

        # Volume spike using hyperopt parameters
        volume_spike = dataframe["volume"] > (volume_ma * self.volume_threshold.value)

        # LONG ENTRY
        dataframe.loc[
            (
                (uptrend)  # Trend up
                & (macd_cross_up)  # Bullish momentum shift
                & (sar_below)  # PSAR below price
                & (volume_spike)  # Volume confirmation
                & (atr > self.atr_min_threshold.value)  # Sufficient volatility
                & (hist_positive_momentum)  # Histogram confirms momentum
                & (dataframe["price_increase"])  # Current candle is green
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (downtrend)  # Trend down
                & (macd_cross_down)  # Bearish momentum shift
                & (sar_above)  # PSAR above price
                & (volume_spike)  # Volume confirmation
                & (atr > self.atr_min_threshold.value)  # Sufficient volatility
                & (hist_negative_momentum)  # Histogram confirms momentum
                & (dataframe["price_decrease"])  # Current candle is red
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
        fast = self.macd_fast.value
        slow = self.macd_slow.value
        signal = self.macd_signal.value
        af = round(self.sar_af.value, 2)
        mx = round(self.sar_max.value, 2)
        ema_period = self.ema_period.value

        # Get pre-calculated indicators for current hyperopt values
        macd = dataframe[f"macd_{fast}_{slow}_{signal}"]
        macd_signal_line = dataframe[f"macd_signal_{fast}_{slow}_{signal}"]
        macd_hist = dataframe[f"macd_hist_{fast}_{slow}_{signal}"]
        sar = dataframe[f"sar_{af}_{mx}"]
        ema = dataframe[f"ema_{ema_period}"]

        # Trend conditions
        uptrend = dataframe["close"] > ema
        downtrend = dataframe["close"] < ema

        # MACD crossovers
        macd_cross_up = (macd > macd_signal_line) & (macd.shift(1) <= macd_signal_line.shift(1))
        macd_cross_down = (macd < macd_signal_line) & (macd.shift(1) >= macd_signal_line.shift(1))

        # SAR conditions
        sar_below = sar < dataframe["close"]
        sar_above = sar > dataframe["close"]

        # LONG EXIT
        dataframe.loc[
            (
                (macd_cross_down)  # MACD bearish cross
                | (sar_above)  # SAR flipped above price
                | (downtrend)  # Trend changed
                | ((macd_hist < 0) & (macd_hist.shift(1) > 0))  # Histogram turned negative
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (macd_cross_up)  # MACD bullish cross
                | (sar_below)  # SAR flipped below price
                | (uptrend)  # Trend changed
                | ((macd_hist > 0) & (macd_hist.shift(1) < 0))  # Histogram turned positive
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
        Custom exit logic for quick scalping
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get hyperopt parameter values
        fast = self.macd_fast.value
        slow = self.macd_slow.value
        signal = self.macd_signal.value
        atr_period = self.atr_period.value

        macd_hist = last_candle[f"macd_hist_{fast}_{slow}_{signal}"]
        atr = last_candle[f"atr_{atr_period}"]

        # Quick profit taking for 1m scalping
        if current_profit > 0.008:
            return "quick_profit"

        # Exit if volatility drops too low (consolidation)
        if atr < self.atr_min_threshold.value * 0.5:
            return "low_volatility"

        # Time-based exit for stuck trades (10 minutes for 1m timeframe)
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=10):
            if current_profit > -0.003:  # Small loss or profit
                return "time_exit"

        # Exit if MACD histogram loses momentum significantly
        if not trade.is_short:
            if macd_hist < -self.hist_threshold.value * 2:
                return "momentum_loss_long"
        else:
            if macd_hist > self.hist_threshold.value * 2:
                return "momentum_loss_short"

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
        Custom stoploss logic for 1m scalping
        """

        # After 3 minutes, if profit > 0.5%, move to breakeven
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=3):
            if current_profit > 0.005:
                return -0.001  # Near breakeven

        # After 5 minutes, if profit > 0.3%, use tight stop
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=5):
            if current_profit > 0.003:
                return -0.003  # Tight stop

        # Progressive stop based on time
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=8):
            return -0.01  # Tighter stop after 8 minutes

        return self.stoploss

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

        # Avoid trading during low liquidity hours (optional)
        hour = current_time.hour
        if hour >= 2 and hour <= 6:  # UTC hours
            return False

        # Check if we have too many open trades on this pair
        open_trades = len([trade for trade in Trade.get_open_trades() if trade.pair == pair])
        if open_trades >= 2:  # Max 2 trades per pair
            return False

        return True

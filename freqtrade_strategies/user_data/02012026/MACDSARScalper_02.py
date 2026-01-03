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
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "1m"

    # Can this strategy go short?
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
        Adds several different TA indicators to the given DataFrame
        """

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(
            dataframe, acceleration=self.sar_af.value, maximum=self.sar_max.value
        )

        # EMA for trend filter
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Volume
        dataframe["volume_ma"] = ta.SMA(
            dataframe["volume"], timeperiod=self.volume_ma_period.value
        )
        dataframe["volume_spike"] = dataframe["volume"] > (
            dataframe["volume_ma"] * self.volume_threshold.value
        )

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Trend conditions
        dataframe["uptrend"] = dataframe["close"] > dataframe["ema50"]
        dataframe["downtrend"] = dataframe["close"] < dataframe["ema50"]

        # MACD crossovers
        dataframe["macd_cross_up"] = (dataframe["macd"] > dataframe["macd_signal"]) & (
            dataframe["macd"].shift(1) <= dataframe["macd_signal"].shift(1)
        )

        dataframe["macd_cross_down"] = (
            dataframe["macd"] < dataframe["macd_signal"]
        ) & (dataframe["macd"].shift(1) >= dataframe["macd_signal"].shift(1))

        # SAR conditions
        dataframe["sar_below"] = dataframe["sar"] < dataframe["close"]
        dataframe["sar_above"] = dataframe["sar"] > dataframe["close"]

        # Histogram momentum
        dataframe["hist_positive_momentum"] = (
            dataframe["macd_hist"] > self.hist_threshold.value
        )
        dataframe["hist_negative_momentum"] = (
            dataframe["macd_hist"] < -self.hist_threshold.value
        )

        # Price action
        dataframe["price_increase"] = dataframe["close"] > dataframe["close"].shift(1)
        dataframe["price_decrease"] = dataframe["close"] < dataframe["close"].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """

        # LONG ENTRY
        dataframe.loc[
            (
                (dataframe["uptrend"])  # Trend up
                & (dataframe["macd_cross_up"])  # Bullish momentum shift
                & (dataframe["sar_below"])  # PSAR below price
                & (dataframe["volume_spike"])  # Volume confirmation
                & (
                    dataframe["atr"] > self.atr_min_threshold.value
                )  # Sufficient volatility
                & (dataframe["hist_positive_momentum"])  # Histogram confirms momentum
                & (dataframe["price_increase"])  # Current candle is green
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY
        dataframe.loc[
            (
                (dataframe["downtrend"])  # Trend down
                & (dataframe["macd_cross_down"])  # Bearish momentum shift
                & (dataframe["sar_above"])  # PSAR above price
                & (dataframe["volume_spike"])  # Volume confirmation
                & (
                    dataframe["atr"] > self.atr_min_threshold.value
                )  # Sufficient volatility
                & (dataframe["hist_negative_momentum"])  # Histogram confirms momentum
                & (dataframe["price_decrease"])  # Current candle is red
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
                (dataframe["macd_cross_down"])  # MACD bearish cross
                | (dataframe["sar_above"])  # SAR flipped above price
                | (dataframe["downtrend"])  # Trend changed
                | (
                    (dataframe["macd_hist"] < 0)  # Histogram turned negative
                    & (dataframe["macd_hist"].shift(1) > 0)
                )
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT
        dataframe.loc[
            (
                (dataframe["macd_cross_up"])  # MACD bullish cross
                | (dataframe["sar_below"])  # SAR flipped below price
                | (dataframe["uptrend"])  # Trend changed
                | (
                    (dataframe["macd_hist"] > 0)  # Histogram turned positive
                    & (dataframe["macd_hist"].shift(1) < 0)
                )
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

        # Quick profit taking for 1m scalping
        if current_profit > 0.008:
            return "quick_profit"

        # Exit if volatility drops too low (consolidation)
        if last_candle["atr"] < self.atr_min_threshold.value * 0.5:
            return "low_volatility"

        # Time-based exit for stuck trades (10 minutes for 1m timeframe)
        if current_time - trade.open_date_utc > pd.Timedelta(minutes=10):
            if current_profit > -0.003:  # Small loss or profit
                return "time_exit"

        # Exit if MACD histogram loses momentum significantly
        if not trade.is_short:
            if last_candle["macd_hist"] < -self.hist_threshold.value * 2:
                return "momentum_loss_long"
        else:
            if last_candle["macd_hist"] > self.hist_threshold.value * 2:
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
        open_trades = len(
            [trade for trade in Trade.get_open_trades() if trade.pair == pair]
        )
        if open_trades >= 2:  # Max 2 trades per pair
            return False

        return True

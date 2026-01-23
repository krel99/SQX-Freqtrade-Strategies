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


class Strategy10012026_1(IStrategy):
    """
    Volatility Squeeze Breakout Strategy
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.1}

    # Optimal stoploss
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # --- Hyperparameters ---
    # Bollinger Bands 1
    buy_bb1_period = IntParameter(10, 50, default=20, space="buy")
    buy_bb1_stddev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    # Keltner Channel 1
    buy_kc1_ema_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc1_atr_period = IntParameter(10, 50, default=20, space="buy")
    buy_kc1_mult = DecimalParameter(1.0, 3.0, default=1.5, space="buy")

    # Bollinger Bands 2
    buy_bb2_period = IntParameter(10, 50, default=30, space="buy")
    buy_bb2_stddev = DecimalParameter(1.5, 3.0, default=2.5, space="buy")

    # Keltner Channel 2
    buy_kc2_ema_period = IntParameter(10, 50, default=30, space="buy")
    buy_kc2_atr_period = IntParameter(10, 50, default=30, space="buy")
    buy_kc2_mult = DecimalParameter(1.0, 3.0, default=2.0, space="buy")

    # Squeeze Lookback
    buy_squeeze_lookback = IntParameter(1, 10, default=3, space="buy")

    # ADX
    buy_adx_period = IntParameter(10, 50, default=14, space="buy")
    buy_adx_threshold = IntParameter(15, 50, default=25, space="buy")

    # RSI
    buy_rsi_period = IntParameter(10, 50, default=14, space="buy")
    buy_rsi_threshold = IntParameter(40, 70, default=55, space="buy")

    # MACD
    buy_macd_fast = IntParameter(6, 24, default=12, space="buy")
    buy_macd_slow = IntParameter(13, 52, default=26, space="buy")
    buy_macd_signal = IntParameter(5, 18, default=9, space="buy")

    # Volume
    buy_volume_ma_period = IntParameter(10, 50, default=20, space="buy")
    buy_volume_spike_factor = DecimalParameter(1.1, 3.0, default=1.5, space="buy")

    # Breakout
    buy_breakout_confirmation_period = IntParameter(1, 5, default=1, space="buy")
    buy_breakout_price_source = CategoricalParameter(
        ["close", "high", "low"], default="close", space="buy"
    )

    # MFI
    buy_mfi_period = IntParameter(10, 50, default=14, space="buy")
    buy_mfi_threshold = IntParameter(30, 70, default=50, space="buy")

    # CCI
    buy_cci_period = IntParameter(10, 50, default=20, space="buy")
    buy_cci_threshold = IntParameter(50, 150, default=100, space="buy")

    # Exit Parameters
    sell_rsi_threshold = IntParameter(60, 90, default=75, space="sell")
    sell_take_profit_pct = DecimalParameter(0.01, 0.1, default=0.03, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        # Pre-calculate EMA for all Keltner periods (10-50)
        for period in range(10, 51):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate ATR for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Pre-calculate Bollinger Bands for all periods (10-50) and stddev combinations
        for period in range(10, 51):
            for stddev in [
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                3.0,
            ]:
                stddev_str = str(stddev).replace(".", "_")
                bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=stddev, nbdevdn=stddev)
                dataframe[f"bb_lower_{period}_{stddev_str}"] = bb["lowerband"]
                dataframe[f"bb_upper_{period}_{stddev_str}"] = bb["upperband"]

        # Pre-calculate ADX for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Pre-calculate RSI for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

        # Pre-calculate Volume MA for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"volume_ma_{period}"] = ta.SMA(dataframe["volume"], timeperiod=period)

        # Pre-calculate MFI for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"mfi_{period}"] = ta.MFI(dataframe, timeperiod=period)

        # Pre-calculate CCI for all periods (10-50)
        for period in range(10, 51):
            dataframe[f"cci_{period}"] = ta.CCI(dataframe, timeperiod=period)

        # Pre-calculate MACD for common combinations
        for fast in range(6, 25):
            for slow in range(13, 53):
                for signal in range(5, 19):
                    if fast < slow:
                        macd = ta.MACD(
                            dataframe, fastperiod=fast, slowperiod=slow, signalperiod=signal
                        )
                        dataframe[f"macd_{fast}_{slow}_{signal}"] = macd["macd"]
                        dataframe[f"macdsignal_{fast}_{slow}_{signal}"] = macd["macdsignal"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        bb1_period = self.buy_bb1_period.value
        bb1_stddev = self.buy_bb1_stddev.value
        bb1_stddev_str = str(round(bb1_stddev, 1)).replace(".", "_")

        bb2_period = self.buy_bb2_period.value
        bb2_stddev = self.buy_bb2_stddev.value
        bb2_stddev_str = str(round(bb2_stddev, 1)).replace(".", "_")

        kc1_ema_period = self.buy_kc1_ema_period.value
        kc1_atr_period = self.buy_kc1_atr_period.value
        kc1_mult = self.buy_kc1_mult.value

        kc2_ema_period = self.buy_kc2_ema_period.value
        kc2_atr_period = self.buy_kc2_atr_period.value
        kc2_mult = self.buy_kc2_mult.value

        adx_period = self.buy_adx_period.value
        rsi_period = self.buy_rsi_period.value
        macd_fast = self.buy_macd_fast.value
        macd_slow = self.buy_macd_slow.value
        macd_signal = self.buy_macd_signal.value
        volume_ma_period = self.buy_volume_ma_period.value
        mfi_period = self.buy_mfi_period.value
        cci_period = self.buy_cci_period.value

        # Select pre-calculated indicators
        bb1_lower = dataframe[f"bb_lower_{bb1_period}_{bb1_stddev_str}"]
        bb1_upper = dataframe[f"bb_upper_{bb1_period}_{bb1_stddev_str}"]

        bb2_lower = dataframe[f"bb_lower_{bb2_period}_{bb2_stddev_str}"]
        bb2_upper = dataframe[f"bb_upper_{bb2_period}_{bb2_stddev_str}"]

        kc1_ema = dataframe[f"ema_{kc1_ema_period}"]
        kc1_atr = dataframe[f"atr_{kc1_atr_period}"]
        kc1_lower = kc1_ema - kc1_atr * kc1_mult
        kc1_upper = kc1_ema + kc1_atr * kc1_mult

        kc2_ema = dataframe[f"ema_{kc2_ema_period}"]
        kc2_atr = dataframe[f"atr_{kc2_atr_period}"]
        kc2_lower = kc2_ema - kc2_atr * kc2_mult
        kc2_upper = kc2_ema + kc2_atr * kc2_mult

        adx = dataframe[f"adx_{adx_period}"]
        rsi = dataframe[f"rsi_{rsi_period}"]
        macd = dataframe[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"]
        macdsignal = dataframe[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"]
        volume_ma = dataframe[f"volume_ma_{volume_ma_period}"]
        mfi = dataframe[f"mfi_{mfi_period}"]
        cci = dataframe[f"cci_{cci_period}"]

        # Calculate squeeze detection
        squeeze_on_1 = (bb1_lower > kc1_lower) & (bb1_upper < kc1_upper)
        squeeze_on_2 = (bb2_lower > kc2_lower) & (bb2_upper < kc2_upper)
        squeeze_on = squeeze_on_1 & squeeze_on_2

        is_squeeze = (
            squeeze_on.rolling(self.buy_squeeze_lookback.value).sum()
            == self.buy_squeeze_lookback.value
        )

        breakout_source = dataframe[self.buy_breakout_price_source.value]
        is_breakout = breakout_source > bb1_upper

        confirmation = (
            (adx > self.buy_adx_threshold.value)
            & (rsi > self.buy_rsi_threshold.value)
            & (macd > macdsignal)
            & (dataframe["volume"] > volume_ma * self.buy_volume_spike_factor.value)
            & (mfi > self.buy_mfi_threshold.value)
            & (cci > self.buy_cci_threshold.value)
        )

        dataframe.loc[
            is_squeeze.shift(self.buy_breakout_confirmation_period.value)
            & ~squeeze_on
            & is_breakout
            & confirmation,
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        rsi_period = self.buy_rsi_period.value
        bb1_period = self.buy_bb1_period.value
        bb1_stddev = self.buy_bb1_stddev.value
        bb1_stddev_str = str(round(bb1_stddev, 1)).replace(".", "_")

        # Select pre-calculated indicators
        rsi = dataframe[f"rsi_{rsi_period}"]
        bb1_upper = dataframe[f"bb_upper_{bb1_period}_{bb1_stddev_str}"]

        rsi_exit = rsi > self.sell_rsi_threshold.value

        # Take profit when price is a certain percentage above the upper BB
        bb_take_profit = dataframe["close"] > bb1_upper * (1 + self.sell_take_profit_pct.value)

        dataframe.loc[rsi_exit | bb_take_profit, "exit_long"] = 1

        return dataframe

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


class MultiIndicatorMomentum(IStrategy):
    """
    Multi-Indicator Momentum Strategy

    This strategy combines multiple technical indicators to identify momentum-based
    trading opportunities. It uses RSI, MACD, EMA, Bollinger Bands, ATR, and volume
    analysis to generate entry and exit signals.

    All parameters are hyperoptimizable for finding the optimal configuration.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3
    load_hyperopt_params = True
    # Timeframe
    timeframe = "15m"

    # Can short
    can_short = True

    # Hyperopt spaces
    # ROI table - will be overridden by hyperopt
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0.01}

    # Stoploss - will be overridden by hyperopt
    stoploss = -0.10

    # Trailing stop - will be overridden by hyperopt
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Exit signal
    use_exit_signal = CategoricalParameter([True, False], default=True, space="sell")

    # -------------------------------------------------------------------------
    # Hyperoptimizable parameters for indicators
    # -------------------------------------------------------------------------

    # RSI parameters
    rsi_period = IntParameter(10, 50, default=14, space="buy")
    rsi_period_slow = IntParameter(20, 60, default=28, space="buy")

    # MACD parameters
    macd_fast = IntParameter(8, 20, default=12, space="buy")
    macd_slow = IntParameter(20, 50, default=26, space="buy")
    macd_signal = IntParameter(5, 20, default=9, space="buy")

    # EMA parameters
    ema_short = IntParameter(5, 25, default=9, space="buy")
    ema_medium = IntParameter(20, 50, default=21, space="buy")
    ema_long = IntParameter(50, 200, default=55, space="buy")

    # Bollinger Bands parameters
    bb_period = IntParameter(10, 40, default=20, space="buy")
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # ATR parameters
    atr_period = IntParameter(7, 21, default=14, space="buy")
    atr_multiplier = DecimalParameter(0.5, 3.0, default=1.5, decimals=1, space="buy")

    # Volume parameters
    volume_ma_period = IntParameter(10, 50, default=20, space="buy")
    volume_threshold = DecimalParameter(0.5, 2.5, default=1.2, decimals=1, space="buy")

    # Stochastic RSI parameters
    stoch_rsi_period = IntParameter(10, 30, default=14, space="buy")
    stoch_rsi_smooth_k = IntParameter(1, 5, default=3, space="buy")
    stoch_rsi_smooth_d = IntParameter(1, 5, default=3, space="buy")

    # -------------------------------------------------------------------------
    # Hyperoptimizable parameters for entry signals
    # -------------------------------------------------------------------------

    # Long entry thresholds
    buy_rsi_lower = IntParameter(20, 50, default=30, space="buy")
    buy_rsi_upper = IntParameter(40, 70, default=50, space="buy")
    buy_stoch_rsi = IntParameter(10, 40, default=20, space="buy")
    buy_bb_width_min = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="buy")
    buy_macd_hist_threshold = DecimalParameter(-0.001, 0.001, default=0.0, decimals=4, space="buy")

    # Short entry thresholds
    sell_rsi_lower = IntParameter(30, 60, default=50, space="sell")
    sell_rsi_upper = IntParameter(50, 80, default=70, space="sell")
    sell_stoch_rsi = IntParameter(60, 90, default=80, space="sell")
    sell_bb_width_min = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="sell")
    sell_macd_hist_threshold = DecimalParameter(
        -0.001, 0.001, default=0.0, decimals=4, space="sell"
    )

    # Exit thresholds
    exit_rsi_long = IntParameter(60, 85, default=70, space="sell")
    exit_rsi_short = IntParameter(15, 40, default=30, space="sell")
    exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell")

    # -------------------------------------------------------------------------
    # Enable/disable conditions
    # -------------------------------------------------------------------------

    # Long conditions
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_volume_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_stoch_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_atr_enabled = CategoricalParameter([True, False], default=True, space="buy")

    # Short conditions
    sell_rsi_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_macd_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_bb_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_ema_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_volume_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_stoch_rsi_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_atr_enabled = CategoricalParameter([True, False], default=True, space="sell")

    # -------------------------------------------------------------------------
    # Protection parameters (optimizable)
    # -------------------------------------------------------------------------

    # StoplossGuard - Stops trading after N stoplosses within a time window
    protection_stoploss_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_stoploss_trade_limit = IntParameter(
        2, 10, default=4, space="protection"
    )  # Number of stoplosses before protection
    protection_stoploss_lookback_period = IntParameter(
        10, 1440, default=60, space="protection"
    )  # Lookback in candles
    protection_stoploss_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )  # How long to stop trading in candles

    # CooldownPeriod - Cooldown after each trade
    protection_cooldown_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_cooldown_period = IntParameter(
        1, 20, default=5, space="protection"
    )  # Cooldown period in candles

    # LowProfitPairs - Stops trading pairs with low profits
    protection_lowprofit_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_lowprofit_trade_limit = IntParameter(
        2, 10, default=4, space="protection"
    )  # Minimum number of trades
    protection_lowprofit_lookback_period = IntParameter(
        10, 1440, default=360, space="protection"
    )  # Lookback in candles
    protection_lowprofit_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )  # How long to stop trading in candles
    protection_lowprofit_required_profit = DecimalParameter(
        -0.05, 0.05, default=0.0, decimals=3, space="protection"
    )  # Minimum required profit

    # MaxDrawdown - Stop trading if max drawdown is reached
    protection_maxdrawdown_enabled = CategoricalParameter(
        [True, False], default=False, space="protection"
    )
    protection_maxdrawdown_trade_limit = IntParameter(
        3, 20, default=8, space="protection"
    )  # Minimum number of trades
    protection_maxdrawdown_lookback_period = IntParameter(
        10, 1440, default=200, space="protection"
    )  # Lookback in candles
    protection_maxdrawdown_stop_duration = IntParameter(
        10, 360, default=60, space="protection"
    )  # How long to stop trading in candles
    protection_maxdrawdown_allowed_drawdown = DecimalParameter(
        0.01, 0.30, default=0.10, decimals=2, space="protection"
    )  # Maximum allowed drawdown

    # -------------------------------------------------------------------------
    # Hyperopt spaces for ROI, Stoploss, and Trailing stop
    # -------------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """
        Initialize the strategy and properly set use_exit_signal value
        when loading from hyperopt parameters.
        """
        super().__init__(config)
        # If use_exit_signal is a hyperopt parameter, extract its value
        if hasattr(self.use_exit_signal, "value"):
            self.use_exit_signal = self.use_exit_signal.value

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table based on hyperopt parameters
        """
        roi_table = {}
        roi_table[0] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"] + params["roi_p4"]
        roi_table[params["roi_t4"]] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"]
        roi_table[params["roi_t4"] + params["roi_t3"]] = params["roi_p1"] + params["roi_p2"]
        roi_table[params["roi_t4"] + params["roi_t3"] + params["roi_t2"]] = params["roi_p1"]
        roi_table[params["roi_t4"] + params["roi_t3"] + params["roi_t2"] + params["roi_t1"]] = 0

        return roi_table

    def roi_space() -> List[Dimension]:
        return [
            Integer(10, 120, name="roi_t1"),
            Integer(10, 120, name="roi_t2"),
            Integer(10, 120, name="roi_t3"),
            Integer(10, 120, name="roi_t4"),
            SKDecimal(0.001, 0.05, decimals=3, name="roi_p1"),
            SKDecimal(0.001, 0.05, decimals=3, name="roi_p2"),
            SKDecimal(0.001, 0.05, decimals=3, name="roi_p3"),
            SKDecimal(0.001, 0.05, decimals=3, name="roi_p4"),
        ]

    def stoploss_space() -> List[Dimension]:
        return [
            SKDecimal(-0.35, -0.02, decimals=3, name="stoploss"),
        ]

    def trailing_space() -> List[Dimension]:
        return [
            Categorical([True, False], name="trailing_stop"),
            SKDecimal(0.001, 0.10, decimals=3, name="trailing_stop_positive"),
            SKDecimal(0.001, 0.05, decimals=3, name="trailing_stop_positive_offset"),
            Categorical([True, False], name="trailing_only_offset_is_reached"),
        ]

    @staticmethod
    def protection_space() -> List[Dimension]:
        """
        Define the hyperoptimization space for protections.
        """
        return [
            # StoplossGuard
            Categorical([True, False], name="protection_stoploss_enabled"),
            Integer(2, 10, name="protection_stoploss_trade_limit"),
            Integer(10, 1440, name="protection_stoploss_lookback_period"),
            Integer(10, 360, name="protection_stoploss_stop_duration"),
            # CooldownPeriod
            Categorical([True, False], name="protection_cooldown_enabled"),
            Integer(1, 20, name="protection_cooldown_period"),
            # LowProfitPairs
            Categorical([True, False], name="protection_lowprofit_enabled"),
            Integer(2, 10, name="protection_lowprofit_trade_limit"),
            Integer(10, 1440, name="protection_lowprofit_lookback_period"),
            Integer(10, 360, name="protection_lowprofit_stop_duration"),
            SKDecimal(-0.05, 0.05, decimals=3, name="protection_lowprofit_required_profit"),
            # MaxDrawdown
            Categorical([True, False], name="protection_maxdrawdown_enabled"),
            Integer(3, 20, name="protection_maxdrawdown_trade_limit"),
            Integer(10, 1440, name="protection_maxdrawdown_lookback_period"),
            Integer(10, 360, name="protection_maxdrawdown_stop_duration"),
            SKDecimal(0.01, 0.30, decimals=2, name="protection_maxdrawdown_allowed_drawdown"),
        ]

    # -------------------------------------------------------------------------
    # Strategy methods
    # -------------------------------------------------------------------------
    def _get_param_value(self, param):
        """
        Safely get parameter value, handling both hyperopt parameters and plain values.
        This is needed for compatibility with hyperopt parallel processing.
        """
        return param.value if hasattr(param, "value") else param

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators using batch calculation to avoid DataFrame fragmentation
        """
        # Store new indicators in a dictionary first
        indicators = {}

        # Get current parameter values
        rsi_period = self._get_param_value(self.rsi_period)
        rsi_period_slow = self._get_param_value(self.rsi_period_slow)
        macd_fast = self._get_param_value(self.macd_fast)
        macd_slow = self._get_param_value(self.macd_slow)
        macd_signal = self._get_param_value(self.macd_signal)
        ema_short = self._get_param_value(self.ema_short)
        ema_medium = self._get_param_value(self.ema_medium)
        ema_long = self._get_param_value(self.ema_long)
        bb_period = self._get_param_value(self.bb_period)
        bb_std = self._get_param_value(self.bb_std)
        atr_period = self._get_param_value(self.atr_period)

        # Calculate RSI variants
        indicators[f"rsi_{rsi_period}"] = ta.RSI(dataframe, timeperiod=rsi_period)
        indicators[f"rsi_slow_{rsi_period_slow}"] = ta.RSI(dataframe, timeperiod=rsi_period_slow)

        # Calculate MACD with current parameters
        macd = ta.MACD(
            dataframe, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
        )
        indicators[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"] = macd["macd"]
        indicators[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"] = macd["macdsignal"]
        indicators[f"macdhist_{macd_fast}_{macd_slow}_{macd_signal}"] = macd["macdhist"]

        # Calculate EMA variants
        indicators[f"ema_short_{ema_short}"] = ta.EMA(dataframe, timeperiod=ema_short)
        indicators[f"ema_medium_{ema_medium}"] = ta.EMA(dataframe, timeperiod=ema_medium)
        indicators[f"ema_long_{ema_long}"] = ta.EMA(dataframe, timeperiod=ema_long)

        # Calculate Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        indicators[f"bb_upper_{bb_period}_{bb_std}"] = bb["upperband"]
        indicators[f"bb_middle_{bb_period}_{bb_std}"] = bb["middleband"]
        indicators[f"bb_lower_{bb_period}_{bb_std}"] = bb["lowerband"]
        indicators[f"bb_width_{bb_period}_{bb_std}"] = (bb["upperband"] - bb["lowerband"]) / bb[
            "middleband"
        ]
        indicators[f"bb_percent_{bb_period}_{bb_std}"] = (dataframe["close"] - bb["lowerband"]) / (
            bb["upperband"] - bb["lowerband"]
        )

        # Calculate ATR
        indicators[f"atr_{atr_period}"] = ta.ATR(dataframe, timeperiod=atr_period)

        # Calculate Stochastic RSI
        stoch_rsi_period = self._get_param_value(self.stoch_rsi_period)
        stoch_rsi_smooth_k = self._get_param_value(self.stoch_rsi_smooth_k)
        stoch_rsi_smooth_d = self._get_param_value(self.stoch_rsi_smooth_d)
        stoch = ta.STOCHRSI(
            dataframe,
            timeperiod=stoch_rsi_period,
            fastk_period=stoch_rsi_smooth_k,
            fastd_period=stoch_rsi_smooth_d,
        )
        indicators[f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"] = (
            stoch["fastk"]
        )
        indicators[f"stoch_rsi_d_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"] = (
            stoch["fastd"]
        )

        # Volume indicators
        volume_ma_period = self._get_param_value(self.volume_ma_period)
        indicators[f"volume_ma_{volume_ma_period}"] = (
            dataframe["volume"].rolling(window=volume_ma_period).mean()
        )
        indicators[f"volume_ratio_{volume_ma_period}"] = (
            dataframe["volume"] / dataframe["volume"].rolling(window=volume_ma_period).mean()
        )

        # Additional derived indicators
        indicators["high_low_ratio"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        indicators["close_open_ratio"] = (dataframe["close"] - dataframe["open"]) / dataframe[
            "open"
        ]

        # Price action indicators
        indicators["price_change"] = dataframe["close"].pct_change()
        indicators["volume_change"] = dataframe["volume"].pct_change()

        # Momentum indicators
        indicators["momentum_10"] = dataframe["close"] - dataframe["close"].shift(10)
        indicators["momentum_20"] = dataframe["close"] - dataframe["close"].shift(20)

        # Convert indicators dict to DataFrame and concatenate at once
        indicators_df = pd.DataFrame(indicators, index=dataframe.index)
        dataframe = pd.concat([dataframe, indicators_df], axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals using hyperoptimizable parameters
        """
        conditions_long = []
        conditions_short = []

        # Get current parameter values
        rsi_period = self._get_param_value(self.rsi_period)
        rsi_period_slow = self._get_param_value(self.rsi_period_slow)
        macd_fast = self._get_param_value(self.macd_fast)
        macd_slow = max(self._get_param_value(self.macd_slow), macd_fast + 1)
        macd_signal = self._get_param_value(self.macd_signal)
        ema_short = self._get_param_value(self.ema_short)
        ema_medium = self._get_param_value(self.ema_medium)
        ema_long = self._get_param_value(self.ema_long)
        bb_period = self._get_param_value(self.bb_period)
        bb_std = self._get_param_value(self.bb_std)
        atr_period = self._get_param_value(self.atr_period)
        volume_ma_period = self._get_param_value(self.volume_ma_period)
        stoch_rsi_period = self._get_param_value(self.stoch_rsi_period)
        stoch_rsi_smooth_k = self._get_param_value(self.stoch_rsi_smooth_k)
        stoch_rsi_smooth_d = self._get_param_value(self.stoch_rsi_smooth_d)

        # Ensure required columns exist
        self._ensure_indicator_columns(
            dataframe,
            rsi_period,
            rsi_period_slow,
            macd_fast,
            macd_slow,
            macd_signal,
            ema_short,
            ema_medium,
            ema_long,
            bb_period,
            bb_std,
            atr_period,
            volume_ma_period,
            stoch_rsi_period,
            stoch_rsi_smooth_k,
            stoch_rsi_smooth_d,
        )

        # ===== LONG ENTRY CONDITIONS =====

        # RSI conditions
        if self._get_param_value(self.buy_rsi_enabled):
            conditions_long.append(
                (dataframe[f"rsi_{rsi_period}"] > self._get_param_value(self.buy_rsi_lower))
                & (dataframe[f"rsi_{rsi_period}"] < self._get_param_value(self.buy_rsi_upper))
                & (dataframe[f"rsi_{rsi_period}"] > dataframe[f"rsi_{rsi_period}"].shift(1))
            )

        # MACD conditions
        if self._get_param_value(self.buy_macd_enabled):
            conditions_long.append(
                (
                    dataframe[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"]
                    > dataframe[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"]
                )
                & (
                    dataframe[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"].shift(1)
                    <= dataframe[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"].shift(1)
                )
                & (
                    dataframe[f"macdhist_{macd_fast}_{macd_slow}_{macd_signal}"]
                    > self._get_param_value(self.buy_macd_hist_threshold)
                )
            )

        # Bollinger Bands conditions
        if self._get_param_value(self.buy_bb_enabled):
            conditions_long.append(
                (dataframe["close"] < dataframe[f"bb_middle_{bb_period}_{bb_std}"])
                & (dataframe["close"] > dataframe[f"bb_lower_{bb_period}_{bb_std}"])
                & (
                    dataframe[f"bb_width_{bb_period}_{bb_std}"]
                    > self._get_param_value(self.buy_bb_width_min)
                )
            )

        # EMA conditions
        if self._get_param_value(self.buy_ema_enabled):
            conditions_long.append(
                (dataframe["close"] > dataframe[f"ema_short_{ema_short}"])
                & (dataframe[f"ema_short_{ema_short}"] > dataframe[f"ema_medium_{ema_medium}"])
            )

        # Volume conditions
        if self._get_param_value(self.buy_volume_enabled):
            conditions_long.append(
                (
                    dataframe[f"volume_ratio_{volume_ma_period}"]
                    > self._get_param_value(self.volume_threshold)
                )
                & (dataframe["volume"] > 0)
            )

        # Stochastic RSI conditions
        if self._get_param_value(self.buy_stoch_rsi_enabled):
            conditions_long.append(
                (
                    dataframe[
                        f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                    < self._get_param_value(self.buy_stoch_rsi)
                )
                & (
                    dataframe[
                        f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                    > dataframe[
                        f"stoch_rsi_d_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                )
            )

        # ATR filter for volatility
        if self._get_param_value(self.buy_atr_enabled):
            conditions_long.append(
                dataframe[f"atr_{atr_period}"] > dataframe[f"atr_{atr_period}"].shift(14).mean()
            )

        # ===== SHORT ENTRY CONDITIONS =====

        # RSI conditions
        if self._get_param_value(self.sell_rsi_enabled):
            conditions_short.append(
                (dataframe[f"rsi_{rsi_period}"] > self._get_param_value(self.sell_rsi_lower))
                & (dataframe[f"rsi_{rsi_period}"] < self._get_param_value(self.sell_rsi_upper))
                & (dataframe[f"rsi_{rsi_period}"] < dataframe[f"rsi_{rsi_period}"].shift(1))
            )

        # MACD conditions
        if self._get_param_value(self.sell_macd_enabled):
            conditions_short.append(
                (
                    dataframe[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"]
                    < dataframe[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"]
                )
                & (
                    dataframe[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"].shift(1)
                    >= dataframe[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"].shift(1)
                )
                & (
                    dataframe[f"macdhist_{macd_fast}_{macd_slow}_{macd_signal}"]
                    < self._get_param_value(self.sell_macd_hist_threshold)
                )
            )

        # Bollinger Bands conditions
        if self._get_param_value(self.sell_bb_enabled):
            conditions_short.append(
                (dataframe["close"] > dataframe[f"bb_middle_{bb_period}_{bb_std}"])
                & (dataframe["close"] < dataframe[f"bb_upper_{bb_period}_{bb_std}"])
                & (
                    dataframe[f"bb_width_{bb_period}_{bb_std}"]
                    > self._get_param_value(self.sell_bb_width_min)
                )
            )

        # EMA conditions
        if self._get_param_value(self.sell_ema_enabled):
            conditions_short.append(
                (dataframe["close"] < dataframe[f"ema_short_{ema_short}"])
                & (dataframe[f"ema_short_{ema_short}"] < dataframe[f"ema_medium_{ema_medium}"])
            )

        # Volume conditions
        if self._get_param_value(self.sell_volume_enabled):
            conditions_short.append(
                (
                    dataframe[f"volume_ratio_{volume_ma_period}"]
                    > self._get_param_value(self.volume_threshold)
                )
                & (dataframe["volume"] > 0)
            )

        # Stochastic RSI conditions
        if self._get_param_value(self.sell_stoch_rsi_enabled):
            conditions_short.append(
                (
                    dataframe[
                        f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                    > self._get_param_value(self.sell_stoch_rsi)
                )
                & (
                    dataframe[
                        f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                    < dataframe[
                        f"stoch_rsi_d_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
                    ]
                )
            )

        # ATR filter for volatility
        if self._get_param_value(self.sell_atr_enabled):
            conditions_short.append(
                dataframe[f"atr_{atr_period}"] > dataframe[f"atr_{atr_period}"].shift(14).mean()
            )

        # Apply conditions
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_long), "enter_long"] = 1

        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_short), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals
        """
        if not self._get_param_value(self.use_exit_signal):
            return dataframe

        conditions_exit_long = []
        conditions_exit_short = []

        # Get current parameter values
        rsi_period = self._get_param_value(self.rsi_period)

        # Ensure RSI column exists
        if f"rsi_{rsi_period}" not in dataframe.columns:
            dataframe[f"rsi_{rsi_period}"] = ta.RSI(dataframe, timeperiod=rsi_period)

        # Exit long conditions
        conditions_exit_long.append(
            (dataframe[f"rsi_{rsi_period}"] > self._get_param_value(self.exit_rsi_long))
            | (
                (dataframe["close"] - dataframe["open"]) / dataframe["open"]
                > self._get_param_value(self.exit_profit_threshold)
            )
        )

        # Exit short conditions
        conditions_exit_short.append(
            (dataframe[f"rsi_{rsi_period}"] < self._get_param_value(self.exit_rsi_short))
            | (
                (dataframe["open"] - dataframe["close"]) / dataframe["open"]
                > self._get_param_value(self.exit_profit_threshold)
            )
        )

        # Apply exit conditions
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_exit_long), "exit_long"] = 1

        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_exit_short), "exit_short"] = 1

        return dataframe

    def _ensure_indicator_columns(
        self,
        dataframe: DataFrame,
        rsi_period: int,
        rsi_period_slow: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        ema_short: int,
        ema_medium: int,
        ema_long: int,
        bb_period: int,
        bb_std: float,
        atr_period: int,
        volume_ma_period: int,
        stoch_rsi_period: int,
        stoch_rsi_smooth_k: int,
        stoch_rsi_smooth_d: int,
    ) -> None:
        """
        Ensure required indicator columns exist in the dataframe
        """
        # Dictionary to collect missing indicators
        missing_indicators = {}

        # Check and calculate missing RSI
        if f"rsi_{rsi_period}" not in dataframe.columns:
            missing_indicators[f"rsi_{rsi_period}"] = ta.RSI(dataframe, timeperiod=rsi_period)

        if f"rsi_slow_{rsi_period_slow}" not in dataframe.columns:
            missing_indicators[f"rsi_slow_{rsi_period_slow}"] = ta.RSI(
                dataframe, timeperiod=rsi_period_slow
            )

        # Check and calculate missing MACD
        if f"macd_{macd_fast}_{macd_slow}_{macd_signal}" not in dataframe.columns:
            macd = ta.MACD(
                dataframe, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
            )
            missing_indicators[f"macd_{macd_fast}_{macd_slow}_{macd_signal}"] = macd["macd"]
            missing_indicators[f"macdsignal_{macd_fast}_{macd_slow}_{macd_signal}"] = macd[
                "macdsignal"
            ]
            missing_indicators[f"macdhist_{macd_fast}_{macd_slow}_{macd_signal}"] = macd["macdhist"]

        # Check and calculate missing EMAs
        if f"ema_short_{ema_short}" not in dataframe.columns:
            missing_indicators[f"ema_short_{ema_short}"] = ta.EMA(dataframe, timeperiod=ema_short)

        if f"ema_medium_{ema_medium}" not in dataframe.columns:
            missing_indicators[f"ema_medium_{ema_medium}"] = ta.EMA(
                dataframe, timeperiod=ema_medium
            )

        if f"ema_long_{ema_long}" not in dataframe.columns:
            missing_indicators[f"ema_long_{ema_long}"] = ta.EMA(dataframe, timeperiod=ema_long)

        # Check and calculate missing Bollinger Bands
        if f"bb_upper_{bb_period}_{bb_std}" not in dataframe.columns:
            bb = ta.BBANDS(dataframe, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
            missing_indicators[f"bb_upper_{bb_period}_{bb_std}"] = bb["upperband"]
            missing_indicators[f"bb_middle_{bb_period}_{bb_std}"] = bb["middleband"]
            missing_indicators[f"bb_lower_{bb_period}_{bb_std}"] = bb["lowerband"]
            missing_indicators[f"bb_width_{bb_period}_{bb_std}"] = (
                bb["upperband"] - bb["lowerband"]
            ) / bb["middleband"]
            missing_indicators[f"bb_percent_{bb_period}_{bb_std}"] = (
                dataframe["close"] - bb["lowerband"]
            ) / (bb["upperband"] - bb["lowerband"])

        # Check and calculate missing ATR
        if f"atr_{atr_period}" not in dataframe.columns:
            missing_indicators[f"atr_{atr_period}"] = ta.ATR(dataframe, timeperiod=atr_period)

        # Check and calculate missing volume indicators
        if f"volume_ma_{volume_ma_period}" not in dataframe.columns:
            missing_indicators[f"volume_ma_{volume_ma_period}"] = (
                dataframe["volume"].rolling(window=volume_ma_period).mean()
            )
            missing_indicators[f"volume_ratio_{volume_ma_period}"] = (
                dataframe["volume"] / dataframe["volume"].rolling(window=volume_ma_period).mean()
            )

        # Check and calculate missing Stochastic RSI
        if (
            f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
            not in dataframe.columns
        ):
            stoch = ta.STOCHRSI(
                dataframe,
                timeperiod=stoch_rsi_period,
                fastk_period=stoch_rsi_smooth_k,
                fastd_period=stoch_rsi_smooth_d,
            )
            missing_indicators[
                f"stoch_rsi_k_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
            ] = stoch["fastk"]
            missing_indicators[
                f"stoch_rsi_d_{stoch_rsi_period}_{stoch_rsi_smooth_k}_{stoch_rsi_smooth_d}"
            ] = stoch["fastd"]

        # Add all missing indicators at once using concatenation
        if missing_indicators:
            missing_df = pd.DataFrame(missing_indicators, index=dataframe.index)
            for col in missing_df.columns:
                if col not in dataframe.columns:
                    dataframe[col] = missing_df[col]

    @property
    def protections(self):
        """
        Define protections for the strategy based on hyperopt parameters.
        These protections help manage risk and prevent overtrading.
        """
        prot = []

        # StoplossGuard - Stop trading after consecutive stoplosses
        if self._get_param_value(self.protection_stoploss_enabled):
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_stoploss_lookback_period
                    ),
                    "trade_limit": self._get_param_value(self.protection_stoploss_trade_limit),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_stoploss_stop_duration
                    ),
                    "only_per_pair": False,  # Apply globally
                }
            )

        # CooldownPeriod - Add cooldown after each trade
        if self._get_param_value(self.protection_cooldown_enabled):
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self._get_param_value(self.protection_cooldown_period),
                }
            )

        # LowProfitPairs - Stop trading low-profit pairs
        if self._get_param_value(self.protection_lowprofit_enabled):
            prot.append(
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_lowprofit_lookback_period
                    ),
                    "trade_limit": self._get_param_value(self.protection_lowprofit_trade_limit),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_lowprofit_stop_duration
                    ),
                    "required_profit": self._get_param_value(
                        self.protection_lowprofit_required_profit
                    ),
                    "only_per_pair": True,  # Apply per pair
                }
            )

        # MaxDrawdown - Stop trading if drawdown is too high
        if self._get_param_value(self.protection_maxdrawdown_enabled):
            prot.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self._get_param_value(
                        self.protection_maxdrawdown_lookback_period
                    ),
                    "trade_limit": self._get_param_value(self.protection_maxdrawdown_trade_limit),
                    "stop_duration_candles": self._get_param_value(
                        self.protection_maxdrawdown_stop_duration
                    ),
                    "max_allowed_drawdown": self._get_param_value(
                        self.protection_maxdrawdown_allowed_drawdown
                    ),
                }
            )

        return prot

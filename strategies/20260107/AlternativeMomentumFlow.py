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


class AlternativeMomentumFlow(IStrategy):
    """
    Alternative Momentum Flow Strategy

    This strategy uses alternative technical indicators to identify momentum-based
    trading opportunities. It replaces common indicators with alternatives:
    - MFI instead of RSI (Money Flow Index for volume-weighted momentum)
    - PPO instead of MACD (Percentage Price Oscillator)
    - SMA instead of EMA (Simple Moving Average)
    - Keltner Channels instead of Bollinger Bands
    - ADR instead of ATR (Average Daily Range)
    - Williams %R instead of Stochastic RSI
    - OBV and CMF for volume analysis

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

    # MFI (Money Flow Index) parameters - replaces RSI
    mfi_period = IntParameter(10, 30, default=14, space="buy")
    mfi_period_slow = IntParameter(20, 50, default=28, space="buy")

    # PPO (Percentage Price Oscillator) parameters - replaces MACD
    ppo_fast = IntParameter(8, 20, default=12, space="buy")
    ppo_slow = IntParameter(20, 50, default=26, space="buy")
    ppo_signal = IntParameter(5, 20, default=9, space="buy")

    # SMA (Simple Moving Average) parameters - replaces EMA
    sma_short = IntParameter(5, 25, default=10, space="buy")
    sma_medium = IntParameter(20, 50, default=25, space="buy")
    sma_long = IntParameter(50, 200, default=60, space="buy")

    # Keltner Channel parameters - replaces Bollinger Bands
    keltner_period = IntParameter(10, 40, default=20, space="buy")
    keltner_atrmultiplier = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    # ADR (Average Daily Range) parameters - replaces ATR
    adr_period = IntParameter(7, 21, default=14, space="buy")
    adr_multiplier = DecimalParameter(0.5, 3.0, default=1.5, decimals=1, space="buy")

    # Williams %R parameters - replaces Stochastic RSI
    willr_period = IntParameter(10, 30, default=14, space="buy")

    # OBV (On-Balance Volume) and CMF (Chaikin Money Flow) parameters
    cmf_period = IntParameter(10, 30, default=20, space="buy")
    obv_ema_period = IntParameter(10, 50, default=21, space="buy")

    # CCI (Commodity Channel Index) parameters
    cci_period = IntParameter(10, 30, default=20, space="buy")

    # DMI (Directional Movement Index) parameters
    dmi_period = IntParameter(10, 30, default=14, space="buy")

    # Volume threshold
    volume_threshold = DecimalParameter(0.5, 2.5, default=1.2, decimals=1, space="buy")

    # -------------------------------------------------------------------------
    # Hyperoptimizable parameters for entry signals
    # -------------------------------------------------------------------------

    # Long entry thresholds
    buy_mfi_lower = IntParameter(10, 40, default=20, space="buy")
    buy_mfi_upper = IntParameter(30, 60, default=45, space="buy")
    buy_willr_threshold = IntParameter(-90, -60, default=-80, space="buy")
    buy_keltner_width_min = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="buy")
    buy_ppo_hist_threshold = DecimalParameter(-0.5, 0.5, default=0.0, decimals=2, space="buy")
    buy_cci_threshold = IntParameter(-200, -50, default=-100, space="buy")
    buy_cmf_threshold = DecimalParameter(-0.3, 0.1, default=-0.05, decimals=2, space="buy")
    buy_dmi_threshold = IntParameter(15, 35, default=25, space="buy")

    # Short entry thresholds
    sell_mfi_lower = IntParameter(40, 70, default=55, space="sell")
    sell_mfi_upper = IntParameter(60, 90, default=80, space="sell")
    sell_willr_threshold = IntParameter(-40, -10, default=-20, space="sell")
    sell_keltner_width_min = DecimalParameter(0.001, 0.05, default=0.01, decimals=3, space="sell")
    sell_ppo_hist_threshold = DecimalParameter(-0.5, 0.5, default=0.0, decimals=2, space="sell")
    sell_cci_threshold = IntParameter(50, 200, default=100, space="sell")
    sell_cmf_threshold = DecimalParameter(-0.1, 0.3, default=0.05, decimals=2, space="sell")
    sell_dmi_threshold = IntParameter(15, 35, default=25, space="sell")

    # Exit thresholds
    exit_mfi_long = IntParameter(60, 90, default=75, space="sell")
    exit_mfi_short = IntParameter(10, 40, default=25, space="sell")
    exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell")

    # -------------------------------------------------------------------------
    # Enable/disable conditions
    # -------------------------------------------------------------------------

    # Long conditions
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ppo_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_keltner_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_volume_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_willr_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_cci_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_cmf_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_dmi_enabled = CategoricalParameter([True, False], default=True, space="buy")

    # Short conditions
    sell_mfi_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_ppo_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_keltner_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_sma_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_volume_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_willr_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_cci_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_cmf_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_dmi_enabled = CategoricalParameter([True, False], default=True, space="sell")

    # -------------------------------------------------------------------------
    # Protection parameters (optimizable)
    # -------------------------------------------------------------------------

    # StoplossGuard - Stops trading after N stoplosses within a time window
    protection_stoploss_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_stoploss_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_stoploss_lookback_period = IntParameter(10, 1440, default=60, space="protection")
    protection_stoploss_stop_duration = IntParameter(10, 360, default=60, space="protection")

    # CooldownPeriod - Cooldown after each trade
    protection_cooldown_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_cooldown_period = IntParameter(1, 20, default=5, space="protection")

    # LowProfitPairs - Stops trading pairs with low profits
    protection_lowprofit_enabled = CategoricalParameter(
        [True, False], default=True, space="protection"
    )
    protection_lowprofit_trade_limit = IntParameter(2, 10, default=4, space="protection")
    protection_lowprofit_lookback_period = IntParameter(10, 1440, default=360, space="protection")
    protection_lowprofit_stop_duration = IntParameter(10, 360, default=60, space="protection")
    protection_lowprofit_required_profit = DecimalParameter(
        -0.05, 0.05, default=0.0, decimals=3, space="protection"
    )

    # MaxDrawdown - Stop trading if max drawdown is reached
    protection_maxdrawdown_enabled = CategoricalParameter(
        [True, False], default=False, space="protection"
    )
    protection_maxdrawdown_trade_limit = IntParameter(3, 20, default=8, space="protection")
    protection_maxdrawdown_lookback_period = IntParameter(10, 1440, default=200, space="protection")
    protection_maxdrawdown_stop_duration = IntParameter(10, 360, default=60, space="protection")
    protection_maxdrawdown_allowed_drawdown = DecimalParameter(
        0.01, 0.30, default=0.10, decimals=2, space="protection"
    )

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
        Populate alternative indicators using batch calculation
        """
        # Store new indicators in a dictionary first
        indicators = {}

        # Get current parameter values
        mfi_period = self._get_param_value(self.mfi_period)
        mfi_period_slow = self._get_param_value(self.mfi_period_slow)
        ppo_fast = self._get_param_value(self.ppo_fast)
        ppo_slow = self._get_param_value(self.ppo_slow)
        ppo_signal = self._get_param_value(self.ppo_signal)
        sma_short = self._get_param_value(self.sma_short)
        sma_medium = self._get_param_value(self.sma_medium)
        sma_long = self._get_param_value(self.sma_long)
        keltner_period = self._get_param_value(self.keltner_period)
        keltner_atrmultiplier = self._get_param_value(self.keltner_atrmultiplier)
        adr_period = self._get_param_value(self.adr_period)
        willr_period = self._get_param_value(self.willr_period)
        cmf_period = self._get_param_value(self.cmf_period)
        obv_ema_period = self._get_param_value(self.obv_ema_period)
        cci_period = self._get_param_value(self.cci_period)
        dmi_period = self._get_param_value(self.dmi_period)

        # Calculate MFI (Money Flow Index) - volume-weighted RSI
        indicators[f"mfi_{mfi_period}"] = ta.MFI(dataframe, timeperiod=mfi_period)
        indicators[f"mfi_slow_{mfi_period_slow}"] = ta.MFI(dataframe, timeperiod=mfi_period_slow)

        # Calculate PPO (Percentage Price Oscillator) - percentage-based MACD
        ppo = ta.PPO(dataframe, fastperiod=ppo_fast, slowperiod=ppo_slow, matype=0)
        indicators[f"ppo_{ppo_fast}_{ppo_slow}"] = ppo
        ppo_signal_line = ta.EMA(ppo, timeperiod=ppo_signal)
        indicators[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"] = ppo_signal_line
        indicators[f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal}"] = ppo - ppo_signal_line

        # Calculate SMA (Simple Moving Average)
        indicators[f"sma_short_{sma_short}"] = ta.SMA(dataframe, timeperiod=sma_short)
        indicators[f"sma_medium_{sma_medium}"] = ta.SMA(dataframe, timeperiod=sma_medium)
        indicators[f"sma_long_{sma_long}"] = ta.SMA(dataframe, timeperiod=sma_long)

        # Calculate Keltner Channels
        keltner_ema = ta.EMA(dataframe, timeperiod=keltner_period)
        keltner_atr = ta.ATR(dataframe, timeperiod=keltner_period)
        indicators[f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}"] = keltner_ema + (
            keltner_atr * keltner_atrmultiplier
        )
        indicators[f"keltner_middle_{keltner_period}_{keltner_atrmultiplier}"] = keltner_ema
        indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"] = keltner_ema - (
            keltner_atr * keltner_atrmultiplier
        )
        indicators[f"keltner_width_{keltner_period}_{keltner_atrmultiplier}"] = (
            2 * keltner_atr * keltner_atrmultiplier
        ) / keltner_ema
        indicators[f"keltner_percent_{keltner_period}_{keltner_atrmultiplier}"] = (
            dataframe["close"]
            - indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"]
        ) / (
            indicators[f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}"]
            - indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"]
        )

        # Calculate ADR (Average Daily Range)
        daily_range = dataframe["high"] - dataframe["low"]
        indicators[f"adr_{adr_period}"] = daily_range.rolling(window=adr_period).mean()

        # Calculate Williams %R
        indicators[f"willr_{willr_period}"] = ta.WILLR(dataframe, timeperiod=willr_period)

        # Calculate OBV (On-Balance Volume)
        obv = ta.OBV(dataframe)
        indicators["obv"] = obv
        indicators[f"obv_ema_{obv_ema_period}"] = ta.EMA(obv, timeperiod=obv_ema_period)
        indicators[f"obv_signal_{obv_ema_period}"] = obv - indicators[f"obv_ema_{obv_ema_period}"]

        # Calculate CMF (Chaikin Money Flow)
        clv = (
            (dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"])
        clv = clv.fillna(0)  # Handle division by zero
        money_flow_volume = clv * dataframe["volume"]
        indicators[f"cmf_{cmf_period}"] = (
            money_flow_volume.rolling(window=cmf_period).sum()
            / dataframe["volume"].rolling(window=cmf_period).sum()
        )

        # Calculate CCI (Commodity Channel Index)
        indicators[f"cci_{cci_period}"] = ta.CCI(dataframe, timeperiod=cci_period)

        # Calculate DMI (Directional Movement Index)
        dmi = ta.DX(dataframe, timeperiod=dmi_period)
        indicators[f"dx_{dmi_period}"] = dmi
        plus_di = ta.PLUS_DI(dataframe, timeperiod=dmi_period)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=dmi_period)
        indicators[f"plus_di_{dmi_period}"] = plus_di
        indicators[f"minus_di_{dmi_period}"] = minus_di
        indicators[f"adx_{dmi_period}"] = ta.ADX(dataframe, timeperiod=dmi_period)

        # Volume analysis
        volume_ma = dataframe["volume"].rolling(window=20).mean()
        indicators["volume_ma_20"] = volume_ma
        indicators["volume_ratio_20"] = dataframe["volume"] / volume_ma

        # Additional derived indicators
        indicators["high_low_ratio"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        indicators["close_open_ratio"] = (dataframe["close"] - dataframe["open"]) / dataframe[
            "open"
        ]

        # Price action indicators
        indicators["price_change"] = dataframe["close"].pct_change()
        indicators["volume_change"] = dataframe["volume"].pct_change()

        # Momentum indicators
        indicators["roc_10"] = ta.ROC(dataframe, timeperiod=10)
        indicators["roc_20"] = ta.ROC(dataframe, timeperiod=20)

        # Convert indicators dict to DataFrame and concatenate at once
        indicators_df = pd.DataFrame(indicators, index=dataframe.index)
        dataframe = pd.concat([dataframe, indicators_df], axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals using alternative indicators
        """
        conditions_long = []
        conditions_short = []

        # Get current parameter values
        mfi_period = self._get_param_value(self.mfi_period)
        mfi_period_slow = self._get_param_value(self.mfi_period_slow)
        ppo_fast = self._get_param_value(self.ppo_fast)
        ppo_slow = max(self._get_param_value(self.ppo_slow), ppo_fast + 1)
        ppo_signal = self._get_param_value(self.ppo_signal)
        sma_short = self._get_param_value(self.sma_short)
        sma_medium = self._get_param_value(self.sma_medium)
        sma_long = self._get_param_value(self.sma_long)
        keltner_period = self._get_param_value(self.keltner_period)
        keltner_atrmultiplier = self._get_param_value(self.keltner_atrmultiplier)
        adr_period = self._get_param_value(self.adr_period)
        willr_period = self._get_param_value(self.willr_period)
        cmf_period = self._get_param_value(self.cmf_period)
        obv_ema_period = self._get_param_value(self.obv_ema_period)
        cci_period = self._get_param_value(self.cci_period)
        dmi_period = self._get_param_value(self.dmi_period)

        # Ensure required columns exist
        self._ensure_indicator_columns(
            dataframe,
            mfi_period,
            mfi_period_slow,
            ppo_fast,
            ppo_slow,
            ppo_signal,
            sma_short,
            sma_medium,
            sma_long,
            keltner_period,
            keltner_atrmultiplier,
            adr_period,
            willr_period,
            cmf_period,
            obv_ema_period,
            cci_period,
            dmi_period,
        )

        # ===== LONG ENTRY CONDITIONS =====

        # MFI conditions
        if self._get_param_value(self.buy_mfi_enabled):
            conditions_long.append(
                (dataframe[f"mfi_{mfi_period}"] > self._get_param_value(self.buy_mfi_lower))
                & (dataframe[f"mfi_{mfi_period}"] < self._get_param_value(self.buy_mfi_upper))
                & (dataframe[f"mfi_{mfi_period}"] > dataframe[f"mfi_{mfi_period}"].shift(1))
            )

        # PPO conditions
        if self._get_param_value(self.buy_ppo_enabled):
            conditions_long.append(
                (
                    dataframe[f"ppo_{ppo_fast}_{ppo_slow}"]
                    > dataframe[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"]
                )
                & (
                    dataframe[f"ppo_{ppo_fast}_{ppo_slow}"].shift(1)
                    <= dataframe[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"].shift(1)
                )
                & (
                    dataframe[f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal}"]
                    > self._get_param_value(self.buy_ppo_hist_threshold)
                )
            )

        # Keltner Channel conditions
        if self._get_param_value(self.buy_keltner_enabled):
            conditions_long.append(
                (
                    dataframe["close"]
                    < dataframe[f"keltner_middle_{keltner_period}_{keltner_atrmultiplier}"]
                )
                & (
                    dataframe["close"]
                    > dataframe[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"]
                )
                & (
                    dataframe[f"keltner_width_{keltner_period}_{keltner_atrmultiplier}"]
                    > self._get_param_value(self.buy_keltner_width_min)
                )
            )

        # SMA conditions
        if self._get_param_value(self.buy_sma_enabled):
            conditions_long.append(
                (dataframe["close"] > dataframe[f"sma_short_{sma_short}"])
                & (dataframe[f"sma_short_{sma_short}"] > dataframe[f"sma_medium_{sma_medium}"])
            )

        # Volume conditions
        if self._get_param_value(self.buy_volume_enabled):
            conditions_long.append(
                (dataframe["volume_ratio_20"] > self._get_param_value(self.volume_threshold))
                & (dataframe["volume"] > 0)
            )

        # Williams %R conditions
        if self._get_param_value(self.buy_willr_enabled):
            conditions_long.append(
                (
                    dataframe[f"willr_{willr_period}"]
                    < self._get_param_value(self.buy_willr_threshold)
                )
                & (dataframe[f"willr_{willr_period}"] > dataframe[f"willr_{willr_period}"].shift(1))
            )

        # CCI conditions
        if self._get_param_value(self.buy_cci_enabled):
            conditions_long.append(
                (dataframe[f"cci_{cci_period}"] < self._get_param_value(self.buy_cci_threshold))
                & (dataframe[f"cci_{cci_period}"] > dataframe[f"cci_{cci_period}"].shift(1))
            )

        # CMF conditions
        if self._get_param_value(self.buy_cmf_enabled):
            conditions_long.append(
                dataframe[f"cmf_{cmf_period}"] > self._get_param_value(self.buy_cmf_threshold)
            )

        # DMI conditions
        if self._get_param_value(self.buy_dmi_enabled):
            conditions_long.append(
                (dataframe[f"plus_di_{dmi_period}"] > dataframe[f"minus_di_{dmi_period}"])
                & (dataframe[f"adx_{dmi_period}"] > self._get_param_value(self.buy_dmi_threshold))
            )

        # ===== SHORT ENTRY CONDITIONS =====

        # MFI conditions
        if self._get_param_value(self.sell_mfi_enabled):
            conditions_short.append(
                (dataframe[f"mfi_{mfi_period}"] > self._get_param_value(self.sell_mfi_lower))
                & (dataframe[f"mfi_{mfi_period}"] < self._get_param_value(self.sell_mfi_upper))
                & (dataframe[f"mfi_{mfi_period}"] < dataframe[f"mfi_{mfi_period}"].shift(1))
            )

        # PPO conditions
        if self._get_param_value(self.sell_ppo_enabled):
            conditions_short.append(
                (
                    dataframe[f"ppo_{ppo_fast}_{ppo_slow}"]
                    < dataframe[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"]
                )
                & (
                    dataframe[f"ppo_{ppo_fast}_{ppo_slow}"].shift(1)
                    >= dataframe[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"].shift(1)
                )
                & (
                    dataframe[f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal}"]
                    < self._get_param_value(self.sell_ppo_hist_threshold)
                )
            )

        # Keltner Channel conditions
        if self._get_param_value(self.sell_keltner_enabled):
            conditions_short.append(
                (
                    dataframe["close"]
                    > dataframe[f"keltner_middle_{keltner_period}_{keltner_atrmultiplier}"]
                )
                & (
                    dataframe["close"]
                    < dataframe[f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}"]
                )
                & (
                    dataframe[f"keltner_width_{keltner_period}_{keltner_atrmultiplier}"]
                    > self._get_param_value(self.sell_keltner_width_min)
                )
            )

        # SMA conditions
        if self._get_param_value(self.sell_sma_enabled):
            conditions_short.append(
                (dataframe["close"] < dataframe[f"sma_short_{sma_short}"])
                & (dataframe[f"sma_short_{sma_short}"] < dataframe[f"sma_medium_{sma_medium}"])
            )

        # Volume conditions
        if self._get_param_value(self.sell_volume_enabled):
            conditions_short.append(
                (dataframe["volume_ratio_20"] > self._get_param_value(self.volume_threshold))
                & (dataframe["volume"] > 0)
            )

        # Williams %R conditions
        if self._get_param_value(self.sell_willr_enabled):
            conditions_short.append(
                (
                    dataframe[f"willr_{willr_period}"]
                    > self._get_param_value(self.sell_willr_threshold)
                )
                & (dataframe[f"willr_{willr_period}"] < dataframe[f"willr_{willr_period}"].shift(1))
            )

        # CCI conditions
        if self._get_param_value(self.sell_cci_enabled):
            conditions_short.append(
                (dataframe[f"cci_{cci_period}"] > self._get_param_value(self.sell_cci_threshold))
                & (dataframe[f"cci_{cci_period}"] < dataframe[f"cci_{cci_period}"].shift(1))
            )

        # CMF conditions
        if self._get_param_value(self.sell_cmf_enabled):
            conditions_short.append(
                dataframe[f"cmf_{cmf_period}"] < self._get_param_value(self.sell_cmf_threshold)
            )

        # DMI conditions
        if self._get_param_value(self.sell_dmi_enabled):
            conditions_short.append(
                (dataframe[f"minus_di_{dmi_period}"] > dataframe[f"plus_di_{dmi_period}"])
                & (dataframe[f"adx_{dmi_period}"] > self._get_param_value(self.sell_dmi_threshold))
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
        mfi_period = self._get_param_value(self.mfi_period)

        # Ensure MFI column exists
        if f"mfi_{mfi_period}" not in dataframe.columns:
            dataframe[f"mfi_{mfi_period}"] = ta.MFI(dataframe, timeperiod=mfi_period)

        # Exit long conditions
        conditions_exit_long.append(
            (dataframe[f"mfi_{mfi_period}"] > self._get_param_value(self.exit_mfi_long))
            | (
                (dataframe["close"] - dataframe["open"]) / dataframe["open"]
                > self._get_param_value(self.exit_profit_threshold)
            )
        )

        # Exit short conditions
        conditions_exit_short.append(
            (dataframe[f"mfi_{mfi_period}"] < self._get_param_value(self.exit_mfi_short))
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
        mfi_period: int,
        mfi_period_slow: int,
        ppo_fast: int,
        ppo_slow: int,
        ppo_signal: int,
        sma_short: int,
        sma_medium: int,
        sma_long: int,
        keltner_period: int,
        keltner_atrmultiplier: float,
        adr_period: int,
        willr_period: int,
        cmf_period: int,
        obv_ema_period: int,
        cci_period: int,
        dmi_period: int,
    ) -> None:
        """
        Ensure required indicator columns exist in the dataframe
        """
        # Dictionary to collect missing indicators
        missing_indicators = {}

        # Check and calculate missing MFI
        if f"mfi_{mfi_period}" not in dataframe.columns:
            missing_indicators[f"mfi_{mfi_period}"] = ta.MFI(dataframe, timeperiod=mfi_period)

        if f"mfi_slow_{mfi_period_slow}" not in dataframe.columns:
            missing_indicators[f"mfi_slow_{mfi_period_slow}"] = ta.MFI(
                dataframe, timeperiod=mfi_period_slow
            )

        # Check and calculate missing PPO - check all three columns
        if (
            f"ppo_{ppo_fast}_{ppo_slow}" not in dataframe.columns
            or f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}" not in dataframe.columns
            or f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal}" not in dataframe.columns
        ):
            # Calculate or retrieve PPO base
            if f"ppo_{ppo_fast}_{ppo_slow}" in dataframe.columns:
                ppo = dataframe[f"ppo_{ppo_fast}_{ppo_slow}"]
            else:
                ppo = ta.PPO(dataframe, fastperiod=ppo_fast, slowperiod=ppo_slow, matype=0)
                missing_indicators[f"ppo_{ppo_fast}_{ppo_slow}"] = ppo

            # Always calculate signal and histogram with current parameters
            ppo_signal_line = ta.EMA(ppo, timeperiod=ppo_signal)
            missing_indicators[f"ppo_signal_{ppo_fast}_{ppo_slow}_{ppo_signal}"] = ppo_signal_line
            missing_indicators[f"ppo_hist_{ppo_fast}_{ppo_slow}_{ppo_signal}"] = (
                ppo - ppo_signal_line
            )

        # Check and calculate missing SMAs
        if f"sma_short_{sma_short}" not in dataframe.columns:
            missing_indicators[f"sma_short_{sma_short}"] = ta.SMA(dataframe, timeperiod=sma_short)

        if f"sma_medium_{sma_medium}" not in dataframe.columns:
            missing_indicators[f"sma_medium_{sma_medium}"] = ta.SMA(
                dataframe, timeperiod=sma_medium
            )

        if f"sma_long_{sma_long}" not in dataframe.columns:
            missing_indicators[f"sma_long_{sma_long}"] = ta.SMA(dataframe, timeperiod=sma_long)

        # Check and calculate missing Keltner Channels
        if f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}" not in dataframe.columns:
            keltner_ema = ta.EMA(dataframe, timeperiod=keltner_period)
            keltner_atr = ta.ATR(dataframe, timeperiod=keltner_period)
            missing_indicators[f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}"] = (
                keltner_ema + (keltner_atr * keltner_atrmultiplier)
            )
            missing_indicators[f"keltner_middle_{keltner_period}_{keltner_atrmultiplier}"] = (
                keltner_ema
            )
            missing_indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"] = (
                keltner_ema - (keltner_atr * keltner_atrmultiplier)
            )
            missing_indicators[f"keltner_width_{keltner_period}_{keltner_atrmultiplier}"] = (
                2 * keltner_atr * keltner_atrmultiplier
            ) / keltner_ema
            missing_indicators[f"keltner_percent_{keltner_period}_{keltner_atrmultiplier}"] = (
                dataframe["close"]
                - missing_indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"]
            ) / (
                missing_indicators[f"keltner_upper_{keltner_period}_{keltner_atrmultiplier}"]
                - missing_indicators[f"keltner_lower_{keltner_period}_{keltner_atrmultiplier}"]
            )

        # Check and calculate missing ADR
        if f"adr_{adr_period}" not in dataframe.columns:
            daily_range = dataframe["high"] - dataframe["low"]
            missing_indicators[f"adr_{adr_period}"] = daily_range.rolling(window=adr_period).mean()

        # Check and calculate missing Williams %R
        if f"willr_{willr_period}" not in dataframe.columns:
            missing_indicators[f"willr_{willr_period}"] = ta.WILLR(
                dataframe, timeperiod=willr_period
            )

        # Check and calculate missing volume indicators
        if "volume_ratio_20" not in dataframe.columns:
            volume_ma = dataframe["volume"].rolling(window=20).mean()
            missing_indicators["volume_ma_20"] = volume_ma
            missing_indicators["volume_ratio_20"] = dataframe["volume"] / volume_ma

        # Check and calculate missing CMF
        if f"cmf_{cmf_period}" not in dataframe.columns:
            clv = (
                (dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])
            ) / (dataframe["high"] - dataframe["low"])
            clv = clv.fillna(0)
            money_flow_volume = clv * dataframe["volume"]
            missing_indicators[f"cmf_{cmf_period}"] = (
                money_flow_volume.rolling(window=cmf_period).sum()
                / dataframe["volume"].rolling(window=cmf_period).sum()
            )

        # Check and calculate missing CCI
        if f"cci_{cci_period}" not in dataframe.columns:
            missing_indicators[f"cci_{cci_period}"] = ta.CCI(dataframe, timeperiod=cci_period)

        # Check and calculate missing DMI
        if f"adx_{dmi_period}" not in dataframe.columns:
            missing_indicators[f"dx_{dmi_period}"] = ta.DX(dataframe, timeperiod=dmi_period)
            missing_indicators[f"plus_di_{dmi_period}"] = ta.PLUS_DI(
                dataframe, timeperiod=dmi_period
            )
            missing_indicators[f"minus_di_{dmi_period}"] = ta.MINUS_DI(
                dataframe, timeperiod=dmi_period
            )
            missing_indicators[f"adx_{dmi_period}"] = ta.ADX(dataframe, timeperiod=dmi_period)

        # Check and calculate missing OBV
        if "obv" not in dataframe.columns:
            obv = ta.OBV(dataframe)
            missing_indicators["obv"] = obv
            missing_indicators[f"obv_ema_{obv_ema_period}"] = ta.EMA(obv, timeperiod=obv_ema_period)
            missing_indicators[f"obv_signal_{obv_ema_period}"] = (
                obv - missing_indicators[f"obv_ema_{obv_ema_period}"]
            )

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

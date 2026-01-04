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
    merge_informative_pair,
)

# --------------------------------
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
import talib.abstract as ta
import pandas_ta as pta

# Suppress pandas_ta FutureWarnings about ChainedAssignment
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_ta")


class Strategy_0_4536(IStrategy):
    """
    Strategy 0.4536 - Converted from StrategyQuantX

    This strategy uses fuzzy logic for entry/exit signals combining MTATR,
    Vortex indicator, VWAP, and various session-based indicators across multiple timeframes.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "15m"

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.10}

    # Optimal stoploss
    stoploss = -0.089  # 8.9% from parameters

    # Trailing stoploss
    trailing_stop = False

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 500

    # Strategy parameters from pseudocode
    # Hyperparameters for optimization
    mtatr_period_1 = IntParameter(40, 60, default=50, space="entry")
    mtatr_period_2 = IntParameter(15, 25, default=20, space="entry")
    vwap_period_1 = IntParameter(10, 25, default=18, space="entry")
    indicator_crs_ma_period = IntParameter(60, 90, default=74, space="entry")
    bb_bar_opens_period = IntParameter(15, 30, default=20, space="entry")
    period_1 = IntParameter(15, 25, default=20, space="entry")
    vwap_period_2 = IntParameter(20, 35, default=29, space="entry")
    vortex_period = IntParameter(15, 25, default=20, space="exit")
    mtatr_period_3 = IntParameter(25, 45, default=34, space="exit")
    close_vwap_period = IntParameter(15, 30, default=22, space="exit")
    sma_period = IntParameter(15, 25, default=20, space="exit")

    # Entry/Exit parameters
    price_entry_mult = DecimalParameter(0.8, 1.6, default=1.2, space="entry")
    exit_after_bars = IntParameter(5, 10, default=7, space="exit")
    profit_target_pct = DecimalParameter(3.0, 7.0, default=5.0, space="exit")
    stop_loss_pct = DecimalParameter(7.0, 11.0, default=8.9, space="exit")
    period_2 = IntParameter(40, 60, default=50, space="entry")
    atr_period = IntParameter(20, 40, default=30, space="entry")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # Calculate main timeframe indicators

        # MTATR (approximation using standard ATR)
        dataframe["mtatr_1"] = ta.ATR(dataframe, timeperiod=self.mtatr_period_1.value)
        dataframe["mtatr_2"] = ta.ATR(dataframe, timeperiod=self.mtatr_period_2.value)
        dataframe["mtatr_3"] = ta.ATR(dataframe, timeperiod=self.mtatr_period_3.value)

        # VWAP - using pandas_ta
        # Temporarily set datetime index for VWAP calculation
        original_index = dataframe.index
        dataframe_with_date_index = dataframe.set_index("date")
        vwap_result = pta.vwap(
            dataframe_with_date_index["high"],
            dataframe_with_date_index["low"],
            dataframe_with_date_index["close"],
            dataframe_with_date_index["volume"],
            anchor=None,  # Don't use anchor to avoid period grouping issues
        )
        # Reset to original index and assign VWAP values
        dataframe["vwap"] = vwap_result.values if vwap_result is not None else dataframe["close"]

        # HMA for indicator crossover
        dataframe["hma"] = pta.hma(dataframe["close"], length=self.indicator_crs_ma_period.value)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe["close"],
            timeperiod=self.bb_bar_opens_period.value,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_lower"] = bb_lower

        # Vortex Indicator
        vortex = pta.vortex(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.vortex_period.value,
        )
        dataframe["vortex_plus"] = vortex[f"VTXP_{self.vortex_period.value}"]
        dataframe["vortex_minus"] = vortex[f"VTXM_{self.vortex_period.value}"]

        # Monthly High/Low approximation
        monthly_period = 30 * 24 * 4  # 30 days in 15-min candles
        dataframe["high_monthly"] = dataframe["high"].rolling(window=monthly_period).max()
        dataframe["low_monthly"] = dataframe["low"].rolling(window=monthly_period).min()

        # Daily High/Low
        daily_period = 24 * 4  # 24 hours in 15-min candles
        dataframe["high_daily"] = dataframe["high"].rolling(window=daily_period).max()
        dataframe["low_daily"] = dataframe["low"].rolling(window=daily_period).min()

        # Highest and Lowest
        dataframe["highest_open"] = dataframe["open"].rolling(window=self.period_2.value).max()
        dataframe["lowest_open"] = dataframe["open"].rolling(window=self.period_2.value).min()

        # ATR for entry calculations
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # SMA
        dataframe["sma"] = ta.SMA(dataframe["close"], timeperiod=self.sma_period.value)

        # Session-based indicators
        dataframe["hour"] = dataframe["date"].dt.hour
        dataframe["minute"] = dataframe["date"].dt.minute

        # Highest/Lowest in range (8:30 to 6:00) approximation
        def is_in_range_8_30_to_6(row):
            hour = row["hour"]
            if hour >= 8.5 or hour <= 6:
                return True
            return False

        dataframe["in_range"] = dataframe.apply(is_in_range_8_30_to_6, axis=1)
        dataframe["highest_in_range"] = dataframe.apply(
            lambda row: row["high"] if row["in_range"] else np.nan, axis=1
        ).ffill()
        dataframe["lowest_in_range"] = dataframe.apply(
            lambda row: row["low"] if row["in_range"] else np.nan, axis=1
        ).ffill()

        # Highest/Lowest in range (19:00 to 14:00) approximation
        def is_in_range_19_to_14(row):
            hour = row["hour"]
            if hour >= 19 or hour <= 14:
                return True
            return False

        dataframe["in_range_19_14"] = dataframe.apply(is_in_range_19_to_14, axis=1)
        dataframe["highest_in_range_19_14"] = dataframe.apply(
            lambda row: row["high"] if row["in_range_19_14"] else np.nan, axis=1
        ).ffill()
        dataframe["lowest_in_range_19_14"] = dataframe.apply(
            lambda row: row["low"] if row["in_range_19_14"] else np.nan, axis=1
        ).ffill()

        # Add 1h informative indicators
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        # Calculate 1h indicators
        # Daily High/Low for 1h
        informative_1h["high_daily_1h"] = informative_1h["high"].rolling(window=24).max()
        informative_1h["low_daily_1h"] = informative_1h["low"].rolling(window=24).min()

        # Highest and Lowest PRICE_MEDIAN
        informative_1h["median_price"] = (informative_1h["high"] + informative_1h["low"]) / 2
        informative_1h["highest_median_1h"] = (
            informative_1h["median_price"].rolling(window=self.period_1.value).max()
        )
        informative_1h["lowest_median_1h"] = (
            informative_1h["median_price"].rolling(window=self.period_1.value).min()
        )

        # SMA for 1h
        informative_1h["sma_1h"] = ta.SMA(informative_1h["close"], timeperiod=self.sma_period.value)

        # Monthly High/Low for 1h
        informative_1h["high_monthly_1h"] = informative_1h["high"].rolling(window=30 * 24).max()
        informative_1h["low_monthly_1h"] = informative_1h["low"].rolling(window=30 * 24).min()

        # Merge informative pair
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        return dataframe

    def fuzzy_logic_long_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long entry - at least 42% of conditions (2 out of 5) must be true
        """
        conditions = []

        # Condition 1: MTATR_1[3] <= MTATR_2[5]
        if index > 5:
            conditions.append(
                dataframe.iloc[index - 3]["mtatr_1"] <= dataframe.iloc[index - 5]["mtatr_2"]
            )
        else:
            conditions.append(False)

        # Condition 2: VWAP[2] <> HighestInRange[1]
        if index > 2:
            conditions.append(
                abs(
                    dataframe.iloc[index - 2]["vwap"]
                    - dataframe.iloc[index - 1]["highest_in_range"]
                )
                > 0.01
            )
        else:
            conditions.append(False)

        # Condition 3: HighMonthly[3] crosses below its HMA
        if index > 3:
            conditions.append(
                (dataframe.iloc[index - 4]["high_monthly"] > dataframe.iloc[index - 4]["hma"])
                and (dataframe.iloc[index - 3]["high_monthly"] < dataframe.iloc[index - 3]["hma"])
            )
        else:
            conditions.append(False)

        # Condition 4: Open[3] above BB Upper[4]
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 3]["open"] > dataframe.iloc[index - 4]["bb_upper"]
            )
        else:
            conditions.append(False)

        # Condition 5: HighDaily(1h)[3] >= Lowest_median(1h)[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["high_daily_1h_1h"]
                >= dataframe.iloc[index - 3]["lowest_median_1h_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 42% of conditions are met (2 out of 5)
        return sum(conditions) >= 2

    def fuzzy_logic_short_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short entry - at least 42% of conditions (2 out of 5) must be true
        """
        conditions = []

        # Condition 1: MTATR_1[3] >= MTATR_2[5]
        if index > 5:
            conditions.append(
                dataframe.iloc[index - 3]["mtatr_1"] >= dataframe.iloc[index - 5]["mtatr_2"]
            )
        else:
            conditions.append(False)

        # Condition 2: VWAP[2] = LowestInRange[1]
        if index > 2:
            conditions.append(
                abs(
                    dataframe.iloc[index - 2]["vwap"] - dataframe.iloc[index - 1]["lowest_in_range"]
                )
                < 0.01
            )
        else:
            conditions.append(False)

        # Condition 3: LowMonthly[3] crosses above its HMA
        if index > 3:
            conditions.append(
                (dataframe.iloc[index - 4]["low_monthly"] < dataframe.iloc[index - 4]["hma"])
                and (dataframe.iloc[index - 3]["low_monthly"] > dataframe.iloc[index - 3]["hma"])
            )
        else:
            conditions.append(False)

        # Condition 4: Open[3] below BB Lower[4]
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 3]["open"] < dataframe.iloc[index - 4]["bb_lower"]
            )
        else:
            conditions.append(False)

        # Condition 5: LowDaily(1h)[3] <= Highest_median(1h)[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["low_daily_1h_1h"]
                <= dataframe.iloc[index - 3]["highest_median_1h_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 42% of conditions are met (2 out of 5)
        return sum(conditions) >= 2

    def fuzzy_logic_long_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long exit - at least 69% of conditions (4 out of 6) must be true
        """
        conditions = []

        # Condition 1: Close[1] is lower than LowestInRange_19_14[2] for 5 bars at 5 bar ago
        if index > 10:
            condition_met = True
            for i in range(5):
                bar_index = index - 5 - i
                if (
                    dataframe.iloc[bar_index]["close"]
                    >= dataframe.iloc[bar_index - 2]["lowest_in_range_19_14"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 2: VWAP[3] is rising
        if index > 3:
            conditions.append(dataframe.iloc[index - 3]["vwap"] > dataframe.iloc[index - 4]["vwap"])
        else:
            conditions.append(False)

        # Condition 3: Vortex.Minus[4] is higher than 978666 for 6 bars at 2 bar ago
        # Using a reasonable threshold instead of the arbitrary number
        if index > 8:
            condition_met = True
            threshold = 0.9  # Reasonable threshold for vortex
            for i in range(6):
                bar_index = index - 2 - i
                if dataframe.iloc[bar_index]["vortex_minus"] <= threshold:
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 4: MTATR_2[3] < MTATR_3[4]
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 3]["mtatr_2"] < dataframe.iloc[index - 4]["mtatr_3"]
            )
        else:
            conditions.append(False)

        # Condition 5: Close is above VWAP[3]
        if index > 3:
            conditions.append(dataframe.iloc[index]["close"] > dataframe.iloc[index - 3]["vwap"])
        else:
            conditions.append(False)

        # Condition 6: SMA(1h)[2] crosses above HighMonthly(1h)[2]
        if index > 2:
            conditions.append(
                (
                    dataframe.iloc[index - 3]["sma_1h_1h"]
                    < dataframe.iloc[index - 3]["high_monthly_1h_1h"]
                )
                and (
                    dataframe.iloc[index - 2]["sma_1h_1h"]
                    > dataframe.iloc[index - 2]["high_monthly_1h_1h"]
                )
            )
        else:
            conditions.append(False)

        # Return True if at least 69% of conditions are met (4 out of 6)
        return sum(conditions) >= 4

    def fuzzy_logic_short_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short exit - at least 69% of conditions (4 out of 6) must be true
        """
        conditions = []

        # Condition 1: Close[1] is higher than HighestInRange_19_14[2] for 5 bars at 5 bar ago
        if index > 10:
            condition_met = True
            for i in range(5):
                bar_index = index - 5 - i
                if (
                    dataframe.iloc[bar_index]["close"]
                    <= dataframe.iloc[bar_index - 2]["highest_in_range_19_14"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 2: VWAP[3] is falling
        if index > 3:
            conditions.append(dataframe.iloc[index - 3]["vwap"] < dataframe.iloc[index - 4]["vwap"])
        else:
            conditions.append(False)

        # Condition 3: Vortex.Minus[4] is lower than 978666 for 6 bars at 2 bar ago
        # Using a reasonable threshold instead of the arbitrary number
        if index > 8:
            condition_met = True
            threshold = 0.9  # Reasonable threshold for vortex
            for i in range(6):
                bar_index = index - 2 - i
                if dataframe.iloc[bar_index]["vortex_minus"] >= threshold:
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 4: MTATR_2[3] > MTATR_3[4]
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 3]["mtatr_2"] > dataframe.iloc[index - 4]["mtatr_3"]
            )
        else:
            conditions.append(False)

        # Condition 5: Close is below VWAP[3]
        if index > 3:
            conditions.append(dataframe.iloc[index]["close"] < dataframe.iloc[index - 3]["vwap"])
        else:
            conditions.append(False)

        # Condition 6: SMA(1h)[2] crosses below LowMonthly(1h)[2]
        if index > 2:
            conditions.append(
                (
                    dataframe.iloc[index - 3]["sma_1h_1h"]
                    > dataframe.iloc[index - 3]["low_monthly_1h_1h"]
                )
                and (
                    dataframe.iloc[index - 2]["sma_1h_1h"]
                    < dataframe.iloc[index - 2]["low_monthly_1h_1h"]
                )
            )
        else:
            conditions.append(False)

        # Return True if at least 69% of conditions are met (4 out of 6)
        return sum(conditions) >= 4

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signals
        """
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        # Apply fuzzy logic for each candle
        for i in range(len(dataframe)):
            if i < self.startup_candle_count:
                continue

            # Long entry
            if self.fuzzy_logic_long_entry(dataframe, i):
                dataframe.loc[i, "enter_long"] = 1
                # Calculate limit order price
                if i > 5:
                    limit_price = dataframe.iloc[i - 5]["highest_open"] - (
                        self.price_entry_mult.value * dataframe.iloc[i - 4]["atr"]
                    )
                    dataframe.loc[i, "enter_long_limit"] = limit_price

            # Short entry (only if not long entry)
            if self.fuzzy_logic_short_entry(dataframe, i) and dataframe.loc[i, "enter_long"] == 0:
                dataframe.loc[i, "enter_short"] = 1
                # Calculate limit order price
                if i > 5:
                    limit_price = dataframe.iloc[i - 5]["lowest_open"] + (
                        self.price_entry_mult.value * dataframe.iloc[i - 4]["atr"]
                    )
                    dataframe.loc[i, "enter_short_limit"] = limit_price

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        # Apply fuzzy logic for each candle
        for i in range(len(dataframe)):
            if i < self.startup_candle_count:
                continue

            # Long exit
            if self.fuzzy_logic_long_exit(dataframe, i) and dataframe.loc[i, "enter_long"] == 0:
                dataframe.loc[i, "exit_long"] = 1

            # Short exit
            if self.fuzzy_logic_short_exit(dataframe, i) and dataframe.loc[i, "enter_short"] == 0:
                dataframe.loc[i, "exit_short"] = 1

        return dataframe

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
        Custom stoploss logic - using the stop_loss_pct parameter
        """
        return -self.stop_loss_pct.value / 100

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
        Custom exit logic for profit target and exit after bars
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Exit after N bars
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration > (self.exit_after_bars.value * timeframe_to_minutes(self.timeframe)):
            return "exit_after_bars"

        # Profit target as percentage (not ATR-based for this strategy)
        if current_profit >= (self.profit_target_pct.value / 100):
            return "profit_target_reached"

        return None

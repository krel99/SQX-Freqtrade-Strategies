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


class Strategy_0_4501(IStrategy):
    """
    Strategy 0.4501 - Converted from StrategyQuantX

    This strategy uses fuzzy logic for entry/exit signals combining HeikenAshi,
    MTATR, Williams %R, Bollinger Bands, and various session indicators.
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
    stoploss = -0.02  # 2% from parameters

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
    indicator_ma_period_1 = IntParameter(50, 85, default=68, space="entry")
    indicator_crs_ma_period = IntParameter(20, 40, default=30, space="entry")
    mt_atr_period = IntParameter(15, 25, default=20, space="entry")
    bb_bar_opens_period = IntParameter(25, 50, default=37, space="entry")
    wpr_cross_period = IntParameter(15, 30, default=20, space="entry")
    indicator_ma_period_2 = IntParameter(60, 95, default=79, space="entry")
    bb_bar_closes_period = IntParameter(35, 65, default=50, space="entry")
    ma_period = IntParameter(35, 65, default=50, space="exit")
    bs_power_period = IntParameter(10, 20, default=15, space="exit")

    # Entry/Exit parameters
    price_entry_mult = DecimalParameter(0.1, 0.5, default=0.3, space="entry")
    exit_after_bars = IntParameter(3, 8, default=5, space="exit")
    profit_target_coef = DecimalParameter(3.5, 6.0, default=4.7, space="exit")
    stop_loss_pct = DecimalParameter(1.0, 3.0, default=2.0, space="exit")
    keltner_channel_period = IntParameter(15, 30, default=20, space="entry")
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

        # HeikenAshi
        ha = pta.ha(
            dataframe["open"],
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
        )
        dataframe["ha_close"] = ha["HA_close"]
        dataframe["ha_open"] = ha["HA_open"]

        # MTATR (approximation using standard ATR)
        dataframe["mtatr"] = ta.ATR(dataframe, timeperiod=self.mt_atr_period.value)

        # EMA for MTATR crossovers
        dataframe["mtatr_ema"] = ta.EMA(
            dataframe["mtatr"], timeperiod=self.indicator_crs_ma_period.value
        )

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe["close"],
            timeperiod=self.bb_bar_opens_period.value,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_lower"] = bb_lower

        # Williams %R
        dataframe["wpr"] = ta.WILLR(dataframe, timeperiod=self.wpr_cross_period.value)

        # True Range
        dataframe["true_range"] = ta.TRANGE(dataframe)

        # SMA for True Range
        dataframe["true_range_sma"] = ta.SMA(
            dataframe["true_range"], timeperiod=self.indicator_ma_period_2.value
        )

        # Monthly High/Low approximation (30 days * 24 hours * 4 15-min candles)
        monthly_period = 30 * 24 * 4
        dataframe["high_monthly"] = dataframe["high"].rolling(window=monthly_period).max()
        dataframe["low_monthly"] = dataframe["low"].rolling(window=monthly_period).min()
        dataframe["close_monthly"] = dataframe["close"].shift(monthly_period)

        # Weekly High/Low approximation (7 days * 24 hours * 4 15-min candles)
        weekly_period = 7 * 24 * 4
        dataframe["high_weekly"] = dataframe["high"].rolling(window=weekly_period).max()
        dataframe["low_weekly"] = dataframe["low"].rolling(window=weekly_period).min()

        # Session High/Low (4:33-12:11 approximation)
        dataframe["hour"] = dataframe["date"].dt.hour
        dataframe["minute"] = dataframe["date"].dt.minute

        def is_in_session(row):
            hour = row["hour"]
            minute = row["minute"]
            # 4:33 to 12:11
            if hour == 4 and minute >= 30:
                return True
            elif 5 <= hour <= 11:
                return True
            elif hour == 12 and minute <= 15:
                return True
            return False

        dataframe["in_session"] = dataframe.apply(is_in_session, axis=1)
        dataframe["session_high"] = dataframe.apply(
            lambda row: row["high"] if row["in_session"] else np.nan, axis=1
        )
        dataframe["session_low"] = dataframe.apply(
            lambda row: row["low"] if row["in_session"] else np.nan, axis=1
        )
        dataframe["session_high"] = dataframe["session_high"].ffill()
        dataframe["session_low"] = dataframe["session_low"].ffill()

        # Session Close (7:38)
        dataframe["session_close"] = dataframe.apply(
            lambda row: row["close"] if (row["hour"] == 7 and 30 <= row["minute"] < 45) else np.nan,
            axis=1,
        ).ffill()

        # Keltner Channel
        kc = pta.kc(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.keltner_channel_period.value,
            scalar=2.0,
        )
        # Keltner channels returns columns: KCLe_<length>_<scalar>, KCBe_<length>_<scalar>, KCUe_<length>_<scalar>
        # Access by column position for reliability: 0=lower, 1=basis, 2=upper
        if kc is not None and len(kc.columns) >= 3:
            dataframe["kc_lower"] = kc.iloc[:, 0]  # Lower band
            dataframe["kc_upper"] = kc.iloc[:, 2]  # Upper band
        else:
            # Fallback if kc calculation fails
            dataframe["kc_upper"] = dataframe["close"] * 1.02
            dataframe["kc_lower"] = dataframe["close"] * 0.98

        # ATR for entry calculations
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe["atr_20"] = ta.ATR(dataframe, timeperiod=20)  # Fixed ATR

        # Smoothed Moving Average
        dataframe["smma"] = ta.EMA(
            dataframe["close"], timeperiod=self.ma_period.value * 2
        )  # SMMA approximation

        # Bears Power and Bulls Power
        ema_period = self.bs_power_period.value
        dataframe["ema_bears"] = ta.EMA(dataframe["close"], timeperiod=ema_period)
        dataframe["bears_power"] = dataframe["low"] - dataframe["ema_bears"]
        dataframe["bulls_power"] = dataframe["high"] - dataframe["ema_bears"]

        # Add 1h informative indicators
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        # Calculate 1h indicators
        informative_1h["high_weekly_1h"] = informative_1h["high"].rolling(window=7 * 24).max()
        informative_1h["low_weekly_1h"] = informative_1h["low"].rolling(window=7 * 24).min()

        # Session High/Low for 1h
        def is_in_session_1h(row):
            hour = row["hour"]
            # 4:33 to 12:11
            if 4 <= hour <= 12:
                return True
            return False

        informative_1h["hour"] = informative_1h["date"].dt.hour
        informative_1h["in_session_1h"] = informative_1h.apply(is_in_session_1h, axis=1)
        informative_1h["session_high_1h"] = informative_1h.apply(
            lambda row: row["high"] if row["in_session_1h"] else np.nan, axis=1
        )
        informative_1h["session_low_1h"] = informative_1h.apply(
            lambda row: row["low"] if row["in_session_1h"] else np.nan, axis=1
        )
        informative_1h["session_high_1h"] = informative_1h["session_high_1h"].ffill()
        informative_1h["session_low_1h"] = informative_1h["session_low_1h"].ffill()

        # Bollinger Bands for 1h
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            informative_1h["close"],
            timeperiod=self.bb_bar_closes_period.value,
            nbdevup=2.1,
            nbdevdn=2.1,
        )
        informative_1h["bb_upper_1h"] = bb_upper
        informative_1h["bb_lower_1h"] = bb_lower

        # HeikenAshi for 1h
        ha_1h = pta.ha(
            informative_1h["open"],
            informative_1h["high"],
            informative_1h["low"],
            informative_1h["close"],
        )
        informative_1h["ha_close_1h"] = ha_1h["HA_close"]

        # Merge informative pair
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        return dataframe

    def fuzzy_logic_long_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long entry - at least 42% of conditions (3 out of 8) must be true
        """
        conditions = []

        # Condition 1: HeikenAshiClose[1] < HighMonthly[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 1]["ha_close"] < dataframe.iloc[index - 2]["high_monthly"]
            )
        else:
            conditions.append(False)

        # Condition 2: -735601814 is above its EMA (ignoring invalid condition)
        conditions.append(False)

        # Condition 3: MTATR[3] crosses below its EMA
        if index > 3:
            conditions.append(
                (dataframe.iloc[index - 4]["mtatr"] > dataframe.iloc[index - 4]["mtatr_ema"])
                and (dataframe.iloc[index - 3]["mtatr"] < dataframe.iloc[index - 3]["mtatr_ema"])
            )
        else:
            conditions.append(False)

        # Condition 4: Open[5] above BB Lower[6]
        if index > 6:
            conditions.append(
                dataframe.iloc[index - 5]["open"] > dataframe.iloc[index - 6]["bb_lower"]
            )
        else:
            conditions.append(False)

        # Condition 5: Williams %R[5] crosses -80 upwards
        if index > 5:
            conditions.append(
                (dataframe.iloc[index - 6]["wpr"] < -80)
                and (dataframe.iloc[index - 5]["wpr"] > -80)
            )
        else:
            conditions.append(False)

        # Condition 6: TrueRange[4] is below its SMA
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 4]["true_range"]
                < dataframe.iloc[index - 4]["true_range_sma"]
            )
        else:
            conditions.append(False)

        # Condition 7: SessionLow[1] < LowWeekly(1h)[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 1]["session_low_1h_1h"]
                < dataframe.iloc[index - 2]["low_weekly_1h_1h"]
            )
        else:
            conditions.append(False)

        # Condition 8: Close[1] below BB Lower(1h)[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["close"] < dataframe.iloc[index - 1]["bb_lower_1h_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 42% of conditions are met (3 out of 8)
        return sum(conditions) >= 3

    def fuzzy_logic_short_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short entry - at least 42% of conditions (3 out of 8) must be true
        """
        conditions = []

        # Condition 1: HeikenAshiClose[1] > LowMonthly[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 1]["ha_close"] > dataframe.iloc[index - 2]["low_monthly"]
            )
        else:
            conditions.append(False)

        # Condition 2: -735601814 is below its EMA (ignoring invalid condition)
        conditions.append(False)

        # Condition 3: MTATR[3] crosses above its EMA
        if index > 3:
            conditions.append(
                (dataframe.iloc[index - 4]["mtatr"] < dataframe.iloc[index - 4]["mtatr_ema"])
                and (dataframe.iloc[index - 3]["mtatr"] > dataframe.iloc[index - 3]["mtatr_ema"])
            )
        else:
            conditions.append(False)

        # Condition 4: Open[5] below BB Upper[6]
        if index > 6:
            conditions.append(
                dataframe.iloc[index - 5]["open"] < dataframe.iloc[index - 6]["bb_upper"]
            )
        else:
            conditions.append(False)

        # Condition 5: Williams %R[5] crosses -20 downwards
        if index > 5:
            conditions.append(
                (dataframe.iloc[index - 6]["wpr"] > -20)
                and (dataframe.iloc[index - 5]["wpr"] < -20)
            )
        else:
            conditions.append(False)

        # Condition 6: TrueRange[4] is above its SMA
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 4]["true_range"]
                > dataframe.iloc[index - 4]["true_range_sma"]
            )
        else:
            conditions.append(False)

        # Condition 7: SessionHigh[1] > HighWeekly(1h)[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 1]["session_high_1h_1h"]
                > dataframe.iloc[index - 2]["high_weekly_1h_1h"]
            )
        else:
            conditions.append(False)

        # Condition 8: Close[1] above BB Upper(1h)[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["close"] > dataframe.iloc[index - 1]["bb_upper_1h_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 42% of conditions are met (3 out of 8)
        return sum(conditions) >= 3

    def fuzzy_logic_long_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long exit - at least 80% of conditions (4 out of 5) must be true
        """
        conditions = []

        # Condition 1: SessionClose[2] crosses below CloseMonthly[2]
        if index > 2:
            conditions.append(
                (
                    dataframe.iloc[index - 3]["session_close"]
                    > dataframe.iloc[index - 3]["close_monthly"]
                )
                and (
                    dataframe.iloc[index - 2]["session_close"]
                    < dataframe.iloc[index - 2]["close_monthly"]
                )
            )
        else:
            conditions.append(False)

        # Condition 2: SMMA[2] is rising
        if index > 2:
            conditions.append(dataframe.iloc[index - 2]["smma"] > dataframe.iloc[index - 3]["smma"])
        else:
            conditions.append(False)

        # Condition 3: BearsPower[4] is rising for 25 bars at 5 bar ago
        if index > 30:
            condition_met = True
            for i in range(25):
                bar_index = index - 5 - i
                if (
                    dataframe.iloc[bar_index]["bears_power"]
                    <= dataframe.iloc[bar_index - 1]["bears_power"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 4: Close[3] crosses above Open[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["close"] > dataframe.iloc[index - 3]["open"]
            )
        else:
            conditions.append(False)

        # Condition 5: HeikenAshiClose(1h)[3] is falling for 5 bars at 3 bar ago
        if index > 8:
            condition_met = True
            for i in range(5):
                bar_index = index - 3 - i
                if (
                    dataframe.iloc[bar_index]["ha_close_1h_1h"]
                    >= dataframe.iloc[bar_index - 1]["ha_close_1h_1h"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Return True if at least 80% of conditions are met (4 out of 5)
        return sum(conditions) >= 4

    def fuzzy_logic_short_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short exit - at least 80% of conditions (4 out of 5) must be true
        """
        conditions = []

        # Condition 1: SessionClose[2] crosses above CloseMonthly[2]
        if index > 2:
            conditions.append(
                (
                    dataframe.iloc[index - 3]["session_close"]
                    < dataframe.iloc[index - 3]["close_monthly"]
                )
                and (
                    dataframe.iloc[index - 2]["session_close"]
                    > dataframe.iloc[index - 2]["close_monthly"]
                )
            )
        else:
            conditions.append(False)

        # Condition 2: SMMA[2] is falling
        if index > 2:
            conditions.append(dataframe.iloc[index - 2]["smma"] < dataframe.iloc[index - 3]["smma"])
        else:
            conditions.append(False)

        # Condition 3: BullsPower[4] is falling for 25 bars at 5 bar ago
        if index > 30:
            condition_met = True
            for i in range(25):
                bar_index = index - 5 - i
                if (
                    dataframe.iloc[bar_index]["bulls_power"]
                    >= dataframe.iloc[bar_index - 1]["bulls_power"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Condition 4: Close[3] crosses below Open[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["close"] < dataframe.iloc[index - 3]["open"]
            )
        else:
            conditions.append(False)

        # Condition 5: HeikenAshiClose(1h)[3] is rising for 5 bars at 3 bar ago
        if index > 8:
            condition_met = True
            for i in range(5):
                bar_index = index - 3 - i
                if (
                    dataframe.iloc[bar_index]["ha_close_1h_1h"]
                    <= dataframe.iloc[bar_index - 1]["ha_close_1h_1h"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Return True if at least 80% of conditions are met (4 out of 5)
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
                    limit_price = dataframe.iloc[i - 1]["kc_upper"] - (
                        self.price_entry_mult.value * dataframe.iloc[i - 5]["atr"]
                    )
                    dataframe.loc[i, "enter_long_limit"] = limit_price

            # Short entry (only if not long entry)
            if self.fuzzy_logic_short_entry(dataframe, i) and dataframe.loc[i, "enter_long"] == 0:
                dataframe.loc[i, "enter_short"] = 1
                # Calculate limit order price
                if i > 5:
                    limit_price = dataframe.iloc[i - 1]["kc_lower"] + (
                        self.price_entry_mult.value * dataframe.iloc[i - 5]["atr"]
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

        # Profit target based on ATR
        if len(dataframe) > 0:
            current_atr = dataframe.iloc[-1]["atr_20"]
            if current_atr > 0:
                profit_target = (self.profit_target_coef.value * current_atr) / trade.open_rate
                if current_profit >= profit_target:
                    return "profit_target_reached"

        return None

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


class Strategy_0_2422(IStrategy):
    """
    Strategy 0.2422 - Converted from StrategyQuantX

    This strategy uses fuzzy logic for entry signals combining TEMA, Parabolic SAR,
    ADX DI, VWAP, and LWMA indicators across multiple timeframes.
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
    stoploss = -0.086  # 8.6% from parameters

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
    tema_period = IntParameter(30, 50, default=40, space="buy")
    di_period = IntParameter(15, 30, default=20, space="buy")
    vwap_period = IntParameter(30, 70, default=50, space="buy")
    lwma_period = IntParameter(30, 70, default=50, space="buy")
    period_1 = IntParameter(15, 35, default=25, space="buy")

    # Entry/Exit parameters
    price_entry_mult = DecimalParameter(0.5, 1.5, default=1.1, space="buy")
    exit_after_bars = IntParameter(2, 8, default=4, space="sell")
    profit_target_coef = DecimalParameter(3.0, 5.5, default=4.2, space="sell")
    stop_loss_pct = DecimalParameter(6.0, 11.0, default=8.6, space="sell")
    ema_period = IntParameter(30, 60, default=45, space="buy")
    atr_period = IntParameter(25, 50, default=37, space="buy")

    # Parabolic SAR parameters
    sar_af_start = DecimalParameter(0.05, 0.15, default=0.109, space="buy")
    sar_af_max = DecimalParameter(0.2, 0.35, default=0.28, space="buy")

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

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(
            dataframe,
            acceleration=self.sar_af_start.value,
            maximum=self.sar_af_max.value,
        )

        # TEMA (Triple Exponential Moving Average)
        dataframe["tema"] = ta.TEMA(
            dataframe["close"], timeperiod=self.tema_period.value
        )

        # ADX and DI
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.di_period.value)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.di_period.value)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.di_period.value)

        # VWAP - using pandas_ta
        dataframe["vwap"] = pta.vwap(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"]
        )

        # Monthly High/Low approximation (30 days * 24 hours * 4 15-min candles)
        monthly_period = 30 * 24 * 4
        dataframe["high_monthly"] = (
            dataframe["high"].rolling(window=monthly_period).max()
        )
        dataframe["low_monthly"] = dataframe["low"].rolling(window=monthly_period).min()

        # EMA for entry calculations
        dataframe["ema_median"] = ta.EMA(
            (dataframe["high"] + dataframe["low"]) / 2, timeperiod=self.ema_period.value
        )

        # ATR for entry calculations
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe["atr_20"] = ta.ATR(
            dataframe, timeperiod=20
        )  # Fixed ATR for profit target

        # Session Open approximation (12:07 UTC)
        # Calculate session open based on time
        dataframe["hour"] = dataframe["date"].dt.hour
        dataframe["minute"] = dataframe["date"].dt.minute
        dataframe["session_open"] = dataframe.apply(
            lambda row: row["open"]
            if (row["hour"] == 12 and row["minute"] >= 0 and row["minute"] < 15)
            else np.nan,
            axis=1,
        ).ffill()

        # Add 1h informative indicators
        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe="1h"
        )

        # Calculate 1h indicators
        # LWMA approximation using WMA
        informative_1h["lwma_1h"] = ta.WMA(
            informative_1h["close"], timeperiod=self.lwma_period.value
        )

        # Highest and Lowest for PRICE_TYPICAL
        informative_1h["typical_price"] = (
            informative_1h["high"] + informative_1h["low"] + informative_1h["close"]
        ) / 3
        informative_1h["highest_typical"] = (
            informative_1h["typical_price"].rolling(window=self.period_1.value).max()
        )
        informative_1h["lowest_typical"] = (
            informative_1h["typical_price"].rolling(window=self.period_1.value).min()
        )

        # Merge informative pair
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        return dataframe

    def fuzzy_logic_long_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long entry - at least 58% of conditions (3 out of 6) must be true
        """
        conditions = []

        # Condition 1: ParabolicSAR[1] <= HighMonthly[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["sar"]
                <= dataframe.iloc[index - 1]["high_monthly"]
            )
        else:
            conditions.append(False)

        # Condition 2: Ask crosses above TEMA[4]
        # Approximating Ask with close
        if index > 4:
            conditions.append(
                (dataframe.iloc[index - 1]["close"] < dataframe.iloc[index - 5]["tema"])
                and (dataframe.iloc[index]["close"] > dataframe.iloc[index - 4]["tema"])
            )
        else:
            conditions.append(False)

        # Condition 3: ADX DI Minus[5] is falling
        if index > 5:
            conditions.append(
                dataframe.iloc[index - 5]["minus_di"]
                > dataframe.iloc[index - 4]["minus_di"]
            )
        else:
            conditions.append(False)

        # Condition 4: VWAP[1] is rising
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["vwap"] > dataframe.iloc[index - 2]["vwap"]
            )
        else:
            conditions.append(False)

        # Condition 5: LWMA(1h)[3] <= High(1h)[2]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["lwma_1h"]
                <= dataframe.iloc[index - 2]["high_1h"]
            )
        else:
            conditions.append(False)

        # Condition 6: Highest(1h, TYPICAL)[2] < SessionOpen[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 2]["highest_typical_1h"]
                < dataframe.iloc[index - 2]["session_open"]
            )
        else:
            conditions.append(False)

        # Return True if at least 58% of conditions are met (3-4 out of 6)
        return sum(conditions) >= 3

    def fuzzy_logic_short_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short entry - at least 58% of conditions (3 out of 6) must be true
        """
        conditions = []

        # Condition 1: ParabolicSAR[1] >= LowMonthly[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["sar"]
                >= dataframe.iloc[index - 1]["low_monthly"]
            )
        else:
            conditions.append(False)

        # Condition 2: Bid crosses below TEMA[4]
        # Approximating Bid with close
        if index > 4:
            conditions.append(
                (dataframe.iloc[index - 1]["close"] > dataframe.iloc[index - 5]["tema"])
                and (dataframe.iloc[index]["close"] < dataframe.iloc[index - 4]["tema"])
            )
        else:
            conditions.append(False)

        # Condition 3: ADX DI Plus[5] is falling
        if index > 5:
            conditions.append(
                dataframe.iloc[index - 5]["plus_di"]
                > dataframe.iloc[index - 4]["plus_di"]
            )
        else:
            conditions.append(False)

        # Condition 4: VWAP[1] is falling
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["vwap"] < dataframe.iloc[index - 2]["vwap"]
            )
        else:
            conditions.append(False)

        # Condition 5: LWMA(1h)[3] >= Low(1h)[2]
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["lwma_1h"]
                >= dataframe.iloc[index - 2]["low_1h"]
            )
        else:
            conditions.append(False)

        # Condition 6: Lowest(1h, TYPICAL)[2] > SessionOpen[2]
        if index > 2:
            conditions.append(
                dataframe.iloc[index - 2]["lowest_typical_1h"]
                > dataframe.iloc[index - 2]["session_open"]
            )
        else:
            conditions.append(False)

        # Return True if at least 58% of conditions are met (3-4 out of 6)
        return sum(conditions) >= 3

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
                    limit_price = dataframe.iloc[i - 5]["ema_median"] + (
                        self.price_entry_mult.value * dataframe.iloc[i - 2]["atr"]
                    )
                    dataframe.loc[i, "enter_long_limit"] = limit_price

            # Short entry (only if not long entry)
            if (
                self.fuzzy_logic_short_entry(dataframe, i)
                and dataframe.loc[i, "enter_long"] == 0
            ):
                dataframe.loc[i, "enter_short"] = 1
                # Calculate limit order price
                if i > 5:
                    limit_price = dataframe.iloc[i - 5]["ema_median"] - (
                        self.price_entry_mult.value * dataframe.iloc[i - 2]["atr"]
                    )
                    dataframe.loc[i, "enter_short_limit"] = limit_price

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signals
        """
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        # Exit signals are false in the original strategy, so we rely on
        # stop loss, profit target, and exit after bars

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
        if trade_duration > (
            self.exit_after_bars.value * timeframe_to_minutes(self.timeframe)
        ):
            return "exit_after_bars"

        # Profit target based on ATR
        if len(dataframe) > 0:
            current_atr = dataframe.iloc[-1]["atr_20"]
            if current_atr > 0:
                profit_target = (
                    self.profit_target_coef.value * current_atr
                ) / trade.open_rate
                if current_profit >= profit_target:
                    return "profit_target_reached"

        return None

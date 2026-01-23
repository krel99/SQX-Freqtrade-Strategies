from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


class BollingerSquigKrumpa(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01,
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 100

    grot_bb_period = IntParameter(10, 40, default=20, space="buy")
    dakka_bb_period = IntParameter(30, 80, default=50, space="buy")
    squig_mult = DecimalParameter(1.5, 3.5, default=2.8, space="buy")
    krump_target = DecimalParameter(1.0, 3.0, default=1.8, space="sell")
    choppa_loss_mult = DecimalParameter(1.5, 3.0, default=2.1, space="buy")
    waaagh_trail = IntParameter(40, 120, default=80, space="sell")
    stompa_range_period = IntParameter(20, 80, default=50, space="buy")

    def calculate_biggest_range(self, dataframe: DataFrame, period: int) -> pd.Series:
        ranges = dataframe["high"] - dataframe["low"]
        return ranges.rolling(window=period).max()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pre-calculate Bollinger Bands for all grot_bb_period values (10-40)
        for period in range(10, 41):
            upper, middle, lower = ta.BBANDS(
                dataframe["close"], timeperiod=period, nbdevup=1.9, nbdevdn=1.9
            )
            dataframe[f"grot_bb_upper_{period}"] = upper
            dataframe[f"grot_bb_middle_{period}"] = middle
            dataframe[f"grot_bb_lower_{period}"] = lower

        # Pre-calculate Bollinger Bands for all dakka_bb_period values (30-80)
        for period in range(30, 81):
            upper, middle, lower = ta.BBANDS(
                dataframe["close"], timeperiod=period, nbdevup=2.0, nbdevdn=2.0
            )
            dataframe[f"dakka_bb_upper_{period}"] = upper
            dataframe[f"dakka_bb_middle_{period}"] = middle
            dataframe[f"dakka_bb_lower_{period}"] = lower

        # Pre-calculate stompa_range for all periods (20-80)
        for period in range(20, 81):
            dataframe[f"stompa_range_{period}"] = self.calculate_biggest_range(dataframe, period)

        dataframe["warboss_atr"] = ta.ATR(dataframe, timeperiod=20)

        dataframe["grot_open_3"] = dataframe["open"].shift(3)
        dataframe["grot_open_1"] = dataframe["open"].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        grot_bb_period = self.grot_bb_period.value
        dakka_bb_period = self.dakka_bb_period.value

        # Select pre-calculated indicators
        grot_bb_lower = dataframe[f"grot_bb_lower_{grot_bb_period}"]
        grot_bb_upper = dataframe[f"grot_bb_upper_{grot_bb_period}"]
        dakka_bb_lower = dataframe[f"dakka_bb_lower_{dakka_bb_period}"]
        dakka_bb_upper = dataframe[f"dakka_bb_upper_{dakka_bb_period}"]

        # Calculate shifted values
        grot_bb_lower_4 = grot_bb_lower.shift(4)
        grot_bb_upper_4 = grot_bb_upper.shift(4)
        dakka_bb_lower_2 = dakka_bb_lower.shift(2)
        dakka_bb_upper_2 = dakka_bb_upper.shift(2)

        dataframe.loc[
            (
                (dataframe["grot_open_3"] > grot_bb_lower_4)
                & (dataframe["grot_open_1"] > dakka_bb_lower_2)
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["grot_open_3"] < grot_bb_upper_4)
                & (dataframe["grot_open_1"] < dakka_bb_upper_2)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        choppa_stop = -(last_candle["warboss_atr"] * self.choppa_loss_mult.value / trade.open_rate)

        return max(choppa_stop, self.stoploss)

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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get current hyperopt parameter value
        stompa_range_period = self.stompa_range_period.value

        # Select pre-calculated indicator and shift
        stompa_range = dataframe[f"stompa_range_{stompa_range_period}"]
        stompa_range_2 = stompa_range.shift(2).iloc[-1]

        if side == "long":
            squig_price = rate + (self.squig_mult.value * stompa_range_2)
            if squig_price > rate * 1.05:
                return False
        else:
            squig_price = rate - (self.squig_mult.value * stompa_range_2)
            if squig_price < rate * 0.95:
                return False

        return True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        krump_profit = self.krump_target.value / 100

        if current_profit >= krump_profit:
            return "krump_target_smashed"

        if current_time - trade.open_date_utc > pd.Timedelta(hours=43):
            if current_profit > 0:
                return "grot_tired_profit"
            elif current_profit > -0.02:
                return "grot_tired_small"

        return None

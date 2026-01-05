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


class IchiKeltnerDakka(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.08,
        "30": 0.05,
        "60": 0.025,
        "120": 0.01,
    }

    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 200

    gorkimoku_tenkan = IntParameter(50, 150, default=97, space="buy")
    gorkimoku_kijun = IntParameter(15, 40, default=26, space="buy")
    gorkimoku_senkou = IntParameter(30, 80, default=52, space="buy")
    squigoth_keltner_period = IntParameter(15, 30, default=20, space="buy")
    squigoth_keltner_mult = DecimalParameter(1.5, 3.0, default=2.25, space="buy")
    dakka_entry_mult = DecimalParameter(0.05, 0.8, default=0.2, space="buy")
    krumpin_profit = DecimalParameter(1.0, 4.0, default=2.1, space="sell")
    gitsmasha_loss = DecimalParameter(2.0, 6.0, default=3.3, space="sell")
    boombastic_bb_period = IntParameter(15, 30, default=20, space="buy")
    boombastic_bb_dev = DecimalParameter(1.5, 2.5, default=1.9, space="buy")

    def calculate_bb_range(self, dataframe: DataFrame, period: int, dev: float) -> pd.Series:
        upper, middle, lower = ta.BBANDS(
            dataframe["close"], timeperiod=period, nbdevup=dev, nbdevdn=dev
        )
        return upper - lower

    def calculate_ichimoku(
        self, dataframe: DataFrame, tenkan: int, kijun: int, senkou: int
    ) -> dict:
        high_tenkan = dataframe["high"].rolling(window=tenkan).max()
        low_tenkan = dataframe["low"].rolling(window=tenkan).min()
        tenkan_sen = (high_tenkan + low_tenkan) / 2

        high_kijun = dataframe["high"].rolling(window=kijun).max()
        low_kijun = dataframe["low"].rolling(window=kijun).min()
        kijun_sen = (high_kijun + low_kijun) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

        high_senkou = dataframe["high"].rolling(window=senkou).max()
        low_senkou = dataframe["low"].rolling(window=senkou).min()
        senkou_span_b = ((high_senkou + low_senkou) / 2).shift(kijun)

        chikou_span = dataframe["close"].shift(-kijun)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ichi = self.calculate_ichimoku(
            dataframe,
            self.gorkimoku_tenkan.value,
            self.gorkimoku_kijun.value,
            self.gorkimoku_senkou.value,
        )
        dataframe["ichi_tenkan"] = ichi["tenkan_sen"]
        dataframe["ichi_kijun"] = ichi["kijun_sen"]
        dataframe["ichi_senkou_a"] = ichi["senkou_span_a"]
        dataframe["ichi_senkou_b"] = ichi["senkou_span_b"]
        dataframe["ichi_chikou"] = ichi["chikou_span"]

        dataframe["ichi_cloud_green"] = dataframe["ichi_senkou_a"] > dataframe["ichi_senkou_b"]
        dataframe["ichi_cloud_red"] = dataframe["ichi_senkou_a"] <= dataframe["ichi_senkou_b"]

        dataframe["kumo_top"] = dataframe[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
        dataframe["kumo_bottom"] = dataframe[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)

        dataframe["kumo_breakout_bearish"] = (dataframe["close"] < dataframe["kumo_bottom"]) & (
            dataframe["close"].shift(1) >= dataframe["kumo_bottom"].shift(1)
        )
        dataframe["kumo_breakout_bullish"] = (dataframe["close"] > dataframe["kumo_top"]) & (
            dataframe["close"].shift(1) <= dataframe["kumo_top"].shift(1)
        )

        dataframe["kumo_breakout_bearish_3"] = dataframe["kumo_breakout_bearish"].shift(3)
        dataframe["kumo_breakout_bullish_3"] = dataframe["kumo_breakout_bullish"].shift(3)

        keltner_ema = ta.EMA(dataframe, timeperiod=self.squigoth_keltner_period.value)
        keltner_atr = ta.ATR(dataframe, timeperiod=self.squigoth_keltner_period.value)
        dataframe["keltner_upper"] = keltner_ema + (self.squigoth_keltner_mult.value * keltner_atr)
        dataframe["keltner_lower"] = keltner_ema - (self.squigoth_keltner_mult.value * keltner_atr)
        dataframe["keltner_middle"] = keltner_ema

        dataframe["close_below_keltner_lower"] = dataframe["close"] < dataframe["keltner_lower"]
        dataframe["close_above_keltner_upper"] = dataframe["close"] > dataframe["keltner_upper"]

        dataframe["close_below_keltner_lower_1"] = dataframe["close_below_keltner_lower"].shift(1)
        dataframe["close_above_keltner_upper_1"] = dataframe["close_above_keltner_upper"].shift(1)

        dataframe["boombastic_range"] = self.calculate_bb_range(
            dataframe, self.boombastic_bb_period.value, self.boombastic_bb_dev.value
        )
        dataframe["boombastic_range_2"] = dataframe["boombastic_range"].shift(2)

        dataframe["warboss_high"] = dataframe["high"].rolling(window=24).max()
        dataframe["warboss_low"] = dataframe["low"].rolling(window=24).min()
        dataframe["warboss_high_3"] = dataframe["warboss_high"].shift(3)
        dataframe["warboss_low_3"] = dataframe["warboss_low"].shift(3)

        dataframe["grot_atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["kumo_breakout_bearish_3"]) & (dataframe["close_below_keltner_lower_1"]),
            "enter_long",
        ] = 1

        dataframe.loc[
            (dataframe["kumo_breakout_bullish_3"]) & (dataframe["close_above_keltner_upper_1"]),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if side == "long":
            orkboy_price = last_candle["warboss_high_3"] - (
                self.dakka_entry_mult.value * last_candle["boombastic_range_2"]
            )
            return min(orkboy_price, proposed_rate * 0.998)
        else:
            orkboy_price = last_candle["warboss_low_3"] + (
                self.dakka_entry_mult.value * last_candle["boombastic_range_2"]
            )
            return max(orkboy_price, proposed_rate * 1.002)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        krumpin_target = self.krumpin_profit.value / 100

        if current_profit >= krumpin_target:
            return "krumpin_complete"

        gitsmasha_stop = -(self.gitsmasha_loss.value / 100)
        if current_profit <= gitsmasha_stop:
            return "gitsmasha_retreat"

        if current_time - trade.open_date_utc > pd.Timedelta(hours=29):
            if current_profit > 0:
                return "dakka_exhausted_profit"
            elif current_profit > -0.015:
                return "dakka_exhausted_small"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        gitsmasha_stoploss = -(self.gitsmasha_loss.value / 100)

        if current_profit > 0.01:
            return -0.005
        elif current_profit > 0.005:
            return -0.008

        if current_time - trade.open_date_utc > pd.Timedelta(hours=20):
            return max(gitsmasha_stoploss * 0.8, -0.03)

        return gitsmasha_stoploss

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

        if side == "long":
            max_distance = rate * 1.06
            orkboy_price = last_candle["warboss_high_3"] - (
                self.dakka_entry_mult.value * last_candle["boombastic_range_2"]
            )
            if orkboy_price > max_distance:
                return False
        else:
            max_distance = rate * 0.94
            orkboy_price = last_candle["warboss_low_3"] + (
                self.dakka_entry_mult.value * last_candle["boombastic_range_2"]
            )
            if orkboy_price < max_distance:
                return False

        if last_candle["boombastic_range_2"] <= 0:
            return False

        return True

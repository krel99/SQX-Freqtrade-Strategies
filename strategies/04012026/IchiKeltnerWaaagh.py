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


class IchiKeltnerWaaagh(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.10,
        "40": 0.06,
        "80": 0.03,
        "160": 0.01,
    }

    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 200

    orkimoku_tenkan = IntParameter(50, 150, default=106, space="buy")
    orkimoku_kijun = IntParameter(15, 40, default=26, space="buy")
    orkimoku_senkou = IntParameter(30, 80, default=52, space="buy")
    keltna_period = IntParameter(15, 30, default=20, space="buy")
    keltna_mult = DecimalParameter(1.5, 3.0, default=2.25, space="buy")
    dakka_entry_mult = DecimalParameter(0.3, 1.5, default=0.8, space="buy")
    waaagh_profit = DecimalParameter(3.0, 8.0, default=5.4, space="sell")
    choppa_loss = DecimalParameter(5.0, 15.0, default=9.6, space="sell")
    mork_atr_period = IntParameter(10, 20, default=14, space="buy")

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
        """
        Pre-calculates all indicator variants for hyperopt compatibility.
        """
        # Pre-calculate rolling highs/lows for Ichimoku tenkan periods (50-150)
        for period in range(50, 151):
            dataframe[f"high_tenkan_{period}"] = dataframe["high"].rolling(window=period).max()
            dataframe[f"low_tenkan_{period}"] = dataframe["low"].rolling(window=period).min()

        # Pre-calculate rolling highs/lows for Ichimoku kijun periods (15-40)
        for period in range(15, 41):
            dataframe[f"high_kijun_{period}"] = dataframe["high"].rolling(window=period).max()
            dataframe[f"low_kijun_{period}"] = dataframe["low"].rolling(window=period).min()

        # Pre-calculate rolling highs/lows for Ichimoku senkou periods (30-80)
        for period in range(30, 81):
            dataframe[f"high_senkou_{period}"] = dataframe["high"].rolling(window=period).max()
            dataframe[f"low_senkou_{period}"] = dataframe["low"].rolling(window=period).min()

        # Pre-calculate EMA for Keltner periods (15-30)
        for period in range(15, 31):
            dataframe[f"keltner_ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Pre-calculate ATR for all periods (10-30 covers keltna_period and mork_atr_period)
        for period in range(10, 31):
            dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)

        # Fixed indicators (no hyperopt params)
        dataframe["gork_high"] = dataframe["high"].rolling(window=24).max()
        dataframe["gork_low"] = dataframe["low"].rolling(window=24).min()
        dataframe["gork_high_3"] = dataframe["gork_high"].shift(3)
        dataframe["gork_low_3"] = dataframe["gork_low"].shift(3)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current hyperopt parameter values
        tenkan = self.orkimoku_tenkan.value
        kijun = self.orkimoku_kijun.value
        senkou = self.orkimoku_senkou.value
        keltna_period = self.keltna_period.value
        keltna_mult = self.keltna_mult.value

        # Calculate Ichimoku lines from pre-calculated rolling values
        tenkan_sen = (dataframe[f"high_tenkan_{tenkan}"] + dataframe[f"low_tenkan_{tenkan}"]) / 2
        kijun_sen = (dataframe[f"high_kijun_{kijun}"] + dataframe[f"low_kijun_{kijun}"]) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = (
            (dataframe[f"high_senkou_{senkou}"] + dataframe[f"low_senkou_{senkou}"]) / 2
        ).shift(kijun)

        # Calculate kumo (cloud) values
        kumo_top = pd.concat([senkou_span_a, senkou_span_b], axis=1).max(axis=1)
        kumo_bottom = pd.concat([senkou_span_a, senkou_span_b], axis=1).min(axis=1)

        # Calculate kumo breakouts
        kumo_breakout_bearish = (dataframe["close"] < kumo_bottom) & (
            dataframe["close"].shift(1) >= kumo_bottom.shift(1)
        )
        kumo_breakout_bullish = (dataframe["close"] > kumo_top) & (
            dataframe["close"].shift(1) <= kumo_top.shift(1)
        )
        kumo_breakout_bearish_2 = kumo_breakout_bearish.shift(2)
        kumo_breakout_bullish_2 = kumo_breakout_bullish.shift(2)

        # Calculate Keltner channels from pre-calculated values
        keltner_ema = dataframe[f"keltner_ema_{keltna_period}"]
        keltner_atr = dataframe[f"atr_{keltna_period}"]
        keltner_upper = keltner_ema + (keltna_mult * keltner_atr)
        keltner_lower = keltner_ema - (keltna_mult * keltner_atr)

        close_below_keltner_lower = dataframe["close"] < keltner_lower
        close_above_keltner_upper = dataframe["close"] > keltner_upper
        close_below_keltner_lower_1 = close_below_keltner_lower.shift(1)
        close_above_keltner_upper_1 = close_above_keltner_upper.shift(1)

        dataframe.loc[
            (kumo_breakout_bearish_2) & (close_below_keltner_lower_1),
            "enter_long",
        ] = 1

        dataframe.loc[
            (kumo_breakout_bullish_2) & (close_above_keltner_upper_1),
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

        # Get ATR using current hyperopt parameter value
        mork_atr_period = self.mork_atr_period.value
        mork_atr = last_candle[f"atr_{mork_atr_period}"]
        mork_atr_3 = dataframe[f"atr_{mork_atr_period}"].shift(3).iloc[-1]

        if side == "long":
            orkish_price = last_candle["gork_high_3"] - (self.dakka_entry_mult.value * mork_atr_3)
            return min(orkish_price, proposed_rate * 0.995)
        else:
            orkish_price = last_candle["gork_low_3"] + (self.dakka_entry_mult.value * mork_atr_3)
            return max(orkish_price, proposed_rate * 1.005)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        waaagh_target = self.waaagh_profit.value / 100

        if current_profit >= waaagh_target:
            return "waaagh_victory"

        choppa_stop = -(self.choppa_loss.value / 100)
        if current_profit <= choppa_stop:
            return "choppa_defeat"

        if current_time - trade.open_date_utc > pd.Timedelta(hours=33):
            if current_profit > 0:
                return "ork_bored_profit"
            elif current_profit > -0.03:
                return "ork_bored_small"

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
        choppa_stoploss = -(self.choppa_loss.value / 100)

        if current_time - trade.open_date_utc > pd.Timedelta(hours=24):
            return max(choppa_stoploss * 0.75, -0.05)

        return choppa_stoploss

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

        # Get ATR using current hyperopt parameter value
        mork_atr_period = self.mork_atr_period.value
        mork_atr_3 = dataframe[f"atr_{mork_atr_period}"].shift(3).iloc[-1]

        if side == "long":
            max_distance = rate * 1.06
            orkish_price = last_candle["gork_high_3"] - (self.dakka_entry_mult.value * mork_atr_3)
            if orkish_price > max_distance:
                return False
        else:
            max_distance = rate * 0.94
            orkish_price = last_candle["gork_low_3"] + (self.dakka_entry_mult.value * mork_atr_3)
            if orkish_price < max_distance:
                return False

        return True

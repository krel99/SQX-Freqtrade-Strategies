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


class Strategy_0_2369(IStrategy):
    """
    Strategy 0.2369 - Converted from StrategyQuantX

    This strategy uses fuzzy logic for entry/exit signals combining multiple timeframes
    and technical indicators including SMMA, SMA, ADX, VWAP, SuperTrend, and Ichimoku.
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
    stoploss = -0.083  # 8.3% from parameters

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
    smma_period = IntParameter(30, 70, default=50, space="buy")
    sma_period = IntParameter(20, 60, default=40, space="buy")
    di_cross_period = IntParameter(30, 80, default=53, space="buy")
    vwap_period = IntParameter(15, 40, default=28, space="buy")
    di_period = IntParameter(20, 50, default=34, space="buy")
    supertrend_atr_period = IntParameter(400, 550, default=480, space="buy")
    ichimoku_tenkan = IntParameter(5, 15, default=9, space="buy")
    ichimoku_kijun = IntParameter(20, 35, default=26, space="buy")
    ichimoku_senkou = IntParameter(40, 65, default=52, space="buy")
    indicator_ma_period = IntParameter(40, 80, default=61, space="buy")
    period_1 = IntParameter(15, 35, default=25, space="buy")
    bb_period = IntParameter(30, 70, default=50, space="buy")
    kc_period = IntParameter(10, 30, default=20, space="buy")
    avg_volume_period = IntParameter(10, 30, default=20, space="buy")

    # Entry/Exit parameters
    price_entry_mult = DecimalParameter(0.5, 2.0, default=1.4, space="buy")
    exit_after_bars = IntParameter(5, 25, default=15, space="sell")
    profit_target_coef = DecimalParameter(1.5, 4.0, default=2.5, space="sell")
    stop_loss_pct = DecimalParameter(5.0, 12.0, default=8.3, space="sell")
    ema_period = IntParameter(10, 30, default=20, space="buy")
    atr_period = IntParameter(10, 30, default=20, space="buy")

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

        # SMMA (Smoothed Moving Average)
        dataframe["smma"] = ta.EMA(
            dataframe["open"], timeperiod=self.smma_period.value * 2
        )  # Approximation of SMMA

        # SMA
        dataframe["sma"] = ta.SMA(dataframe["open"], timeperiod=self.sma_period.value)

        # ADX and DI
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.di_period.value)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.di_period.value)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.di_period.value)

        # VWAP - using pandas_ta
        dataframe["vwap"] = pta.vwap(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"]
        )

        # SuperTrend
        supertrend = pta.supertrend(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.supertrend_atr_period.value,
            multiplier=1.3,
        )
        dataframe["supertrend"] = supertrend[
            f"SUPERT_{self.supertrend_atr_period.value}_1.3"
        ]

        # Ichimoku
        ichimoku = pta.ichimoku(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            tenkan=self.ichimoku_tenkan.value,
            kijun=self.ichimoku_kijun.value,
            senkou=self.ichimoku_senkou.value,
        )
        dataframe["tenkan_sen"] = ichimoku[0][f"ITS_{self.ichimoku_tenkan.value}"]
        dataframe["kijun_sen"] = ichimoku[0][f"IKS_{self.ichimoku_kijun.value}"]

        # EMA for indicators
        dataframe["ema_indicator"] = ta.EMA(
            dataframe["close"], timeperiod=self.indicator_ma_period.value
        )

        # Bollinger Bands
        bb = ta.BBANDS(
            dataframe["close"],
            timeperiod=self.bb_period.value,
            nbdevup=2.1,
            nbdevdn=2.1,
        )
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]

        # Keltner Channel
        kc = pta.kc(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.kc_period.value,
            scalar=2.5,
        )
        dataframe["kc_upper"] = kc[f"KCUp_{self.kc_period.value}_2.5"]
        dataframe["kc_lower"] = kc[f"KCLe_{self.kc_period.value}_2.5"]

        # Volume average
        dataframe["avg_volume"] = ta.SMA(
            dataframe["volume"], timeperiod=self.avg_volume_period.value
        )

        # Laguerre RSI approximation (using standard RSI as proxy)
        dataframe["laguerre_rsi"] = ta.RSI(dataframe["close"], timeperiod=14)

        # EMA and ATR for entry calculations
        dataframe["ema_high"] = ta.EMA(
            dataframe["high"], timeperiod=self.ema_period.value
        )
        dataframe["ema_low"] = ta.EMA(
            dataframe["low"], timeperiod=self.ema_period.value
        )
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Highest and Lowest
        dataframe["highest"] = (
            dataframe["low"].rolling(window=self.period_1.value).max()
        )
        dataframe["lowest"] = (
            dataframe["high"].rolling(window=self.period_1.value).min()
        )

        # Weekly close approximation (every 7*24*4 = 672 15-min candles)
        dataframe["close_weekly"] = dataframe["close"].shift(672)

        # Add 1h informative indicators
        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe="1h"
        )

        # Calculate 1h indicators
        informative_1h["plus_di_1h"] = ta.PLUS_DI(
            informative_1h, timeperiod=self.di_period.value
        )
        informative_1h["minus_di_1h"] = ta.MINUS_DI(
            informative_1h, timeperiod=self.di_period.value
        )
        informative_1h["kc_upper_1h"] = pta.kc(
            informative_1h["high"],
            informative_1h["low"],
            informative_1h["close"],
            length=self.kc_period.value,
            scalar=2.5,
        )[f"KCUp_{self.kc_period.value}_2.5"]
        informative_1h["kc_lower_1h"] = pta.kc(
            informative_1h["high"],
            informative_1h["low"],
            informative_1h["close"],
            length=self.kc_period.value,
            scalar=2.5,
        )[f"KCLe_{self.kc_period.value}_2.5"]
        informative_1h["avg_volume_1h"] = ta.SMA(
            informative_1h["volume"], timeperiod=self.avg_volume_period.value
        )
        informative_1h["laguerre_rsi_1h"] = ta.RSI(
            informative_1h["close"], timeperiod=14
        )

        # Merge informative pair
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        return dataframe

    def fuzzy_logic_long_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long entry - at least 52% of conditions (3 out of 5) must be true
        """
        conditions = []

        # Condition 1: CloseWeekly[4] = SMMA[3]
        if index > 4:
            conditions.append(
                abs(
                    dataframe.iloc[index - 4]["close_weekly"]
                    - dataframe.iloc[index - 3]["smma"]
                )
                < 0.01
            )
        else:
            conditions.append(False)

        # Condition 2: SMA[3] is in lower 51.7% of values over 334 bars
        if index > 337:
            sma_val = dataframe.iloc[index - 3]["sma"]
            historical_sma = dataframe.iloc[index - 337 : index - 3]["sma"]
            percentile = (historical_sma < sma_val).sum() / len(historical_sma)
            conditions.append(percentile <= 0.517)
        else:
            conditions.append(False)

        # Condition 3: ADX DI Plus[1] crosses below ADX DI Minus[1]
        if index > 1:
            conditions.append(
                (
                    dataframe.iloc[index - 2]["plus_di"]
                    > dataframe.iloc[index - 2]["minus_di"]
                )
                and (
                    dataframe.iloc[index - 1]["plus_di"]
                    < dataframe.iloc[index - 1]["minus_di"]
                )
            )
        else:
            conditions.append(False)

        # Condition 4: Close is below VWAP[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index]["close"] < dataframe.iloc[index - 3]["vwap"]
            )
        else:
            conditions.append(False)

        # Condition 5: ADX DI Plus (1h)[3] is rising
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["plus_di_1h"]
                > dataframe.iloc[index - 4]["plus_di_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 52% of conditions are met (3 out of 5)
        return sum(conditions) >= 3

    def fuzzy_logic_short_entry(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short entry - at least 52% of conditions (3 out of 5) must be true
        """
        conditions = []

        # Condition 1: CloseWeekly[4] <> SMMA[3]
        if index > 4:
            conditions.append(
                abs(
                    dataframe.iloc[index - 4]["close_weekly"]
                    - dataframe.iloc[index - 3]["smma"]
                )
                > 0.01
            )
        else:
            conditions.append(False)

        # Condition 2: SMA[3] is in upper 51.7% of values over 334 bars
        if index > 337:
            sma_val = dataframe.iloc[index - 3]["sma"]
            historical_sma = dataframe.iloc[index - 337 : index - 3]["sma"]
            percentile = (historical_sma < sma_val).sum() / len(historical_sma)
            conditions.append(percentile >= 0.517)
        else:
            conditions.append(False)

        # Condition 3: ADX DI Plus[1] crosses above ADX DI Minus[1]
        if index > 1:
            conditions.append(
                (
                    dataframe.iloc[index - 2]["plus_di"]
                    < dataframe.iloc[index - 2]["minus_di"]
                )
                and (
                    dataframe.iloc[index - 1]["plus_di"]
                    > dataframe.iloc[index - 1]["minus_di"]
                )
            )
        else:
            conditions.append(False)

        # Condition 4: Close is above VWAP[3]
        if index > 3:
            conditions.append(
                dataframe.iloc[index]["close"] > dataframe.iloc[index - 3]["vwap"]
            )
        else:
            conditions.append(False)

        # Condition 5: ADX DI Minus (1h)[3] is rising
        if index > 3:
            conditions.append(
                dataframe.iloc[index - 3]["minus_di_1h"]
                > dataframe.iloc[index - 4]["minus_di_1h"]
            )
        else:
            conditions.append(False)

        # Return True if at least 52% of conditions are met (3 out of 5)
        return sum(conditions) >= 3

    def fuzzy_logic_long_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for long exit - at least 51% of conditions (3 out of 6) must be true
        """
        conditions = []

        # Condition 1: SuperTrend[5] crosses above Ichimoku TenkanSen[2]
        if index > 5:
            conditions.append(
                (
                    dataframe.iloc[index - 6]["supertrend"]
                    < dataframe.iloc[index - 3]["tenkan_sen"]
                )
                and (
                    dataframe.iloc[index - 5]["supertrend"]
                    > dataframe.iloc[index - 2]["tenkan_sen"]
                )
            )
        else:
            conditions.append(False)

        # Condition 2: Close[4] is below its EMA
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 4]["close"]
                < dataframe.iloc[index - 4]["ema_indicator"]
            )
        else:
            conditions.append(False)

        # Condition 3: Highest[5] is in lower 2.9% of values over 255 bars
        if index > 260:
            highest_val = dataframe.iloc[index - 5]["highest"]
            historical_highest = dataframe.iloc[index - 260 : index - 5]["highest"]
            percentile = (historical_highest < highest_val).sum() / len(
                historical_highest
            )
            conditions.append(percentile <= 0.029)
        else:
            conditions.append(False)

        # Condition 4: Open[5] above BB Lower[6]
        if index > 6:
            conditions.append(
                dataframe.iloc[index - 5]["open"]
                > dataframe.iloc[index - 6]["bb_lower"]
            )
        else:
            conditions.append(False)

        # Condition 5: Bar closes above KC Lower (1h)[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["close"]
                > dataframe.iloc[index - 1]["kc_lower_1h"]
            )
        else:
            conditions.append(False)

        # Condition 6: AvgVolume(1h)[1] < LaguerreRSI(1h)[1] for 10 bars at 2 bars ago
        if index > 12:
            condition_met = True
            for i in range(10):
                bar_index = index - 2 - i
                if (
                    dataframe.iloc[bar_index]["avg_volume_1h"]
                    >= dataframe.iloc[bar_index]["laguerre_rsi_1h"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Return True if at least 51% of conditions are met (3 out of 6)
        return sum(conditions) >= 3

    def fuzzy_logic_short_exit(self, dataframe: DataFrame, index: int) -> bool:
        """
        Fuzzy logic for short exit - at least 51% of conditions (3 out of 6) must be true
        """
        conditions = []

        # Condition 1: SuperTrend[5] crosses below Ichimoku TenkanSen[2]
        if index > 5:
            conditions.append(
                (
                    dataframe.iloc[index - 6]["supertrend"]
                    > dataframe.iloc[index - 3]["tenkan_sen"]
                )
                and (
                    dataframe.iloc[index - 5]["supertrend"]
                    < dataframe.iloc[index - 2]["tenkan_sen"]
                )
            )
        else:
            conditions.append(False)

        # Condition 2: Close[4] is above its EMA
        if index > 4:
            conditions.append(
                dataframe.iloc[index - 4]["close"]
                > dataframe.iloc[index - 4]["ema_indicator"]
            )
        else:
            conditions.append(False)

        # Condition 3: Lowest[5] is in upper 2.9% of values over 255 bars
        if index > 260:
            lowest_val = dataframe.iloc[index - 5]["lowest"]
            historical_lowest = dataframe.iloc[index - 260 : index - 5]["lowest"]
            percentile = (historical_lowest < lowest_val).sum() / len(historical_lowest)
            conditions.append(percentile >= 0.971)
        else:
            conditions.append(False)

        # Condition 4: Open[5] below BB Upper[6]
        if index > 6:
            conditions.append(
                dataframe.iloc[index - 5]["open"]
                < dataframe.iloc[index - 6]["bb_upper"]
            )
        else:
            conditions.append(False)

        # Condition 5: Bar closes below KC Upper (1h)[1]
        if index > 1:
            conditions.append(
                dataframe.iloc[index - 1]["close"]
                < dataframe.iloc[index - 1]["kc_upper_1h"]
            )
        else:
            conditions.append(False)

        # Condition 6: AvgVolume(1h)[1] > LaguerreRSI(1h)[1] for 10 bars at 2 bars ago
        if index > 12:
            condition_met = True
            for i in range(10):
                bar_index = index - 2 - i
                if (
                    dataframe.iloc[bar_index]["avg_volume_1h"]
                    <= dataframe.iloc[bar_index]["laguerre_rsi_1h"]
                ):
                    condition_met = False
                    break
            conditions.append(condition_met)
        else:
            conditions.append(False)

        # Return True if at least 51% of conditions are met (3 out of 6)
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
                    limit_price = dataframe.iloc[i - 5]["ema_high"] + (
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
                    limit_price = dataframe.iloc[i - 5]["ema_low"] - (
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

        # Apply fuzzy logic for each candle
        for i in range(len(dataframe)):
            if i < self.startup_candle_count:
                continue

            # Long exit
            if (
                self.fuzzy_logic_long_exit(dataframe, i)
                and dataframe.loc[i, "enter_long"] == 0
            ):
                dataframe.loc[i, "exit_long"] = 1

            # Short exit
            if (
                self.fuzzy_logic_short_exit(dataframe, i)
                and dataframe.loc[i, "enter_short"] == 0
            ):
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
        if trade_duration > (
            self.exit_after_bars.value * timeframe_to_minutes(self.timeframe)
        ):
            return "exit_after_bars"

        # Profit target based on ATR
        if len(dataframe) > 0:
            current_atr = dataframe.iloc[-1]["atr"]
            if current_atr > 0:
                profit_target = (
                    self.profit_target_coef.value * current_atr
                ) / trade.open_rate
                if current_profit >= profit_target:
                    return "profit_target_reached"

        return None

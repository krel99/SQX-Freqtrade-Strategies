from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class BBTimeWindowBreakout(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = True

    minimal_roi = {"0": 0.02}
    stoploss = -0.03

    allowed_minutes = (0, 30, 51)

    # -------- BUY space --------
    # Clear bounds (no huge grids)
    bb_length = IntParameter(10, 60, default=20, space="buy")
    bb_mult = DecimalParameter(1.2, 2.2, default=1.6, decimals=2, space="buy")

    long_break = DecimalParameter(0.998, 1.010, default=1.001, decimals=3, space="buy")
    short_break = DecimalParameter(0.990, 1.002, default=0.999, decimals=3, space="buy")
    min_bb_width = DecimalParameter(0.000, 0.020, default=0.000, decimals=3, space="buy")

    start_hour = IntParameter(0, 23, default=4, space="buy")
    end_hour = IntParameter(0, 23, default=12, space="buy")

    # Hard cap: 2..5 trades/day
    max_trades_per_day = IntParameter(2, 5, default=3, space="buy")

    # -------- SELL space --------
    exit_mid_offset = DecimalParameter(0.990, 1.010, default=1.000, decimals=3, space="sell")
    max_hold_minutes = IntParameter(10, 720, default=240, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Only 2 lightweight columns
        dataframe["minute"] = dataframe["date"].dt.minute
        dataframe["hour"] = dataframe["date"].dt.hour
        return dataframe

    def _in_hour_window(self, dataframe: DataFrame):
        h = dataframe["hour"]
        start = self.start_hour.value
        end = self.end_hour.value
        if start <= end:
            return (h >= start) & (h <= end)
        return (h >= start) | (h <= end)

    @staticmethod
    def _bbands(close: pd.Series, length: int, mult: float):
        # Rolling mean + std. ddof=0 is fine for trading indicators and is faster.
        mid = close.rolling(window=length, min_periods=length).mean()
        std = close.rolling(window=length, min_periods=length).std(ddof=0)
        upper = mid + mult * std
        lower = mid - mult * std
        width = (upper - lower) / mid
        return lower, mid, upper, width

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        length = int(self.bb_length.value)
        mult = float(self.bb_mult.value)

        lower, mid, upper, width = self._bbands(dataframe["close"], length, mult)

        time_ok = dataframe["minute"].isin(self.allowed_minutes) & self._in_hour_window(dataframe)
        width_ok = width > self.min_bb_width.value

        dataframe.loc[
            time_ok & width_ok & (dataframe["close"] > upper * self.long_break.value), "enter_long"
        ] = 1

        dataframe.loc[
            time_ok & width_ok & (dataframe["close"] < lower * self.short_break.value),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        length = int(self.bb_length.value)
        mult = float(self.bb_mult.value)
        _, mid, _, _ = self._bbands(dataframe["close"], length, mult)

        dataframe.loc[dataframe["close"] < mid * self.exit_mid_offset.value, "exit_long"] = 1
        dataframe.loc[dataframe["close"] > mid * self.exit_mid_offset.value, "exit_short"] = 1
        return dataframe

    # -------- hard limits --------
    @staticmethod
    def _hour_floor_utc(dt: datetime) -> datetime:
        dt = dt.astimezone(timezone.utc)
        return dt.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _day_floor_utc(dt: datetime) -> datetime:
        dt = dt.astimezone(timezone.utc)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        try:
            now_utc = current_time.astimezone(timezone.utc)
            hour0 = self._hour_floor_utc(now_utc)
            day0 = self._day_floor_utc(now_utc)
            day1 = day0 + timedelta(days=1)

            last = (
                Trade.get_trades([Trade.pair == pair]).order_by(Trade.open_date_utc.desc()).first()
            )
            if last and last.open_date_utc and self._hour_floor_utc(last.open_date_utc) == hour0:
                return False

            today_count = Trade.get_trades(
                [Trade.pair == pair, Trade.open_date_utc >= day0, Trade.open_date_utc < day1]
            ).count()
            return today_count < int(self.max_trades_per_day.value)
        except Exception:
            return True

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        if trade.open_date_utc is None:
            return None
        hold_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if hold_minutes >= float(self.max_hold_minutes.value):
            return "time_exit"
        return None

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class IchimokuTimeCapped(IStrategy):
    """
    Standard Ichimoku trend strategy (futures, long+short), with hour-window filter and trade caps.

    LONG:
      - Price above Kumo (cloud top)
      - Tenkan > Kijun
      - Chikou above close(displacement) (confirmation)

    SHORT:
      - Price below Kumo (cloud bottom)
      - Tenkan < Kijun
      - Chikou below close(displacement)
    """

    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = True

    minimal_roi = {"0": 0.02}
    stoploss = -0.03

    # ---------------- BUY space (clear bounds) ----------------
    tenkan_len = IntParameter(7, 14, default=9, space="buy")  # classic 9
    kijun_len = IntParameter(20, 40, default=26, space="buy")  # classic 26
    senkou_b_len = IntParameter(40, 80, default=52, space="buy")  # classic 52
    displacement = IntParameter(20, 40, default=26, space="buy")  # classic 26

    # Buffer around cloud to reduce chop (0% .. 0.8%)
    kumo_buffer = DecimalParameter(0.000, 0.008, default=0.001, decimals=3, space="buy")

    # Optimizable hour window (default 4–12)
    start_hour = IntParameter(0, 23, default=4, space="buy")
    end_hour = IntParameter(0, 23, default=12, space="buy")

    # Hard daily cap (2..5)
    max_trades_per_day = IntParameter(2, 5, default=3, space="buy")

    # ---------------- SELL space ----------------
    # Exit logic toggle (kept int so it's easy to hyperopt):
    # 1 = exit on Tenkan/Kijun cross-back
    # 0 = exit on price crossing Kijun (more forgiving)
    exit_mode = IntParameter(0, 1, default=1, space="sell")

    # Time-based safety exit
    max_hold_minutes = IntParameter(30, 720, default=240, space="sell")

    # ---------- helpers ----------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hour"] = dataframe["date"].dt.hour
        return dataframe

    def _in_hour_window(self, dataframe: DataFrame):
        h = dataframe["hour"]
        start = int(self.start_hour.value)
        end = int(self.end_hour.value)
        if start <= end:
            return (h >= start) & (h <= end)
        return (h >= start) | (h <= end)

    @staticmethod
    def _hour_floor_utc(dt: datetime) -> datetime:
        dt = dt.astimezone(timezone.utc)
        return dt.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _day_floor_utc(dt: datetime) -> datetime:
        dt = dt.astimezone(timezone.utc)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _donchian_mid(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
        hh = high.rolling(length, min_periods=length).max()
        ll = low.rolling(length, min_periods=length).min()
        return (hh + ll) / 2.0

    def _ichimoku(self, df: DataFrame):
        t = int(self.tenkan_len.value)
        k = int(self.kijun_len.value)
        sb = int(self.senkou_b_len.value)
        d = int(self.displacement.value)

        tenkan = self._donchian_mid(df["high"], df["low"], t)
        kijun = self._donchian_mid(df["high"], df["low"], k)

        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = self._donchian_mid(df["high"], df["low"], sb)

        # Cloud is projected forward by displacement, but for trading "now" we compare price to
        # the cloud values that were computed displacement candles ago.
        # So we shift cloud BACK by displacement to align with current candle.
        cloud_a_now = senkou_a.shift(d)
        cloud_b_now = senkou_b.shift(d)

        cloud_top = pd.concat([cloud_a_now, cloud_b_now], axis=1).max(axis=1)
        cloud_bot = pd.concat([cloud_a_now, cloud_b_now], axis=1).min(axis=1)

        # Chikou span is current close shifted back by displacement.
        # For confirmation "chikou above past price", we can compare current close to close.shift(d).
        close_past = df["close"].shift(d)

        return tenkan, kijun, cloud_top, cloud_bot, close_past

    # ---------- entries / exits ----------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tenkan, kijun, cloud_top, cloud_bot, close_past = self._ichimoku(dataframe)
        buf = float(self.kumo_buffer.value)

        time_ok = self._in_hour_window(dataframe)

        # LONG conditions
        long_trend = dataframe["close"] > cloud_top * (1.0 + buf)
        long_align = tenkan > kijun
        long_chikou = dataframe["close"] > close_past

        dataframe.loc[time_ok & long_trend & long_align & long_chikou, "enter_long"] = 1

        # SHORT conditions
        short_trend = dataframe["close"] < cloud_bot * (1.0 - buf)
        short_align = tenkan < kijun
        short_chikou = dataframe["close"] < close_past

        dataframe.loc[time_ok & short_trend & short_align & short_chikou, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tenkan, kijun, cloud_top, cloud_bot, close_past = self._ichimoku(dataframe)

        if int(self.exit_mode.value) == 1:
            # Exit on cross-back
            dataframe.loc[(tenkan < kijun), "exit_long"] = 1
            dataframe.loc[(tenkan > kijun), "exit_short"] = 1
        else:
            # Exit on price crossing Kijun
            dataframe.loc[(dataframe["close"] < kijun), "exit_long"] = 1
            dataframe.loc[(dataframe["close"] > kijun), "exit_short"] = 1

        return dataframe

    # ---------- HARD LIMITS: 1/hour + 2..5/day ----------
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
        """
        Enforce per-pair caps:
          - max 1 entry per hour
          - max N entries per day (N=2..5, default 3)
        """
        try:
            now_utc = current_time.astimezone(timezone.utc)
            hour0 = self._hour_floor_utc(now_utc)
            day0 = self._day_floor_utc(now_utc)
            day1 = day0 + timedelta(days=1)

            # 1 per hour
            last = (
                Trade.get_trades([Trade.pair == pair]).order_by(Trade.open_date_utc.desc()).first()
            )
            if last and last.open_date_utc and self._hour_floor_utc(last.open_date_utc) == hour0:
                return False

            # 2..5 per day
            today_count = Trade.get_trades(
                [Trade.pair == pair, Trade.open_date_utc >= day0, Trade.open_date_utc < day1]
            ).count()
            return today_count < int(self.max_trades_per_day.value)

        except Exception:
            # If persistence isn't available, don't block entries.
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

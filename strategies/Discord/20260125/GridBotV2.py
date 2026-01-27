import warnings
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


# ✅ pandas-safe SettingWithCopyWarning import
try:
    from pandas.errors import SettingWithCopyWarning

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")


class GridBotV2(IStrategy):
    """
    Close to original GridV6_tmp7, but improved for edge:
    - No martingale scaling: additive pieces (1, +2, +3, ...)
    - Volatility regime gate: post-move + low-vol-for-N-candles
    - Extra light filters (only for FIRST entry):
        * KAMA slope: direction bias
        * VIDYA slope: responsiveness / momentum
        * FRAMA compression: regime (prefer compressed/noise zone after impulse)
    - Candle-count based time handling (hyperopt-safe)
    - Avoid populate_indicators() calculations
    - Updated config keys:
        * use_sell_signal -> use_exit_signal
        * sell_profit_only -> exit_profit_only
    - Avoid unnecessary threshold recalculation: recalc only when state/grid/avg changes
    """

    INTERFACE_VERSION = 2

    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # ----- Core behavior (kept close to original) -----
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75

    debug = True

    # DCA config (freqtrade position adjust)
    position_adjustment_enable = True

    # Max entries INCLUDING initial. Do not exceed (7 adds + 1 base) -> bounds 6..8 as requested
    max_entries = IntParameter(6, 8, default=7, load=True, space="buy", optimize=True)

    # ROI/SL
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # ✅ Updated exit/signal flags (deprecations fixed)
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 120  # need enough candles for FRAMA/KAMA/VIDYA + gating

    # Grid params
    grid_down_spacing_pct = DecimalParameter(
        3.0, 5.0, default=4.0, decimals=1, load=True, space="sell", optimize=False
    )
    grid_up_spacing_pct = DecimalParameter(
        1.4, 3.0, default=1.6, decimals=1, load=True, space="sell", optimize=True
    )

    # ----- Volatility regime (candle-based, hyperopt-safe) -----
    atr_period = IntParameter(7, 50, default=14, load=True, space="buy", optimize=True)

    move_lookback_candles = IntParameter(12, 288, default=48, load=True, space="buy", optimize=True)
    move_threshold_pct = DecimalParameter(
        0.5, 6.0, default=2.0, decimals=2, load=True, space="buy", optimize=True
    )

    low_vol_candles = IntParameter(12, 288, default=24, load=True, space="buy", optimize=True)
    low_vol_atr_pct = DecimalParameter(
        0.10, 3.00, default=0.60, decimals=2, load=True, space="buy", optimize=True
    )

    post_move_max_candles = IntParameter(12, 576, default=72, load=True, space="buy", optimize=True)

    # ----- Extra light filters (ONLY for first entry) -----
    kama_period = IntParameter(10, 50, default=21, load=True, space="buy", optimize=True)
    vidya_period = IntParameter(10, 50, default=21, load=True, space="buy", optimize=True)
    frama_period = IntParameter(10, 50, default=16, load=True, space="buy", optimize=True)

    ma_slope_lookback = IntParameter(2, 10, default=3, load=True, space="buy", optimize=True)
    frama_compression_pct = DecimalParameter(
        0.10, 2.00, default=0.60, decimals=2, load=True, space="buy", optimize=True
    )

    plot_config = {
        "main_plot": {
            "grid_up": {"grid_up": {"color": "green"}},
            "grid_down": {"grid_down": {"color": "red"}},
        },
        "subplots": {
            "bot_state": {"bot_state": {"color": "yellow"}},
            "trade_allowed": {"trade_allowed": {"color": "white"}},
            "atr_pct": {"atr_pct": {"color": "blue"}},
            "kama_slope": {"kama_slope": {"color": "green"}},
            "vidya_slope": {"vidya_slope": {"color": "orange"}},
            "frama_dist": {"frama_dist": {"color": "red"}},
        },
    }

    # storage dict for live state
    custom_info: Dict = {}

    # ---------------------------------------------------------------------
    # Per your requirement: DO NOTHING here
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ---------------------------------------------------------------------
    def _max_adds(self) -> int:
        return max(int(self.max_entries.value) - 1, 0)

    def _max_stake_multiplier(self) -> float:
        # sum(1..N) = N(N+1)/2
        n = float(max(int(self.max_entries.value), 1))
        return (n * (n + 1.0)) / 2.0

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float:
        mult = self._max_stake_multiplier()
        if self.config.get("stake_amount") == "unlimited":
            return (self.wallets.get_total_stake_amount() / mult) * self.stake_to_wallet_ratio
        return (proposed_stake / mult) * self.stake_to_wallet_ratio

    # ---------------------------------------------------------------------
    # Exotic MA helpers (implemented locally to avoid dependency/version issues)
    @staticmethod
    def _vidya(series: pd.Series, period: int) -> pd.Series:
        """
        VIDYA via CMO-based adaptive alpha.
        alpha = 2/(period+1) * abs(CMO(period))
        """
        s = series.astype("float64")
        diff = s.diff()
        up = diff.clip(lower=0.0)
        down = (-diff).clip(lower=0.0)

        sum_up = up.rolling(period, min_periods=period).sum()
        sum_down = down.rolling(period, min_periods=period).sum()
        denom = (sum_up + sum_down).replace(0.0, np.nan)
        cmo = (sum_up - sum_down) / denom  # [-1..1]
        alpha = (2.0 / (period + 1.0)) * cmo.abs()
        alpha = alpha.fillna(0.0).clip(0.0, 1.0)

        out = pd.Series(np.nan, index=s.index, dtype="float64")
        if len(s) == 0:
            return out

        out.iloc[0] = s.iloc[0]
        a = alpha.to_numpy()
        x = s.to_numpy()
        y = out.to_numpy()

        for i in range(1, len(s)):
            y[i] = y[i - 1] + a[i] * (x[i] - y[i - 1])

        out[:] = y
        return out

    @staticmethod
    def _frama(series: pd.Series, period: int) -> pd.Series:
        """
        FRAMA (Fractal Adaptive Moving Average), simplified and stable.
        Uses rolling highs/lows to estimate fractal dimension and adaptive alpha.
        """
        s = series.astype("float64")
        n = int(period)
        if n < 4:
            return s.ewm(span=max(n, 2), adjust=False).mean()

        # ensure even half window
        n2 = n // 2
        if n2 < 2:
            n2 = 2

        # Rolling highs/lows
        high_n = s.rolling(n, min_periods=n).max()
        low_n = s.rolling(n, min_periods=n).min()
        high_1 = s.rolling(n2, min_periods=n2).max()
        low_1 = s.rolling(n2, min_periods=n2).min()

        # shift for second half
        high_2 = s.shift(n2).rolling(n2, min_periods=n2).max()
        low_2 = s.shift(n2).rolling(n2, min_periods=n2).min()

        n1 = (high_1 - low_1) / float(n2)
        n2v = (high_2 - low_2) / float(n2)
        n3 = (high_n - low_n) / float(n)

        # fractal dimension
        # D = (log(n1+n2) - log(n3)) / log(2)
        eps = 1e-12
        d = (np.log((n1 + n2v).clip(lower=eps)) - np.log(n3.clip(lower=eps))) / np.log(2.0)
        d = d.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(1.0, 2.0)

        alpha = np.exp(-4.6 * (d - 1.0))
        alpha = alpha.clip(0.01, 1.0)

        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.iloc[0] = s.iloc[0]

        a = alpha.to_numpy()
        x = s.to_numpy()
        y = out.to_numpy()

        for i in range(1, len(s)):
            if np.isnan(a[i]):
                y[i] = y[i - 1]
            else:
                y[i] = y[i - 1] + a[i] * (x[i] - y[i - 1])

        out[:] = y
        return out

    # ---------------------------------------------------------------------
    def _compute_gates(self, df: DataFrame) -> DataFrame:
        """
        Returns df with:
          - trade_allowed: for FIRST entry (vol regime + light MA/regime filters)
          - add_allowed: for DCA adds (vol regime only)
        """
        out = df.copy()

        # --- Volatility regime ---
        atr = ta.ATR(out, timeperiod=int(self.atr_period.value))
        out["atr_pct"] = (atr / out["close"]) * 100.0

        lb = int(self.move_lookback_candles.value)
        out["move_pct"] = (out["close"] / out["close"].shift(lb) - 1.0) * 100.0
        out["large_move"] = out["move_pct"].abs() >= float(self.move_threshold_pct.value)

        idx = np.arange(len(out), dtype=float)
        last_move_idx = np.where(out["large_move"].to_numpy(), idx, np.nan)
        last_move_idx = pd.Series(last_move_idx, index=out.index).ffill()

        candles_since = pd.Series(idx, index=out.index) - last_move_idx
        out["post_move"] = (candles_since >= 0) & (
            candles_since <= int(self.post_move_max_candles.value)
        )

        out["low_vol_now"] = out["atr_pct"] <= float(self.low_vol_atr_pct.value)
        lv = int(self.low_vol_candles.value)
        out["low_vol_for_period"] = out["low_vol_now"].rolling(lv, min_periods=lv).min() == 1

        vol_regime = (out["post_move"] & out["low_vol_for_period"]).fillna(False)
        out["add_allowed"] = vol_regime

        # --- Light filters (FIRST entry only) ---
        slope_lb = int(self.ma_slope_lookback.value)

        kama = ta.KAMA(out, timeperiod=int(self.kama_period.value))
        vidya = self._vidya(out["close"], int(self.vidya_period.value))
        frama = self._frama(out["close"], int(self.frama_period.value))

        out["kama_slope"] = kama - kama.shift(slope_lb)
        out["vidya_slope"] = vidya - vidya.shift(slope_lb)

        out["frama_dist"] = (out["close"].sub(frama).abs() / out["close"]) * 100.0

        # Minimal, non-overcomplicated:
        # - avoid fighting downtrend: KAMA slope >= 0
        # - require some responsiveness: VIDYA slope >= 0
        # - prefer compression/noise zone after impulse: FRAMA close distance small
        trend_filter = (
            (out["kama_slope"] >= 0)
            & (out["vidya_slope"] >= 0)
            & (out["frama_dist"] <= float(self.frama_compression_pct.value))
        )

        out["trade_allowed"] = (vol_regime & trend_filter).fillna(False)

        return out

    # ---------------------------------------------------------------------
    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if pair not in self.custom_info:
            # {DATESTAMP, GRID, BOT_STATE, LIVE_DATE, INIT_COUNT, AVG_PRICE, UNITS}
            self.custom_info[pair] = ["", 0.0, 0, "", self.live_candles, 0.0, 0.0]

        df = self._compute_gates(dataframe)

        last_row = df.tail(1).index.item()
        init_count = self.custom_info[pair][self.INIT_COUNT]

        # --- init bot state for live/dry vs backtest/hyperopt (kept close to original) ---
        if self.dp.runmode.value in ("live", "dry_run"):
            if self.custom_info[pair][self.LIVE_DATE] == "":
                init_count = 0
                row = last_row
                bot_state = 0
                grid = df["close"].iloc[row]
                avg_price = grid
                units = 0.0
            else:
                live_date_candle = df.loc[df["date"] == self.custom_info[pair][self.LIVE_DATE]]
                if len(live_date_candle) > 0:
                    row = live_date_candle.index[0]
                    bot_state = int(self.custom_info[pair][self.BOT_STATE])
                    grid = float(self.custom_info[pair][self.GRID])
                    avg_price = float(self.custom_info[pair][self.AVG_PRICE])
                    units = float(self.custom_info[pair][self.UNITS])
                else:
                    init_count = 0
                    row = last_row
                    bot_state = 0
                    grid = df["close"].iloc[row]
                    avg_price = grid
                    units = 0.0
        else:
            row = 0
            bot_state = 0
            grid = df["close"].iloc[0]
            avg_price = grid
            units = 0.0

        max_adds = self._max_adds()

        # thresholds calculator (same idea as original)
        def recalc_thresholds(grid_: float, avg_: float, state_: int):
            grid_up_shift = grid_ * ((state_ * self.grid_shift_pct) / 100.0)
            grid_up = avg_ * (1.0 + (self.grid_up_spacing_pct.value / 100.0)) + grid_up_shift
            grid_down = grid_ * (1.0 - (self.grid_down_spacing_pct.value / 100.0))
            grid_trigger_up = grid_ * (1.0 + (self.grid_trigger_pct / 100.0))
            grid_trigger_down = grid_ * (1.0 - (self.grid_trigger_pct / 100.0))
            return grid_up, grid_down, grid_trigger_up, grid_trigger_down

        grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
            grid, avg_price, bot_state
        )

        # output arrays
        Buy_1 = np.zeros((last_row + 1), dtype=int)
        Buy_2 = np.zeros((last_row + 1), dtype=int)
        Sell_1 = np.zeros((last_row + 1), dtype=int)

        Grid_up = np.full((last_row + 1), np.nan)
        Grid_down = np.full((last_row + 1), np.nan)
        Bot_state = np.full((last_row + 1), np.nan)

        Close = df["close"].values
        EntryAllowed = df["trade_allowed"].values  # first entry filter
        AddAllowed = df["add_allowed"].values  # add filter (lighter)

        # Avoid unnecessary threshold recalcs: only recalc after changes
        need_recalc = True

        while row <= last_row:
            close = float(Close[row])
            entry_allowed = bool(EntryAllowed[row])
            add_allowed = bool(AddAllowed[row])

            # Save live state at boundary (same as original)
            if self.dp.runmode.value in ("live", "dry_run"):
                if row == (last_row - init_count):
                    self.custom_info[pair][self.BOT_STATE] = bot_state
                    self.custom_info[pair][self.GRID] = grid
                    self.custom_info[pair][self.LIVE_DATE] = df["date"].iloc[row]
                    self.custom_info[pair][self.AVG_PRICE] = avg_price
                    self.custom_info[pair][self.UNITS] = units

            if need_recalc:
                grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
                    grid, avg_price, bot_state
                )
                need_recalc = False

            new_bot_state = bot_state
            new_grid = grid
            new_units = units
            new_avg_price = avg_price

            # -----------------------------
            # state 0: waiting for first entry
            if bot_state == 0:
                if close > grid_trigger_up:
                    new_grid = close
                    new_units = 0.0
                    new_avg_price = close
                    need_recalc = True

                if entry_allowed and close <= grid_trigger_down:
                    new_bot_state = 1
                    Buy_1[row] = 1
                    new_units = 1.0 / close
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

            # -----------------------------
            # in position: can exit or add
            if 1 <= bot_state <= max_adds:
                # exit (profit side)
                if close > grid_up:
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0.0
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

                # add only if vol regime still valid (lighter than first entry)
                if add_allowed and close <= grid_down:
                    new_bot_state = bot_state + 1
                    Buy_2[row] = 1

                    # additive pieces: 1, +2, +3...
                    add_piece = float(bot_state + 1)
                    total_stake = (float(new_bot_state) * (float(new_bot_state) + 1.0)) / 2.0

                    new_units = units + (add_piece / close)
                    new_avg_price = total_stake / new_units
                    new_grid = close
                    need_recalc = True

            # terminal safety
            if bot_state == (max_adds + 1):
                if (close > grid_up) or (close <= grid_down):
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0.0
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

            bot_state, grid, units, avg_price = new_bot_state, new_grid, new_units, new_avg_price

            # plot levels
            if bot_state == 0:
                Grid_up[row] = grid_trigger_up
                Grid_down[row] = grid_trigger_down
            else:
                Grid_up[row] = grid_up
                Grid_down[row] = grid_down

            Bot_state[row] = bot_state
            row += 1

        if init_count < self.live_candles:
            init_count += 1
        self.custom_info[pair][self.INIT_COUNT] = init_count

        df["buy_1"] = Buy_1
        df["buy_2"] = Buy_2
        df["sell_1"] = Sell_1
        df["grid_up"] = Grid_up
        df["grid_down"] = Grid_down
        df["bot_state"] = Bot_state

        return df

    # ---------------------------------------------------------------------
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ):
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is None or len(df) < 5:
            df = self.dp.get_pair_dataframe(trade.pair, self.timeframe)
            if df is None or len(df) < self.startup_candle_count:
                return None

        if "buy_2" not in df.columns or "bot_state" not in df.columns:
            df = self.calculate_state(df, {"pair": trade.pair})

        last_candle = df.iloc[-1].squeeze()

        # new candle guard
        if self.custom_info[trade.pair][self.DATESTAMP] != last_candle["date"]:
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle["date"]

            if int(last_candle.get("buy_2", 0)) == 1:
                filled_buys = trade.select_filled_orders("buy")
                count_of_buys = len(filled_buys)  # includes initial buy

                # next order piece = count_of_buys + 1 (1->2->3...)
                if 1 <= count_of_buys < int(self.max_entries.value):
                    try:
                        base_stake = filled_buys[0].cost
                        next_piece = float(count_of_buys + 1)
                        return base_stake * next_piece
                    except Exception:
                        return None
        return None

    # ---------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = self.calculate_state(dataframe, metadata)
        df.loc[:, "enter_long"] = 0
        df.loc[(df["buy_1"] == 1), "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        if "sell_1" not in df.columns:
            df = self.calculate_state(dataframe, metadata)

        df.loc[:, "exit_long"] = 0
        df.loc[(df["sell_1"] == 1), "exit_long"] = 1
        return df

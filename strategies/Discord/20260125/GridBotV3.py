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


class GridBotV3(IStrategy):
    """
    GridBotV3 goals:
    - Keep GridBotV2 structure (state machine + additive DCA).
    - Improve robustness: adaptive spacing based on long-term trend strength.
    - Reduce "death by ROI exits": enable exit_profit_only by default (prevents loss exits).
    - Keep it simple: same gating idea (post-move + low-vol) + light MA filters for first entry.
    - No populate_indicators() calculations (hyperopt-safe as you requested).

    Core changes vs V2:
    - Adaptive grid spacing per candle:
        effective_down_spacing_pct = base_down * down_mult(trend)
        effective_up_spacing_pct   = base_up   * up_mult(trend)
      In strong downtrend => down spacing increases (safer), up spacing slightly increases (avoid tiny take-profits).
      In strong uptrend   => down spacing reduces mildly (more responsive), up spacing reduces mildly (faster profit-taking).
    - Optional entry throttling in strong downtrend (still DCA allowed once in, but initial entry becomes harder).
    """

    INTERFACE_VERSION = 2

    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # ----- Core behavior -----
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75

    debug = True

    # DCA enabled
    position_adjustment_enable = True

    # Max entries INCLUDING initial (kept exactly as requested: <= 8 total)
    max_entries = IntParameter(6, 8, default=8, load=True, space="buy", optimize=True)

    # ROI/SL (you hyperopt these anyway)
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # ✅ Updated exit/signal flags (deprecations fixed)
    use_exit_signal = True

    # IMPORTANT robustness change: prevent loss exits due to ROI/exit-signal mechanics.
    # Hyperopt can still override via config/params if you want, but default should avoid losses.
    exit_profit_only = True
    exit_profit_offset = 0.001

    ignore_roi_if_entry_signal = False

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 240  # long-term EMA + FRAMA/KAMA/VIDYA + gates

    # ----- Base grid params -----
    # Keep down spacing non-optimized by default (safer). We adapt it dynamically instead.
    grid_down_spacing_pct = DecimalParameter(
        3.0, 6.0, default=4.0, decimals=1, load=True, space="sell", optimize=False
    )
    # Allow hyperopt to tune profit spacing.
    grid_up_spacing_pct = DecimalParameter(
        1.4, 4.0, default=2.2, decimals=1, load=True, space="sell", optimize=True
    )

    # ----- Volatility regime (candle-based) -----
    atr_period = IntParameter(7, 60, default=44, load=True, space="buy", optimize=True)

    move_lookback_candles = IntParameter(12, 288, default=54, load=True, space="buy", optimize=True)
    move_threshold_pct = DecimalParameter(
        0.5, 8.0, default=5.39, decimals=2, load=True, space="buy", optimize=True
    )

    low_vol_candles = IntParameter(12, 288, default=18, load=True, space="buy", optimize=True)
    low_vol_atr_pct = DecimalParameter(
        0.10, 4.00, default=2.89, decimals=2, load=True, space="buy", optimize=True
    )

    post_move_max_candles = IntParameter(12, 576, default=17, load=True, space="buy", optimize=True)

    # ----- Light entry filters (FIRST entry only) -----
    kama_period = IntParameter(10, 60, default=15, load=True, space="buy", optimize=True)
    vidya_period = IntParameter(10, 60, default=11, load=True, space="buy", optimize=True)
    frama_period = IntParameter(10, 60, default=43, load=True, space="buy", optimize=True)

    ma_slope_lookback = IntParameter(2, 12, default=10, load=True, space="buy", optimize=True)
    frama_compression_pct = DecimalParameter(
        0.10, 2.50, default=0.90, decimals=2, load=True, space="buy", optimize=True
    )

    # NEW: soften the slope filter to avoid over-filtering / flat equity
    slope_min_pct = DecimalParameter(
        -1.0, 1.0, default=-0.05, decimals=3, load=True, space="buy", optimize=True
    )

    # ----- NEW: Long-term trend adaptive spacing -----
    # Long EMA defines macro bias and "how dangerous" mean-revert entries are.
    trend_ema_period = IntParameter(80, 400, default=200, load=True, space="buy", optimize=True)
    trend_slope_lookback = IntParameter(5, 60, default=20, load=True, space="buy", optimize=True)

    # How aggressively spacing adapts to trend. Higher => wider down spacing in downtrend, tighter in uptrend.
    trend_adapt_strength = DecimalParameter(
        0.0, 3.0, default=1.2, decimals=2, load=True, space="buy", optimize=True
    )

    # Optional: block new FIRST entries in strong downtrend (DCA adds still allowed once in).
    block_entry_trend_score = DecimalParameter(
        0.0, 1.0, default=0.55, decimals=2, load=True, space="buy", optimize=True
    )

    plot_config = {
        "main_plot": {
            "grid_up": {"grid_up": {"color": "green"}},
            "grid_down": {"grid_down": {"color": "red"}},
            "ema_trend": {"ema_trend": {"color": "blue"}},
        },
        "subplots": {
            "bot_state": {"bot_state": {"color": "yellow"}},
            "trade_allowed": {"trade_allowed": {"color": "white"}},
            "add_allowed": {"add_allowed": {"color": "gray"}},
            "trend_score": {"trend_score": {"color": "purple"}},
        },
    }

    custom_info: Dict = {}

    # ---------------------------------------------------------------------
    # Per your requirement: do NOTHING here
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
    # Exotic MA helpers (kept local to avoid dependency/version issues)
    @staticmethod
    def _vidya(series: pd.Series, period: int) -> pd.Series:
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
        s = series.astype("float64")
        n = int(period)
        if n < 4:
            return s.ewm(span=max(n, 2), adjust=False).mean()

        half = max(n // 2, 2)

        high_n = s.rolling(n, min_periods=n).max()
        low_n = s.rolling(n, min_periods=n).min()
        high_1 = s.rolling(half, min_periods=half).max()
        low_1 = s.rolling(half, min_periods=half).min()
        high_2 = s.shift(half).rolling(half, min_periods=half).max()
        low_2 = s.shift(half).rolling(half, min_periods=half).min()

        n1 = (high_1 - low_1) / float(half)
        n2v = (high_2 - low_2) / float(half)
        n3 = (high_n - low_n) / float(n)

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
    def _compute_gates_and_trend(self, df: DataFrame) -> DataFrame:
        """
        Adds:
          - trade_allowed: first-entry gate (vol regime + light filters + optional downtrend block)
          - add_allowed: add gate (vol regime only)
          - trend_score: [-1..+1] (negative = downtrend)
          - down_mult / up_mult: adaptive spacing multipliers per candle
        """
        out = df.copy()

        close = out["close"].astype("float64")

        # ----- long-term trend -----
        ema_p = int(self.trend_ema_period.value)
        out["ema_trend"] = ta.EMA(out, timeperiod=ema_p)

        slope_lb = int(self.trend_slope_lookback.value)
        ema = out["ema_trend"]
        # slope in % (approx)
        ema_slope_pct = (
            (ema - ema.shift(slope_lb)) / ema.shift(slope_lb).replace(0, np.nan)
        ) * 100.0
        ema_slope_pct = ema_slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # price distance to ema in %
        ema_dist_pct = ((close - ema) / ema.replace(0, np.nan)) * 100.0
        ema_dist_pct = ema_dist_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Trend score in [-1..1], smooth and bounded:
        # negative when price below ema and slope negative; positive in opposite case.
        raw = (0.6 * (ema_dist_pct / 5.0)) + (0.4 * (ema_slope_pct / 2.0))
        out["trend_score"] = np.tanh(raw).astype("float64")

        # Adaptive multipliers (kept simple):
        # In downtrend => widen down spacing (safer), widen up spacing slightly (avoid tiny exits that churn)
        # In uptrend   => tighten down spacing mildly (more responsive), tighten up spacing mildly (realize profits sooner)
        strength = float(self.trend_adapt_strength.value)
        ts = out["trend_score"]

        # down_mult >= ~0.7 .. <= ~1 + strength
        out["down_mult"] = (1.0 + strength * (-ts).clip(lower=0.0)).clip(0.7, 1.0 + strength)
        # up_mult between ~0.7..1.3
        out["up_mult"] = (1.0 + 0.5 * strength * (-ts) - 0.3 * strength * (ts)).clip(0.7, 1.3)

        # ----- volatility regime (same as V2) -----
        atr = ta.ATR(out, timeperiod=int(self.atr_period.value))
        out["atr_pct"] = (atr / close) * 100.0

        lb = int(self.move_lookback_candles.value)
        out["move_pct"] = (close / close.shift(lb) - 1.0) * 100.0
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

        # ----- light filters for FIRST entry only (slightly relaxed) -----
        slb = int(self.ma_slope_lookback.value)
        kama = ta.KAMA(out, timeperiod=int(self.kama_period.value))
        vidya = self._vidya(close, int(self.vidya_period.value))
        frama = self._frama(close, int(self.frama_period.value))

        # slope in % terms (so slope_min_pct makes sense across price scales)
        kama_slope_pct = ((kama - kama.shift(slb)) / kama.shift(slb).replace(0, np.nan)) * 100.0
        vidya_slope_pct = ((vidya - vidya.shift(slb)) / vidya.shift(slb).replace(0, np.nan)) * 100.0
        kama_slope_pct = kama_slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vidya_slope_pct = vidya_slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["kama_slope"] = kama_slope_pct
        out["vidya_slope"] = vidya_slope_pct

        out["frama_dist"] = (close.sub(frama).abs() / close) * 100.0

        slope_min = float(self.slope_min_pct.value)

        trend_filter = (
            (out["kama_slope"] >= slope_min)
            & (out["vidya_slope"] >= slope_min)
            & (out["frama_dist"] <= float(self.frama_compression_pct.value))
        )

        # Optional downtrend block for FIRST entries:
        # If trend_score is too negative, block entry. (Adds still allowed once in trade.)
        block_thr = float(self.block_entry_trend_score.value)
        not_strong_down = out["trend_score"] >= -block_thr

        out["trade_allowed"] = (vol_regime & trend_filter & not_strong_down).fillna(False)

        return out

    # ---------------------------------------------------------------------
    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if pair not in self.custom_info:
            # {DATESTAMP, GRID, BOT_STATE, LIVE_DATE, INIT_COUNT, AVG_PRICE, UNITS}
            self.custom_info[pair] = ["", 0.0, 0, "", self.live_candles, 0.0, 0.0]

        df = self._compute_gates_and_trend(dataframe)

        last_row = df.tail(1).index.item()
        init_count = self.custom_info[pair][self.INIT_COUNT]

        # --- init bot state for live/dry vs backtest/hyperopt ---
        if self.dp.runmode.value in ("live", "dry_run"):
            if self.custom_info[pair][self.LIVE_DATE] == "":
                init_count = 0
                row = last_row
                bot_state = 0
                grid = float(df["close"].iloc[row])
                avg_price = grid
                units = 0.0
            else:
                live_date_candle = df.loc[df["date"] == self.custom_info[pair][self.LIVE_DATE]]
                if len(live_date_candle) > 0:
                    row = int(live_date_candle.index[0])
                    bot_state = int(self.custom_info[pair][self.BOT_STATE])
                    grid = float(self.custom_info[pair][self.GRID])
                    avg_price = float(self.custom_info[pair][self.AVG_PRICE])
                    units = float(self.custom_info[pair][self.UNITS])
                else:
                    init_count = 0
                    row = last_row
                    bot_state = 0
                    grid = float(df["close"].iloc[row])
                    avg_price = grid
                    units = 0.0
        else:
            row = 0
            bot_state = 0
            grid = float(df["close"].iloc[0])
            avg_price = grid
            units = 0.0

        max_adds = self._max_adds()

        Close = df["close"].values
        EntryAllowed = df["trade_allowed"].values
        AddAllowed = df["add_allowed"].values

        # adaptive multipliers per candle
        DownMult = df["down_mult"].values
        UpMult = df["up_mult"].values

        # output arrays
        Buy_1 = np.zeros((last_row + 1), dtype=int)
        Buy_2 = np.zeros((last_row + 1), dtype=int)
        Sell_1 = np.zeros((last_row + 1), dtype=int)

        Grid_up = np.full((last_row + 1), np.nan)
        Grid_down = np.full((last_row + 1), np.nan)
        Bot_state = np.full((last_row + 1), np.nan)

        # avoid unnecessary threshold recalcs: only when state/grid/avg changes OR spacing multiplier changes a lot.
        need_recalc = True
        prev_dm = None
        prev_um = None

        def recalc_thresholds(grid_: float, avg_: float, state_: int, dm: float, um: float):
            # adaptive spacing
            up_spacing = float(self.grid_up_spacing_pct.value) * float(um)
            down_spacing = float(self.grid_down_spacing_pct.value) * float(dm)

            grid_up_shift = grid_ * ((state_ * self.grid_shift_pct) / 100.0)
            grid_up = avg_ * (1.0 + (up_spacing / 100.0)) + grid_up_shift
            grid_down = grid_ * (1.0 - (down_spacing / 100.0))
            grid_trigger_up = grid_ * (1.0 + (self.grid_trigger_pct / 100.0))
            grid_trigger_down = grid_ * (1.0 - (self.grid_trigger_pct / 100.0))
            return grid_up, grid_down, grid_trigger_up, grid_trigger_down

        # initialize thresholds
        dm0 = float(DownMult[row]) if len(DownMult) > row else 1.0
        um0 = float(UpMult[row]) if len(UpMult) > row else 1.0
        grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
            grid, avg_price, bot_state, dm0, um0
        )
        prev_dm, prev_um = dm0, um0

        while row <= last_row:
            close = float(Close[row])
            entry_allowed = bool(EntryAllowed[row])
            add_allowed = bool(AddAllowed[row])

            dm = float(DownMult[row]) if len(DownMult) > row else 1.0
            um = float(UpMult[row]) if len(UpMult) > row else 1.0

            # save live state at boundary
            if self.dp.runmode.value in ("live", "dry_run"):
                if row == (last_row - init_count):
                    self.custom_info[pair][self.BOT_STATE] = bot_state
                    self.custom_info[pair][self.GRID] = grid
                    self.custom_info[pair][self.LIVE_DATE] = df["date"].iloc[row]
                    self.custom_info[pair][self.AVG_PRICE] = avg_price
                    self.custom_info[pair][self.UNITS] = units

            # if multipliers changed meaningfully, refresh thresholds
            if (
                (prev_dm is None)
                or (prev_um is None)
                or (abs(dm - prev_dm) > 0.05)
                or (abs(um - prev_um) > 0.05)
            ):
                need_recalc = True

            if need_recalc:
                grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
                    grid, avg_price, bot_state, dm, um
                )
                need_recalc = False
                prev_dm, prev_um = dm, um

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
                if close > grid_up:
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0.0
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

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

        # update init_count (live)
        if init_count < self.live_candles:
            init_count += 1
        self.custom_info[pair][self.INIT_COUNT] = init_count

        # attach state outputs
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
        # Try analyzed df first (but populate_indicators is empty, so it may not contain our columns)
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

                if 1 <= count_of_buys < int(self.max_entries.value):
                    try:
                        base_stake = filled_buys[0].cost
                        next_piece = float(count_of_buys + 1)  # 1->2->3...
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

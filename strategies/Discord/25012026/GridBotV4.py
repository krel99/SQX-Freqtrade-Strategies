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


class GridBotV4(IStrategy):
    """
    V4 goals (based on your feedback):
    - Trade MORE often (closer to V2 frequency).
    - Still robust: "further positioning" via adaptive spacing:
        * long-term trend adaptive spacing
        * ATR adaptive spacing (current regime)
        * spacing widens as bot_state increases (deeper adds further apart)
    - Keep additive DCA: 1, +2, +3 ... (no martingale)
    - Keep max_entries constraint exactly (6..8, max = 8 total = 7 adds + base)
    - Keep candle-count / hyperopt safe time handling.
    - Do NOT use populate_indicators().

    Key changes vs V3:
    - Remove hard downtrend block (it killed frequency).
    - Replace strict "low vol consecutive" with "low vol ratio" (so you get more setups).
    - Keep the impulse idea, but make it softer: allow "post-move" OR "very low ATR" fallback.
    - Adaptive spacing is the main robustness tool now, not heavy filtering.
    """

    INTERFACE_VERSION = 2

    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # ----- core behavior (close to original) -----
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75

    debug = True
    position_adjustment_enable = True

    # Max entries INCLUDING initial (must remain <= 8)
    max_entries = IntParameter(6, 8, default=8, load=True, space="buy", optimize=True)

    # ROI / SL (hyperopt can override)
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Updated flags (deprecations fixed)
    use_exit_signal = True
    exit_profit_only = True  # keeps robustness (avoid loss exits)
    exit_profit_offset = 0.001  # tiny profit requirement
    ignore_roi_if_entry_signal = False

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 260

    # ----- base grid params -----
    grid_down_spacing_pct = DecimalParameter(
        2.5, 7.0, default=4.0, decimals=1, load=True, space="sell", optimize=False
    )
    grid_up_spacing_pct = DecimalParameter(
        1.0, 4.5, default=2.2, decimals=1, load=True, space="sell", optimize=True
    )

    # ----- volatility / setup regime (higher frequency than V3) -----
    atr_period = IntParameter(7, 60, default=28, load=True, space="buy", optimize=True)

    # "Impulse" signal (kept, but softened)
    move_lookback_candles = IntParameter(6, 144, default=36, load=True, space="buy", optimize=True)
    move_threshold_pct = DecimalParameter(
        0.5, 8.0, default=3.5, decimals=2, load=True, space="buy", optimize=True
    )
    post_move_max_candles = IntParameter(6, 192, default=48, load=True, space="buy", optimize=True)

    # Low-vol gate (ratio-based instead of strict consecutive)
    low_vol_candles = IntParameter(6, 96, default=18, load=True, space="buy", optimize=True)
    low_vol_atr_pct = DecimalParameter(
        0.10, 4.50, default=2.0, decimals=2, load=True, space="buy", optimize=True
    )
    low_vol_min_ratio = DecimalParameter(
        0.30, 1.00, default=0.65, decimals=2, load=True, space="buy", optimize=True
    )

    # If there's no clear impulse, allow entries when ATR% is *extremely* low (higher frequency, still sensible)
    allow_no_impulse_atr_pct = DecimalParameter(
        0.10, 2.50, default=0.80, decimals=2, load=True, space="buy", optimize=True
    )

    # ----- light filters (kept but relaxed) -----
    # Keep KAMA direction, but soft: allow slightly negative slopes.
    kama_period = IntParameter(10, 60, default=20, load=True, space="buy", optimize=True)
    ma_slope_lookback = IntParameter(2, 12, default=6, load=True, space="buy", optimize=True)
    kama_slope_min_pct = DecimalParameter(
        -1.0, 1.0, default=-0.20, decimals=3, load=True, space="buy", optimize=True
    )

    # ----- adaptive spacing controls (main robustness lever) -----
    # Long-term trend
    trend_ema_period = IntParameter(80, 400, default=200, load=True, space="buy", optimize=True)
    trend_slope_lookback = IntParameter(5, 80, default=20, load=True, space="buy", optimize=True)
    trend_adapt_strength = DecimalParameter(
        0.0, 3.0, default=1.0, decimals=2, load=True, space="buy", optimize=True
    )

    # ATR adaptive spacing
    atr_adapt_strength = DecimalParameter(
        0.0, 2.5, default=0.8, decimals=2, load=True, space="buy", optimize=True
    )
    atr_ref_pct = DecimalParameter(
        0.5, 6.0, default=2.0, decimals=2, load=True, space="buy", optimize=True
    )

    # "Further positioning": widen spacing as bot_state increases
    level_spacing_increase = DecimalParameter(
        0.0, 0.60, default=0.18, decimals=2, load=True, space="buy", optimize=True
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
            "atr_pct": {"atr_pct": {"color": "blue"}},
            "trend_score": {"trend_score": {"color": "purple"}},
        },
    }

    custom_info: Dict = {}

    # ------------------------------------------------------------
    # Per your requirement: DO NOTHING here
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    def _compute_context(self, df: DataFrame) -> DataFrame:
        """
        Adds:
          - atr_pct
          - trend_score ([-1..+1])
          - trade_allowed (for first entry)
          - add_allowed (for adds)
          - down_mult_base, up_mult_base (trend+atr based, NOT level-based)
        """
        out = df.copy()
        close = out["close"].astype("float64")

        # ATR%
        atr = ta.ATR(out, timeperiod=int(self.atr_period.value))
        out["atr_pct"] = (atr / close) * 100.0
        atr_pct = out["atr_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Trend score from EMA distance + slope
        ema_p = int(self.trend_ema_period.value)
        out["ema_trend"] = ta.EMA(out, timeperiod=ema_p)
        ema = out["ema_trend"].replace([np.inf, -np.inf], np.nan).ffill()
        ema = ema.fillna(close)

        slope_lb = int(self.trend_slope_lookback.value)
        ema_prev = ema.shift(slope_lb).replace(0, np.nan)
        ema_slope_pct = ((ema - ema_prev) / ema_prev) * 100.0
        ema_slope_pct = ema_slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ema_dist_pct = ((close - ema) / ema.replace(0, np.nan)) * 100.0
        ema_dist_pct = ema_dist_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        raw = (0.6 * (ema_dist_pct / 5.0)) + (0.4 * (ema_slope_pct / 2.0))
        out["trend_score"] = np.tanh(raw).astype("float64")
        ts = out["trend_score"]

        # Trend multipliers
        t_strength = float(self.trend_adapt_strength.value)
        # downtrend => increase down spacing
        trend_down_mult = (1.0 + t_strength * (-ts).clip(lower=0.0)).clip(0.7, 1.0 + t_strength)
        # downtrend => slightly increase up spacing to avoid churn
        trend_up_mult = (1.0 + 0.40 * t_strength * (-ts) - 0.20 * t_strength * ts).clip(0.7, 1.35)

        # ATR multipliers
        a_strength = float(self.atr_adapt_strength.value)
        atr_ref = float(self.atr_ref_pct.value)
        atr_factor = (1.0 + a_strength * (atr_pct / max(atr_ref, 1e-6))).clip(
            0.7, 1.0 + 3.0 * a_strength
        )

        out["down_mult_base"] = (trend_down_mult * atr_factor).astype("float64")
        out["up_mult_base"] = (trend_up_mult * (0.8 + 0.2 * atr_factor)).astype("float64")

        # ---------------- setup regime (higher frequency) ----------------
        # "Impulse" detection: use lookback shift (as in V2 style)
        lb = int(self.move_lookback_candles.value)
        move_pct = (close / close.shift(lb) - 1.0) * 100.0
        out["move_pct"] = move_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["large_move"] = out["move_pct"].abs() >= float(self.move_threshold_pct.value)

        # last impulse index
        idx = np.arange(len(out), dtype=float)
        last_move_idx = np.where(out["large_move"].to_numpy(), idx, np.nan)
        last_move_idx = pd.Series(last_move_idx, index=out.index).ffill()
        candles_since = pd.Series(idx, index=out.index) - last_move_idx

        post_move = (candles_since >= 0) & (candles_since <= int(self.post_move_max_candles.value))
        out["post_move"] = post_move.fillna(False)

        # Low-vol ratio gate
        out["low_vol_now"] = atr_pct <= float(self.low_vol_atr_pct.value)
        lv = int(self.low_vol_candles.value)
        low_vol_ratio = out["low_vol_now"].rolling(lv, min_periods=lv).mean()
        low_vol_ok = (low_vol_ratio >= float(self.low_vol_min_ratio.value)).fillna(False)
        out["low_vol_ratio"] = low_vol_ratio.fillna(0.0)

        # If no impulse, allow entries when ATR% is extremely low
        no_impulse_ok = (atr_pct <= float(self.allow_no_impulse_atr_pct.value)).fillna(False)

        vol_gate = (low_vol_ok & (out["post_move"] | no_impulse_ok)).fillna(False)
        out["add_allowed"] = vol_gate  # adds need only regime

        # Very light direction check (KAMA slope in %)
        slb = int(self.ma_slope_lookback.value)
        kama = ta.KAMA(out, timeperiod=int(self.kama_period.value))
        kama_prev = kama.shift(slb).replace(0, np.nan)
        kama_slope_pct = ((kama - kama_prev) / kama_prev) * 100.0
        kama_slope_pct = kama_slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["kama_slope"] = kama_slope_pct

        # Only for FIRST entry: require KAMA slope >= min (relaxed)
        out["trade_allowed"] = (
            vol_gate & (kama_slope_pct >= float(self.kama_slope_min_pct.value))
        ).fillna(False)

        return out

    # ------------------------------------------------------------
    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if pair not in self.custom_info:
            # {DATESTAMP, GRID, BOT_STATE, LIVE_DATE, INIT_COUNT, AVG_PRICE, UNITS}
            self.custom_info[pair] = ["", 0.0, 0, "", self.live_candles, 0.0, 0.0]

        df = self._compute_context(dataframe)

        last_row = df.tail(1).index.item()
        init_count = self.custom_info[pair][self.INIT_COUNT]

        # init for live/dry vs backtest/hyperopt
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

        DownBase = df["down_mult_base"].values
        UpBase = df["up_mult_base"].values

        # output arrays
        Buy_1 = np.zeros((last_row + 1), dtype=int)
        Buy_2 = np.zeros((last_row + 1), dtype=int)
        Sell_1 = np.zeros((last_row + 1), dtype=int)

        Grid_up = np.full((last_row + 1), np.nan)
        Grid_down = np.full((last_row + 1), np.nan)
        Bot_state = np.full((last_row + 1), np.nan)

        # Recalc thresholds only when needed
        need_recalc = True
        prev_dm = None
        prev_um = None
        prev_state_for_spacing = None

        def recalc_thresholds(
            grid_: float, avg_: float, state_: int, dm_base: float, um_base: float
        ):
            # level-based widening ("further positioning")
            lvl_inc = float(self.level_spacing_increase.value)
            level_mult = 1.0 + lvl_inc * float(
                max(state_ - 1, 0)
            )  # state 1 => 1.0, state 2 => 1+inc, ...

            # adaptive spacing (trend+atr) and level widening
            down_spacing = float(self.grid_down_spacing_pct.value) * float(dm_base) * level_mult
            up_spacing = float(self.grid_up_spacing_pct.value) * float(um_base)

            grid_up_shift = grid_ * ((state_ * self.grid_shift_pct) / 100.0)
            grid_up = avg_ * (1.0 + (up_spacing / 100.0)) + grid_up_shift
            grid_down = grid_ * (1.0 - (down_spacing / 100.0))
            grid_trigger_up = grid_ * (1.0 + (self.grid_trigger_pct / 100.0))
            grid_trigger_down = grid_ * (1.0 - (self.grid_trigger_pct / 100.0))
            return grid_up, grid_down, grid_trigger_up, grid_trigger_down

        # init thresholds
        dm0 = float(DownBase[row]) if len(DownBase) > row else 1.0
        um0 = float(UpBase[row]) if len(UpBase) > row else 1.0
        grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
            grid, avg_price, bot_state, dm0, um0
        )
        prev_dm, prev_um = dm0, um0
        prev_state_for_spacing = bot_state

        while row <= last_row:
            close = float(Close[row])
            entry_allowed = bool(EntryAllowed[row])
            add_allowed = bool(AddAllowed[row])

            dm = float(DownBase[row]) if len(DownBase) > row else 1.0
            um = float(UpBase[row]) if len(UpBase) > row else 1.0

            # save live state at boundary
            if self.dp.runmode.value in ("live", "dry_run"):
                if row == (last_row - init_count):
                    self.custom_info[pair][self.BOT_STATE] = bot_state
                    self.custom_info[pair][self.GRID] = grid
                    self.custom_info[pair][self.LIVE_DATE] = df["date"].iloc[row]
                    self.custom_info[pair][self.AVG_PRICE] = avg_price
                    self.custom_info[pair][self.UNITS] = units

            # refresh if base multipliers move meaningfully or state changed (state changes affect level spacing)
            if (
                (prev_dm is None)
                or (prev_um is None)
                or (abs(dm - prev_dm) > 0.08)
                or (abs(um - prev_um) > 0.08)
            ):
                need_recalc = True
            if prev_state_for_spacing != bot_state:
                need_recalc = True

            if need_recalc:
                grid_up, grid_down, grid_trigger_up, grid_trigger_down = recalc_thresholds(
                    grid, avg_price, bot_state, dm, um
                )
                need_recalc = False
                prev_dm, prev_um = dm, um
                prev_state_for_spacing = bot_state

            new_bot_state = bot_state
            new_grid = grid
            new_units = units
            new_avg_price = avg_price

            # state 0: waiting
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

            # in position
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

    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
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

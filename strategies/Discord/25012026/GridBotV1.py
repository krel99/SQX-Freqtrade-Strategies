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


class GridBotV1(IStrategy):
    """
    Close to original GridV6_tmp7, but:
    - No martingale scaling: additive pieces (1, +2, +3, ...)
    - Volatility regime gate: only allow entry/add when post-move + low-vol-for-N-candles
    - Candle-count based time handling (hyperopt-safe)
    - Avoid populate_indicators() calculations
    - Fix deprecations:
        sell_profit_only -> exit_profit_only
        use_sell_signal   -> use_exit_signal
    - Avoid unnecessary threshold recalculation: only recalc when state/grid/avg changes
    """

    INTERFACE_VERSION = 2

    DATESTAMP = 0
    GRID = 1
    BOT_STATE = 2
    LIVE_DATE = 3
    INIT_COUNT = 4
    AVG_PRICE = 5
    UNITS = 6

    # --- original-ish fixed params ---
    grid_trigger_pct = 2.0
    grid_shift_pct = 0.4
    live_candles = 200
    stake_to_wallet_ratio = 0.75

    debug = True

    # --- position adjustment / DCA ---
    position_adjustment_enable = True

    # Total number of buys including initial (1 piece + 2 pieces + 3 pieces ...)
    max_entries = IntParameter(6, 8, default=7, load=True, space="buy", optimize=True)

    # ROI / SL
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

    # Timeframe
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 60

    # Grid params
    grid_down_spacing_pct = DecimalParameter(
        3.0, 5.0, default=4.0, decimals=1, load=True, space="sell", optimize=False
    )
    grid_up_spacing_pct = DecimalParameter(
        1.4, 3.0, default=1.6, decimals=1, load=True, space="sell", optimize=True
    )

    # Volatility regime (candle-count based)
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

    plot_config = {
        "main_plot": {
            "grid_up": {"grid_up": {"color": "green"}},
            "grid_down": {"grid_down": {"color": "red"}},
        },
        "subplots": {
            "bot_state": {"bot_state": {"color": "yellow"}},
            "trade_allowed": {"trade_allowed": {"color": "white"}},
            "atr_pct": {"atr_pct": {"color": "blue"}},
        },
    }

    # storage dict for live state
    custom_info: Dict = {}

    # ---------------------------------------------------------------------
    # DO NOT compute anything here (per your requirement)
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
    def _compute_trade_allowed(self, df: DataFrame) -> DataFrame:
        """
        Vectorized gate:
        - detect "large move" over move_lookback_candles
        - require ATR% <= threshold for low_vol_candles consecutively
        - only within post_move_max_candles after last large move
        """
        out = df.copy()

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

        out["trade_allowed"] = (out["post_move"] & out["low_vol_for_period"]).fillna(False)
        return out

    # ---------------------------------------------------------------------
    def calculate_state(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if pair not in self.custom_info:
            # {DATESTAMP, GRID, BOT_STATE, LIVE_DATE, INIT_COUNT, AVG_PRICE, UNITS}
            self.custom_info[pair] = ["", 0.0, 0, "", self.live_candles, 0.0, 0.0]

        df = self._compute_trade_allowed(dataframe)

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

        Buy_1 = np.zeros((last_row + 1), dtype=int)
        Buy_2 = np.zeros((last_row + 1), dtype=int)
        Sell_1 = np.zeros((last_row + 1), dtype=int)

        Grid_up = np.full((last_row + 1), np.nan)
        Grid_down = np.full((last_row + 1), np.nan)
        Bot_state = np.full((last_row + 1), np.nan)

        Close = df["close"].values
        Allowed = df["trade_allowed"].values

        need_recalc = True

        while row <= last_row:
            close = float(Close[row])
            allowed = bool(Allowed[row])

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

            if bot_state == 0:
                if close > grid_trigger_up:
                    new_grid = close
                    new_units = 0.0
                    new_avg_price = close
                    need_recalc = True

                if allowed and close <= grid_trigger_down:
                    new_bot_state = 1
                    Buy_1[row] = 1
                    new_units = 1.0 / close
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

            if 1 <= bot_state <= max_adds:
                if close > grid_up:
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0.0
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

                if allowed and close <= grid_down:
                    new_bot_state = bot_state + 1
                    Buy_2[row] = 1

                    add_piece = float(bot_state + 1)
                    total_stake = (float(new_bot_state) * (float(new_bot_state) + 1.0)) / 2.0

                    new_units = units + (add_piece / close)
                    new_avg_price = total_stake / new_units
                    new_grid = close
                    need_recalc = True

            if bot_state == (max_adds + 1):
                if (close > grid_up) or (close <= grid_down):
                    new_bot_state = 0
                    Sell_1[row] = 1
                    new_units = 0.0
                    new_avg_price = close
                    new_grid = close
                    need_recalc = True

            bot_state, grid, units, avg_price = new_bot_state, new_grid, new_units, new_avg_price

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

        if self.custom_info[trade.pair][self.DATESTAMP] != last_candle["date"]:
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle["date"]

            if int(last_candle.get("buy_2", 0)) == 1:
                filled_buys = trade.select_filled_orders("buy")
                count_of_buys = len(filled_buys)

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

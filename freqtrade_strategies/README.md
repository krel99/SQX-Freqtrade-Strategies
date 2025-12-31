# Freqtrade Strategies - Converted from StrategyQuantX

This directory contains 4 crypto trading strategies converted from StrategyQuantX pseudocode to freqtrade-compatible Python format. These strategies were originally backtested on BTCUSDT (Binance) from 2017.08.17 to 2025.12.29.

## Strategies Overview

### Strategy_0_2369
- **Timeframe**: 15m (with 1h informative pairs)
- **Key Indicators**: SMMA, SMA, ADX/DI, VWAP, SuperTrend, Ichimoku
- **Entry Logic**: Fuzzy logic with 52% threshold (3 out of 5 conditions)
- **Exit Logic**: Fuzzy logic with 51% threshold (3 out of 6 conditions)
- **Stop Loss**: 8.3%
- **Features**: Complex multi-timeframe analysis with comprehensive exit conditions

### Strategy_0_2422
- **Timeframe**: 15m (with 1h informative pairs)
- **Key Indicators**: TEMA, Parabolic SAR, ADX/DI, VWAP, LWMA
- **Entry Logic**: Fuzzy logic with 58% threshold (3 out of 6 conditions)
- **Exit Logic**: No specific exit signals (relies on stop loss and profit targets)
- **Stop Loss**: 8.6%
- **Features**: Simple entry-focused strategy with clear trend following

### Strategy_0_4501
- **Timeframe**: 15m (with 1h informative pairs)
- **Key Indicators**: HeikenAshi, MTATR, Williams %R, Bollinger Bands, Session indicators
- **Entry Logic**: Fuzzy logic with 42% threshold (3 out of 8 conditions)
- **Exit Logic**: Fuzzy logic with 80% threshold (4 out of 5 conditions)
- **Stop Loss**: 2.0%
- **Features**: Session-based analysis with tight stop loss

### Strategy_0_4536
- **Timeframe**: 15m (with 1h informative pairs)
- **Key Indicators**: MTATR, Vortex, VWAP, Session ranges
- **Entry Logic**: Fuzzy logic with 42% threshold (2 out of 5 conditions)
- **Exit Logic**: Fuzzy logic with 69% threshold (4 out of 6 conditions)
- **Stop Loss**: 8.9%
- **Features**: Range-based trading with multiple MTATR periods

## Installation

### Prerequisites

1. Install freqtrade:
```bash
# Using pip
pip install freqtrade

# Or using docker
docker pull freqtradeorg/freqtrade:stable
```

2. Install required dependencies:
```bash
pip install pandas pandas-ta numpy talib
```

### Setup

1. Copy the strategies to your freqtrade user_data directory:
```bash
cp -r user_data/strategies/* /path/to/freqtrade/user_data/strategies/
```

2. Create a configuration file for freqtrade:
```json
{
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "timeframe": "15m",
    "dry_run": true,
    "dry_run_wallet": 10000,
    "cancel_open_orders_on_exit": false,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": "your_api_key",
        "secret": "your_api_secret",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": [
            "BTC/USDT:USDT"
        ],
        "pair_blacklist": []
    },
    "pairlists": [
        {
            "method": "StaticPairList"
        }
    ],
    "edge": {
        "enabled": false
    },
    "telegram": {
        "enabled": false
    },
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "somethingrandom",
        "CORS_origins": [],
        "username": "freqtrader",
        "password": "freqtrader"
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "force_entry_enable": false,
    "internals": {
        "process_throttle_secs": 5
    }
}
```

## Usage

### Running a Strategy

```bash
# Backtesting
freqtrade backtesting --strategy Strategy_0_2369 --timeframe 15m

# Dry run (paper trading)
freqtrade trade --strategy Strategy_0_2369 --config config.json

# Live trading (use with caution!)
freqtrade trade --strategy Strategy_0_2369 --config config.json --db-url sqlite:///tradesv3.sqlite
```

### Hyperparameter Optimization

All strategies include hyperparameters that can be optimized using freqtrade's hyperopt:

```bash
# Optimize buy parameters
freqtrade hyperopt --strategy Strategy_0_2369 --hyperopt-loss SharpeHyperOptLoss --spaces buy -e 100

# Optimize sell parameters
freqtrade hyperopt --strategy Strategy_0_2369 --hyperopt-loss SharpeHyperOptLoss --spaces sell -e 100

# Optimize all parameters
freqtrade hyperopt --strategy Strategy_0_2369 --hyperopt-loss SharpeHyperOptLoss --spaces all -e 200
```

### Strategy Comparison

```bash
# Compare all strategies
freqtrade backtesting --strategy-list Strategy_0_2369 Strategy_0_2422 Strategy_0_4501 Strategy_0_4536 --timeframe 15m
```

## Important Notes

### Conversion Considerations

1. **Fuzzy Logic Implementation**: The original strategies use fuzzy logic with percentage thresholds. This has been faithfully implemented using Python conditions.

2. **Indicator Approximations**:
   - SMMA is approximated using EMA with double the period
   - MTATR is approximated using standard ATR
   - Session indicators are calculated based on hour/minute timestamps
   - Some MT4-specific indicators have been replaced with closest equivalents

3. **Limit Orders**: The strategies use limit orders for entry with specific price calculations based on ATR and moving averages.

4. **Exit Mechanisms**:
   - Stop loss percentage
   - Profit target based on ATR or percentage
   - Exit after N bars
   - Fuzzy logic exit conditions

### Risk Warning

⚠️ **IMPORTANT**: These strategies are provided for educational purposes only. Past performance does not guarantee future results. Always:
- Test thoroughly on demo accounts before live trading
- Start with small positions
- Never invest more than you can afford to lose
- Consider transaction costs and slippage
- Monitor and adjust strategies based on current market conditions

### Optimization Tips

1. **Start with default parameters**: The provided defaults are from the original backtest
2. **Optimize one strategy at a time**: Focus on understanding each strategy's behavior
3. **Use appropriate loss functions**: 
   - SharpeHyperOptLoss for risk-adjusted returns
   - SortinoHyperOptLoss for downside risk focus
   - ProfitDrawDownHyperOptLoss for balanced optimization
4. **Consider market conditions**: These strategies were tested on historical data; current market conditions may differ
5. **Paper trade first**: Always validate optimized parameters with paper trading before going live

## GeneTrader Integration

These strategies can be further optimized using GeneTrader (genetic algorithm optimization):

1. Export strategy parameters to GeneTrader format
2. Run genetic optimization
3. Import optimized parameters back to freqtrade

For GeneTrader integration, ensure your strategies follow the required format with clearly defined hyperparameters and fitness functions.

## Support and Contributions

For issues, questions, or improvements:
1. Check freqtrade documentation: https://www.freqtrade.io/
2. Review the original pseudocode in the `pseudocode` directory
3. Test modifications thoroughly before deployment

## License

These strategies are provided as-is without any warranty. Use at your own risk.

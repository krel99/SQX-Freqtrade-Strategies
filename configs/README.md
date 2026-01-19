# Freqtrade Configuration Files

This document provides an overview of the purpose-built configuration files for testing Freqtrade strategies.

## Naming Convention

The configuration files follow a naming convention of `config_futures_DDMMYYYY-f-<type>.json`, where:

- `DDMMYYYY`: The date the configuration was created.
- `f`: Indicates that the configuration uses the feather data format.
- `<type>`: A short code to describe the purpose of the configuration.

## Configurations

### `config_futures_23072022-f-long.json`

- **Purpose**: This configuration is designed for testing long-only trading strategies.
- **Key Settings**:
    - `strategy_direction`: `"long"`
    - `stake_amount`: `100` (fixed stake)
    - `slippage_percent`: `0.05` (higher slippage)

### `config_futures_23072022-f-short.json`

- **Purpose**: This configuration is designed for testing short-only trading strategies.
- **Key Settings**:
    - `strategy_direction`: `"short"`
    - `stake_amount`: `100` (fixed stake)
    - `slippage_percent`: `0.05` (higher slippage)

### `config_futures_23072022-f-lev.json`

- **Purpose**: This configuration is designed for testing strategies with leverage.
- **Key Settings**:
    - `leverage`: `3.0`
    - `stake_amount`: `100` (fixed stake)
    - `slippage_percent`: `0.05` (higher slippage)

### `config_futures_23072022-f-pairs.json`

- **Purpose**: This configuration is designed for testing strategies with a pair list updated to include popular pairs from 2022.
- **Key Settings**:
    - `pair_whitelist`: Contains a list of 10 popular cryptocurrency pairs.
    - `stake_amount`: `100` (fixed stake)
    - `slippage_percent`: `0.05` (higher slippage)

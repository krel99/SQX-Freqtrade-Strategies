# ForexDog Strategy Pseudocode

This document outlines the logic for three variations of the ForexDog strategy. The strategy is based on a series of moving averages to identify trends and potential entry points.

## Common Elements for all Variations:

### Indicators:
- A series of 12 exponential moving averages (EMAs) with configurable periods. The default periods are 5, 20, 40, 50, 80, 100, 200, 400, 640, 1600, 1920, 3200.
- Each period will be optimizable within a range of +/- 30% of the default value, ensuring the ranges do not overlap.
- Average True Range (ATR) for stop-loss calculation.

### Exit Conditions:
- Take profit when the price reaches the next slower moving average.
- A time-based stop-loss will exit a losing trade after a certain number of candles.

## Variation 1: ForexDogV1 - Basic Crossover

### Entry Condition:
1. The price is above the first 6 EMAs (ema_5 to ema_100).
2. The close price has just crossed above the `ema_20`.
3. The distance between the current price and the next slow EMA (`ema_200`) is greater than a certain percentage (e.g., 1%) of the current price.
4. The distance between `ema_200` and `ema_400` is also significant, indicating a non-consolidating market.

### Exit Condition:
1. Take profit if the price touches or crosses above `ema_200`.
2. Stop-loss is placed at a multiple of ATR below the `ema_40`.
3. If the trade is losing after `X` candles, exit the trade.

## Variation 2: ForexDogV2 - Momentum Confirmation

### Entry Condition:
1. The price is above the first 8 EMAs (ema_5 to ema_400).
2. The close price has just crossed above the `ema_40`.
3. The distance to the next slow EMA (`ema_640`) is greater than a certain percentage.
4. A momentum indicator (e.g., RSI or MACD) confirms the upward trend (e.g., RSI > 60).
5. The faster EMAs are ordered correctly (e.g., ema_5 > ema_20 > ema_40), indicating strong momentum.

### Exit Condition:
1. Take profit if the price touches or crosses above `ema_640`.
2. Stop-loss is placed at a multiple of ATR below the `ema_80`.
3. If the trade is losing after `Y` candles, exit the trade.

## Variation 3: ForexDogV3 - Volatility-Adapted Entry

### Entry Condition:
1. The price is above the first 10 EMAs (ema_5 to ema_1600).
2. The close price has just crossed above the `ema_80`.
3. The distance to the next slow EMA (`ema_1920`) is greater than a certain percentage.
4. The ATR is above a certain threshold, indicating sufficient volatility for a trade.
5. The volume is above its moving average, confirming the strength of the move.

### Exit Condition:
1. Take profit if the price touches or crosses above `ema_1920`.
2. Stop-loss is a trailing stop-loss based on ATR, starting below `ema_100`.
3. If the trade is losing after `Z` candles, exit the trade.

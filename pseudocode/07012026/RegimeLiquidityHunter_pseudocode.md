# RegimeLiquidityHunter Strategy Pseudocode

## Strategy Overview
A regime-switching strategy that identifies whether the market is trending or ranging, then hunts for liquidity pools where stop losses cluster. Uses statistical analysis to find high-probability reversals at these liquidity levels.

## Core Concepts

### 1. Market Regime Detection
- Uses ADX to determine trend strength
- Classifies market as: Trending Up (+1), Ranging (0), or Trending Down (-1)
- Applies smoothing to avoid regime whipsaws

### 2. Liquidity Pool Hunting
- Identifies where stop losses likely cluster (above recent highs, below recent lows)
- Detects stop hunts when price briefly exceeds these levels then reverses
- Uses volume spikes to confirm liquidity grabs

### 3. Statistical Analysis
- Z-score to identify statistical extremes
- Percentile rank to gauge position within historical range
- Different thresholds for different market regimes

### 4. Adaptive Behavior
- Different entry logic for trending vs ranging markets
- Dynamic leverage based on volatility
- Adjustable stops based on market conditions

## Indicator Calculations

### Regime Detection
```
FUNCTION detect_regime(candles):
    // Calculate trend strength
    adx = CALCULATE_ADX(candles, regime_lookback)
    
    // Determine if trending
    is_trending = adx > (25 * trend_threshold)
    
    // Get trend direction
    ema_fast = EMA(close, trend_fast_period)
    ema_slow = EMA(close, trend_slow_period)
    
    regime = 0  // Default to ranging
    IF is_trending:
        IF ema_fast > ema_slow:
            regime = 1   // Uptrend
        ELSE:
            regime = -1  // Downtrend
    
    // Smooth to avoid whipsaws
    regime_smooth = MEDIAN(regime, regime_switch_delay)
    
    RETURN regime_smooth
```

### Liquidity Pool Detection
```
FUNCTION detect_liquidity_pools(candles):
    // Identify volume spikes (potential liquidity)
    volume_ma = MOVING_AVERAGE(volume, 20)
    volume_spike = volume / volume_ma
    
    // Find high volume price levels
    high_volume_levels = []
    FOR each candle:
        IF volume_spike > volume_spike_threshold:
            high_volume_levels.APPEND(close_price)
    
    // Identify stop loss clusters
    recent_high = MAX(high, liquidity_lookback)
    recent_low = MIN(low, liquidity_lookback)
    
    // Detect stop hunts
    stop_hunt_long = (low < recent_low * 1.002) AND (close > recent_low)
    stop_hunt_short = (high > recent_high * 0.998) AND (close < recent_high)
    
    // Calculate liquidity score
    liquidity_score = (volume_percentile * 0.3 +
                      (1 - distance_from_high_volume) * 0.3 +
                      volume_spike_normalized * 0.4)
    
    RETURN liquidity_indicators
```

### Statistical Indicators
```
FUNCTION calculate_statistics(candles, period):
    // Z-score of price
    price_mean = MEAN(close, period)
    price_std = STANDARD_DEVIATION(close, period)
    price_zscore = (close - price_mean) / price_std
    
    // Percentile rank
    price_percentile = PERCENTILE_RANK(close, lookback)
    volume_percentile = PERCENTILE_RANK(volume, lookback)
    
    // Distance from mean
    sma = SIMPLE_MOVING_AVERAGE(close, mean_period)
    mean_distance = (close - sma) / sma
    
    RETURN statistical_indicators
```

## Entry Logic

### Protection Filters
```
FUNCTION check_protection_filters():
    // Volatility protection
    IF volatility >= atr_pct * max_volatility_enter:
        RETURN FALSE
    
    // Liquidity protection
    IF liquidity_score < min_liquidity_score:
        RETURN FALSE
    
    // Regime confidence
    IF regime_stability < min_regime_confidence:
        RETURN FALSE
    
    // Drawdown protection
    IF current_drawdown > max_drawdown_percent:
        RETURN FALSE
    
    // Trade limit protection
    IF consecutive_losses >= max_consecutive_losses:
        RETURN FALSE
    
    // Cooldown protection
    IF candles_since_last_trade < min_time_between_trades:
        RETURN FALSE
    
    // Daily limit
    IF trades_today >= max_trades_per_day:
        RETURN FALSE
    
    // Trade quality check
    IF trade_quality < 0.5:
        RETURN FALSE
    
    RETURN TRUE
```

### Long Entry - Ranging Market
```
FUNCTION check_ranging_long_entry():
    IF NOT check_protection_filters():
        RETURN NO_ENTRY
    
    IF regime_smooth == 0:  // Ranging market
        IF price_zscore < zscore_entry_long AND      // Statistically oversold
           price_percentile < 0.2 AND                 // Bottom 20% of range
           rsi < 35 AND                               // Momentum oversold
           mean_distance < -0.02 * reversion_strength AND  // Below mean
           liquidity_score > 0.5:                     // Good liquidity
            RETURN ENTER_LONG
```

### Long Entry - Trending Market
```
FUNCTION check_trending_long_entry():
    IF NOT check_protection_filters():
        RETURN NO_ENTRY
    
    IF regime_smooth == 1:  // Uptrend
        IF price_zscore > -0.5 AND price_zscore < 1.0 AND  // Not overextended
           rsi > 40 AND rsi < 70 AND                       // Healthy momentum
           close > keltner_lower AND                       // Above support
           keltner_position < 0.7 AND                      // Room to move up
           volume_percentile > 0.4:                        // Decent volume
            RETURN ENTER_LONG
```

### Long Entry - Liquidity Hunt
```
FUNCTION check_liquidity_hunt_long():
    IF NOT check_protection_filters():
        RETURN NO_ENTRY
    
    IF stop_hunt_long == TRUE:  // Stop hunt detected
        IF volume_spike > threshold AND  // High volume confirms
           close > open AND              // Bullish recovery
           trade_quality > 0.5:          // Minimum quality score
            RETURN ENTER_LONG
```

### Short Entry - Ranging Market
```
FUNCTION check_ranging_short_entry():
    IF regime_smooth == 0:  // Ranging market
        IF price_zscore > zscore_entry_short AND      // Statistically overbought
           price_percentile > 0.8 AND                  // Top 20% of range
           rsi > 65 AND                                // Momentum overbought
           mean_distance > 0.02 * reversion_strength AND  // Above mean
           liquidity_score > 0.5:                      // Good liquidity
            RETURN ENTER_SHORT
```

### Short Entry - Trending Market
```
FUNCTION check_trending_short_entry():
    IF regime_smooth == -1:  // Downtrend
        IF price_zscore < 0.5 AND price_zscore > -1.0 AND  // Not overextended
           rsi < 60 AND rsi > 30 AND                       // Healthy momentum
           close < keltner_upper AND                       // Below resistance
           keltner_position > 0.3 AND                      // Room to move down
           volume_percentile > 0.4:                        // Decent volume
            RETURN ENTER_SHORT
```

## Exit Logic

### Statistical Exit
```
FUNCTION check_statistical_exit(trade):
    IF trade.is_long:
        IF price_zscore > 1.5 OR              // Statistically overbought
           price_percentile > 0.9:             // Top of range
            RETURN EXIT_LONG
    
    IF trade.is_short:
        IF price_zscore < -1.5 OR             // Statistically oversold
           price_percentile < 0.1:             // Bottom of range
            RETURN EXIT_SHORT
```

### Regime Change Exit
```
FUNCTION check_regime_exit(trade):
    current_regime = regime_smooth
    previous_regime = regime_smooth[5_bars_ago]
    
    IF trade.is_long:
        IF current_regime == -1 AND previous_regime >= 0:
            RETURN EXIT_LONG  // Switched to downtrend
    
    IF trade.is_short:
        IF current_regime == 1 AND previous_regime <= 0:
            RETURN EXIT_SHORT  // Switched to uptrend
```

### Target Exit
```
FUNCTION check_target_exit(trade):
    IF trade.is_long:
        IF mean_distance > 0.02 * take_profit_multiplier:
            RETURN EXIT_LONG
    
    IF trade.is_short:
        IF mean_distance < -0.02 * take_profit_multiplier:
            RETURN EXIT_SHORT
```

## Custom Exit Logic
```
FUNCTION custom_exit(trade, profit, indicators):
    // Update consecutive losses tracker
    IF profit < -0.02:
        consecutive_losses += 1
    ELSE IF profit > 0.01:
        consecutive_losses = 0
    
    // Regime change exit
    IF trade.is_short AND regime_smooth > 0.5:
        RETURN EXIT_REGIME_CHANGE
    IF trade.is_long AND regime_smooth < -0.5:
        RETURN EXIT_REGIME_CHANGE
    
    // Statistical extreme exit
    IF abs(price_zscore) > 3.0 AND profit > 0:
        RETURN EXIT_STATISTICAL_EXTREME
    
    // Low liquidity exit
    IF liquidity_score < 0.2 AND profit > 0.005:
        RETURN EXIT_LOW_LIQUIDITY
    
    // Stop hunt in opposite direction
    IF trade.is_short AND stop_hunt_long:
        RETURN EXIT_LIQUIDITY_GRAB
    IF trade.is_long AND stop_hunt_short:
        RETURN EXIT_LIQUIDITY_GRAB
    
    // Emergency volatility exit
    IF volatility > atr_pct * 4 AND profit > -0.01:
        RETURN EXIT_EXTREME_VOLATILITY
    
    // Maximum drawdown exit
    IF current_drawdown > max_drawdown_percent * 1.5:
        RETURN EXIT_MAX_DRAWDOWN
    
    // Trade quality deterioration exit
    IF trade_quality < 0.2 AND profit > -0.005:
        RETURN EXIT_LOW_QUALITY
```

## Risk Management

### Dynamic Stop Loss
```
FUNCTION calculate_stoploss(profit, atr, regime):
    // Tighter stops after consecutive losses
    loss_multiplier = 1.0
    IF consecutive_losses >= reduce_position_after_losses:
        loss_multiplier = 0.5  // Halve stop distance
    
    // Base stop on ATR
    atr_stop = (atr * 2 * stop_loss_multiplier * loss_multiplier) / price
    
    // Adjust for regime
    IF abs(regime_smooth) > 0.5:  // Trending
        atr_stop = atr_stop * 1.5  // Wider stops
    ELSE:  // Ranging
        atr_stop = atr_stop * 0.75  // Tighter stops
    
    // Tighten with profit
    IF profit > 3%:
        RETURN -min(atr_stop, 1%)
    ELSE IF profit > 1.5%:
        RETURN -min(atr_stop, 2%)
    ELSE:
        RETURN -min(atr_stop, 4%)
```

### Adaptive Leverage
```
FUNCTION calculate_leverage(volatility, regime):
    // Base on volatility
    IF volatility < 1%:
        base_leverage = 3.0
    ELSE IF volatility < 2%:
        base_leverage = 2.0
    ELSE IF volatility < 3%:
        base_leverage = 1.5
    ELSE:
        base_leverage = 1.0
    
    // Adjust for regime
    IF abs(regime_smooth) < 0.5:  // Ranging
        base_leverage *= 1.2  // Higher leverage for mean reversion
    ELSE:  // Trending
        base_leverage *= 0.8  // Lower leverage for trends
    
    RETURN min(base_leverage * volatility_multiplier, max_leverage)
```

## Trade Quality Scoring
```
FUNCTION calculate_trade_quality():
    trade_quality = (liquidity_score * 0.3 +
                    regime_stability * 0.3 +
                    (1 - volatility/max_recent_volatility) * 0.4)
    RETURN CLAMP(trade_quality, 0, 1)
```

## Key Parameters

### Regime Detection
- `regime_lookback`: 20-60 (default: 40)
- `trend_threshold`: 0.5-1.5 (default: 1.0)
- `volatility_lookback`: 10-30 (default: 20)
- `regime_switch_delay`: 2-10 (default: 5)

### Liquidity Detection
- `liquidity_lookback`: 30-100 (default: 50)
- `liquidity_sensitivity`: 1.5-3.0 (default: 2.0)
- `volume_spike_threshold`: 1.5-3.0 (default: 2.0)

### Statistical Parameters
- `zscore_period`: 10-30 (default: 20)
- `zscore_entry_long`: -3.0 to -1.0 (default: -2.0)
- `zscore_entry_short`: 1.0-3.0 (default: 2.0)

### Mean Reversion
- `mean_period`: 20-50 (default: 30)
- `reversion_strength`: 0.5-2.0 (default: 1.0)

### Risk Management
- `volatility_multiplier`: 0.5-2.0 (default: 1.0)
- `take_profit_multiplier`: 1.0-3.0 (default: 1.5)
- `stop_loss_multiplier`: 0.5-1.5 (default: 1.0)

### Protection Parameters
- `max_consecutive_losses`: 2-5 (default: 3)
- `cooldown_after_stoploss`: 0-20 candles (default: 10)
- `min_time_between_trades`: 0-10 candles (default: 3)
- `max_trades_per_day`: 3-10 (default: 5)
- `max_volatility_enter`: 2.0-5.0 (default: 3.0)
- `min_liquidity_score`: 0.2-0.6 (default: 0.3)
- `max_drawdown_percent`: 5.0-15.0 (default: 10.0)
- `reduce_position_after_losses`: 1-3 (default: 2)
- `min_regime_confidence`: 0.5-0.9 (default: 0.7)

## Protection Methods
```
FUNCTION define_protections():
    protections = [
        StoplossGuard(lookback=48, limit=max_consecutive_losses),
        MaxDrawdown(lookback=200, max=max_drawdown_percent),
        LowProfitPairs(lookback=360, min_profit=-5%),
        CooldownPeriod(duration=min_time_between_trades)
    ]
    RETURN protections
```

## Strategy Strengths
1. **Adaptive**: Switches behavior based on market conditions
2. **Statistical Edge**: Uses z-scores and percentiles for objective entries
3. **Liquidity Focused**: Hunts for stop loss clusters
4. **Risk Aware**: Dynamic stops and leverage based on volatility
5. **Regime Aware**: Different logic for trending vs ranging markets
6. **Protection Heavy**: Multiple layers of drawdown and loss protection
7. **Quality Filtering**: Trade quality scoring prevents low-probability entries
8. **Cooldown Logic**: Prevents overtrading after losses

## Best Market Conditions
- Works well in both trending and ranging markets
- Excels when liquidity pools are clear (high volume markets)
- Best in futures markets with good volume
- Performs well during regular trading hours
- Adapts to changing volatility conditions

## Potential Weaknesses
- May struggle in extremely choppy, directionless markets
- Regime detection can lag during transitions
- Requires sufficient historical data for statistics
- Stop hunts may be less effective in low-volume markets
- Parameter optimization needed for different assets

## Unique Features vs Other Strategies
- **Regime Switching**: Unlike fixed strategies, adapts to market state
- **Liquidity Hunting**: Specifically targets stop loss clusters
- **Statistical Foundation**: Uses z-scores instead of just technical indicators
- **Multi-Mode Entry**: Different logic for different market conditions
- **Volatility Adaptation**: Dynamically adjusts risk based on market volatility

## Testing Recommendations
1. Test regime detection accuracy separately
2. Optimize statistical thresholds for specific markets
3. Validate stop hunt detection with actual volume data
4. Backtest across different market conditions (trending/ranging)
5. Monitor regime switching frequency to avoid overtrading
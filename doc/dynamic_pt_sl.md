# Dynamic PT and SL Adjustment Logic
## Case 1: Price Trend is Upward
1) If the position is a BUY, the current price trend is upward (i.e., ClosePrice(N) - PositionPrice > 0), and the action signal is also BUY:

### Increase PT: Adjust PT dynamically to allow for more profit. 

**For example:**

$$𝑃𝑇=max(𝑃𝑇,𝐶𝑙𝑜𝑠𝑒𝑃𝑟𝑖𝑐𝑒(𝑁)+𝐾⋅𝐴𝑇𝑅)$$


**ATR** (Average True Range) is used to account for market volatility.

**K** is a multiplier (e.g., 1.5 to 2.0) that ensures PT is extended proportionally to the trend's strength.


### Tighten SL: Adjust SL to protect profits:

$$𝑆𝐿=max⁡(𝑆𝐿,𝐶𝑙𝑜𝑠𝑒𝑃𝑟𝑖𝑐𝑒(𝑁)−𝛼⋅𝐴𝑇𝑅)$$

**\alpha** is a smaller multiplier (e.g., 0.5) to keep SL closer to the current price.

## Case 2: Price Trend is Downward
If the position is a BUY, but the price trend is downward (i.e., ClosePrice(N) - PositionPrice < 0):
### Reduce PT: 
Lower the PT to lock in any potential recovery:
$$𝑃
𝑇
=
min
⁡
(
𝑃
𝑇
,
𝐶
𝑙
𝑜
𝑠
𝑒
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑁
)
+
𝛽
⋅
𝐴
𝑇
𝑅
)$$

**\beta** is a smaller multiplier (e.g., 0.8).

### Widen SL: 
Increase SL to provide more room for price fluctuations, minimizing the risk of premature stop-out:
$$𝑆
𝐿
=
min
⁡
(
𝑆
𝐿
,
𝐶
𝑙
𝑜
𝑠
𝑒
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑁
)
−
𝛾
⋅
𝐴
𝑇
𝑅
)$$

SL=min(SL,ClosePrice(N)−γ⋅ATR)
**\gamma** is larger than \alpha (e.g., 1.0 to 1.5).
### Case 3: Position Reversal
If the action signal indicates SELL while holding a BUY position:
Set Immediate SL: Tighten the SL to a point just below the current price to minimize loss or lock in small profits:
$$𝑆
𝐿
=
𝐶
𝑙
𝑜
𝑠
𝑒
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑁
)
−
𝛿
⋅
𝐴
𝑇
𝑅$$
SL=ClosePrice(N)−δ⋅ATR
**\delta** is a very small multiplier (e.g., 0.2).
Reset PT: Adjust PT to a conservative level near the current price to lock in gains or exit at minimal loss.

## Implementation Outline
Here is a Python pseudocode implementation using these principles:

```python
def optimize_pt_sl(position, close_price, position_price, atr, action_signal, pt, sl):
    K, alpha, beta, gamma, delta = 1.5, 0.5, 0.8, 1.5, 0.2  # Dynamic tuning factors

    if position == "BUY":
        price_trend = close_price - position_price

        if price_trend > 0:  # Upward trend
            if action_signal == "BUY":
                # Increase PT and tighten SL
                pt = max(pt, close_price + K * atr)
                sl = max(sl, close_price - alpha * atr)
            else:  # Action signal is SELL
                # Tighten SL and reset PT
                sl = close_price - delta * atr
                pt = close_price + beta * atr
        else:  # Downward trend
            # Reduce PT and widen SL
            pt = min(pt, close_price + beta * atr)
            sl = min(sl, close_price - gamma * atr)

    elif position == "SELL":
        # Similar logic for SELL positions (inverse of BUY)
        price_trend = position_price - close_price

        if price_trend > 0:  # Downward trend
            if action_signal == "SELL":
                pt = max(pt, close_price - K * atr)
                sl = max(sl, close_price + alpha * atr)
            else:  # Action signal is BUY
                sl = close_price + delta * atr
                pt = close_price - beta * atr
        else:  # Upward trend
            pt = min(pt, close_price - beta * atr)
            sl = min(sl, close_price + gamma * atr)

    return pt, sl
```
### Advantages of the Method
Volatility-Aware: By using ATR, the adjustments are proportional to market conditions.
Trend Following: Encourages riding profitable trends while cutting losses early.

Dynamic Flexibility: Adapts to both positive and negative price movements and market signals.
### Example Usage
Assume:

Position: BUY

ClosePrice(N): 0.7520

PositionPrice: 0.7500

ATR: 0.0015 (150 pips)

Action Signal: BUY

Initial PT: 50 pips (0.7550)

Initial SL: -20 pips (0.7480)

```python
pt, sl = optimize_pt_sl(
    position="BUY",
    close_price=0.7520,
    position_price=0.7500,
    atr=0.0015,
    action_signal="BUY",
    pt=0.7550,
    sl=0.7480
)
print(f"Optimized PT: {pt}, Optimized SL: {sl}")
```

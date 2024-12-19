## Make a Forex Trading App based on LLM for day trader

## fundamental principles

### Bars
The trading price over a period of time can be condensed into a bar (such as the commonly used candlestick). In this context, a bar represents this concept. A bar should contain information about both the random fluctuations in price and certain patterns (like trends). Therefore, we need a vector to map the information contained in the bar, and we can perform a bar-to-vector transformation (bar2vec).

Candlesticks compress price changes over a period of time, providing only four basic representations of the price during that time period (open, close, high, low). It does not include the distribution state of the price within that time frame. We can consider adding an indicator value for the normal distribution of the bar during the pre-processing stage, which reflects the shape distribution of the price over time. Although this information may not be particularly useful for graphical analysis, it could be very valuable for machine learning.

A series of bars over a period of time (regardless of the time unit) represents the price changes during that period, essentially forming a language of price changes (LPC). However, unlike NLP, not every bar in the sequence is meaningful. A large portion of bars do not represent price trends, or their patterns are completely drowned out by noise. Particularly, when the time unit of the bar is smaller, the random fluctuations (i.e., white noise) component becomes larger. As we all know, noise is random and unpredictable. Therefore, in smaller time units, only strong and clear price trend signals are meaningful. This is why models based on 5-minute intervals may perform better than 1-minute models because the price trend representations are more prominent in the former. Even so, in the feature space of price changes, the effect of mapping remains a long-tail distribution. Most bars have feature space mapping values that are drowned out by noise and are unpredictable. Only the "head" of the long tail can provide support for price predictions. This is why we consider using Probability Sparse Self-Attention for encoding and decoding.

### Trading Market
From the perspective of the market, what drives price changes, and why do prices fluctuate? Price theory suggests:

1. Market behavior contains all information.
2. Prices move along a trend.
3. History repeats itself.
If price theory is correct, then based on points 2 and 3, prices move along a trend, which can be understood as prices having a trend over a given period of time, making them predictable. Market behavior can also be viewed as price fluctuations. According to point 1, the factors driving price changes include the following key elements:

1. The judgment of buyers and sellers on market prices (banks, institutions, market makers).
2. News (interest rate changes, economic outlook, etc.), unexpected events.
3. Seasonal and large economic cycles.
   
A bar sequence only reflects price changes, without including the driving forces behind these changes. Moreover, one of the most important drivers of price changes (point 1) lacks clear, quantifiable information that can be incorporated. This highlights that our predictions are based on incomplete and asymmetric information. To improve the reliability of predictions, we should aim to include as much of the information from points 2 and 3 in historical and future forecast data. Specifically, more information should be added to the time dimension, such as (hour, day, day of the week, day of the month, month, year, which market, news).

Transformers have indeed shown remarkable progress in both NLP (BERT, GPT) and CV (ViT, etc.), and despite the models getting larger (reaching terabytes of parameters), overfitting has not been a major concern, indicating the robustness of the transformer architecture.

### Applying Transformers to the Trading Market
It is indeed possible to apply transformers to the trading market. Trading data, such as historical data from stocks, forex, and cryptocurrency markets, provides a large and diverse dataset that could be used to train such models. The volume and variety of this data make it well-suited for transformer-based models, which thrive on large-scale datasets.

Using transformers in the trading domain has several advantages:

Sufficient Data Availability: Markets generate huge amounts of data every day, providing the necessary scale for training large models. This data includes price movements, volumes, trends, and various market indicators, all of which are structured in time-series format, similar to the sequential nature of text data in NLP.

Feature Representation: Transformers are highly effective in learning long-range dependencies, which can be critical in financial markets. Market trends often extend over varying time frames (from short-term to long-term), and transformers can learn these dependencies well, compared to traditional models like LSTMs, which may struggle with longer sequences.

Fine-tuning for Specific Assets or Pairs: Once a general model has been trained using large datasets, it can be fine-tuned on specific assets, trading pairs, or time periods. This would allow the model to adapt to the characteristics of a specific stock or trading pair, providing more accurate predictions and perhaps outperforming other traditional methods such as LSTM-based models, technical analysis, or fundamental analysis.

Handling Noisy Data: Financial markets have a significant amount of noise (random fluctuations in prices), making prediction a challenge. The long-tail distribution of bar data and the noise within small time units make it crucial to focus on meaningful trend data. Transformers, with their attention mechanisms, can potentially identify important signals while ignoring irrelevant noise.

### Model Considerations
When applying transformers to trading markets, some specific considerations should be made:

Self-Attention and the Long-Tail Distribution: In the trading market, only a small portion of the data (the "head" of the long tail) contains meaningful information for predictions, while much of the data is noise. A multi-head probabilistic sparse self-attention (probSparse) could be used to focus on the most important parts of the feature map. This can reduce memory and computation costs by only attending to significant trends and patterns, which would be particularly beneficial when dealing with long sequences of trading data.

Encoding Options: The original transformer encoding mechanism may work well in the trading domain, but adjustments like probSparse self-attention could optimize it further. This method could potentially allow the model to capture the key signals in trading data more efficiently by ignoring random fluctuations.

### Expected Performance
Given these factors, it's reasonable to expect that a transformer-based approach could outperform other methods like LSTM or traditional technical analysis in market prediction tasks, particularly for long-term dependencies and complex relationships between data points. By fine-tuning the model for specific assets and time periods, the predictive performance could be further enhanced, making transformers a promising tool for financial market analysis and trading strategies.

Overall, the application of transformers to trading markets represents an exciting opportunity to leverage their robustness and scalability, provided that the models are carefully designed to handle the unique challenges of market data.

### Input Pre-processing: Bar Preprocessing Strategy
In this approach, we treat the length of bars (measured in days) as equivalent to sentence lengths in NLP tasks. The goal is to avoid sequences that are too long, as this would exponentially increase the complexity (dimensionality), similar to how sentence length affects NLP models. Using natural day boundaries for segmentation is an ideal choice.

### Maximum Number of Bars in One Day (Granularity of Time)
1 min bars: Minimal information loss, but high noise, resulting in 1,440 bars per day (60 bars/hour × 24 hours). The sequence length is large.
5 min bars: Minimal information loss with reasonable noise levels, resulting in 288 bars per day. The sequence length is reasonable, making it a good compromise.
15 min bars: Greater information loss, reasonable noise, with 96 bars per day. Sequence length is still manageable.
1 hour bars: Significant information loss, low noise, with only 24 bars per day. The sequence length is small, but important details might be lost.
Conclusion: 5-minute bars provide a good balance between minimal information loss, lower noise levels, and a manageable sequence length.

Bar Encoding Design (Under Improvement)
The design of the bar encoding is evolving to better capture the patterns within the data. A relative price representation is employed to make the encoded bar information more generalized across different assets. This approach improves the model's ability to learn from price changes without being tied to absolute price values.

Encoding Example:

plaintext
Copy code
[stock_name][r_bar[0]]...[r_bar[n]]
Relative Price Encoding:
Each bar is represented by the relative values of open, high, low, and close prices, based on the opening price.

𝑟
_
𝑏
𝑎
𝑟
[
𝑜
𝑝
𝑒
𝑛
,
ℎ
𝑖
𝑔
ℎ
,
𝑙
𝑜
𝑤
,
𝑐
𝑙
𝑜
𝑠
𝑒
]
[
𝑛
]
=
𝑏
𝑎
𝑟
[
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
,
ℎ
𝑖
𝑔
ℎ
−
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
,
𝑙
𝑜
𝑤
−
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
,
𝑐
𝑙
𝑜
𝑠
𝑒
−
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
]
[
𝑛
]
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
r_bar[open,high,low,close][n]= 
open_price
bar[open_price,high−open_price,low−open_price,close−open_price][n]
​
 
This creates a more universal representation that is independent of absolute price values, allowing the model to focus on price patterns rather than magnitude.

Further Pre-processing of Bars
De-noising:
Removing or filtering out noise from the data to ensure that the model captures meaningful trends and signals, rather than random fluctuations.

Normalization:
Normalizing the data to a consistent range, ensuring that price values are scaled uniformly across different assets and time periods.

Similarity Calculation:
Identifying similarities between bars to detect recurring patterns or trends that may be indicative of future price movements.

Tokenization:
Converting bars into tokens, effectively creating a vocabulary that is independent of absolute prices. This bar vocabulary (similar to word tokens in NLP) will allow the model to focus on patterns and trends.

Outcome:
Through the above steps, the model creates a bar vocabulary that is free from the biases introduced by absolute prices. The relative and de-noised representation provides a compact set of tokens that can be used in modeling price patterns.

Absolute Price Representation
In contrast to the relative encoding, the absolute price representation of each bar can still be maintained and normalized. For instance:

𝑏
𝑎
𝑟
[
𝑛
]
=
𝑛
𝑜
𝑟
𝑚
(
𝑏
𝑎
𝑟
[
𝑛
]
[
𝑜
𝑝
𝑒
𝑛
_
𝑝
𝑟
𝑖
𝑐
𝑒
]
,
𝑑
𝑎
𝑖
𝑙
𝑦
[
ℎ
𝑖
𝑔
ℎ
,
𝑙
𝑜
𝑤
]
)
bar[n]=norm(bar[n][open_price],daily[high,low])
Positional Encoding for Bars
Local Time Stamp:
Each bar is associated with a local time stamp, ensuring that its position within a day is captured (e.g., 5-minute intervals). This can be represented through a uniform input representation, ensuring consistent time granularity (e.g., taking the 5-minute interval as the smallest unit).

Global Time Stamp (Learnable):
A global time stamp is also incorporated, which allows the model to understand the relative position of each bar across multiple days or longer periods.

By treating time events within a day as a limited vocabulary (using 5-minute intervals as the finest granularity), the model can effectively capture time-based patterns.

This strategy forms a robust foundation for applying transformers in trading, as it provides a compact, noise-filtered, and normalized representation of market data that can be efficiently processed by modern deep learning architectures.

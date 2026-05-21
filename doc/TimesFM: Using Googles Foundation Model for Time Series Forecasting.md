# TimesFM: Using Google’s Foundation Model for Time Series Forecasting
Siduo Jiang

Sep 17, 2025

Forecasting the future is a fundamental human endeavor. From predicting next quarter’s sales to anticipating disease outbreaks, our ability to look ahead is critical. For decades, this task has been the domain of statistical methods like ARIMA and, more recently, specialized deep learning models. However, these approaches often require significant expertise and must be trained or tuned for each new dataset.

What if we could have a single, powerful model that understands the language of time — the rhythms, seasons, and trends inherent in time series data — and could forecast for a brand-new dataset without any prior training on it?

That’s the promise of TimesFM (Time-series Foundation Model), a new decoder-only foundation model for forecasting from Google Research. This article will serve as your guide to understanding and using this groundbreaking model. We’ll dive into the theory, explore how it was built, and walk through practical code examples for forecasting, both with and without external factors.

## What is Time Series Forecasting?
At its core, time series forecasting is the practice of using a model to predict future values based on historical data. The data points are collected at successive, equally spaced points in time.

Think about:

Retail: Predicting the daily demand for a product to optimize inventory.
Finance: Forecasting the stock market to make investment decisions.
Healthcare: Estimating the number of weekly flu cases to allocate resources, as we’ll see in our code example.
Energy: Predicting hourly electricity consumption to manage the power grid.
The key challenge is to identify and extrapolate the underlying patterns in the data, such as trends (long-term increase or decrease), seasonality (predictable, repeating fluctuations), and cycles (unpredictable fluctuations).

## What is a Decoder?
To understand TimesFM, we first need to understand its architectural foundation: the decoder. Decoders are a cornerstone of modern AI, famously powering Large Language Models (LLMs) like GPT.

Imagine you’re translating a sentence from English to French. You read the English sentence (the “context”) and then begin writing the French translation word by word.

A decoder works similarly. It’s a neural network architecture designed for generative tasks — creating new sequences based on an input context. In the case of an LLM, it takes a sequence of words (the prompt) and generates the next word. It then appends this new word to the sequence and generates the next one, and so on, in an autoregressive fashion.

TimesFM applies this exact principle to numbers instead of words. The “context” is the historical time series data. The model “generates” the next value (or a patch of values) in the sequence, which is our forecast. It learns the “grammar” of time — the intricate temporal patterns — instead of human language.

## Introducing TimesFM: The Foundation Model for Time
TimesFM is designed to be a general-purpose, zero-shot forecaster. This means it can make surprisingly accurate predictions for datasets it has never seen before, without any fine-tuning. This is a paradigm shift from traditional models.

The key question the researchers at Google sought to answer was: “Can a large model pretrained on massive amounts of time-series data learn universal temporal patterns useful for forecasting on unseen datasets?”

The answer is a resounding yes. Here are the key architectural choices that make TimesFM so effective:

1. Decoder-Only Architecture: By using a decoder-only model, TimesFM is naturally suited for generating future sequences from past context, and it can gracefully handle inputs of varying lengths.

2. Input Patching: Instead of looking at one data point at a time, TimesFM groups the input time series into contiguous, non-overlapping “patches.” Think of it as reading a sentence instead of a single word. This has two major benefits:
Improved Performance: It allows the model to learn local patterns within a patch more effectively.
Inference Speed: The transformer has fewer “tokens” (patches) to process, making it much faster. The paper found a patch length of 32 to be a sweet spot.

1. Longer Output Patches: Unlike many LLMs that generate one token at a time, TimesFM can be configured to predict an output patch that is longer than its input patches. For example, it can process 32-point input patches to generate a 128-point forecast in a single step. This dramatically reduces the number of steps needed for long-horizon forecasting, improving speed and accuracy.

### How Well Does It Perform?

This is where TimesFM truly shines. The paper evaluates the model against a suite of well-known forecasting benchmarks (Monash, Darts, Informer ETT) that it was not trained on.

The results are remarkable. In zero-shot mode, TimesFM’s performance is close to or even surpasses state-of-the-art supervised models that were specifically trained and tuned for each of those datasets. This demonstrates that the universal patterns learned during pre-training are highly effective and transferable across different domains.

## The Data Behind the Model
A foundation model is only as good as the data it’s trained on. For TimesFM to learn universal patterns, it needed a massive and incredibly diverse dataset. The researchers compiled a corpus with over 100 billion timepoints from a variety of sources:


### Real-World Data:

Google Trends: Billions of points representing search interest over time for thousands of queries, capturing public interest and seasonality at various granularities (hourly, daily, weekly).
Wikipedia Pageviews: Billions of points tracking the views of Wikipedia pages, reflecting cultural events and trends.
Public Datasets: Well-known forecasting datasets like M4, Electricity, and Traffic were also included.
Synthetic Data: To ensure the model doesn’t just memorize patterns common in real-world data (e.g., 24-hour cycles), the researchers generated a huge amount of synthetic time series data. This included various trends, seasonal patterns (sine/cosine waves), and ARMA processes, teaching the model to generalize to patterns it might not have seen in the wild.
This combination of scale and diversity is what enables TimesFM to be such a powerful generalist.

Getting Started: Zero-Shot Forecasting with TimesFM
Now, let’s get practical. The simplest way to use TimesFM is for zero-shot forecasting, where you only need your historical data.

First, ensure you have the timesfm library installed.

```shell
!pip install --quiet git+https://github.com/google-research/timesfm
```

Now, let’s perform a forecast.

```python
import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Bike Rental Data ---
def create_bike_rental_data():
    """Creates a realistic, synthetic daily bike rental dataset."""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n_days = len(dates)

    # Baseline trend (rentals slowly increasing over the years)
    trend = np.linspace(start=150, stop=350, num=n_days)

    # Yearly seasonality (more rentals in summer, less in winter)
    yearly_seasonality = 180 * (1 + np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi/2))

    # Weekly seasonality (more rentals on weekends)
    weekly_seasonality = 120 * (dates.dayofweek >= 5)

    # Noise
    noise = np.random.normal(0, 40, n_days)

    # Combine to create the rental count
    rentals = trend + yearly_seasonality + weekly_seasonality + noise
    rentals = np.maximum(0, rentals).astype(int) # Ensure no negative rentals

    # Create weather covariates
    temp = 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi/2) + np.random.randn(n_days) * 2
    is_weekend = (dates.dayofweek >= 5).astype(int)

    # Create the final DataFrame
    df = pd.DataFrame({
        'rentals': rentals,
        'temperature': temp,
        'is_weekend': is_weekend
    }, index=dates)
    return df

rental_df = create_bike_rental_data()
time_series_data = rental_df['rentals'].values

# --- 2. Set Up the Forecast ---
horizon_len = 90  # We want to forecast the next 90 days
context_len = 365 # The model can see up to 365 past days
historical_data = time_series_data[:-horizon_len]
true_future_values = time_series_data[-horizon_len:]

# --- 3. Initialize TimesFM and Forecast ---
timesfm_backend = 'cpu' #can change to gpu for faster performance
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend=timesfm_backend,
          per_core_batch_size=32,
          horizon_len=90,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )

# The frequency '1' indicates a regular, unspecified interval (in our case, daily)
forecast_results, _ = tfm.forecast([historical_data], freq=[1])
forecast_values = forecast_results[0]

# --- 4. Visualize the Results ---
dates = rental_df.index
plt.figure(figsize=(15, 7))
plt.plot(dates[:-horizon_len], historical_data, label="Historical Data", color="black")
plt.plot(dates[-horizon_len:], true_future_values, label="True Future Values", color="blue", linestyle='--')
plt.plot(dates[-horizon_len:], forecast_values, label="TimesFM Forecast", color="red")

plt.legend()
plt.title("TimesFM Zero-Shot Forecast for Daily Bike Rentals")
plt.xlabel("Date")
plt.ylabel("Number of Rentals")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((true_future_values - forecast_values)**2))
print(f"Zero-Shot Forecast RMSE: {rmse:.2f}")
```



Just like that, with a few lines of code, we have a robust forecast. The model had no prior knowledge of this specific bike rental dataset but was able to identify and extrapolate the trend and seasonal patterns by leveraging its vast pre-trained knowledge.

Leveling Up: Forecasting with Covariates
Sometimes, the value you want to predict is influenced by other known factors. For example, ice cream sales are heavily influenced by temperature. This external factor is called a covariate.

In simple terms, this is a regression problem: Y ~ X, where you predict a target Y based on a feature X. TimesFM can incorporate these covariates to produce more accurate forecasts, provided you have the future values for those covariates.

```python
# --- 1. Prepare Target and Covariate Data ---
# NOTE: The covariate data MUST be available for the full period (history + horizon)
target_series = rental_df["rentals"].values
temp_series = rental_df["temperature"].values
weekend_series = rental_df["is_weekend"].values

# --- 2. Forecast with Covariates ---
# The model uses historical target and historical+future covariates
covariate_forecast, _ = tfm.forecast_with_covariates(
    inputs=[target_series[:-horizon_len]],
    dynamic_numerical_covariates={
        "temperature": [temp_series],
        "is_weekend": [weekend_series],
    },
    freq=[1],
)
covariate_forecast_values = covariate_forecast[0]

# --- 3. Visualize the Comparison ---
plt.figure(figsize=(15, 7))
plt.plot(dates[-horizon_len:], true_future_values, label="True Future Values", color="blue", linewidth=2.5)
plt.plot(dates[-horizon_len:], forecast_values, label="Zero-Shot Forecast (No Covariates)", color="red", linestyle='--')
plt.plot(dates[-horizon_len:], covariate_forecast_values, label="Forecast with Temp & Weekend Data", color="green", linestyle='-.')
plt.legend()
plt.title("TimesFM Forecast Comparison: With and Without Covariates")
plt.xlabel("Date")
plt.ylabel("Number of Rentals")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Calculate RMSE for the covariate model
covariate_rmse = np.sqrt(np.mean((true_future_values - covariate_forecast_values)**2))
print(f"Zero-Shot Forecast RMSE: {rmse:.2f}")
print(f"Forecast with Covariates RMSE: {covariate_rmse:.2f}")
```

Why does providing extra data like temperature and weekend information lead to a better forecast? A model that only looks at the history of bike rentals — a univariate model — is essentially working with blinders on. It can recognize that rentals spike on a seven-day cycle and rise during certain months, but it has no understanding of the underlying causes. It’s simply extrapolating past patterns. When we introduce covariates, we remove the blinders. The model is no longer just seeing a pattern; it’s learning a relationship: higher temperatures correlate with more rentals, and weekend days have a fundamentally different demand profile than weekdays. This allows for more nuanced and accurate predictions. For example, if a future Saturday has a forecast for unusually cold weather, the model can intelligently lower its rental prediction for that day, something a univariate model would struggle with. In essence, covariates transform unexplained variance (or ‘noise’) in the historical data into a predictable signal, grounding the forecast in the real-world factors that drive demand.

For Maximum Performance: Finetuning
While TimesFM’s zero-shot performance is its most impressive feature, you can squeeze out even more accuracy by finetuning it on your specific dataset.

Finetuning means taking the pre-trained model and training it for a few more epochs on your own data. This allows the model to adapt its generalized knowledge to the specific nuances, scale, and idiosyncrasies of your time series.

The paper demonstrates that finetuning on just 10% of a dataset can lead to state-of-the-art results, significantly outperforming models trained from scratch.

While the timesfm library doesn’t provide a simple high-level .finetune() API in its initial release, the conceptual process would look like this using a framework like JAX or PyTorch

# 1. Load the pre-trained model
model = load_pretrained_timesfm_model()
optimizer = create_optimizer(model.parameters())

# 2. Prepare your specific training data
# For each sample, you need the historical context and the true future values
train_data_loader = prepare_finetuning_data(my_historical_data, my_future_data)

# 3. Run the training loop
for epoch in range(num_epochs):
    for context, ground_truth_horizon in train_data_loader:
        # Forward pass: get the model's prediction
        predicted_horizon = model(context)

        # Calculate loss (e.g., Mean Squared Error)
        loss = mse_loss(predicted_horizon, ground_truth_horizon)

        # Backward pass: update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 4. Save your newly finetuned model
save_model(model, "my_finetuned_timesfm.chkpt")
Finetuning is the path to take when you have a critical forecasting task and need the absolute best performance possible.

For those interested in the technical details, the paper by Das et. al. reveals a few more insights:

Scaling Laws: Just like LLMs, TimesFM benefits from scale. The researchers found a clear trend: as the model size (parameters) and the amount of computation (FLOPS) increase, the forecasting error consistently decreases. This suggests that future, even larger versions of TimesFM could be more powerful still.
The Importance of Synthetic Data: An ablation study showed that training without the synthetic dataset caused a performance drop, especially on datasets with uncommon granularities (like quarterly or 10-minute intervals). This highlights the crucial role synthetic data plays in teaching the model to be a true generalist.
Autoregressive Decoding Matters: The paper confirmed that for long horizons, generating a larger output patch in one go is more effective than predicting one single point at a time. This is a key design decision that differentiates TimesFM from standard autoregressive models.
The journey for time series foundation models is just beginning. The researchers outline several exciting avenues for future work:
Probabilistic Forecasting: The current model provides a point forecast. A future version could predict a range of possible outcomes (quantiles), which is crucial for risk management.
Advanced Covariate Handling: Pre-training the model from scratch with covariate data could make its core reasoning about external factors even more powerful.
Prompting: Exploring how to “prompt” a time series model — perhaps by providing examples or instructions in-context — is a fascinating open question.
Conclusion
TimesFM represents a significant milestone in the field of time series analysis. By successfully applying the principles of large-scale pre-training from the world of NLP, Google has created a powerful, versatile, and accessible forecasting tool.

Its strong zero-shot performance democratizes forecasting, allowing practitioners to get reliable predictions without deep domain expertise or extensive model training. Its ability to incorporate covariates and be finetuned provides a path to state-of-the-art results for mission-critical applications.

As foundation models for time series continue to evolve, they promise to unlock new capabilities and make sophisticated forecasting more accessible than ever before. TimesFM is a powerful first step into that future.

Forecasting



Siduo Jiang
Written by Siduo Jiang

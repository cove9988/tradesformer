The issue you're encountering arises because the yfinance library has limitations when fetching intraday data (intervals shorter than one day). Specifically, for intervals like '5m', Yahoo Finance only provides data for the past 60 days. When you request data outside this range, the resulting DataFrame will be empty, which is why only the header is saved to your CSV file.

Explanation
Yahoo Finance Limitations: Yahoo Finance restricts the amount of historical intraday data available. For intervals less than '1d', you can only retrieve data for approximately the last 60 days.

Your Date Range: You're requesting data from '2022-01-01' to '2023-01-01', which is a one-year period in the past. Since this is well beyond the 60-day limit, yfinance cannot fetch this data, resulting in an empty DataFrame.

How to Fix the Issue
Option 1: Adjust the Date Range
If you need intraday data (like '5m' intervals), you'll have to adjust your date range to be within the last 60 days.

python
Copy code
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the currency pair and period
pair = 'EURUSD'
symbol = 'EURUSD=X'  # EUR/USD exchange rate
end_date = datetime.today()
start_date = end_date - timedelta(days=59)  # Last 60 days
interval = '5m'  # 5-minute intervals

# Download data
data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
data = data.dropna()

# Ensure the 'data' directory exists
import os
os.makedirs('data', exist_ok=True)

# Save to CSV
filename = f'data/yahoo-{pair}-{start_date.strftime("%Y-%m-%d")}-{end_date.strftime("%Y-%m-%d")}-{interval}.csv'
data.to_csv(filename)
print(f'Data saved to {filename}')
Option 2: Use a Daily Interval
If you need data for a longer period, consider using a daily interval ('1d'). Daily data is available for much longer historical periods.

python
Copy code
import yfinance as yf
import pandas as pd

# Define the currency pair and period
pair = 'EURUSD'
symbol = 'EURUSD=X'  # EUR/USD exchange rate
start_date = '2022-01-01'
end_date = '2023-01-01'
interval = '1d'  # Daily intervals

# Download data
data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
data = data.dropna()

# Ensure the 'data' directory exists
import os
os.makedirs('data', exist_ok=True)

# Save to CSV
filename = f'data/yahoo-{pair}-{start_date}-{end_date}-{interval}.csv'
data.to_csv(filename)
print(f'Data saved to {filename}')
Option 3: Use an Alternative Data Source
If you require historical intraday data beyond 60 days, consider using alternative data providers:

Alpha Vantage: Provides free intraday Forex data with certain limitations.
Polygon.io: Offers comprehensive historical data but may require a paid subscription.
OANDA API: Provides Forex data through their API (may require an account).
Dukascopy: Offers historical tick data for Forex pairs.
Example using Alpha Vantage:

python
Copy code
pip install alpha_vantage
python
Copy code
from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd

api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
cc = ForeignExchange(key=api_key)

# Fetch intraday data
data, _ = cc.get_currency_exchange_intraday(from_symbol='EUR', to_symbol='USD', interval='5min', outputsize='full')

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Rename columns
df.columns = ['Open', 'High', 'Low', 'Close']

# Save to CSV
df.to_csv('data/alpha_vantage_EURUSD_5min.csv')
Note: Alpha Vantage has API call limits and may not provide extensive historical intraday data.

Summary
Why Only Headers Were Saved: The DataFrame data is empty because yfinance cannot fetch '5m' interval data for dates beyond the last 60 days.

Solutions:

Adjust Date Range: Limit your date range to within the last 60 days for intraday data.
Use Daily Data: If historical data over a longer period is needed, switch to a daily interval.
Alternative Data Sources: Use other APIs or data providers that offer historical intraday data.
Additional Information
Check DataFrame Content: Before saving, you can check if the DataFrame is empty:

python
Copy code
if data.empty:
    print("No data fetched. DataFrame is empty.")
else:
    data.to_csv(filename)
    print(f'Data saved to {filename}')
Error Handling: Implement error handling to manage scenarios where data retrieval fails.

Next Steps
Decide on Data Needs: Determine whether you need intraday data or if daily data suffices for your research.

Choose the Right Tool: Based on your data requirements, select the appropriate data source or API.

Data Storage: Ensure you handle data storage efficiently, especially if working with large datasets.

Example with Adjusted Date Range
python
Copy code
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the currency pair and period
pair = 'EURUSD'
symbol = 'EURUSD=X'  # EUR/USD exchange rate
end_date = datetime.today()
start_date = end_date - timedelta(days=59)  # Last 60 days
interval = '5m'  # 5-minute intervals

# Download data
data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
data = data.dropna()

# Check if data is empty
if data.empty:
    print("No data fetched. Please check the date range and interval.")
else:
    # Ensure the 'data' directory exists
    import os
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    filename = f'data/yahoo-{pair}-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}-{interval}.csv'
    data.to_csv(filename)
    print(f'Data saved to {filename}')
Alternative: Using Daily Data for the Past Year
python
Copy code
import yfinance as yf
import pandas as pd

# Define the currency pair and period
pair = 'EURUSD'
symbol = 'EURUSD=X'
start_date = '2022-01-01'
end_date = '2023-01-01'
interval = '1d'  # Daily intervals

# Download data
data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
data = data.dropna()

# Check if data is empty
if data.empty:
    print("No data fetched. Please check the date range and interval.")
else:
    # Ensure the 'data' directory exists
    import os
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    filename = f'data/yahoo-{pair}-{start_date}-{end_date}-{interval}.csv'
    data.to_csv(filename)
    print(f'Data saved to {filename}')
Conclusion
The problem you're facing is due to the limitations of the data provider (Yahoo Finance) when it comes to historical intraday data. By adjusting your date range or interval, or by choosing an alternative data source, you can successfully retrieve and save the desired data for your research.


Why This Matters for RL Training
Approach	Pros	Cons	RL Suitability
Global Normalization (Before Split)	Consistent scale across episodes	Data leakage between weeks	❌ Poor - leaks future info
Per-Week Normalization (After Split)	Each week is self-contained	Harder to compare across weeks	✅ Excellent - matches RL's episodic nature

import yfinance as yf
import pandas as pd
import numpy as np

def generate_mock_nifty_data(output_path="NIFTY 50.csv"):
    # Fetch Nifty 50 data from Yahoo Finance (^NSEI)
    print("Fetching NIFTY 50 data from Yahoo Finance...")
    nifty = yf.download("^NSEI", start="2000-01-01", end="2024-01-01", progress=False)

    # Note: latest yfinance returns MultiIndex columns. We need to flatten them.
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    # Reset index to make Date a column
    nifty = nifty.reset_index()

    # Generate mock data for P/E, P/B, Div Yield
    np.random.seed(42)
    nifty['P/E'] = np.random.uniform(15, 30, size=len(nifty))
    nifty['P/B'] = np.random.uniform(2.5, 4.5, size=len(nifty))
    nifty['Div Yield %'] = np.random.uniform(1.0, 2.5, size=len(nifty))

    nifty.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
    # Reorder columns to match user spec
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'P/E', 'P/B', 'Div Yield %']
    nifty = nifty[cols]

    nifty.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path} with mock P/E, P/B, and Div Yield.")

if __name__ == "__main__":
    generate_mock_nifty_data()

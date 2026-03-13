import pandas as pd
import numpy as np

def load_data(filepath="NIFTY 50.csv"):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def compute_advanced_eda(df, risk_free_rate=0.06):
    """
    Computes daily returns, rolling volatility, Sharpe ratio, max drawdown, and 5-yr rolling CAGR.
    """
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Annualized rolling volatility (assume 252 trading days)
    df['Vol_30D'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
    df['Vol_90D'] = df['Daily_Return'].rolling(window=90).std() * np.sqrt(252)
    df['Vol_252D'] = df['Daily_Return'].rolling(window=252).std() * np.sqrt(252)
    
    # Sharpe Ratio (rolling 252 day)
    # Average daily return annualized minus risk-free rate, divided by annualized volatility
    rolling_ann_ret = df['Daily_Return'].rolling(window=252).mean() * 252
    df['Sharpe_252D'] = (rolling_ann_ret - risk_free_rate) / df['Vol_252D']
    
    # Max Drawdown (rolling 252 day)
    roll_max = df['Close'].rolling(window=252, min_periods=1).max()
    daily_dd = df['Close'] / roll_max - 1.0
    df['Drawdown'] = daily_dd
    df['Max_Drawdown_252D'] = daily_dd.rolling(window=252, min_periods=1).min()
    
    # 5-year rolling CAGR (approx 1260 trading days)
    # CAGR = (Ending Value / Beginning Value) ** (1/5) - 1
    df['CAGR_5Y'] = (df['Close'] / df['Close'].shift(1260)) ** (1/5) - 1
    
    return df

def compute_technical_indicators(df):
    """
    Computes SMA, MACD, RSI, Bollinger Bands, and Golden/Death Cross.
    """
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Golden / Death Cross Signal
    # 1 for Golden Cross (50 > 200), -1 for Death Cross (50 < 200)
    df['Cross_Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    
    # Bollinger Bands (20-day)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

def compute_valuation_bands(df):
    """
    Computes P/E and P/B historical percentiles. Also inverts Div Yield for valuation cycles.
    """
    # Expanding percentiles to show historical valuation zones
    if 'P/E' in df.columns:
        df['PE_Avg'] = df['P/E'].expanding().mean()
        df['PE_Upper_Band'] = df['PE_Avg'] + df['P/E'].expanding().std()
        df['PE_Lower_Band'] = df['PE_Avg'] - df['P/E'].expanding().std()
        
    if 'P/B' in df.columns:
        df['PB_Avg'] = df['P/B'].expanding().mean()
        df['PB_Upper_Band'] = df['PB_Avg'] + df['P/B'].expanding().std()
        df['PB_Lower_Band'] = df['PB_Avg'] - df['P/B'].expanding().std()
        
    if 'Div Yield %' in df.columns:
        # Inverse Div Yield (Price is often high when yield is low, and vice versa)
        df['Inv_Div_Yield'] = 1 / df['Div Yield %']
        
    return df

def get_seasonality_and_anomalies(df):
    """
    Returns a pivot table for monthly returns heatmap and top best/worst anomalies.
    """
    # Create Year and Month columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Calculate Monthly Returns (Start to end of month)
    monthly_data = df.groupby(['Year', 'Month']).agg({'Close': ['first', 'last']})
    monthly_data.columns = ['Start', 'End']
    monthly_data['Monthly_Return'] = (monthly_data['End'] / monthly_data['Start'] - 1) * 100
    
    # Pivot for Heatmap
    monthly_returns_heatmap = monthly_data.reset_index().pivot(index='Year', columns='Month', values='Monthly_Return')
    
    # Anomalies
    best_days = df.nlargest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return']].copy()
    worst_days = df.nsmallest(10, 'Daily_Return')[['Date', 'Close', 'Daily_Return']].copy()
    
    # Market events dictionary (simplified explanation mapping based on date ranges in Indian context)
    def explain_event(date):
        year = date.year
        if year == 2008 or year == 2009:
            return "Global Financial Crisis"
        elif year == 2004:
            return "2004 Election Surprise"
        elif year == 2020:
            return "COVID-19 Pandemic"
        elif year == 2016:
            return "Demonetization / US Elections"
        return "Other Macro Volatility"
        
    best_days['Event'] = best_days['Date'].apply(explain_event)
    worst_days['Event'] = worst_days['Date'].apply(explain_event)
    
    return monthly_returns_heatmap, best_days, worst_days

def process_all_data(filepath="NIFTY 50.csv"):
    df = load_data(filepath)
    df = compute_advanced_eda(df)
    df = compute_technical_indicators(df)
    df = compute_valuation_bands(df)
    heatmap, best, worst = get_seasonality_and_anomalies(df)
    return df, heatmap, best, worst

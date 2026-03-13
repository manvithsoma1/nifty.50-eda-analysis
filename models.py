import pandas as pd
import numpy as np
from hmmlearn import hmm
from prophet import Prophet

def get_regime_hmm(df):
    """
    Uses Gaussian HMM to classify market into Bull / Bear / Sideways regimes.
    Features: Daily Return and Rolling Volatility (30D).
    """
    df = df.copy()
    
    # Ensure features exist
    if 'Daily_Return' not in df.columns or 'Vol_30D' not in df.columns:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Vol_30D'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
        
    features = ['Daily_Return', 'Vol_30D']
    clean_df = df.dropna(subset=features).copy()
    
    if len(clean_df) == 0:
        df['Regime'] = 'Unknown'
        return df
        
    X = clean_df[features].values
    
    # Fit HMM
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    # Map the hidden states based on the mean returns and volatility to Bull, Bear, Sideways
    means = model.means_
    returns = means[:, 0]
    
    # Sort by returns to identify Bear (lowest) and Bull (highest)
    sorted_idx = np.argsort(returns)
    bear_idx = sorted_idx[0]
    sideways_idx = sorted_idx[1]
    bull_idx = sorted_idx[2]
    
    state_map = {
        bull_idx: 'Bull',
        bear_idx: 'Bear',
        sideways_idx: 'Sideways'
    }
    
    clean_df['Regime'] = [state_map[s] for s in hidden_states]
    
    # Merge back to original df
    df = df.join(clean_df[['Regime']])
    df['Regime'] = df['Regime'].fillna('Unknown')
    return df

def get_prophet_forecast(df, periods=90):
    """
    Forecasting next 90 days of Nifty 50 closing prices using Prophet.
    Returns:
    - forecast_df: Prophet forecast dataframe containing 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    - model: The fitted Prophet model (for plotting if needed)
    """
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    m = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return forecast, m

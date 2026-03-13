import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# SECTION 1 & 4: PAGE CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Nifty 50 Analytics", layout="wide", page_icon="📈")

st.markdown("""
<style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #0a0e1a;
        color: #ffffff;
    }
    .css-1d391kg, .css-1dp5vir, .css-18e3th9 {
        background-color: #0f172a;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #00d4aa;
    }
    div[data-testid="stMetric"] > div {
        background-color: #111827;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(90deg, #0f172a 0%, #111827 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #00d4aa;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & DATA GENERATION
# -----------------------------------------------------------------------------
@st.cache_data
def generate_sample_data():
    """Generates realistic NIFTY 50 daily data from 2000 to 2024"""
    np.random.seed(42)
    dates = pd.bdate_range(start="2000-01-01", end="2024-01-01")
    n = len(dates)
    
    # Generate random walk with drift for prices
    returns = np.random.normal(loc=0.0004, scale=0.015, size=n)
    price_index = 1500 * np.exp(np.cumsum(returns))
    
    open_p = price_index * np.random.normal(1, 0.005, n)
    high_p = np.maximum(open_p, price_index) * np.random.normal(1.005, 0.002, n)
    low_p = np.minimum(open_p, price_index) * np.random.normal(0.995, 0.002, n)
    
    # Volume
    volume = np.random.lognormal(mean=15, sigma=1, size=n)
    
    # Valuation metrics
    pe = 20 + 5 * np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 1, n)
    pb = 3 + 1 * np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.2, n)
    div_yield = 1.5 - 0.5 * np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_p,
        'High': high_p,
        'Low': low_p,
        'Close': price_index,
        'Volume': volume,
        'P/E': pe,
        'P/B': pb,
        'Div Yield %': div_yield
    })
    return df

@st.cache_data
def process_data(df):
    """Computes all required technical and statistical features"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cum_Return'] = (1 + df['Daily_Return'].fillna(0)).cumprod() - 1
    
    # Volatility
    df['Vol_30d'] = df['Daily_Return'].rolling(30).std() * np.sqrt(252)
    df['Vol_90d'] = df['Daily_Return'].rolling(90).std() * np.sqrt(252)
    df['Vol_252d'] = df['Daily_Return'].rolling(252).std() * np.sqrt(252)
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    # Golden / Death Cross
    df['Cross_Signal'] = 0
    mask_bull = (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1))
    mask_bear = (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1))
    df.loc[mask_bull, 'Cross_Signal'] = 1
    df.loc[mask_bear, 'Cross_Signal'] = -1
    
    # Bollinger Bands
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['BB_std']
    df['BB_Lower'] = df['MA20'] - 2 * df['BB_std']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Max Drawdown
    rolling_max = df['Close'].expanding().max()
    df['Drawdown'] = df['Close'] / rolling_max - 1
    
    # Date parts for seasonality
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    return df

@st.cache_data
def get_annual_summary(df):
    annual = []
    years = sorted(df['Year'].unique())
    for y in years:
        ydf = df[df['Year'] == y]
        if len(ydf) < 10:
            continue
        ret = (ydf['Close'].iloc[-1] / ydf['Close'].iloc[0]) - 1
        vol = ydf['Daily_Return'].std() * np.sqrt(252)
        max_dd = ydf['Drawdown'].min()
        avg_pe = ydf['P/E'].mean()
        sharpe = (ret - 0.06) / vol if vol > 0 else 0
        
        annual.append({
            'Year': str(y),
            'Return': ret,
            'Volatility': vol,
            'Max Drawdown': max_dd,
            'Avg P/E': avg_pe,
            'Sharpe Ratio': sharpe
        })
    return pd.DataFrame(annual)

def apply_plotly_layout(fig, title):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="white")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", gridwidth=1),
        hoverlabel=dict(bgcolor="#111827", font_size=13, font_family="sans-serif")
    )
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/NIFTY_50_logo.svg/1200px-NIFTY_50_logo.svg.png", width=150)
    st.title("⚙️ Controls")
    
    uploaded_file = st.file_uploader("Upload Custom CSV (Optional)", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = generate_sample_data()
        
    df_processed = process_data(raw_df)
    
    min_date = df_processed['Date'].min().date()
    max_date = df_processed['Date'].max().date()
    
    date_range = st.sidebar.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    
    st.markdown("### Overlays (Price Chart)")
    show_ma50 = st.checkbox("Show MA50", value=True)
    show_ma200 = st.checkbox("Show MA200", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=False)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **📊 Price Chart**: Candlesticks, volume, MAs, and Golden/Death crosses.
    **📉 Returns & Risk**: Cumulative returns, drawdowns, volatility, and distribution.
    **🗓️ Seasonality**: Heatmaps and average returns by month/day.
    **⚖️ Valuation**: P/E, P/B bands, and Dividend Yield cycles.
    **📅 Annual Breakdown**: Year-over-year performance summaries.
    """)

# Filter data
mask = (df_processed['Date'].dt.date >= date_range[0]) & (df_processed['Date'].dt.date <= date_range[1])
df = df_processed[mask].copy()

# -----------------------------------------------------------------------------
# HERO BANNER & KPI CARDS
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <h2>🚀 Nifty 50 Market Analytics Engine</h2>
    <p style="color:#A0AEC0; margin:0;">Analysis Period: {date_range[0]} to {date_range[1]}</p>
</div>
""", unsafe_allow_html=True)

# Calculate KPIs
last_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
day_change = (last_close / prev_close - 1) * 100

ytd_mask = (df['Year'] == df['Date'].dt.year.iloc[-1])
if len(df[ytd_mask]) > 0:
    ytd_ret = (last_close / df[ytd_mask]['Close'].iloc[0] - 1) * 100
else:
    ytd_ret = 0.0

max_dd = df['Drawdown'].min() * 100
cur_rsi = df['RSI'].iloc[-1]

cur_pe = df['P/E'].iloc[-1]
med_pe = df['P/E'].median()

kpi_cols = st.columns(6)
kpi_cols[0].metric("Current Close", f"{last_close:,.1f}")
kpi_cols[1].metric("Day Change", f"{day_change:+.2f}%", delta_color="normal" if day_change>0 else "inverse")
kpi_cols[2].metric("YTD Return", f"{ytd_ret:+.2f}%")
kpi_cols[3].metric("Max Drawdown", f"{max_dd:.2f}%")
kpi_cols[4].metric("Current RSI", f"{cur_rsi:.1f}")
kpi_cols[5].metric("P/E vs Median", f"{cur_pe:.1f}", f"{cur_pe - med_pe:+.1f}", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Chart", 
    "📉 Returns & Risk", 
    "🗓️ Seasonality", 
    "⚖️ Valuation", 
    "📅 Annual Breakdown"
])

# -------------------- TAB 1: PRICE CHART --------------------
with tab1:
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05)
    
    # Candlestick
    fig_price.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price", increasing_line_color='#00d4aa', decreasing_line_color='#ef4444'
    ), row=1, col=1)
    
    # Overlays
    if show_ma50:
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='MA50', line=dict(color='#fbbf24', width=1.5)), row=1, col=1)
    if show_ma200:
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], mode='lines', name='MA200', line=dict(color='#ec4899', width=1.5)), row=1, col=1)
    if show_bb:
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', fill='tonexty', fillcolor='rgba(0, 212, 170, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Bollinger Bands'), row=1, col=1)
        
    # Crosses
    golden_crosses = df[df['Cross_Signal'] == 1]
    death_crosses = df[df['Cross_Signal'] == -1]
    
    fig_price.add_trace(go.Scatter(x=golden_crosses['Date'], y=golden_crosses['MA50'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00d4aa'), name='Golden Cross'), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=death_crosses['Date'], y=death_crosses['MA50'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef4444'), name='Death Cross'), row=1, col=1)

    # Volume
    colors = ['#00d4aa' if row['Close'] >= row['Open'] else '#ef4444' for _, row in df.iterrows()]
    fig_price.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig_price.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_price = apply_plotly_layout(fig_price, "Nifty 50 Price Action & Indicators")
    st.plotly_chart(fig_price, use_container_width=True)

# -------------------- TAB 2: RETURNS & RISK --------------------
with tab2:
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df['Date'], y=df['Cum_Return']*100, fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.2)', line=dict(color='#00d4aa')))
        fig_cum = apply_plotly_layout(fig_cum, "Cumulative Return (%)")
        st.plotly_chart(fig_cum, use_container_width=True)
        
    with r1c2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df['Date'], y=df['Drawdown']*100, fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)', line=dict(color='#ef4444')))
        fig_dd = apply_plotly_layout(fig_dd, "Max Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)
        
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['Vol_30d']*100, name='30-Day', line=dict(color='#60a5fa')))
        fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['Vol_90d']*100, name='90-Day', line=dict(color='#fbbf24')))
        fig_vol = apply_plotly_layout(fig_vol, "Rolling Annualized Volatility (%)")
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with r2c2:
        clean_ret = df['Daily_Return'].dropna()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=clean_ret, histnorm='probability density', nbinsx=100, marker_color='#94a3b8', name='Returns'))
        # Normal Dist overlay
        mu, std = stats.norm.fit(clean_ret)
        xmin, xmax = clean_ret.min(), clean_ret.max()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        fig_hist.add_trace(go.Scatter(x=x, y=p, mode='lines', line=dict(color='#00d4aa', width=2), name='Normal Dist'))
        fig_hist = apply_plotly_layout(fig_hist, "Daily Returns Distribution")
        fig_hist.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Market Extremes")
    c1, c2 = st.columns(2)
    top_days = df.nlargest(5, 'Daily_Return')
    worst_days = df.nsmallest(5, 'Daily_Return')
    
    with c1:
        fig_top = px.bar(top_days, x='Daily_Return', y=top_days['Date'].dt.strftime('%Y-%m-%d'), orientation='h')
        fig_top.update_traces(marker_color='#00d4aa')
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_top = apply_plotly_layout(fig_top, "Top 5 Best Days")
        fig_top.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_top, use_container_width=True)
        
    with c2:
        fig_worst = px.bar(worst_days, x='Daily_Return', y=worst_days['Date'].dt.strftime('%Y-%m-%d'), orientation='h')
        fig_worst.update_traces(marker_color='#ef4444')
        fig_worst.update_layout(yaxis={'categoryorder':'total descending'})
        fig_worst = apply_plotly_layout(fig_worst, "Top 5 Worst Days")
        fig_worst.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_worst, use_container_width=True)

# -------------------- TAB 3: SEASONALITY --------------------
with tab3:
    # Monthly Heatmap
    monthly_ret = df.groupby(['Year', 'Month'])['Daily_Return'].apply(lambda x: (np.prod(1+x)-1)*100).reset_index()
    pivot = monthly_ret.pivot(index='Year', columns='Month', values='Daily_Return')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[i-1] for i in pivot.columns]
    
    fig_heat = px.imshow(pivot, color_continuous_scale='RdYlGn', text_auto=".1f", aspect='auto')
    fig_heat = apply_plotly_layout(fig_heat, "Monthly Returns Heatmap (%)")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        avg_month = monthly_ret.groupby('Month')['Daily_Return'].mean().reset_index()
        avg_month['Month_Name'] = [month_names[i-1] for i in avg_month['Month']]
        colors = ['#00d4aa' if r > 0 else '#ef4444' for r in avg_month['Daily_Return']]
        fig_avg_mo = go.Figure(go.Bar(x=avg_month['Month_Name'], y=avg_month['Daily_Return'], marker_color=colors))
        fig_avg_mo = apply_plotly_layout(fig_avg_mo, "Average Monthly Return (%)")
        st.plotly_chart(fig_avg_mo, use_container_width=True)
        
    with c2:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        avg_dow = df.groupby('DayOfWeek')['Daily_Return'].mean().reindex(dow_order) * 100
        colors = ['#00d4aa' if r > 0 else '#ef4444' for r in avg_dow]
        fig_dow = go.Figure(go.Bar(x=avg_dow.index, y=avg_dow.values, marker_color=colors))
        fig_dow = apply_plotly_layout(fig_dow, "Average Day-of-Week Return (%)")
        st.plotly_chart(fig_dow, use_container_width=True)

# -------------------- TAB 4: VALUATION --------------------
with tab4:
    p25 = df['P/E'].quantile(0.25)
    p75 = df['P/E'].quantile(0.75)
    
    if cur_pe > p75:
        badge = "🔴 EXPENSIVE"
    elif cur_pe < p25:
        badge = "🟢 CHEAP"
    else:
        badge = "🟡 FAIR"
        
    st.markdown(f"### Current Market Valuation: {badge}")
    
    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(x=df['Date'], y=df['P/E'], name='P/E Ratio', line=dict(color='#8b5cf6')))
    fig_pe.add_hline(y=med_pe, line_dash="dash", line_color="#cbd5e1", annotation_text="Median")
    fig_pe.add_hrect(y0=p25, y1=p75, fillcolor="#334155", opacity=0.3, layer="below", line_width=0, annotation_text="Interquartile Range")
    fig_pe = apply_plotly_layout(fig_pe, "Historical P/E Ratio")
    st.plotly_chart(fig_pe, use_container_width=True)
    
    fig_pb = go.Figure()
    fig_pb.add_trace(go.Scatter(x=df['Date'], y=df['P/B'], name='P/B Ratio', line=dict(color='#ec4899')))
    fig_pb.add_hline(y=df['P/B'].median(), line_dash="dash", line_color="#cbd5e1")
    fig_pb.add_hrect(y0=df['P/B'].quantile(0.25), y1=df['P/B'].quantile(0.75), fillcolor="#334155", opacity=0.3, layer="below", line_width=0)
    fig_pb = apply_plotly_layout(fig_pb, "Historical P/B Ratio")
    st.plotly_chart(fig_pb, use_container_width=True)
    
    fig_div = make_subplots(specs=[[{"secondary_y": True}]])
    fig_div.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='#64748b')), secondary_y=False)
    fig_div.add_trace(go.Scatter(x=df['Date'], y=df['Div Yield %'], name='Dividend Yield %', line=dict(color='#f59e0b')), secondary_y=True)
    fig_div.update_yaxes(title_text="Close Price", secondary_y=False)
    fig_div.update_yaxes(title_text="Div Yield (Inverted)", autorange="reversed", secondary_y=True)
    fig_div = apply_plotly_layout(fig_div, "Price vs Dividend Yield Cycles")
    st.plotly_chart(fig_div, use_container_width=True)

# -------------------- TAB 5: ANNUAL BREAKDOWN --------------------
with tab5:
    annual_df = get_annual_summary(df)
    
    c1, c2 = st.columns(2)
    with c1:
        colors = ['#00d4aa' if r > 0 else '#ef4444' for r in annual_df['Return']]
        fig_ann_bar = go.Figure(go.Bar(x=annual_df['Year'], y=annual_df['Return']*100, marker_color=colors))
        fig_ann_bar = apply_plotly_layout(fig_ann_bar, "Annual Returns (%)")
        st.plotly_chart(fig_ann_bar, use_container_width=True)
        
    with c2:
        fig_scatter = px.scatter(
            annual_df, x='Volatility', y='Return', size=annual_df['Return'].abs(), color='Return',
            color_continuous_scale='RdYlGn', text='Year', size_max=40
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter = apply_plotly_layout(fig_scatter, "Risk vs Return (Annual)")
        fig_scatter.update_xaxes(tickformat='.1%')
        fig_scatter.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("### Annual Metrics Table")
    st.dataframe(
        annual_df.style.format({
            'Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Max Drawdown': '{:.2%}',
            'Avg P/E': '{:.2f}',
            'Sharpe Ratio': '{:.2f}'
        }).map(lambda x: 'color: #00d4aa' if x > 0 else 'color: #ef4444', subset=['Return']),
        use_container_width=True, hide_index=True
    )

# -----------------------------------------------------------------------------
# DEBUG DATA EXPANDER
# -----------------------------------------------------------------------------
with st.expander("🔍 Debug Data"):
    st.write("Sample of Computed Technical Indicators (Last 5 days)")
    cols_to_show = ['Date', 'Close', 'MA50', 'MA200', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
    st.dataframe(df[cols_to_show].tail(5))


# -----------------------------------------------------------------------------
# DEPLOYMENT COMMANDS
# -----------------------------------------------------------------------------
"""
# =============================================================================
# DEPLOYMENT INSTRUCTIONS
# =============================================================================

# 1. Install Dependencies
# pip install streamlit plotly pandas numpy scipy

# 2. Run Locally
# streamlit run app.py

# 3. Deploy to Streamlit Cloud
# - Push your code to a public or private GitHub repository
# - Include a requirements.txt file with the following contents:
#   streamlit
#   plotly
#   pandas
#   numpy
#   scipy
# - Go to https://share.streamlit.io/
# - Click "New app" -> "Deploy an app"
# - Select your repository, branch, and specify 'app.py' as the main file
# - Click "Deploy"

# =============================================================================
"""

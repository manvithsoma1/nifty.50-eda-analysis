# 📈 Nifty 50 EDA — Interactive Analytics Dashboard

> A professional-grade Exploratory Data Analysis dashboard for the **Nifty 50 Index** (2000–2024), built with **Python · Streamlit · Plotly**.  
> 🔗 **Live App:** [nifty50-eda-analysis.streamlit.app](https://nifty50-eda-analysis-cgtpm3utywnjdnzaf8gysh.streamlit.app/)

---

## 🖥️ Dashboard Preview

| Tab | What You'll See |
|-----|----------------|
| 📊 Price & Indicators | Candlestick chart · MA50/200 · Bollinger Bands · RSI · MACD |
| 📉 Returns & Volatility | Cumulative returns · Drawdown · Rolling volatility · Return distribution |
| 🗓️ Seasonality | Year × Month heatmap · Average monthly return bar chart |
| ⚖️ Valuation | P/E & P/B with percentile bands · Price vs Dividend Yield overlay |
| 📅 Annual Breakdown | Year-by-year return bars · Risk vs Return scatter · Summary table |

---

## 📁 Project Structure

```
nifty-50-eda-analysis/
│
├── nifty50_dashboard.py   # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # You are here
└── data/
    └── nifty50.csv        # (Optional) Your own Nifty 50 CSV data
```

---

## 📊 Dataset

The dashboard works with any CSV containing the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Date` | Trading date | `2000-01-03` |
| `Open` | Opening price | `1482.15` |
| `High` | Intraday high | `1592.90` |
| `Low` | Intraday low | `1482.15` |
| `Close` | Closing price | `1592.20` |
| `P/E` | Price-to-Earnings ratio | `25.91` |
| `P/B` | Price-to-Book ratio | `4.63` |
| `Div Yield %` | Dividend Yield percentage | `0.95` |

> 💡 **No CSV? No problem.** The app auto-generates realistic Nifty 50 sample data so you can explore all features immediately.

---

## ⚙️ Technical Indicators Computed

- **Moving Averages** — MA20, MA50, MA200
- **Bollinger Bands** — 20-day, ±2 standard deviations
- **RSI** — 14-day Relative Strength Index
- **MACD** — 12/26/9 exponential moving average crossover
- **Rolling Volatility** — 30-day and 90-day annualized
- **Drawdown** — Rolling peak-to-trough decline series
- **Daily & Cumulative Returns** — Simple and log returns
- **P/E & P/B Percentile Bands** — Historical valuation zones

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nifty-50-eda-analysis.git
cd nifty-50-eda-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run nifty50_dashboard.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit==1.35.0
altair==5.3.0
plotly==5.22.0
pandas==2.2.2
numpy==1.26.4
```

> **Python version:** 3.9 or higher recommended. Tested on Python 3.11.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set the main file as `nifty50_dashboard.py`
4. Click **Deploy** — live in ~2 minutes ✅

---

## 🔍 Key Insights You Can Explore

- **Identify bull & bear market regimes** using MA200 crossovers
- **Spot overvalued/undervalued entry points** using P/E percentile bands
- **Find the best months to be invested** using the seasonality heatmap
- **Quantify risk-adjusted performance** using drawdown and volatility charts
- **Compare any time period** using the sidebar date range slider

---

## 🛠️ Built With

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [NumPy](https://numpy.org) | Numerical computing |

---

## 👤 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## ⚠️ Disclaimer

This dashboard is built for **educational and research purposes only**.  
It does not constitute financial advice. Past market performance is not indicative of future results.

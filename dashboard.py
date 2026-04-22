import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import pandas as pd
import vectorbt as vbt
import plotly.express as px
import numpy as np
import requests

st.set_page_config(page_title="Sentiment Alpha Engine", layout="wide")
st.title("📈 Sentiment Alpha Engine")

# Load Model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=600) 
def get_latest_headline(symbol):
    try:
        t = yf.Ticker(symbol)
        # Pull the most recent news item
        latest_news = t.news[0]
        return latest_news['title']
    except Exception:
        return f"No recent news found for {symbol}. Enter a headline manually."

# Corrected Fetch Function
def fetch_auto_news(ticker, limit=50):
    # Standard Alpha Vantage URL format: https://alphavantage.co...
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={ALPHA_VANTAGE_API_KEY}'
    r = requests.get(url)
    return r.json()

# Corrected Trend Function
@st.cache_data(ttl=600)
def get_sentiment_trend_av(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=50&apikey={api_key}'
    
    try:
        r = requests.get(url)
        data = r.json()
        
        feed = data.get("feed", [])
        if not feed:
            return pd.DataFrame() 
            
        data_points = []
        for item in feed:
            # Map Alpha Vantage fields
            data_points.append({
                "Date": pd.to_datetime(item.get('time_published')),
                "Headline": item.get('title'),
                # AV uses overall_sentiment_score
                "Sentiment Score": float(item.get('overall_sentiment_score', 0))
            })
        return pd.DataFrame(data_points)
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_sector_comparison(symbol):
    s_obj = yf.Ticker(symbol)
    s_news = s_obj.news[:10]
    scores = []
    for n in s_news:
        title = n.get('title') or n.get('content', {}).get('title')
        if title:
            res = sentiment_pipe(title)
            scores.append(res[0]['score'] if res[0]['label'] == 'positive' else -res[0]['score'] if res[0]['label'] == 'negative' else 0)
    return sum(scores) / len(scores) if scores else 0

@st.cache_data(ttl=3600)
def get_dynamic_sector_etf(symbol):
    try:
        t_info = yf.Ticker(symbol).info
        sector_name = t_info.get('sector', 'Unknown')
        return SECTOR_MAP.get(sector_name, "SPY"), sector_name # Fallback to S&P 500
    except:
        return "SPY", "Market"

def run_realtime_backtest(price_data, sentiment_df, threshold=0.3):
    # Align Sentiment with Price
    combined = pd.merge_asof(price_data, sentiment_df, left_index=True, right_on='Date')
    
    # Generate Signals
    entries = combined['Sentiment Score'] > threshold
    exits = combined['Sentiment Score'] < -threshold
    
    # VectorBT Portfolio Execution
        # 3. Create Signals
    entries = news_copy['Sentiment Score'] > threshold
    exits = news_copy['Sentiment Score'] < 0
        
        # --- NEW SAFETY CHECK ---
    if not entries.any():
        st.warning(f"No headlines found above {threshold}. Try lowering the slider.")
    else:
        # Run VectorBT Portfolio
        pf = vbt.Portfolio.from_signals(
            aligned_prices, 
            entries.vbt.reindex_like(aligned_prices), 
            exits.vbt.reindex_like(aligned_prices), 
            init_cash=init_cash,
            fees=0.001,
            freq='1D'
        )

            # Display Results
        if len(pf.trades) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Return", f"{float(pf.total_return()):.2%}")
            c2.metric("Sharpe Ratio", f"{float(pf.sharpe_ratio()):.2f}" if not pd.isna(pf.sharpe_ratio()) else "N/A")
            c3.metric("Total Trades", len(pf.trades))
            
            st.line_chart(pf.value())
        else:
            st.info("Signals generated, but no trades executed. This usually means the 'Sell' signal happened before the 'Buy'.")

def create_sentiment_heatmap(news_df):
    # Create a local copy and extract time features
    df_copy = news_df.copy()
    df_copy['Hour'] = df_copy['Date'].dt.hour
    df_copy['Day'] = df_copy['Date'].dt.day_name()
    
    # Sort days of the week logically
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create Pivot Table: Rows = Days, Columns = Hours, Values = Mean Sentiment
    heatmap_data = df_copy.pivot_table(
        index='Day', 
        columns='Hour', 
        values='Sentiment Score', 
        aggfunc='mean'
    ).reindex(day_order)
    
    # 4. Fill missing hours with 0 to keep the grid clean
    heatmap_data = heatmap_data.fillna(np.nan)
    
    return heatmap_data

sentiment_pipe = load_model()

# 2. Sidebar Controls
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# 2. Dynamic Headline Fetching
ALPHA_VANTAGE_API_KEY = "67U3KG9YDLA1ILZB"
trend_df = get_sentiment_trend_av(ticker, ALPHA_VANTAGE_API_KEY)

if not trend_df.empty:
    # Automatically pick the newest headline from Alpha Vantage
    auto_headline = trend_df.iloc[0]['Headline']
else:
    auto_headline = "Waiting for news feed..."

headline = st.sidebar.text_area("Analyze News Headline", value=auto_headline)

# Sentiment Analysis Logic
if st.sidebar.button("Run Inference"):
    result = sentiment_pipe(headline)[0]
    label = result['label']
    score = result['score']
    
    col1, col2 = st.columns(2)
    col1.metric("Sentiment", label.upper())
    col2.metric("Confidence", f"{score:.2%}")

if st.sidebar.button("💾 Archive News Session"):
    trend_df.to_csv("alpha_engine_history.csv", mode='a', header=False, index=False)
    st.sidebar.success("Data archived for long-term backtesting!")

# Market Data Visualization
st.subheader(f"{ticker} Price Action")
data = yf.download(ticker, period="1mo", interval="1d")

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    current_price = data['Close'].iloc[-1]
    open_price = data['Open'].iloc[-1]
    change = float(current_price - open_price)
    st.sidebar.metric("Live Price", f"${float(current_price):.2f}", f"{change:.2f}")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    )])
    
    fig.update_layout(
        template="plotly_dark", 
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider() # Adds a nice visual line between sections
st.subheader(f"Recent Sentiment Trend: {ticker}")

if not trend_df.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Date'], 
        y=trend_df['Sentiment Score'],
        mode='lines+markers',
        name='Sentiment',
        hovertext=trend_df['Headline'],
        line=dict(color='#00FFCC', width=3)
    ))
    fig_trend.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_trend.update_layout(
        template="plotly_dark",
        yaxis=dict(title="Score (Neg -1 to Pos +1)", range=[-1.1, 1.1]),
        height=350
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Gathering news data...")

# --- Alpha Signal Logic ---
st.divider()
st.subheader("Alpha Signal Analysis")

if not trend_df.empty:
    avg_sentiment = trend_df['Sentiment Score'].head(3).mean() # Average of last 3 headlines
    
    # Simple Logic: Sentiment + Price Momentum
    if avg_sentiment > 0.2:
        signal = "🚀 BULLISH"
        color = "green"
        reason = "Recent news narrative is strongly positive."
    elif avg_sentiment < -0.2:
        signal = "⚠️ BEARISH"
        color = "red"
        reason = "Recent news narrative is showing significant pressure."
    else:
        signal = "⚖️ NEUTRAL"
        color = "gray"
        reason = "Sentiment is sideways; no strong news catalyst detected."

    st.markdown(f"### Signal: :{color}[{signal}]")
    st.info(f"**Reasoning:** {reason}")
else:
    st.write("Waiting for more news data to generate an Alpha Signal...")

st.divider()
st.subheader("Top Alpha Drivers")
if not trend_df.empty:
    top_movers = trend_df.sort_values(by="Sentiment Score", ascending=False)
    st.table(top_movers[['Date', 'Headline', 'Sentiment Score']].head(10))

SECTOR_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC"
}

# --- Dynamic Sector Benchmarking ---
st.subheader("Relative Sentiment: Stock vs. Dynamic Sector")

# Get the dynamic ticker and sector name
dynamic_etf, sector_display = get_dynamic_sector_etf(ticker)

# Fetch the comparison scores
sector_avg = get_sector_comparison(dynamic_etf)
stock_avg = trend_df['Sentiment Score'].mean()

# UI Display
st.markdown(f"Benchmarking **{ticker}** against the **{sector_display}** sector (via {dynamic_etf}).")
c1, c2 = st.columns(2)
c1.metric(f"{ticker} Avg", f"{stock_avg:.2f}")
c2.metric(f"{dynamic_etf} ({sector_display})", f"{sector_avg:.2f}", delta=f"{stock_avg - sector_avg:.2f}")

if stock_avg > sector_avg:
    st.success(f"{ticker} is outperforming its peers in the **{sector_display}** sector.")
else:
    st.warning(f"{ticker} is lagging the **{sector_display}** sector narrative.")


# Sidebar Controls for the Backtest
st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Performance Analysis")
threshold = st.sidebar.slider("Sentiment Buy Threshold", 0.0, 1.0, 0.3)
init_cash = st.sidebar.number_input("Initial Cash ($)", value=10000)

st.divider()
st.subheader("Real-Time Backtest Performance")

if not trend_df.empty and not data.empty:
    try:
        # Standardize Data
        price_series = data['Close'].copy().tz_localize(None)
        news_df = trend_df.copy()
        news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)

        # Create an index 
        full_index = pd.date_range(start=price_series.index.min(), end=news_df['Date'].max(), freq='D')
        
        # 'Forward Fill' Friday's price into Saturday so the engine has a price to trade
        extended_prices = price_series.reindex(full_index, method='ffill')
        
        # Group news by day and align to the new timeline
        daily_sentiment = news_df.groupby(news_df['Date'].dt.normalize())['Sentiment Score'].mean()
        aligned_sentiment = daily_sentiment.reindex(full_index, fill_value=0)

        # Signals & Portfolio
        entries = aligned_sentiment > threshold
        exits = aligned_sentiment < -0.1

        pf = vbt.Portfolio.from_signals(
            extended_prices, 
            entries, 
            exits, 
            init_cash=init_cash,
            fees=0.001,
            freq='1D',
            upon_long_conflict='Entry'
        )

        # 4. Display Results
        if len(pf.trades) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Return", f"{float(pf.total_return()):.2%}")
            c2.metric("Sharpe Ratio", f"{float(pf.sharpe_ratio()):.2f}" if not pd.isna(pf.sharpe_ratio()) else "N/A")
            c3.metric("Total Trades", len(pf.trades))
            st.line_chart(pf.value())
        else:
            st.warning("Still waiting for a signal. Try setting the threshold slider lower.")

    except Exception as e:
        st.error(f"Engine Alignment Error: {e}")
else:
    st.info("Gathering historical data to initialize backtest...")


# --- Sentiment Heatmap Section ---
st.divider()
st.subheader(f"Sentiment Heatmap: {ticker} (By Hour/Day)")

if not trend_df.empty:
    heatmap_df = create_sentiment_heatmap(trend_df)
    
    # Use a simpler imshow call to avoid layout conflicts
    fig_heatmap = px.imshow(
        heatmap_df,
        labels=dict(x="Hour of Day (24h)", y="Day of Week", color="Sentiment"),
        x=list(heatmap_df.columns),
        y=list(heatmap_df.index),
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    
    # Update layout separately to ensure compatibility
    fig_heatmap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.write("Insufficient news data to generate a heatmap.")

# --- Volume Correlation Section ---
st.divider()
st.subheader(f"News Sentiment vs. Trading Volume: {ticker}")

if not trend_df.empty and not data.empty:
    # 1. Standardize both to be timezone-naive
    # Ensure volume index is clean
    daily_vol = data['Volume'].resample('D').mean()
    if daily_vol.index.tz is not None:
        daily_vol.index = daily_vol.index.tz_localize(None)
    
    # Ensure sentiment index is clean
    daily_sent = trend_df.groupby(trend_df['Date'].dt.normalize())['Sentiment Score'].mean()
    if daily_sent.index.tz is not None:
        daily_sent.index = daily_sent.index.tz_localize(None)
    
    # 2. Join the data now that they match
    corr_df = pd.DataFrame({
        'Volume': daily_vol,
        'Sentiment': daily_sent
    }).dropna()

    if not corr_df.empty:
        # 2. Create a Dual-Axis Chart
        fig_corr = go.Figure()

        # Add Volume Bar Chart
        fig_corr.add_trace(go.Bar(
            x=corr_df.index, y=corr_df['Volume'],
            name="Trading Volume", marker_color='rgba(100, 100, 100, 0.3)',
            yaxis="y2"
        ))

        # Add Sentiment Line Chart
        fig_corr.add_trace(go.Scatter(
            x=corr_df.index, y=corr_df['Sentiment'],
            name="Avg Sentiment", line=dict(color='#00FFCC', width=3)
        ))

        # Setup Dual Axis Layout
        fig_corr.update_layout(
            template="plotly_dark",
            yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1]),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            height=400
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough overlapping daily data yet to calculate volume correlation.")
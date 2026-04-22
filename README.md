# Sentiment-Alpha-Engine
[https://sentiment-alpha-engine-equities-forecasting.streamlit.app/]

An end-to-end financial intelligence dashboard that uses **NLP (Natural Language Processing)** to quantify market sentiment and run real-time trading backtests. 

This engine bridges the gap between news headlines and price action, using **FinBERT** (a BERT model pre-trained on financial data) to detect market-moving narratives before they reflect in the price.

---

## ✨ Key Features

*   **Real-Time Sentiment Analysis:** Leverages HuggingFace's `ProsusAI/finbert` to analyze headlines for Bullish, Bearish, or Neutral sentiment.
*   **Dynamic Sector Benchmarking:** Automatically identifies a stock's sector (e.g., Technology, Energy) and compares its sentiment score against the relevant sector ETF (e.g., XLK, XLE).
*   **VectorBT Backtesting:** Simulates a trading strategy where "Buy" signals are triggered by sentiment spikes above a user-defined threshold.
*   **Sentiment Heatmap:** Visualizes "sentiment clusters" by day of the week and hour of the day to identify when news impact is highest.
*   **Volume Correlation:** Overlays average sentiment against trading volume to identify high-conviction market moves.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io)
*   **LLM/NLP:** [HuggingFace Transformers](https://huggingface.co) (FinBERT)
*   **Financial Data:** [yfinance](https://github.com) & [Alpha Vantage API](https://alphavantage.co)
*   **Backtesting:** [VectorBT](https://vectorbt.dev)
*   **Visualization:** Plotly & Plotly Express

## 🚦 Getting Started

### Prerequisites
* Python 3.9+
* An Alpha Vantage API Key ([Get API Key](https://alphavantage.cosupport/#api-key))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com
    cd sentiment-alpha-engine
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Secrets:**
    To run locally, create a `.streamlit/secrets.toml` file:
    ```toml
    ALPHA_VANTAGE_API_KEY = "your_api_key_here"
    ```

5.  **Run the app:**
    ```bash
    streamlit run dashboard.py
    ```

## 📈 Analysis Methodology

The engine calculates an **"Alpha Signal"** by analyzing the narrative density of recent news:
*   **Bullish:** Sentiment Score > 0.2 (Strong positive narrative)
*   **Bearish:** Sentiment Score < -0.2 (Significant news pressure)
*   **Neutral:** Sentiment between -0.2 and 0.2 (Sideways narrative)

---

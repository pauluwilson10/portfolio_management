import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info
from data_fetcher import fetch_stock_data
from sentiment_analysis import analyze_sentiment
from portfolio_analyzer import (
    calculate_enhanced_risk_score, 
    rebalance_portfolio_enhanced, 
    get_rebalancing_insights,
    calculate_portfolio_metrics
)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from utils import plot_price_trends

st.set_page_config(page_title="AI Portfolio Manager", layout="wide", initial_sidebar_state="expanded")
st.title("📈 AI-Powered Active Portfolio Manager")
 

# Add fallback functions at the top after imports
def calculate_fallback_risk_score(data, sentiment_scores):
    """Fallback risk calculation if the enhanced version fails"""
    risk_scores = {}
    
    for ticker, metrics in data.items():
        try:
            history = metrics.get("history")
            if history is None or len(history) < 10:
                risk_scores[ticker] = 0.5
                continue
            
            returns = history['Close'].pct_change().dropna()
            if len(returns) == 0:
                risk_scores[ticker] = 0.5
                continue
            
            # Basic volatility calculation
            recent_returns = returns.tail(min(30, len(returns)))
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            prices = history['Close'].tail(min(60, len(history)))
            cumulative = (1 + prices.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Sentiment adjustment
            sentiment = sentiment_scores.get(ticker, 0)
            sentiment_risk = (1 - sentiment) * 0.2
            
            # Combine risk factors
            combined_risk = (volatility * 0.5 + max_drawdown * 0.3 + sentiment_risk * 0.2)
            risk_score = min(max(combined_risk, 0.1), 0.9)
            risk_scores[ticker] = round(risk_score, 4)
            
        except Exception as e:
            print(f"Error calculating risk for {ticker}: {e}")
            risk_scores[ticker] = 0.5
    
    return risk_scores

def calculate_simple_portfolio_metrics(allocations, data):
    """Simplified portfolio metrics calculation"""
    try:
        all_returns = {}
        weights = []
        tickers = []
        
        for ticker, weight in allocations.items():
            if ticker in data and weight > 0:
                returns = data[ticker]['history']['Close'].pct_change().dropna()
                if len(returns) >= 30:
                    all_returns[ticker] = returns.tail(60)
                    weights.append(weight)
                    tickers.append(ticker)
        
        if not all_returns:
            return None
        
        min_length = min(len(returns) for returns in all_returns.values())
        if min_length < 20:
            return None
        
        portfolio_returns = np.zeros(min_length)
        for i, (ticker, weight) in enumerate(zip(tickers, weights)):
            returns_aligned = all_returns[ticker].tail(min_length).values
            portfolio_returns += weight * returns_aligned
        
        portfolio_returns = pd.Series(portfolio_returns)
        volatility = portfolio_returns.std() * np.sqrt(252)
        mean_return = portfolio_returns.mean() * 252
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        return {
            'returns': portfolio_returns,
            'volatility': volatility,
            'sharpe': sharpe_ratio,
            'mean_return': mean_return
        }
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.risk-high { color: #ff4444; }
.risk-medium { color: #ffaa00; }
.risk-low { color: #00aa44; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("🔍 Portfolio Configuration")

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    lstm_epochs = st.slider("LSTM Training Epochs", 20, 100, 50)
    look_back_days = st.slider("Look-back Period (days)", 10, 30, 20)
    target_return = st.slider("Target Annual Return", 0.05, 0.25, 0.12, 0.01)
    include_market_beta = st.checkbox("Include Market Beta Analysis", value=True)

# --- Market Data for Beta Calculation ---
market_data = None
if include_market_beta:
    try:
        market_ticker = st.sidebar.selectbox(
            "Market Benchmark:", 
            ["^GSPC", "^IXIC", "^DJI", "^RUT"],
            format_func=lambda x: {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^RUT": "Russell 2000"}[x]
        )
        market_data = yf.download(market_ticker, period="1y")['Close']
    except:
        st.sidebar.warning("Could not fetch market data for beta calculation")

# --- Ticker Helper with Enhanced Features ---
st.sidebar.header("🔍 Ticker Helper")
all_tickers = stock_info.tickers_nasdaq() + stock_info.tickers_other()
ticker_choice = st.sidebar.selectbox(
    "Type or select a ticker:",
    options=[""] + all_tickers
)

if ticker_choice:
    try:
        ticker_obj = yf.Ticker(ticker_choice)
        info = ticker_obj.info
        if info.get("shortName"):
            st.sidebar.success(f"**{info['shortName']}** ({ticker_choice})")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Price", f"${info.get('currentPrice', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A')
                st.write(f"**Beta:** {info.get('beta', 'N/A')}")
        else:
            st.sidebar.error("Ticker not recognized.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- Portfolio Upload & Processing ---
st.header("📤 Portfolio Upload")
uploaded = st.file_uploader(
    "Upload your portfolio CSV (columns: Ticker, Quantity)", 
    type="csv",
    help="CSV should contain 'Ticker' and 'Quantity' columns"
)

if uploaded:
    try:
        portfolio = pd.read_csv(uploaded)
        
        # Validate portfolio format
        required_columns = ['Ticker', 'Quantity']
        if not all(col in portfolio.columns for col in required_columns):
            st.error(f"Portfolio CSV must contain columns: {required_columns}")
            st.stop()
        
        tickers = portfolio['Ticker'].str.upper().tolist()
        quantities = portfolio['Quantity'].tolist()
        
        # Portfolio summary
        st.subheader("📋 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(tickers))
        with col2:
            st.metric("Total Shares", sum(quantities))
        with col3:
            st.metric("Avg Shares/Stock", f"{np.mean(quantities):.1f}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data
        status_text.text("🔄 Fetching real-time market data...")
        progress_bar.progress(20)
        data, sector_map = fetch_stock_data(tickers)
        
        # Analyze sentiment
        status_text.text("🧠 Analyzing market sentiment...")
        progress_bar.progress(40)
        sentiments = {}
        for i, ticker in enumerate(tickers):
            sentiments[ticker] = analyze_sentiment(ticker)
            progress_bar.progress(40 + int(20 * (i + 1) / len(tickers)))
        
        # Calculate enhanced risk scores
        status_text.text("📊 Calculating enhanced risk scores...")
        progress_bar.progress(70)
        try:
            risk_scores = calculate_enhanced_risk_score(data, sentiments, market_data)
            # Validate risk scores
            if all(score == 0.5 for score in risk_scores.values()):
                raise ValueError("All risk scores are default values - using fallback calculation")
        except Exception as e:
            risk_scores = calculate_fallback_risk_score(data, sentiments)
        
        # Calculate current allocations
        current_allocations = {}
        total_value = 0
        for ticker, quantity in zip(tickers, quantities):
            if ticker in data:
                value = quantity * data[ticker]['current_price']
                total_value += value
        
        for ticker, quantity in zip(tickers, quantities):
            if ticker in data:
                value = quantity * data[ticker]['current_price']
                current_allocations[ticker] = value / total_value if total_value > 0 else 1/len(tickers)
        
        # Portfolio rebalancing
        status_text.text("🔄 Optimizing portfolio allocation...")
        progress_bar.progress(90)
        recommendation = rebalance_portfolio_enhanced(
            current_allocations, risk_scores, data, target_return
        )
        
        # Get insights
        insights = get_rebalancing_insights(current_allocations, recommendation, risk_scores, data)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators after a brief pause
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # === RESULTS DISPLAY ===
        st.header("📊 Portfolio Analysis Results")
        
        # Calculate portfolio metrics
        try:
            portfolio_metrics = calculate_portfolio_metrics(current_allocations, data)
            if portfolio_metrics is None:
                portfolio_metrics = calculate_simple_portfolio_metrics(current_allocations, data)
        except Exception as e:
            portfolio_metrics = calculate_simple_portfolio_metrics(current_allocations, data)
        
        # Key Metrics
        col1, col2 = st.columns(2)
        with col1:
            avg_risk = np.mean(list(risk_scores.values()))
            risk_color = "🔴" if avg_risk > 0.6 else "🟡" if avg_risk > 0.3 else "🟢"
            st.metric("Average Risk Score", f"{avg_risk:.3f} {risk_color}")
        
        with col2:
            total_portfolio_value = sum(q * data[t]['current_price'] for t, q in zip(tickers, quantities) if t in data)
            st.metric("Portfolio Value", f"${total_portfolio_value:,.2f}")
        
         
        # Risk Scores Table
        st.subheader("📊 Enhanced Risk Analysis")
        risk_df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Risk Score': f"{score:.3f}",
                'Risk Level': 'High' if score > 0.6 else 'Medium' if score > 0.3 else 'Low',
                'Current Price': f"${data[ticker]['current_price']:.2f}" if ticker in data else 'N/A',
                'Sector': sector_map.get(ticker, 'Unknown')
            }
            for ticker, score in risk_scores.items()
        ])
        
        st.dataframe(
            risk_df.style.applymap(
                lambda x: 'color: red' if 'High' in str(x) else 'color: orange' if 'Medium' in str(x) else 'color: green' if 'Low' in str(x) else '',
                subset=['Risk Level']
            ),
            use_container_width=True
        )
        
        # Portfolio Allocation Comparison
        st.subheader("🔄 Portfolio Rebalancing Recommendations")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Ticker': list(current_allocations.keys()),
            'Current Allocation': [f"{current_allocations[t]:.1%}" for t in current_allocations.keys()],
            'Recommended Allocation': [f"{recommendation.get(t, 0):.1%}" for t in current_allocations.keys()],
            'Change': [f"{(recommendation.get(t, 0) - current_allocations[t]):.1%}" for t in current_allocations.keys()],
            #'Risk Score': [f"{risk_scores.get(t, 0):.3f}" for t in current_allocations.keys()]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Rebalancing Insights
        if insights:
            st.subheader("💡 Key Rebalancing Insights")
            for insight in insights:
                st.info(insight)
        
        # Visualizations
        st.subheader("📈 Portfolio Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Allocation Comparison", "Risk Distribution", "Price Trends", "Sector Analysis"])
        
        with tab1:
            # Allocation comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                subplot_titles=("Current Allocation", "Recommended Allocation")
            )
            
            # Current allocation pie chart
            fig.add_trace(
                go.Pie(
                    labels=list(current_allocations.keys()),
                    values=list(current_allocations.values()),
                    name="Current"
                ),
                row=1, col=1
            )
            
            # Recommended allocation pie chart
            fig.add_trace(
                go.Pie(
                    labels=list(recommendation.keys()),
                    values=list(recommendation.values()),
                    name="Recommended"
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Risk distribution
            risk_fig = px.bar(
                x=list(risk_scores.keys()),
                y=list(risk_scores.values()),
                title="Risk Score Distribution",
                labels={'x': 'Ticker', 'y': 'Risk Score'},
                color=list(risk_scores.values()),
                color_continuous_scale='RdYlGn_r'
            )
            risk_fig.update_layout(height=400)
            st.plotly_chart(risk_fig, use_container_width=True)
        
        with tab3:
            # Price trends (using your existing function)
            st.pyplot(plot_price_trends(data))
        
        with tab4:
            # Sector analysis
            sector_risk = {}
            for ticker, sector in sector_map.items():
                if sector not in sector_risk:
                    sector_risk[sector] = []
                if ticker in risk_scores:
                    sector_risk[sector].append(risk_scores[ticker])
            
            sector_avg_risk = {sector: np.mean(risks) for sector, risks in sector_risk.items() if risks}
            
            if sector_avg_risk:
                sector_fig = px.bar(
                    x=list(sector_avg_risk.keys()),
                    y=list(sector_avg_risk.values()),
                    title="Average Risk Score by Sector",
                    labels={'x': 'Sector', 'y': 'Average Risk Score'},
                    color=list(sector_avg_risk.values()),
                    color_continuous_scale='RdYlGn_r'
                )
                sector_fig.update_layout(height=400)
                st.plotly_chart(sector_fig, use_container_width=True)
        
        # Download recommendations
        st.subheader("💾 Export Results")
        
        # Create comprehensive results dataframe
        results_df = pd.DataFrame({
            'Ticker': list(current_allocations.keys()),
            'Current_Allocation': list(current_allocations.values()),
            'Recommended_Allocation': [recommendation.get(t, 0) for t in current_allocations.keys()],
            'Risk_Score': [risk_scores.get(t, 0) for t in current_allocations.keys()],
            'Current_Price': [data[t]['current_price'] if t in data else 0 for t in current_allocations.keys()],
            'Sector': [sector_map.get(t, 'Unknown') for t in current_allocations.keys()]
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Analysis",
            data=csv,
            file_name="portfolio_analysis_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing portfolio: {str(e)}")
        st.info("Please ensure your CSV has 'Ticker' and 'Quantity' columns with valid data.")

# Footer
st.markdown("---")
 
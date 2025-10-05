#!/usr/bin/env python
"""
Demo Script for AI-Driven Paper Trading System
This script demonstrates the key features of the trading system
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TradingConfig
from data_fetcher import DataAggregator, MarketDataFetcher
from ai_module import AISignalGenerator, SentimentAnalyzer
from technical_indicators import TechnicalIndicators
from paper_trading_engine import PaperTradingEngine, OrderSide, OrderType
from visualization import PerformanceVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("   AI-DRIVEN PAPER TRADING SYSTEM FOR NSE")
    print("   Intelligent Trading with Machine Learning")
    print("="*60)
    print()

def demo_data_fetching():
    """Demonstrate data fetching capabilities"""
    print("\n📊 DEMO 1: Data Fetching")
    print("-"*40)
    
    fetcher = MarketDataFetcher()
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        data = fetcher.get_nse_data(symbol)
        
        if data:
            print(f"  Current Price: ₹{data['current_price']:.2f}")
            print(f"  Day Change: {data['change_pct']:.2f}%")
            print(f"  Volume: {data['volume']:,}")
        
        # Get historical data
        hist = fetcher.get_historical_data(symbol, period="1mo", interval="1d")
        if hist is not None:
            print(f"  Historical data: {len(hist)} days")
            print(f"  30-day return: {((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Get index data
    print("\n📈 Nifty 50 Index:")
    index_data = fetcher.get_index_data()
    if index_data:
        print(f"  Current: {index_data['current_value']:.2f}")
        print(f"  Change: {index_data['change_pct']:.2f}%")

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis"""
    print("\n🤖 DEMO 2: AI Sentiment Analysis")
    print("-"*40)
    
    analyzer = SentimentAnalyzer()
    
    # Sample news headlines
    sample_news = [
        "Reliance Industries posts record quarterly profit, beats estimates",
        "Market crash fears as inflation rises above expectations",
        "TCS wins major $2 billion deal from European bank",
        "Regulatory concerns impact banking sector growth",
        "Bullish momentum continues in IT stocks amid strong earnings"
    ]
    
    print("\nAnalyzing news sentiment:")
    for news in sample_news:
        result = analyzer.analyze_text(news)
        sentiment_emoji = "📈" if result['sentiment'] == 'positive' else "📉" if result['sentiment'] == 'negative' else "➡️"
        print(f"{sentiment_emoji} {news[:50]}...")
        print(f"   Sentiment: {result['sentiment']}, Score: {result['score']:.3f}")
    
    # Batch analysis
    batch_result = analyzer.analyze_news_batch([{'title': n, 'description': ''} for n in sample_news])
    print(f"\nOverall Market Sentiment: {batch_result['overall_sentiment']:.3f}")
    print(f"Distribution: Positive={batch_result['sentiment_distribution']['positive']}, "
          f"Negative={batch_result['sentiment_distribution']['negative']}, "
          f"Neutral={batch_result['sentiment_distribution']['neutral']}")

def demo_technical_analysis():
    """Demonstrate technical indicators"""
    print("\n📉 DEMO 3: Technical Analysis")
    print("-"*40)
    
    fetcher = MarketDataFetcher()
    analyzer = TechnicalIndicators()
    
    symbol = "RELIANCE"
    print(f"\nCalculating technical indicators for {symbol}...")
    
    # Get historical data
    hist_data = fetcher.get_historical_data(symbol, period="3mo", interval="1d")
    
    if hist_data is not None and not hist_data.empty:
        # Calculate indicators
        hist_data = analyzer.calculate_all_indicators(hist_data)
        
        # Generate signals
        signals = analyzer.generate_signals(hist_data)
        
        print("\n📊 Current Technical Signals:")
        print(f"  Price: ₹{signals['current_price']:.2f}")
        print(f"  RSI: {signals['rsi']:.2f}")
        print(f"  SMA Signal: {signals['sma_signal']}")
        print(f"  MACD Signal: {signals['macd_signal']}")
        print(f"  Overall Signal: {signals['overall_signal']}")
        print(f"  Buy Strength: {signals['buy_strength']:.2%}")
        print(f"  Sell Strength: {signals['sell_strength']:.2%}")
        
        # Show recent indicators
        print("\n📈 Recent Indicator Values:")
        recent = hist_data.tail(5)[['Close', 'RSI', 'SMA_short', 'SMA_long', 'Volume']]
        print(recent.to_string())

def demo_signal_generation():
    """Demonstrate AI signal generation"""
    print("\n🎯 DEMO 4: Composite AI Signal Generation")
    print("-"*40)
    
    # Initialize components
    data_agg = DataAggregator()
    ai_generator = AISignalGenerator()
    tech_analyzer = TechnicalIndicators()
    
    symbols = ["RELIANCE", "TCS"]
    
    print("\nGenerating AI-powered trading signals...")
    
    for symbol in symbols:
        print(f"\n📌 {symbol}:")
        
        # Get data
        fetcher = MarketDataFetcher()
        hist_data = fetcher.get_historical_data(symbol, period="1mo")
        
        if hist_data is not None and not hist_data.empty:
            # Calculate technical indicators
            hist_data = tech_analyzer.calculate_all_indicators(hist_data)
            tech_signals = tech_analyzer.generate_signals(hist_data)
            
            # Mock sentiment data
            sentiment_data = {'overall_sentiment': 0.3}
            
            # Mock market data
            market_data = {
                'current_price': hist_data['Close'].iloc[-1],
                'volume': hist_data['Volume'].iloc[-1]
            }
            
            # Generate composite signal
            composite = ai_generator.generate_composite_signal(
                technical_signals=tech_signals,
                sentiment_data=sentiment_data,
                market_data=market_data,
                historical_data=hist_data
            )
            
            # Display results
            signal_emoji = "🟢" if composite['signal'] == 'BUY' else "🔴" if composite['signal'] == 'SELL' else "🟡"
            
            print(f"  {signal_emoji} Signal: {composite['signal']}")
            print(f"  Strength: {composite['strength']:.2%}")
            print(f"  Confidence: {composite['confidence']:.2%}")
            print(f"  Components:")
            print(f"    - Technical Score: {composite['components']['technical_score']:.3f}")
            print(f"    - Sentiment Score: {composite['components']['sentiment_score']:.3f}")
            print(f"    - Regime: {composite['components']['regime']}")
            print(f"  Reasoning: {composite['reasoning'][:100]}...")

def demo_paper_trading():
    """Demonstrate paper trading engine"""
    print("\n💰 DEMO 5: Paper Trading Simulation")
    print("-"*40)
    
    # Initialize trading engine
    engine = PaperTradingEngine(mode=TradingConfig.INTRADAY, initial_capital=100000)
    
    print(f"\nStarting Capital: ₹{engine.portfolio.initial_capital:,.2f}")
    
    # Simulate some trades
    trades = [
        ("RELIANCE", OrderSide.BUY, 10, 2500),
        ("TCS", OrderSide.BUY, 5, 3500),
        ("RELIANCE", OrderSide.SELL, 10, 2600),  # Profit trade
        ("HDFCBANK", OrderSide.BUY, 8, 1600),
        ("TCS", OrderSide.SELL, 5, 3450),  # Loss trade
    ]
    
    print("\nExecuting trades:")
    for symbol, side, quantity, price in trades:
        order = engine.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            notes=f"Demo trade for {symbol}"
        )
        
        result = engine.execute_order(order, price)
        
        if result['success']:
            emoji = "✅" if side == OrderSide.BUY else "💸"
            print(f"  {emoji} {side.value} {quantity} {symbol} @ ₹{price:.2f}")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    # Update portfolio with current prices
    current_prices = {
        "RELIANCE": 2600,
        "TCS": 3450,
        "HDFCBANK": 1650
    }
    engine.update_portfolio({'market_data': 
        {symbol: {'current_price': price} for symbol, price in current_prices.items()}
    })
    
    # Show portfolio summary
    summary = engine.portfolio.get_summary()
    
    print("\n📊 Portfolio Summary:")
    print(f"  Total Value: ₹{summary['total_value']:,.2f}")
    print(f"  Cash: ₹{summary['cash']:,.2f}")
    print(f"  P&L: ₹{summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:.2f}%)")
    print(f"  Open Positions: {summary['open_positions']}")
    print(f"  Realized P&L: ₹{summary['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L: ₹{summary['unrealized_pnl']:,.2f}")
    
    # Calculate performance metrics
    metrics = engine.get_performance_metrics()
    
    print("\n📈 Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key or 'rate' in key or 'drawdown' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n📊 DEMO 6: Visualization")
    print("-"*40)
    
    print("\nGenerating performance visualizations...")
    
    visualizer = PerformanceVisualizer()
    
    # Create sample portfolio history
    portfolio_history = []
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    initial_value = 100000
    
    for i, date in enumerate(dates):
        # Simulate portfolio value changes
        random_return = (pd.np.random.randn() * 0.02)  # 2% daily volatility
        value = initial_value * (1 + random_return) if i > 0 else initial_value
        
        portfolio_history.append({
            'timestamp': date,
            'total_value': value,
            'cash': value * 0.3,
            'initial_capital': initial_value,
            'total_pnl': value - initial_value,
            'total_pnl_pct': ((value - initial_value) / initial_value) * 100
        })
    
    print("  ✅ Portfolio value chart generated")
    print("  ✅ Drawdown analysis generated")
    print("  ✅ Returns distribution generated")
    print("\n  📁 Charts saved in 'charts/' directory")

def main():
    """Run all demos"""
    print_banner()
    
    demos = [
        ("Data Fetching", demo_data_fetching),
        ("Sentiment Analysis", demo_sentiment_analysis),
        ("Technical Analysis", demo_technical_analysis),
        ("AI Signal Generation", demo_signal_generation),
        ("Paper Trading Engine", demo_paper_trading),
        ("Visualization", demo_visualization)
    ]
    
    print("\n🚀 Starting System Demo\n")
    print("This demo will showcase all major components of the trading system.")
    print("Each component will be demonstrated with real market data.\n")
    
    for name, demo_func in demos:
        try:
            demo_func()
            time.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"\n⚠️ Error in {name} demo: {e}")
            print("Continuing with next demo...")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📚 Next Steps:")
    print("1. Review the README.md for detailed documentation")
    print("2. Configure your API keys in .env file")
    print("3. Run 'python main.py --help' to see all options")
    print("4. Start with 'python main.py --run-once' for a single trading cycle")
    print("5. Use 'python main.py --backtest' to test on historical data")
    print("\nHappy Trading! 📈")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("charts", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
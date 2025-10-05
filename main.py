"""
Main Paper Trading System Orchestrator
Advanced AI-driven trading system for NSE markets
"""

"""
Main Paper Trading System Orchestrator
Advanced AI-driven trading system for NSE markets
"""

import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print(os.getenv('NEWS_API_KEY'))  # This will print your API key if everything is set up correctly

import argparse
from typing import Dict, List, Optional
import schedule
import threading
# ... rest of your imports and code


# Load environment variables from .env file
load_dotenv()

# Import all modules
from config import TradingConfig, MARKET_HOURS, SECTOR_MAPPING
from data_fetcher import DataAggregator, MarketDataFetcher, NewsDataFetcher
from ai_module import AISignalGenerator, SentimentAnalyzer, MarketRegimeDetector, LLMAnalyzer
from technical_indicators import TechnicalIndicators
from paper_trading_engine import PaperTradingEngine, OrderSide
from visualization import PerformanceVisualizer

# ... your entire existing class and function definitions stay as they are ...


# Setup logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(TradingConfig.LOG_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, TradingConfig.LOG_LEVEL),
        format=TradingConfig.LOG_FORMAT,
        handlers=[
            logging.FileHandler(
                f"{TradingConfig.LOG_DIR}/trading_{datetime.now().strftime('%Y%m%d')}.log"
            ),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class TradingSystemOrchestrator:
    """
    Main orchestrator for the paper trading system
    Combines all modules and manages the trading workflow
    """
    
    def __init__(self, mode: str = TradingConfig.INTRADAY, 
                 symbols: List[str] = None,
                 enable_llm: bool = False):
        """
        Initialize the trading system
        
        Args:
            mode: Trading mode (intraday/swing)
            symbols: List of symbols to trade
            enable_llm: Whether to enable LLM analysis
        """
        self.mode = mode
        self.symbols = symbols or TradingConfig.DEFAULT_SYMBOLS
        self.config = TradingConfig.get_config(mode)
        
        # Initialize components
        logger.info("Initializing Trading System Components...")
        
        self.data_aggregator = DataAggregator()
        self.technical_analyzer = TechnicalIndicators()
        self.ai_signal_generator = AISignalGenerator()
        self.trading_engine = PaperTradingEngine(mode)
        self.visualizer = PerformanceVisualizer()
        
        # Optional LLM analyzer
        self.llm_analyzer = None
        if enable_llm:
            api_key = TradingConfig.OPENAI_API_KEY or TradingConfig.CLAUDE_API_KEY
            if api_key:
                provider = "openai" if TradingConfig.OPENAI_API_KEY else "claude"
                self.llm_analyzer = LLMAnalyzer(api_key, provider)
                logger.info(f"LLM Analyzer enabled with {provider}")
        
        # State tracking
        self.is_running = False
        self.last_update = None
        self.signal_history = []
        self.benchmark_data = None
        
        # Create necessary directories
        os.makedirs("charts", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        logger.info(f"Trading System initialized in {mode} mode with {len(self.symbols)} symbols")
    
    def fetch_and_process_data(self) -> Dict:
        """
        Fetch and process all required data
        """
        logger.info("Fetching market data...")
        
        # Get comprehensive data
        data = self.data_aggregator.get_comprehensive_data(self.symbols)
        
        # Process technical indicators for each symbol
        for symbol in self.symbols:
            if symbol in data['historical_data']:
                hist_df = data['historical_data'][symbol]
                
                # Calculate technical indicators
                hist_df = self.technical_analyzer.calculate_all_indicators(hist_df)
                data['historical_data'][symbol] = hist_df
                
                # Generate technical signals
                signals = self.technical_analyzer.generate_signals(hist_df)
                
                if 'technical_signals' not in data:
                    data['technical_signals'] = {}
                data['technical_signals'][symbol] = signals
        
        # Analyze sentiment
        if data.get('news_data'):
            sentiment_results = self.ai_signal_generator.sentiment_analyzer.calculate_market_sentiment(
                data.get('market_news', []),
                data.get('news_data', {})
            )
            data['sentiment_analysis'] = sentiment_results
        
        self.last_update = datetime.now()
        return data
    
    def generate_trading_signals(self, data: Dict) -> Dict[str, Dict]:
        """
        Generate trading signals for all symbols
        """
        signals = {}
        
        for symbol in self.symbols:
            try:
                # Check if we have required data
                if symbol not in data.get('market_data', {}):
                    logger.warning(f"No market data for {symbol}")
                    continue
                
                if symbol not in data.get('technical_signals', {}):
                    logger.warning(f"No technical signals for {symbol}")
                    continue
                
                # Get sentiment for this symbol
                symbol_sentiment = {}
                if 'sentiment_analysis' in data:
                    if symbol in data['sentiment_analysis'].get('stocks', {}):
                        symbol_sentiment = data['sentiment_analysis']['stocks'][symbol]
                
                # Generate composite signal
                signal = self.ai_signal_generator.generate_composite_signal(
                    technical_signals=data['technical_signals'][symbol],
                    sentiment_data=symbol_sentiment,
                    market_data=data['market_data'][symbol],
                    historical_data=data.get('historical_data', {}).get(symbol)
                )
                
                # Add LLM analysis if available
                if self.llm_analyzer and signal['signal'] != 'HOLD':
                    llm_insight = self.llm_analyzer.analyze_market_context(
                        data['market_data'][symbol],
                        data.get('news_data', {}).get(symbol, [])[:3]
                    )
                    if llm_insight:
                        signal['llm_analysis'] = llm_insight
                
                signals[symbol] = signal
                
                # Log signal
                logger.info(f"Signal for {symbol}: {signal['signal']} "
                          f"(Strength: {signal['strength']:.2f}, "
                          f"Confidence: {signal['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def execute_trades(self, signals: Dict[str, Dict], market_data: Dict) -> List[Dict]:
        """
        Execute trades based on signals
        """
        executed_trades = []
        
        for symbol, signal in signals.items():
            try:
                # Get market data for symbol
                symbol_data = market_data.get('market_data', {}).get(symbol)
                if not symbol_data:
                    continue
                
                # Process signal through trading engine
                order = self.trading_engine.process_signal(symbol, signal, symbol_data)
                
                if order:
                    executed_trades.append({
                        'symbol': symbol,
                        'order': order.to_dict(),
                        'signal': signal
                    })
                    
                    # Store in signal history
                    self.signal_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': signal,
                        'executed': order is not None
                    })
                    
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
        
        return executed_trades
    
    def update_portfolio(self, market_data: Dict):
        """
        Update portfolio with latest prices
        """
        self.trading_engine.update_portfolio(market_data.get('market_data', {}))
    
    def calculate_performance(self) -> Dict:
        """
        Calculate performance metrics
        """
        # Get benchmark data if available
        benchmark_returns = None
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data.pct_change().dropna()
        
        # Calculate metrics
        metrics = self.trading_engine.get_performance_metrics(benchmark_returns)
        
        return metrics
    
    def generate_report(self, data: Dict, signals: Dict, metrics: Dict):
        """
        Generate comprehensive trading report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'symbols': self.symbols,
            'portfolio_summary': self.trading_engine.portfolio.get_summary(),
            'performance_metrics': metrics,
            'active_signals': signals,
            'market_breadth': data.get('market_breadth', {}),
            'sentiment_summary': data.get('sentiment_analysis', {}).get('market', {}),
            'recent_trades': self.trading_engine.portfolio.transaction_history[-10:],
            'signal_statistics': self._calculate_signal_statistics()
        }
        
        # Save report
        report_file = f"reports/trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_file}")
        
        return report
    
    def _calculate_signal_statistics(self) -> Dict:
        """
        Calculate statistics from signal history
        """
        if not self.signal_history:
            return {}
        
        total_signals = len(self.signal_history)
        executed_signals = sum(1 for s in self.signal_history if s['executed'])
        
        buy_signals = sum(1 for s in self.signal_history if s['signal']['signal'] == 'BUY')
        sell_signals = sum(1 for s in self.signal_history if s['signal']['signal'] == 'SELL')
        hold_signals = sum(1 for s in self.signal_history if s['signal']['signal'] == 'HOLD')
        
        return {
            'total_signals': total_signals,
            'executed_signals': executed_signals,
            'execution_rate': executed_signals / total_signals if total_signals > 0 else 0,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_signal_strength': np.mean([s['signal']['strength'] for s in self.signal_history]),
            'avg_signal_confidence': np.mean([s['signal']['confidence'] for s in self.signal_history])
        }
    
    def visualize_performance(self):
        """
        Create performance visualizations
        """
        try:
            # Get data
            portfolio_history = self.trading_engine.portfolio.portfolio_history
            closed_positions = self.trading_engine.portfolio.closed_positions
            metrics = self.calculate_performance()
            
            # Create visualizations
            if portfolio_history:
                self.visualizer.plot_portfolio_value(portfolio_history)
                self.visualizer.plot_drawdown(portfolio_history)
            
            if closed_positions:
                self.visualizer.plot_trade_analysis(closed_positions)
            
            if metrics:
                self.visualizer.plot_metrics_comparison(metrics)
            
            # Create dashboard
            if portfolio_history and metrics:
                self.visualizer.create_dashboard(
                    portfolio_history,
                    closed_positions,
                    metrics
                )
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def run_trading_cycle(self):
        """
        Run one complete trading cycle
        """
        try:
            logger.info("=" * 50)
            logger.info(f"Starting trading cycle at {datetime.now()}")
            
            # 1. Fetch and process data
            data = self.fetch_and_process_data()
            
            # 2. Generate trading signals
            signals = self.generate_trading_signals(data)
            
            # 3. Execute trades
            executed_trades = self.execute_trades(signals, data)
            
            if executed_trades:
                logger.info(f"Executed {len(executed_trades)} trades")
                for trade in executed_trades:
                    logger.info(f"  - {trade['symbol']}: {trade['order']['side']} "
                              f"{trade['order']['quantity']} @ {trade['order']['price']:.2f}")
            
            # 4. Update portfolio
            self.update_portfolio(data)
            
            # 5. Calculate performance
            metrics = self.calculate_performance()
            
            # 6. Log summary
            summary = self.trading_engine.portfolio.get_summary()
            logger.info(f"Portfolio Value: ₹{summary['total_value']:.2f} "
                       f"(P&L: ₹{summary['total_pnl']:.2f}, "
                       f"{summary['total_pnl_pct']:.2f}%)")
            
            # 7. Generate report (every 10 cycles)
            if len(self.signal_history) % 10 == 0:
                self.generate_report(data, signals, metrics)
            
            # 8. Save state
            self.trading_engine.save_state("data/trading_state.json")
            
            logger.info("Trading cycle completed")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def is_market_open(self) -> bool:
        """
        Check if market is open
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours
        return MARKET_HOURS['market_open'] <= current_time <= MARKET_HOURS['market_close']
    
    def start(self, run_continuous: bool = True):
        """
        Start the trading system
        """
        self.is_running = True
        logger.info("Trading System Started")
        
        if run_continuous:
            # Schedule based on mode
            if self.mode == TradingConfig.INTRADAY:
                # Run every minute during market hours
                schedule.every(1).minutes.do(self.run_if_market_open)
            else:
                # Run every hour for swing trading
                schedule.every().hour.do(self.run_trading_cycle)
            
            # Run visualization every 30 minutes
            schedule.every(30).minutes.do(self.visualize_performance)
            
            # Main loop
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        else:
            # Run once
            self.run_trading_cycle()
            self.visualize_performance()
    
    def run_if_market_open(self):
        """
        Run trading cycle only if market is open
        """
        if self.is_market_open():
            self.run_trading_cycle()
        else:
            logger.info("Market is closed")
    
    def stop(self):
        """
        Stop the trading system
        """
        self.is_running = False
        
        # Final report
        metrics = self.calculate_performance()
        self.generate_report({}, {}, metrics)
        
        # Final visualization
        self.visualize_performance()
        
        logger.info("Trading System Stopped")
    
    def backtest(self, start_date: str, end_date: str):
        """
        Run backtest on historical data
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # This is a simplified backtest framework
        # In production, you'd implement proper event-driven backtesting
        
        for symbol in self.symbols:
            try:
                # Fetch historical data
                fetcher = MarketDataFetcher()
                hist_data = fetcher.get_historical_data(
                    symbol,
                    period="6mo",  # Get 6 months of data
                    interval="1d" if self.mode == TradingConfig.SWING else "5m"
                )
                
                if hist_data is None or hist_data.empty:
                    continue
                
                # Filter date range
                hist_data = hist_data[start_date:end_date]
                
                # Calculate indicators
                hist_data = self.technical_analyzer.calculate_all_indicators(hist_data)
                
                # Simulate trading
                for date in hist_data.index[50:]:  # Start after warmup period
                    # Get data up to this date
                    current_data = hist_data[:date]
                    
                    # Generate signals
                    signals = self.technical_analyzer.generate_signals(current_data)
                    
                    # Create mock market data
                    market_data = {
                        'current_price': current_data['Close'].iloc[-1],
                        'volume': current_data['Volume'].iloc[-1]
                    }
                    
                    # Process signal
                    composite_signal = self.ai_signal_generator.generate_composite_signal(
                        technical_signals=signals,
                        sentiment_data={},  # No sentiment in backtest
                        market_data=market_data,
                        historical_data=current_data
                    )
                    
                    # Execute trade
                    self.trading_engine.process_signal(symbol, composite_signal, market_data)
                    
                    # Update portfolio
                    self.trading_engine.update_portfolio({symbol: market_data})
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
        
        # Calculate final metrics
        metrics = self.calculate_performance()
        
        logger.info("Backtest completed")
        logger.info(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        
        return metrics


def main():
    """
    Main entry point for the trading system
    """
    parser = argparse.ArgumentParser(description='AI-Driven Paper Trading System for NSE')
    
    parser.add_argument('--mode', type=str, default='intraday',
                       choices=['intraday', 'swing'],
                       help='Trading mode')
    
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=TradingConfig.DEFAULT_SYMBOLS,
                       help='List of symbols to trade')
    
    parser.add_argument('--enable-llm', action='store_true',
                       help='Enable LLM analysis')
    
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest mode')
    
    parser.add_argument('--start-date', type=str,
                       default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                       help='Backtest start date')
    
    parser.add_argument('--end-date', type=str,
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='Backtest end date')
    
    parser.add_argument('--run-once', action='store_true',
                       help='Run one cycle only')
    
    args = parser.parse_args()
    
    # Create trading system
    system = TradingSystemOrchestrator(
        mode=args.mode,
        symbols=args.symbols,
        enable_llm=args.enable_llm
    )
    
    try:
        if args.backtest:
            # Run backtest
            metrics = system.backtest(args.start_date, args.end_date)
            
            # Save backtest results
            with open('reports/backtest_results.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
        else:
            # Run live paper trading
            system.start(run_continuous=not args.run_once)
            
    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
        system.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        system.stop()


if __name__ == "__main__":
    main()

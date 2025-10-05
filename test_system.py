#!/usr/bin/env python
"""
Test Suite for AI-Driven Paper Trading System
Run this to verify all components are working correctly
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TradingConfig
from technical_indicators import TechnicalIndicators
from paper_trading_engine import PaperTradingEngine, Order, OrderSide, OrderType, Position, Portfolio
from ai_module import SentimentAnalyzer, MarketRegimeDetector, AISignalGenerator

class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculations"""
    
    def setUp(self):
        """Set up test data"""
        self.indicators = TechnicalIndicators()
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        self.df = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.5,
            'High': prices + abs(np.random.randn(100) * 2),
            'Low': prices - abs(np.random.randn(100) * 2),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_sma_calculation(self):
        """Test Simple Moving Average"""
        sma = self.indicators.sma(self.df['Close'], 10)
        self.assertIsNotNone(sma)
        self.assertEqual(len(sma), len(self.df))
        self.assertTrue(sma.iloc[-1] > 0)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.indicators.rsi(self.df['Close'], 14)
        self.assertIsNotNone(rsi)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd = self.indicators.macd(self.df['Close'])
        self.assertIn('macd', macd)
        self.assertIn('signal', macd)
        self.assertIn('histogram', macd)
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        df_with_indicators = self.indicators.calculate_all_indicators(self.df.copy())
        signals = self.indicators.generate_signals(df_with_indicators)
        
        self.assertIn('overall_signal', signals)
        self.assertIn('buy_strength', signals)
        self.assertIn('sell_strength', signals)
        self.assertTrue(0 <= signals['buy_strength'] <= 1)
        self.assertTrue(0 <= signals['sell_strength'] <= 1)

class TestPaperTradingEngine(unittest.TestCase):
    """Test paper trading engine functionality"""
    
    def setUp(self):
        """Initialize trading engine"""
        self.engine = PaperTradingEngine(initial_capital=100000)
    
    def test_order_placement(self):
        """Test order placement"""
        order = self.engine.place_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=100
        )
        
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, "TEST")
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.side, OrderSide.BUY)
    
    def test_order_execution(self):
        """Test order execution"""
        order = self.engine.place_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            price=100
        )
        
        result = self.engine.execute_order(order, market_price=100)
        
        self.assertTrue(result['success'])
        self.assertIn('TEST', self.engine.portfolio.positions)
        self.assertEqual(self.engine.portfolio.positions['TEST'].quantity, 100)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        initial_cash = self.engine.portfolio.cash
        
        # Buy position
        order = self.engine.place_order("TEST", OrderSide.BUY, 100, price=100)
        self.engine.execute_order(order, 100)
        
        # Update prices
        self.engine.portfolio.update_prices({"TEST": 110})
        
        # Check portfolio value
        expected_value = (initial_cash - 100 * 100 * 1.001) + (100 * 110)  # Including slippage
        self.assertAlmostEqual(
            self.engine.portfolio.total_value,
            expected_value,
            delta=100  # Allow small difference for commission
        )
    
    def test_stop_loss_trigger(self):
        """Test stop loss functionality"""
        # Buy with stop loss
        order = self.engine.place_order("TEST", OrderSide.BUY, 100, price=100)
        self.engine.execute_order(order, 100)
        
        # Set stop loss
        position = self.engine.portfolio.positions['TEST']
        position.stop_loss = 95
        
        # Trigger stop loss
        triggered = self.engine.portfolio.check_stop_loss_take_profit({"TEST": 94})
        
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]['type'], 'STOP_LOSS')
        self.assertNotIn('TEST', self.engine.portfolio.positions)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute some trades
        trades = [
            ("TEST1", OrderSide.BUY, 100, 100),
            ("TEST1", OrderSide.SELL, 100, 110),  # Profit
            ("TEST2", OrderSide.BUY, 50, 200),
            ("TEST2", OrderSide.SELL, 50, 190),   # Loss
        ]
        
        for symbol, side, qty, price in trades:
            order = self.engine.place_order(symbol, side, qty, price=price)
            self.engine.execute_order(order, price)
        
        metrics = self.engine.get_performance_metrics()
        
        self.assertIn('total_return_pct', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_trades', metrics)
        self.assertEqual(metrics['total_trades'], 2)
        self.assertGreater(metrics['win_rate'], 0)

class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality"""
    
    def setUp(self):
        """Initialize sentiment analyzer"""
        # Use mock for testing without downloading model
        with patch('ai_module.AutoTokenizer.from_pretrained'):
            with patch('ai_module.AutoModelForSequenceClassification.from_pretrained'):
                self.analyzer = SentimentAnalyzer()
    
    def test_rule_based_sentiment(self):
        """Test rule-based sentiment fallback"""
        # Force rule-based by setting pipeline to None
        self.analyzer.sentiment_pipeline = None
        
        positive_text = "Stock shows bullish momentum with strong growth"
        negative_text = "Market crash fears amid weak earnings"
        
        pos_result = self.analyzer.analyze_text(positive_text)
        neg_result = self.analyzer.analyze_text(negative_text)
        
        self.assertEqual(pos_result['sentiment'], 'positive')
        self.assertEqual(neg_result['sentiment'], 'negative')
        self.assertGreater(pos_result['score'], 0)
        self.assertLess(neg_result['score'], 0)
    
    def test_batch_sentiment_analysis(self):
        """Test batch sentiment analysis"""
        self.analyzer.sentiment_pipeline = None  # Use rule-based
        
        news = [
            {'title': 'Market rallies on strong earnings', 'description': ''},
            {'title': 'Stocks fall amid recession fears', 'description': ''},
            {'title': 'Trading remains flat', 'description': ''}
        ]
        
        result = self.analyzer.analyze_news_batch(news)
        
        self.assertIn('overall_sentiment', result)
        self.assertIn('sentiment_distribution', result)
        self.assertEqual(
            result['sentiment_distribution']['positive'] +
            result['sentiment_distribution']['negative'] +
            result['sentiment_distribution']['neutral'],
            3
        )

class TestMarketRegimeDetector(unittest.TestCase):
    """Test market regime detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = MarketRegimeDetector()
        
        # Create trending up data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        trend = np.linspace(100, 150, 50)
        noise = np.random.randn(50) * 2
        
        self.trending_up = pd.DataFrame({
            'Open': trend + noise,
            'High': trend + abs(noise) + 2,
            'Low': trend - abs(noise) - 2,
            'Close': trend + noise * 0.5,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Create volatile data
        volatile_prices = 100 + np.cumsum(np.random.randn(50) * 5)
        self.volatile = pd.DataFrame({
            'Open': volatile_prices,
            'High': volatile_prices + abs(np.random.randn(50) * 10),
            'Low': volatile_prices - abs(np.random.randn(50) * 10),
            'Close': volatile_prices + np.random.randn(50) * 5,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
    
    def test_trending_regime_detection(self):
        """Test detection of trending market"""
        regime = self.detector.detect_regime(self.trending_up)
        
        self.assertEqual(regime['regime'], 'trending_up')
        self.assertGreater(regime['confidence'], 0.5)
        self.assertGreater(regime['slope'], 0)
    
    def test_volatile_regime_detection(self):
        """Test detection of volatile market"""
        regime = self.detector.detect_regime(self.volatile)
        
        self.assertIn(regime['regime'], ['volatile', 'ranging'])
        self.assertGreater(regime['volatility'], 0)
    
    def test_regime_parameters(self):
        """Test regime-specific parameters"""
        params = self.detector.get_regime_parameters('trending_up')
        
        self.assertIn('position_size_multiplier', params)
        self.assertIn('stop_loss_multiplier', params)
        self.assertIn('signal_threshold', params)
        self.assertGreater(params['position_size_multiplier'], 0)

class TestAISignalGenerator(unittest.TestCase):
    """Test AI signal generation"""
    
    def setUp(self):
        """Initialize signal generator"""
        with patch('ai_module.AutoTokenizer.from_pretrained'):
            with patch('ai_module.AutoModelForSequenceClassification.from_pretrained'):
                self.generator = AISignalGenerator()
    
    def test_composite_signal_generation(self):
        """Test composite signal generation"""
        # Mock technical signals
        technical_signals = {
            'sma_signal': 'BUY',
            'rsi': 35,
            'macd_signal': 'BUY',
            'volume_signal': 'HIGH'
        }
        
        # Mock sentiment data
        sentiment_data = {'overall_sentiment': 0.5}
        
        # Mock market data
        market_data = {'current_price': 100, 'volume': 1000000}
        
        # Create sample historical data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        historical_data = pd.DataFrame({
            'Close': prices,
            'High': prices + 2,
            'Low': prices - 2,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Generate signal
        signal = self.generator.generate_composite_signal(
            technical_signals,
            sentiment_data,
            market_data,
            historical_data
        )
        
        self.assertIn('signal', signal)
        self.assertIn('strength', signal)
        self.assertIn('confidence', signal)
        self.assertIn('composite_score', signal)
        self.assertIn(signal['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertTrue(0 <= signal['strength'] <= 1)
        self.assertTrue(0 <= signal['confidence'] <= 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @patch('data_fetcher.MarketDataFetcher.get_nse_data')
    @patch('data_fetcher.MarketDataFetcher.get_historical_data')
    def test_end_to_end_signal_generation(self, mock_hist, mock_data):
        """Test complete signal generation pipeline"""
        
        # Mock market data
        mock_data.return_value = {
            'symbol': 'TEST',
            'current_price': 100,
            'volume': 1000000,
            'change_pct': 2.5
        }
        
        # Mock historical data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        mock_hist.return_value = pd.DataFrame({
            'Open': prices,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Initialize components
        from main import TradingSystemOrchestrator
        
        with patch('ai_module.AutoTokenizer.from_pretrained'):
            with patch('ai_module.AutoModelForSequenceClassification.from_pretrained'):
                system = TradingSystemOrchestrator(symbols=['TEST'])
                
                # Fetch and process data
                data = system.fetch_and_process_data()
                
                self.assertIn('market_data', data)
                self.assertIn('technical_signals', data)
                
                # Generate signals
                signals = system.generate_trading_signals(data)
                
                self.assertIn('TEST', signals)
                self.assertIn('signal', signals['TEST'])

def run_tests():
    """Run all tests"""
    print("="*60)
    print("   RUNNING TEST SUITE")
    print("="*60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestPaperTradingEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketRegimeDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestAISignalGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run tests
    success = run_tests()
    sys.exit(0 if success else 1)
"""
AI/ML Module for Sentiment Analysis and Signal Generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

from config import TradingConfig

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes news and social media sentiment using transformer models
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize sentiment analyzer with FinBERT or similar model
        """
        if model_name is None:
            model_name = TradingConfig.SENTIMENT_CONFIG['model_name']
        
        try:
            # Initialize FinBERT for financial sentiment analysis
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer
            )
            logger.info(f"Initialized sentiment analyzer with {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}, using default sentiment: {e}")
            self.sentiment_pipeline = None
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        Returns: {'sentiment': 'positive/negative/neutral', 'score': float}
        """
        try:
            if self.sentiment_pipeline is None:
                # Fallback to rule-based sentiment
                return self._rule_based_sentiment(text)
            
            # Use transformer model
            result = self.sentiment_pipeline(text[:512])[0]  # Truncate to max length
            
            # Map label to standardized format
            label_map = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral'
            }
            
            sentiment = label_map.get(result['label'].lower(), result['label'].lower())
            
            # Convert to numerical score (-1 to 1)
            if sentiment == 'positive':
                score = result['score']
            elif sentiment == 'negative':
                score = -result['score']
            else:
                score = 0
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': result['score']
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict:
        """
        Fallback rule-based sentiment analysis
        """
        text_lower = text.lower()
        
        # Simple keyword-based sentiment
        positive_words = ['bullish', 'growth', 'profit', 'gain', 'rise', 'surge', 
                         'rally', 'strong', 'outperform', 'upgrade', 'buy']
        negative_words = ['bearish', 'loss', 'decline', 'fall', 'drop', 'weak',
                         'sell', 'downgrade', 'crash', 'recession', 'risk']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(pos_count / 10, 1.0)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = -min(neg_count / 10, 1.0)
        else:
            sentiment = 'neutral'
            score = 0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': 0.5  # Low confidence for rule-based
        }
    
    def analyze_news_batch(self, news_articles: List[Dict]) -> Dict:
        """
        Analyze sentiment for multiple news articles
        """
        if not news_articles:
            return {'overall_sentiment': 0, 'sentiment_distribution': {}}
        
        sentiments = []
        
        for article in news_articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if text.strip():
                sentiment_result = self.analyze_text(text)
                sentiments.append(sentiment_result)
        
        if not sentiments:
            return {'overall_sentiment': 0, 'sentiment_distribution': {}}
        
        # Calculate overall sentiment
        scores = [s['score'] for s in sentiments]
        overall_sentiment = np.mean(scores)
        
        # Calculate distribution
        distribution = {
            'positive': sum(1 for s in sentiments if s['sentiment'] == 'positive'),
            'negative': sum(1 for s in sentiments if s['sentiment'] == 'negative'),
            'neutral': sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        }
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': distribution,
            'average_confidence': np.mean([s['confidence'] for s in sentiments]),
            'individual_sentiments': sentiments
        }
    
    def calculate_market_sentiment(self, market_news: List[Dict], 
                                  company_news: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate overall market and individual stock sentiment
        """
        results = {}
        
        # Overall market sentiment
        market_sentiment = self.analyze_news_batch(market_news)
        results['market'] = market_sentiment
        
        # Individual stock sentiments
        results['stocks'] = {}
        for symbol, news in company_news.items():
            if news:
                results['stocks'][symbol] = self.analyze_news_batch(news)
        
        return results


class MarketRegimeDetector:
    """
    Detects market regime (trending, ranging, volatile) using ML
    """
    
    def __init__(self):
        self.regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict:
        """
        Detect current market regime using various indicators
        """
        if price_data is None or len(price_data) < 20:
            return {'regime': 'unknown', 'confidence': 0}
        
        try:
            # Calculate various metrics
            returns = price_data['Close'].pct_change().dropna()
            
            # Trend strength using linear regression
            x = np.arange(len(price_data))
            y = price_data['Close'].values
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = slope / np.mean(y)
            
            # Volatility
            volatility = returns.std()
            avg_volatility = returns.rolling(20).std().mean()
            
            # Price range
            price_range = (price_data['High'] - price_data['Low']).mean()
            avg_price = price_data['Close'].mean()
            normalized_range = price_range / avg_price
            
            # Determine regime
            if abs(normalized_slope) > 0.001:  # Trending
                if normalized_slope > 0:
                    regime = 'trending_up'
                else:
                    regime = 'trending_down'
                confidence = min(abs(normalized_slope) * 100, 1.0)
            elif volatility > avg_volatility * 1.5:  # High volatility
                regime = 'volatile'
                confidence = min(volatility / avg_volatility, 1.0)
            else:  # Ranging
                regime = 'ranging'
                confidence = 0.7
            
            return {
                'regime': regime,
                'confidence': confidence,
                'slope': normalized_slope,
                'volatility': volatility,
                'range': normalized_range
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """
        Get trading parameters based on market regime
        """
        params = {
            'trending_up': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5,
                'signal_threshold': 0.6
            },
            'trending_down': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.0,
                'signal_threshold': 0.7
            },
            'ranging': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 0.9,
                'take_profit_multiplier': 1.2,
                'signal_threshold': 0.65
            },
            'volatile': {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.8,
                'signal_threshold': 0.75
            },
            'unknown': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'signal_threshold': 0.8
            }
        }
        
        return params.get(regime, params['unknown'])


class AISignalGenerator:
    """
    Combines technical, sentiment, and regime analysis for signal generation
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.config = TradingConfig.SENTIMENT_CONFIG
    
    def generate_composite_signal(self, 
                                 technical_signals: Dict,
                                 sentiment_data: Dict,
                                 market_data: Dict,
                                 historical_data: pd.DataFrame) -> Dict:
        """
        Generate composite trading signal using AI/ML
        """
        try:
            # Get market regime
            regime_info = self.regime_detector.detect_regime(historical_data)
            regime_params = self.regime_detector.get_regime_parameters(regime_info['regime'])
            
            # Calculate component scores
            technical_score = self._calculate_technical_score(technical_signals)
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            regime_score = self._calculate_regime_score(regime_info, technical_signals)
            
            # Weight the scores
            weights = {
                'technical': self.config['technical_weight'],
                'sentiment': self.config['news_impact_weight'],
                'regime': self.config['market_regime_weight']
            }
            
            # Calculate composite score
            composite_score = (
                technical_score * weights['technical'] +
                sentiment_score * weights['sentiment'] +
                regime_score * weights['regime']
            )
            
            # Generate signal
            signal_threshold = regime_params['signal_threshold']
            
            if composite_score > signal_threshold:
                signal = 'BUY'
                strength = min((composite_score - signal_threshold) / (1 - signal_threshold), 1.0)
            elif composite_score < -signal_threshold:
                signal = 'SELL'
                strength = min((abs(composite_score) - signal_threshold) / (1 - signal_threshold), 1.0)
            else:
                signal = 'HOLD'
                strength = abs(composite_score) / signal_threshold
            
            # Calculate position sizing
            base_position = 1.0
            position_size = base_position * regime_params['position_size_multiplier'] * strength
            
            # Risk parameters
            stop_loss = regime_params['stop_loss_multiplier'] * \
                       TradingConfig.get_config(TradingConfig.INTRADAY)['stop_loss_pct']
            take_profit = regime_params['take_profit_multiplier'] * \
                         TradingConfig.get_config(TradingConfig.INTRADAY)['target_pct']
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': self._calculate_confidence(technical_score, sentiment_score, regime_info),
                'composite_score': composite_score,
                'components': {
                    'technical_score': technical_score,
                    'sentiment_score': sentiment_score,
                    'regime_score': regime_score,
                    'regime': regime_info['regime']
                },
                'position_size': position_size,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'reasoning': self._generate_reasoning(signal, technical_signals, sentiment_data, regime_info)
            }
            
        except Exception as e:
            logger.error(f"Error generating composite signal: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0,
                'confidence': 0,
                'composite_score': 0,
                'reasoning': f"Error in signal generation: {e}"
            }
    
    def _calculate_technical_score(self, technical_signals: Dict) -> float:
        """
        Calculate score from technical indicators
        """
        scores = []
        
        # Moving average signals
        if technical_signals.get('sma_signal'):
            scores.append(1 if technical_signals['sma_signal'] == 'BUY' else -1)
        
        if technical_signals.get('ema_signal'):
            scores.append(1 if technical_signals['ema_signal'] == 'BUY' else -1)
        
        # RSI signal
        rsi = technical_signals.get('rsi', 50)
        if rsi < 30:
            scores.append(1)  # Oversold
        elif rsi > 70:
            scores.append(-1)  # Overbought
        else:
            scores.append(0)
        
        # MACD signal
        if technical_signals.get('macd_signal'):
            scores.append(1 if technical_signals['macd_signal'] == 'BUY' else -1)
        
        # Volume signal
        if technical_signals.get('volume_signal'):
            scores.append(0.5 if technical_signals['volume_signal'] == 'HIGH' else 0)
        
        # Calculate average score
        if scores:
            return np.mean(scores)
        return 0
    
    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """
        Calculate score from sentiment analysis
        """
        if not sentiment_data:
            return 0
        
        # Get overall sentiment
        overall = sentiment_data.get('overall_sentiment', 0)
        
        # Normalize to -1 to 1 range
        return np.clip(overall, -1, 1)
    
    def _calculate_regime_score(self, regime_info: Dict, technical_signals: Dict) -> float:
        """
        Calculate score based on market regime
        """
        regime = regime_info.get('regime', 'unknown')
        
        if regime == 'trending_up':
            # Favor long positions in uptrend
            return 0.5
        elif regime == 'trending_down':
            # Favor short positions or cash in downtrend
            return -0.3
        elif regime == 'ranging':
            # Neutral in ranging market
            return 0
        elif regime == 'volatile':
            # Be cautious in volatile markets
            return -0.2
        else:
            return 0
    
    def _calculate_confidence(self, technical_score: float, 
                            sentiment_score: float, 
                            regime_info: Dict) -> float:
        """
        Calculate overall confidence in the signal
        """
        # Agreement between indicators increases confidence
        scores = [technical_score, sentiment_score]
        
        # Check if all signals agree
        if all(s > 0 for s in scores) or all(s < 0 for s in scores):
            agreement_score = 1.0
        else:
            agreement_score = 0.5
        
        # Regime confidence
        regime_confidence = regime_info.get('confidence', 0.5)
        
        # Overall confidence
        confidence = (agreement_score * 0.6 + regime_confidence * 0.4)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, signal: str, technical_signals: Dict,
                          sentiment_data: Dict, regime_info: Dict) -> str:
        """
        Generate human-readable reasoning for the signal
        """
        reasons = []
        
        # Technical reasons
        if technical_signals.get('sma_signal') == 'BUY':
            reasons.append("Price above moving averages")
        elif technical_signals.get('sma_signal') == 'SELL':
            reasons.append("Price below moving averages")
        
        rsi = technical_signals.get('rsi', 50)
        if rsi < 30:
            reasons.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 70:
            reasons.append(f"RSI overbought at {rsi:.1f}")
        
        # Sentiment reasons
        sentiment = sentiment_data.get('overall_sentiment', 0)
        if sentiment > 0.3:
            reasons.append(f"Positive sentiment ({sentiment:.2f})")
        elif sentiment < -0.3:
            reasons.append(f"Negative sentiment ({sentiment:.2f})")
        
        # Regime reasons
        regime = regime_info.get('regime', 'unknown')
        reasons.append(f"Market regime: {regime}")
        
        return f"{signal} signal generated. Factors: {'; '.join(reasons)}"


class LLMAnalyzer:
    """
    Optional integration with LLM APIs for advanced analysis
    """
    
    def __init__(self, api_key: str = None, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        self.enabled = api_key is not None
        
        if self.enabled:
            logger.info(f"LLM Analyzer enabled with {provider}")
    
    def analyze_market_context(self, market_data: Dict, news: List[Dict]) -> Optional[Dict]:
        """
        Use LLM to analyze market context and provide insights
        """
        if not self.enabled:
            return None
        
        try:
            # Prepare prompt
            prompt = self._prepare_market_analysis_prompt(market_data, news)
            
            # Get LLM response (implementation depends on provider)
            # This is a placeholder - actual implementation would call the API
            response = self._call_llm_api(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return None
    
    def _prepare_market_analysis_prompt(self, market_data: Dict, news: List[Dict]) -> str:
        """
        Prepare prompt for LLM analysis
        """
        prompt = "Analyze the following market data and news:\n\n"
        
        # Add market data summary
        prompt += "Market Data:\n"
        for key, value in market_data.items():
            prompt += f"- {key}: {value}\n"
        
        # Add news headlines
        prompt += "\nRecent News:\n"
        for article in news[:5]:
            prompt += f"- {article.get('title', '')}\n"
        
        prompt += "\nProvide a brief analysis of market conditions and potential opportunities."
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> Dict:
        """
        Placeholder for LLM API call
        """
        # This would be implemented based on the chosen provider (OpenAI, Claude, etc.)
        return {
            'analysis': 'Market analysis placeholder',
            'opportunities': [],
            'risks': []
        }

"""
Data Fetcher Module for Market Data and News
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import json
import time
from bs4 import BeautifulSoup
from config import TradingConfig, NEWS_API_KEY

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetches market data from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_nse_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch NSE data for a symbol using alternative methods
        """
        try:
            # Method 1: Try yfinance (works for NSE stocks)
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            
            data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'prev_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'market_cap': info.get('marketCap', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            # Calculate change percentage
            if data['prev_close'] > 0:
                data['change_pct'] = ((data['current_price'] - data['prev_close']) / 
                                     data['prev_close']) * 100
            else:
                data['change_pct'] = 0
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching NSE data for {symbol}: {e}")
            
            # Method 2: Try alternative API
            try:
                return self._fetch_from_alternative_api(symbol)
            except:
                return None
    
    def _fetch_from_alternative_api(self, symbol: str) -> Optional[Dict]:
        """
        Fallback method using alternative free APIs
        """
        try:
            # Using Google Finance as fallback
            url = f"https://www.google.com/finance/quote/{symbol}:NSE"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Parse basic data from HTML (simplified example)
                # In production, you'd need more robust parsing
                
                data = {
                    'symbol': symbol,
                    'current_price': 0,
                    'day_high': 0,
                    'day_low': 0,
                    'volume': 0,
                    'prev_close': 0,
                    'change_pct': 0
                }
                return data
        except Exception as e:
            logger.error(f"Alternative API failed for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1mo", 
                          interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for technical analysis
        """
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
                
            # Add some calculated fields
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_options_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch F&O (Options) data for a symbol
        """
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get options expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            # Get nearest expiry options chain
            nearest_expiry = expirations[0]
            calls = ticker.option_chain(nearest_expiry).calls
            puts = ticker.option_chain(nearest_expiry).puts
            
            # Calculate Put-Call Ratio (PCR)
            total_put_oi = puts['openInterest'].sum()
            total_call_oi = calls['openInterest'].sum()
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Find max pain strike
            max_pain = self._calculate_max_pain(calls, puts)
            
            return {
                'pcr': pcr,
                'max_pain': max_pain,
                'call_oi': total_call_oi,
                'put_oi': total_put_oi,
                'expiry': nearest_expiry,
                'iv_calls': calls['impliedVolatility'].mean() if len(calls) > 0 else 0,
                'iv_puts': puts['impliedVolatility'].mean() if len(puts) > 0 else 0,
            }
            
        except Exception as e:
            logger.debug(f"No options data for {symbol}: {e}")
            return None
    
    def _calculate_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame) -> float:
        """Calculate max pain strike price"""
        try:
            strikes = calls['strike'].unique()
            pain_values = []
            
            for strike in strikes:
                # Calculate pain for calls (ITM calls)
                call_pain = ((calls[calls['strike'] < strike]['openInterest'] * 
                            (strike - calls[calls['strike'] < strike]['strike'])).sum())
                
                # Calculate pain for puts (ITM puts)
                put_pain = ((puts[puts['strike'] > strike]['openInterest'] * 
                           (puts[puts['strike'] > strike]['strike'] - strike)).sum())
                
                total_pain = call_pain + put_pain
                pain_values.append((strike, total_pain))
            
            # Find strike with minimum pain
            if pain_values:
                max_pain_strike = min(pain_values, key=lambda x: x[1])[0]
                return float(max_pain_strike)
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating max pain: {e}")
            return 0
    
    def get_index_data(self, index_name: str = "^NSEI") -> Optional[Dict]:
        """
        Fetch NIFTY 50 index data
        """
        try:
            ticker = yf.Ticker(index_name)
            info = ticker.info
            
            # Get recent history for index
            hist = ticker.history(period="1d", interval="5m")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                day_high = hist['High'].max()
                day_low = hist['Low'].min()
                volume = hist['Volume'].sum()
            else:
                current_price = info.get('regularMarketPrice', 0)
                day_high = info.get('dayHigh', 0)
                day_low = info.get('dayLow', 0)
                volume = info.get('volume', 0)
            
            data = {
                'index': index_name,
                'current_value': current_price,
                'day_high': day_high,
                'day_low': day_low,
                'volume': volume,
                'prev_close': info.get('previousClose', 0),
                'change_pct': info.get('regularMarketChangePercent', 0)
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            return None
    
    def get_market_breadth(self, symbols: List[str]) -> Dict:
        """
        Calculate market breadth indicators
        """
        try:
            advances = 0
            declines = 0
            unchanged = 0
            
            for symbol in symbols:
                data = self.get_nse_data(symbol)
                if data:
                    change = data.get('change_pct', 0)
                    if change > 0.1:
                        advances += 1
                    elif change < -0.1:
                        declines += 1
                    else:
                        unchanged += 1
            
            total = advances + declines + unchanged
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advances / declines if declines > 0 else float('inf'),
                'bullish_percent': (advances / total * 100) if total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {}


class NewsDataFetcher:
    """Fetches news and sentiment data"""
    
    def __init__(self, api_key: str = NEWS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def get_market_news(self, query: str = "NSE OR Nifty OR Indian stocks", 
                       page_size: int = 20) -> List[Dict]:
        """
        Fetch latest market news
        """
        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': page_size,
                'from': (datetime.now() - timedelta(days=1)).isoformat()
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Process articles
                processed_articles = []
                for article in articles:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'content': article.get('content', '')
                    })
                
                return processed_articles
            else:
                logger.error(f"News API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_company_news(self, company_name: str) -> List[Dict]:
        """
        Fetch news for specific company
        """
        return self.get_market_news(query=company_name, page_size=10)
    
    def get_economic_calendar(self) -> List[Dict]:
        """
        Get economic events (mock implementation - would need proper API)
        """
        # This is a simplified mock - in production, use APIs like:
        # - Trading Economics API
        # - Investing.com scraping
        # - Alpha Vantage economic indicators
        
        events = [
            {
                'event': 'RBI Monetary Policy',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'importance': 'high',
                'forecast': '6.5%',
                'previous': '6.5%'
            },
            {
                'event': 'GDP Growth Rate',
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'importance': 'high',
                'forecast': '7.2%',
                'previous': '7.0%'
            }
        ]
        
        return events
    
    def get_social_sentiment(self, query: str) -> Dict:
        """
        Get social media sentiment (simplified implementation)
        """
        # In production, you could use:
        # - Twitter API
        # - Reddit API (praw)
        # - StockTwits API
        
        # Mock implementation
        return {
            'overall_sentiment': 0.2,  # -1 to 1 scale
            'volume': 150,
            'bullish_count': 90,
            'bearish_count': 60
        }


class DataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.news_fetcher = NewsDataFetcher()
        
    def get_comprehensive_data(self, symbols: List[str]) -> Dict:
        """
        Get all relevant data for analysis
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'market_data': {},
                'options_data': {},
                'historical_data': {},
                'news_data': {},
                'market_breadth': {},
                'index_data': {}
            }
            
            # Fetch market data for each symbol
            for symbol in symbols:
                logger.info(f"Fetching data for {symbol}")
                
                # Current market data
                market_data = self.market_fetcher.get_nse_data(symbol)
                if market_data:
                    data['market_data'][symbol] = market_data
                
                # Historical data
                hist_data = self.market_fetcher.get_historical_data(symbol, period="1mo")
                if hist_data is not None and not hist_data.empty:
                    data['historical_data'][symbol] = hist_data
                
                # Options data
                options = self.market_fetcher.get_options_data(symbol)
                if options:
                    data['options_data'][symbol] = options
                
                # Company news
                news = self.news_fetcher.get_company_news(symbol)
                if news:
                    data['news_data'][symbol] = news[:5]  # Top 5 news
            
            # Market breadth
            data['market_breadth'] = self.market_fetcher.get_market_breadth(symbols)
            
            # Index data
            data['index_data'] = self.market_fetcher.get_index_data()
            
            # General market news
            data['market_news'] = self.news_fetcher.get_market_news()
            
            # Economic calendar
            data['economic_events'] = self.news_fetcher.get_economic_calendar()
            
            return data
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return {}
    
    def get_realtime_updates(self, symbols: List[str], interval: int = 60) -> Dict:
        """
        Get realtime updates for intraday trading
        """
        updates = {
            'timestamp': datetime.now().isoformat(),
            'prices': {},
            'volumes': {},
            'changes': {}
        }
        
        for symbol in symbols:
            try:
                data = self.market_fetcher.get_nse_data(symbol)
                if data:
                    updates['prices'][symbol] = data['current_price']
                    updates['volumes'][symbol] = data['volume']
                    updates['changes'][symbol] = data['change_pct']
            except Exception as e:
                logger.error(f"Error getting realtime data for {symbol}: {e}")
        
        return updates

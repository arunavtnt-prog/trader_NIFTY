"""
Technical Indicators Module
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from config import TradingConfig

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Calculate various technical indicators for trading signals
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = TradingConfig.TECHNICAL_CONFIG
        self.config = config
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataframe
        """
        if df is None or df.empty:
            return df
        
        try:
            # Moving Averages
            df['SMA_short'] = self.sma(df['Close'], self.config['sma_short'])
            df['SMA_long'] = self.sma(df['Close'], self.config['sma_long'])
            df['EMA_short'] = self.ema(df['Close'], self.config['ema_short'])
            df['EMA_long'] = self.ema(df['Close'], self.config['ema_long'])
            
            # RSI
            df['RSI'] = self.rsi(df['Close'], self.config['rsi_period'])
            
            # MACD
            macd_result = self.macd(df['Close'], 
                                   self.config['macd_fast'],
                                   self.config['macd_slow'],
                                   self.config['macd_signal'])
            df['MACD'] = macd_result['macd']
            df['MACD_signal'] = macd_result['signal']
            df['MACD_histogram'] = macd_result['histogram']
            
            # Bollinger Bands
            bb_result = self.bollinger_bands(df['Close'],
                                            self.config['bollinger_period'],
                                            self.config['bollinger_std'])
            df['BB_upper'] = bb_result['upper']
            df['BB_middle'] = bb_result['middle']
            df['BB_lower'] = bb_result['lower']
            df['BB_width'] = bb_result['width']
            df['BB_percent'] = bb_result['percent']
            
            # Volume indicators
            df['Volume_MA'] = self.sma(df['Volume'], self.config['volume_ma'])
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
            
            # ATR (Average True Range)
            df['ATR'] = self.atr(df, period=14)
            
            # Stochastic
            stoch_result = self.stochastic(df, period=14)
            df['Stoch_K'] = stoch_result['K']
            df['Stoch_D'] = stoch_result['D']
            
            # Support and Resistance
            sr_levels = self.support_resistance(df)
            df['Support'] = sr_levels['support']
            df['Resistance'] = sr_levels['resistance']
            
            # Price patterns
            df['Doji'] = self.detect_doji(df)
            df['Hammer'] = self.detect_hammer(df)
            df['Shooting_star'] = self.detect_shooting_star(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, series: pd.Series, fast: int = 12, 
             slow: int = 26, signal: int = 9) -> Dict:
        """
        MACD (Moving Average Convergence Divergence)
        """
        ema_fast = self.ema(series, fast)
        ema_slow = self.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, series: pd.Series, period: int = 20, 
                       std_dev: float = 2) -> Dict:
        """
        Bollinger Bands
        """
        middle = self.sma(series, period)
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = upper - lower
        
        # Bollinger %B
        percent_b = (series - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent': percent_b
        }
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def stochastic(self, df: pd.DataFrame, period: int = 14, 
                  smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """
        Stochastic Oscillator
        """
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    def support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate dynamic support and resistance levels
        """
        # Method 1: Recent pivots
        highs = df['High'].rolling(window=window).max()
        lows = df['Low'].rolling(window=window).min()
        
        # Method 2: Volume-weighted levels
        volume_price = df['Close'] * df['Volume']
        cum_volume = df['Volume'].rolling(window=window).sum()
        vwap = volume_price.rolling(window=window).sum() / cum_volume
        
        # Combine methods
        resistance = highs
        support = lows
        
        return {
            'support': support,
            'resistance': resistance,
            'vwap': vwap
        }
    
    def detect_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Detect Doji candlestick pattern
        """
        body = abs(df['Close'] - df['Open'])
        range_hl = df['High'] - df['Low']
        
        doji = (body <= range_hl * threshold).astype(int)
        
        return doji
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Hammer candlestick pattern
        """
        body = abs(df['Close'] - df['Open'])
        range_hl = df['High'] - df['Low']
        
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        hammer = ((lower_shadow > body * 2) & 
                 (upper_shadow < body * 0.3) & 
                 (body < range_hl * 0.3)).astype(int)
        
        return hammer
    
    def detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Shooting Star candlestick pattern
        """
        body = abs(df['Close'] - df['Open'])
        range_hl = df['High'] - df['Low']
        
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        shooting_star = ((upper_shadow > body * 2) & 
                        (lower_shadow < body * 0.3) & 
                        (body < range_hl * 0.3)).astype(int)
        
        return shooting_star
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signals based on technical indicators
        """
        if df is None or len(df) < 50:
            return {}
        
        signals = {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Moving Average Signals
        if latest['SMA_short'] > latest['SMA_long'] and prev['SMA_short'] <= prev['SMA_long']:
            signals['sma_signal'] = 'BUY'
            signals['sma_crossover'] = True
        elif latest['SMA_short'] < latest['SMA_long'] and prev['SMA_short'] >= prev['SMA_long']:
            signals['sma_signal'] = 'SELL'
            signals['sma_crossover'] = True
        else:
            signals['sma_signal'] = 'HOLD'
            signals['sma_crossover'] = False
        
        # EMA Signals
        if latest['EMA_short'] > latest['EMA_long']:
            signals['ema_signal'] = 'BUY'
        elif latest['EMA_short'] < latest['EMA_long']:
            signals['ema_signal'] = 'SELL'
        else:
            signals['ema_signal'] = 'HOLD'
        
        # RSI Signals
        signals['rsi'] = latest['RSI']
        if latest['RSI'] < self.config['rsi_oversold']:
            signals['rsi_signal'] = 'BUY'
        elif latest['RSI'] > self.config['rsi_overbought']:
            signals['rsi_signal'] = 'SELL'
        else:
            signals['rsi_signal'] = 'HOLD'
        
        # MACD Signals
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals['macd_signal'] = 'BUY'
            signals['macd_crossover'] = True
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals['macd_signal'] = 'SELL'
            signals['macd_crossover'] = True
        else:
            signals['macd_signal'] = 'HOLD'
            signals['macd_crossover'] = False
        
        # Bollinger Bands Signals
        if latest['Close'] < latest['BB_lower']:
            signals['bb_signal'] = 'BUY'
        elif latest['Close'] > latest['BB_upper']:
            signals['bb_signal'] = 'SELL'
        else:
            signals['bb_signal'] = 'HOLD'
        
        signals['bb_percent'] = latest['BB_percent']
        
        # Volume Signal
        if latest['Volume_ratio'] > 1.5:
            signals['volume_signal'] = 'HIGH'
        elif latest['Volume_ratio'] < 0.5:
            signals['volume_signal'] = 'LOW'
        else:
            signals['volume_signal'] = 'NORMAL'
        
        # Stochastic Signals
        if latest['Stoch_K'] < 20:
            signals['stoch_signal'] = 'OVERSOLD'
        elif latest['Stoch_K'] > 80:
            signals['stoch_signal'] = 'OVERBOUGHT'
        else:
            signals['stoch_signal'] = 'NEUTRAL'
        
        # Candlestick Patterns
        if latest['Doji'] == 1:
            signals['candle_pattern'] = 'DOJI'
        elif latest['Hammer'] == 1:
            signals['candle_pattern'] = 'HAMMER'
        elif latest['Shooting_star'] == 1:
            signals['candle_pattern'] = 'SHOOTING_STAR'
        else:
            signals['candle_pattern'] = None
        
        # Support/Resistance
        signals['near_support'] = abs(latest['Close'] - latest['Support']) / latest['Close'] < 0.02
        signals['near_resistance'] = abs(latest['Close'] - latest['Resistance']) / latest['Close'] < 0.02
        
        # Overall Signal Strength
        buy_signals = sum([
            signals.get('sma_signal') == 'BUY',
            signals.get('ema_signal') == 'BUY',
            signals.get('rsi_signal') == 'BUY',
            signals.get('macd_signal') == 'BUY',
            signals.get('bb_signal') == 'BUY',
            signals.get('stoch_signal') == 'OVERSOLD',
        ])
        
        sell_signals = sum([
            signals.get('sma_signal') == 'SELL',
            signals.get('ema_signal') == 'SELL',
            signals.get('rsi_signal') == 'SELL',
            signals.get('macd_signal') == 'SELL',
            signals.get('bb_signal') == 'SELL',
            signals.get('stoch_signal') == 'OVERBOUGHT',
        ])
        
        signals['buy_strength'] = buy_signals / 6
        signals['sell_strength'] = sell_signals / 6
        
        if buy_signals >= 4:
            signals['overall_signal'] = 'STRONG_BUY'
        elif buy_signals >= 3:
            signals['overall_signal'] = 'BUY'
        elif sell_signals >= 4:
            signals['overall_signal'] = 'STRONG_SELL'
        elif sell_signals >= 3:
            signals['overall_signal'] = 'SELL'
        else:
            signals['overall_signal'] = 'NEUTRAL'
        
        # Add current price info
        signals['current_price'] = latest['Close']
        signals['atr'] = latest['ATR']
        
        return signals
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength using ADX or similar
        """
        if len(df) < 14:
            return 0
        
        # Simplified trend strength calculation
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        volatility = df['Close'].pct_change().std()
        
        if volatility > 0:
            trend_strength = abs(price_change) / volatility
        else:
            trend_strength = 0
        
        return min(trend_strength, 1.0)

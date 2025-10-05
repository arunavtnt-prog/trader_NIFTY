"""
Configuration file for Paper Trading System
"""

import os
from datetime import datetime
from typing import Dict, List

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsapi_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Optional
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")  # Optional

# Trading Configuration
class TradingConfig:
    # Trading Modes
    INTRADAY = "intraday"
    SWING = "swing"
    
    # Default symbols to track
    DEFAULT_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
        "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ITC",
        "AXISBANK", "LT", "DMART", "SUNPHARMA", "BAJFINANCE"
    ]
    
    # Index for comparison
    BENCHMARK_INDEX = "NIFTY 50"
    
    # Trading parameters
    INTRADAY_CONFIG = {
        "timeframe": "5min",
        "max_positions": 5,
        "stop_loss_pct": 1.0,  # 1% stop loss
        "target_pct": 2.0,     # 2% target
        "position_size_pct": 20,  # 20% of capital per position
    }
    
    SWING_CONFIG = {
        "timeframe": "1d",
        "max_positions": 10,
        "stop_loss_pct": 5.0,  # 5% stop loss
        "target_pct": 15.0,    # 15% target
        "position_size_pct": 10,  # 10% of capital per position
        "holding_period_days": 10,  # Max holding period
    }
    
    # Capital
    INITIAL_CAPITAL = 1000000  # 10 Lakhs INR
    
    # Technical Indicators Config
    TECHNICAL_CONFIG = {
        "sma_short": 10,
        "sma_long": 50,
        "ema_short": 12,
        "ema_long": 26,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "volume_ma": 20,
    }
    
    # Sentiment Analysis Config
    SENTIMENT_CONFIG = {
        "model_name": "ProsusAI/finbert",  # Financial BERT model
        "news_impact_weight": 0.3,  # 30% weight to news sentiment
        "technical_weight": 0.5,     # 50% weight to technical indicators
        "market_regime_weight": 0.2, # 20% weight to market regime
        "sentiment_threshold_buy": 0.6,
        "sentiment_threshold_sell": -0.6,
    }
    
    # Risk Management
    RISK_CONFIG = {
        "max_portfolio_risk": 0.02,  # 2% max portfolio risk per day
        "max_correlation": 0.7,       # Max correlation between positions
        "min_liquidity_volume": 100000,  # Min volume for stock selection
        "max_sector_exposure": 0.4,   # Max 40% in one sector
    }
    
    # Logging
    LOG_DIR = "logs"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Metrics
    METRICS_CONFIG = {
        "risk_free_rate": 0.06,  # 6% annual risk-free rate
        "trading_days": 252,      # Trading days in a year
        "slippage_pct": 0.1,      # 0.1% slippage
        "commission_pct": 0.03,   # 0.03% brokerage
    }
    
    # Data fetch intervals
    DATA_FETCH_INTERVALS = {
        "intraday": 60,   # Fetch every 60 seconds for intraday
        "swing": 3600,    # Fetch every hour for swing trading
    }
    
    @classmethod
    def get_config(cls, mode: str) -> Dict:
        """Get configuration based on trading mode"""
        if mode == cls.INTRADAY:
            return cls.INTRADAY_CONFIG
        elif mode == cls.SWING:
            return cls.SWING_CONFIG
        else:
            raise ValueError(f"Invalid trading mode: {mode}")

# Market Hours (IST)
MARKET_HOURS = {
    "pre_open_start": "09:00",
    "pre_open_end": "09:15",
    "market_open": "09:15",
    "market_close": "15:30",
    "post_market_start": "15:30",
    "post_market_end": "16:00",
}

# Sectors mapping for NSE stocks
SECTOR_MAPPING = {
    "ACMESOLAR": "Power",
    "AADHARHFC": "Financial Services",
    "AARTIIND": "Chemicals",
    "AAVAS": "Financial Services",
    "ACE": "Capital Goods",
    "ABFRL": "Consumer Services",
    "ABLBL": "Consumer Services",
    "ABREL": "Forest Materials",
    "ABSLAMC": "Financial Services",
    "AEGISLOG": "Oil Gas & Consumable Fuels",
    "AEGISVOPAK": "Oil Gas & Consumable Fuels",
    "AFCONS": "Construction",
    "AFFLE": "Information Technology",
    "AKUMS": "Healthcare",
    "AKZOINDIA": "Consumer Durables",
    "APLLTD": "Healthcare",
    "ALKYLAMINE": "Chemicals",
    "ALOKINDS": "Textiles",
    "ARE&M": "Automobile and Auto Components",
    "AMBER": "Consumer Durables",
    "ANANDRATHI": "Financial Services",
    "ANANTRAJ": "Realty",
    "ANGELONE": "Financial Services",
    "APTUS": "Financial Services",
    "ASAHIINDIA": "Automobile and Auto Components",
    "ASTERDM": "Healthcare",
    "ASTRAZEN": "Healthcare",
    "ATHERENERG": "Automobile and Auto Components",
    "ATUL": "Chemicals",
    "AIIL": "Financial Services",
    "BASF": "Chemicals",
    "BEML": "Capital Goods",
    "BLS": "Consumer Services",
    "BALRAMCHIN": "Fast Moving Consumer Goods",
    "BANDHANBNK": "Financial Services",
    "BATAINDIA": "Consumer Durables",
    "BAYERCROP": "Chemicals",
    "BIKAJI": "Fast Moving Consumer Goods",
    "BSOFT": "Information Technology",
    "BLUEDART": "Services",
    "BLUEJET": "Healthcare",
    "BBTC": "Fast Moving Consumer Goods",
    "FIRSTCRY": "Consumer Services",
    "BRIGADE": "Realty",
    "MAPMYINDIA": "Information Technology",
    "CCL": "Fast Moving Consumer Goods",
    "CESC": "Power",
    "CAMPUS": "Consumer Durables",
    "CANFINHOME": "Financial Services",
    "CAPLIPOINT": "Healthcare",
    "CGCL": "Financial Services",
    "CARBORUNIV": "Capital Goods",
    "CASTROLIND": "Oil Gas & Consumable Fuels",
    "CEATLTD": "Automobile and Auto Components",
    "CENTRALBK": "Financial Services",
    "CDSL": "Financial Services",
    "CENTURYPLY": "Consumer Durables",
    "CERA": "Consumer Durables",
    "CHALET": "Consumer Services",
    "CHAMBLFERT": "Chemicals",
    "CHENNPETRO": "Oil Gas & Consumable Fuels",
    "CHOICEIN": "Financial Services",
    "CHOLAHLDNG": "Financial Services",
    "CUB": "Financial Services",
    "CLEAN": "Chemicals",
    "COHANCE": "Healthcare",
    "CAMS": "Financial Services",
    "CONCORDBIO": "Healthcare",
    "CRAFTSMAN": "Automobile and Auto Components",
    "CREDITACC": "Financial Services",
    "CROMPTON": "Consumer Durables",
    "CYIENT": "Information Technology",
    "DCMSHRIRAM": "Diversified",
    "DOMS": "Fast Moving Consumer Goods",
    "DATAPATTNS": "Capital Goods",
    "DEEPAKFERT": "Chemicals",
    "DELHIVERY": "Services",
    "DEVYANI": "Consumer Services",
    "AGARWALEYE": "Healthcare",
    "LALPATHLAB": "Healthcare",
    "DUMMYDBRLT": "Consumer Services",
    "EIDPARRY": "Fast Moving Consumer Goods",
    "EIHOTEL": "Consumer Services",
    "ELECON": "Capital Goods",
    "ELGIEQUIP": "Capital Goods",
    "EMAMILTD": "Fast Moving Consumer Goods",
    "EMCURE": "Healthcare",
    "ENGINERSIN": "Construction",
    "ERIS": "Healthcare",
    "FINCABLES": "Capital Goods",
    "FINPIPE": "Capital Goods",
    "FSL": "Services",
    "FIVESTAR": "Financial Services",
    "FORCEMOT": "Automobile and Auto Components",
    "GRSE": "Capital Goods",
    "GILLETTE": "Fast Moving Consumer Goods",
    "GLAND": "Healthcare",
    "GODIGIT": "Financial Services",
    "GPIL": "Capital Goods",
    "GODREJAGRO": "Fast Moving Consumer Goods",
    "GRANULES": "Healthcare",
    "GRAPHITE": "Capital Goods",
    "GRAVITA": "Metals & Mining",
    "GESHIP": "Services",
    "GMDCLTD": "Metals & Mining",
    "GSPL": "Oil Gas & Consumable Fuels",
    "HEG": "Capital Goods",
    "HBLENGINE": "Capital Goods",
    "HFCL": "Telecommunication",
    "HAPPSTMNDS": "Information Technology",
    "HSCL": "Chemicals",
    "HINDCOPPER": "Metals & Mining",
    "HOMEFIRST": "Financial Services",
    "HONASA": "Fast Moving Consumer Goods",
    "IFCI": "Financial Services",
    "IIFL": "Financial Services",
    "INOXINDIA": "Capital Goods",
    "IRCON": "Construction",
    "ITI": "Telecommunication",
    "INDGN": "Healthcare",
    "INDIACEM": "Construction Materials",
    "INDIAMART": "Consumer Services",
    "IEX": "Financial Services",
    "INOXWIND": "Capital Goods",
    "INTELLECT": "Information Technology",
    "IGIL": "Services",
    "IKS": "Information Technology",
    "JBCHEPHARM": "Healthcare",
    "JBMA": "Automobile and Auto Components",
    "JKTYRE": "Automobile and Auto Components",
    "JMFINANCIL": "Financial Services",
    "JPPOWER": "Power",
    "J&KBANK": "Financial Services",
    "JINDALSAW": "Capital Goods",
    "JUBLINGREA": "Chemicals",
    "JUBLPHARMA": "Healthcare",
    "JWL": "Capital Goods",
    "JYOTHYLAB": "Fast Moving Consumer Goods",
    "JYOTICNC": "Capital Goods",
    "KSB": "Capital Goods",
    "KAJARIACER": "Consumer Durables",
    "KPIL": "Construction",
    "KARURVYSYA": "Financial Services",
    "KAYNES": "Capital Goods",
    "KEC": "Construction",
    "KFINTECH": "Financial Services",
    "KIRLOSBROS": "Capital Goods",
    "KIRLOSENG": "Capital Goods",
    "KIMS": "Healthcare",
    "LTFOODS": "Fast Moving Consumer Goods",
    "LATENTVIEW": "Information Technology",
    "LAURUSLABS": "Healthcare",
    "LEMONTREE": "Consumer Services",
    "MMTC": "Services",
    "MGL": "Oil Gas & Consumable Fuels",
    "MAHSCOOTER": "Financial Services",
    "MAHSEAMLES": "Capital Goods",
    "MANAPPURAM": "Financial Services",
    "MRPL": "Oil Gas & Consumable Fuels",
    "METROPOLIS": "Healthcare",
    "MINDACORP": "Automobile and Auto Components",
    "MSUMI": "Automobile and Auto Components",
    "MCX": "Financial Services",
    "NATCOPHARM": "Healthcare",
    "NBCC": "Construction",
    "NCC": "Construction",
    "NSLNISP": "Metals & Mining",
    "NH": "Healthcare",
    "NAVA": "Power",
    "NAVINFLUOR": "Chemicals",
    "NETWEB": "Information Technology",
    "NEULANDLAB": "Healthcare",
    "NEWGEN": "Information Technology",
    "NIVABUPA": "Financial Services",
    "NUVAMA": "Financial Services",
    "NUVOCO": "Construction Materials",
    "OLAELEC": "Automobile and Auto Components",
    "OLECTRA": "Automobile and Auto Components",
    "ONESOURCE": "Healthcare",
    "PCBL": "Chemicals",
    "PGEL": "Consumer Durables",
    "PNBHOUSING": "Financial Services",
    "PTCIL": "Capital Goods",
    "PVRINOX": "Media Entertainment & Publication",
    "PFIZER": "Healthcare",
    "PPLPHARMA": "Healthcare",
    "POLYMED": "Healthcare",
    "POONAWALLA": "Financial Services",
    "PRAJIND": "Capital Goods",
    "RRKABEL": "Capital Goods",
    "RBLBANK": "Financial Services",
    "RHIM": "Capital Goods",
    "RITES": "Construction",
    "RADICO": "Fast Moving Consumer Goods",
    "RAILTEL": "Telecommunication",
    "RAINBOW": "Healthcare",
    "RKFORGE": "Automobile and Auto Components",
    "RCF": "Chemicals",
    "REDINGTON": "Services",
    "RELINFRA": "Power",
    "RPOWER": "Power",
    "SBFC": "Financial Services",
    "SKFINDIA": "Capital Goods",
    "SAGILITY": "Information Technology",
    "SAILIFE": "Healthcare",
    "SAMMAANCAP": "Financial Services",
    "SAPPHIRE": "Consumer Services",
    "SARDAEN": "Metals & Mining",
    "SAREGAMA": "Media Entertainment & Publication",
    "THELEELA": "Consumer Services",
    "SCHNEIDER": "Capital Goods",
    "SCI": "Services",
    "SHYAMMETL": "Capital Goods",
    "SIGNATURE": "Realty",
    "SOBHA": "Realty",
    "SONATSOFTW": "Information Technology",
    "STARHEALTH": "Financial Services",
    "SUMICHEM": "Chemicals",
    "SUNTV": "Media Entertainment & Publication",
    "SUNDRMFAST": "Automobile and Auto Components",
    "SWANCORP": "Chemicals",
    "SYRMA": "Capital Goods",
    "TBOTEK": "Consumer Services",
    "TATACHEM": "Chemicals",
    "TTML": "Telecommunication",
    "TECHNOE": "Construction",
    "TEJASNET": "Telecommunication",
    "RAMCOCEM": "Construction Materials",
    "TIMKEN": "Capital Goods",
    "TITAGARH": "Capital Goods",
    "TARIL": "Capital Goods",
    "TRIDENT": "Textiles",
    "TRIVENI": "Fast Moving Consumer Goods",
    "TRITURBINE": "Capital Goods",
    "UTIAMC": "Financial Services",
    "USHAMART": "Capital Goods",
    "VGUARD": "Consumer Durables",
    "DBREALTY": "Consumer Services",
    "VTL": "Textiles",
    "MANYAVAR": "Consumer Services",
    "VENTIVE": "Consumer Services",
    "VIJAYA": "Healthcare",
    "WELCORP": "Capital Goods",
    "WELSPUNLIV": "Textiles",
    "WHIRLPOOL": "Consumer Durables",
    "WOCKPHARMA": "Healthcare",
    "ZFCVINDIA": "Automobile and Auto Components",
    "ZEEL": "Media Entertainment & Publication",
    "ZENTEC": "Capital Goods",
    "ZENSARTECH": "Information Technology",
    "ECLERX": "Services",
}

SYMBOLS = [
    "ACMESOLAR","AADHARHFC","AARTIIND","AAVAS","ACE","ABFRL","ABLBL","ABREL","ABSLAMC","AEGISLOG","AEGISVOPAK","AFCONS","AFFLE","AKUMS","AKZOINDIA","APLLTD","ALKYLAMINE","ALOKINDS","ARE&M","AMBER",
    "ANANDRATHI","ANANTRAJ","ANGELONE","APTUS","ASAHIINDIA","ASTERDM","ASTRAZEN","ATHERENERG","ATUL","AIIL","BASF","BEML","BLS","BALRAMCHIN","BANDHANBNK","BATAINDIA","BAYERCROP","BIKAJI","BSOFT","BLUEDART","BLUEJET","BBTC","FIRSTCRY","BRIGADE","MAPMYINDIA","CCL","CESC","CAMPUS","CANFINHOME","CAPLIPOINT",
    "CGCL","CARBORUNIV","CASTROLIND","CEATLTD","CENTRALBK","CDSL","CENTURYPLY","CERA","CHALET","CHAMBLFERT","CHENNPETRO","CHOICEIN","CHOLAHLDNG","CUB","CLEAN","COHANCE","CAMS","CONCORDBIO","CRAFTSMAN",
    "CREDITACC","CROMPTON","CYIENT","DCMSHRIRAM","DOMS","DATAPATTNS","DEEPAKFERT","DELHIVERY","DEVYANI","AGARWALEYE","LALPATHLAB","DUMMYDBRLT","EIDPARRY","EIHOTEL","ELECON","ELGIEQUIP","EMAMILTD",
    "EMCURE","ENGINERSIN","ERIS","FINCABLES","FINPIPE","FSL","FIVESTAR","FORCEMOT","GRSE","GILLETTE","GLAND","GODIGIT","GPIL","GODREJAGRO","GRANULES","GRAPHITE","GRAVITA","GESHIP","GMDCLTD",
    "GSPL","HEG","HBLENGINE","HFCL","HAPPSTMNDS","HSCL","HINDCOPPER","HOMEFIRST","HONASA","IFCI","IIFL","INOXINDIA","IRCON","ITI","INDGN","INDIACEM","INDIAMART","IEX","INOXWIND","INTELLECT","IGIL",
    "IKS","JBCHEPHARM","JBMA","JKTYRE","JMFINANCIL","JPPOWER","J&KBANK","JINDALSAW","JUBLINGREA","JUBLPHARMA","JWL","JYOTHYLAB","JYOTICNC","KSB","KAJARIACER","KPIL","KARURVYSYA","KAYNES",
    "KEC","KFINTECH","KIRLOSBROS","KIRLOSENG","KIMS","LTFOODS","LATENTVIEW","LAURUSLABS","LEMONTREE","MMTC","MGL","MAHSCOOTER","MAHSEAMLES","MANAPPURAM","MRPL","METROPOLIS","MINDACORP",
    "MSUMI","MCX","NATCOPHARM","NBCC","NCC","NSLNISP","NH","NAVA","NAVINFLUOR","NETWEB","NEULANDLAB","NEWGEN","NIVABUPA","NUVAMA","NUVOCO","OLAELEC","OLECTRA","ONESOURCE","PCBL","PGEL","PNBHOUSING","PTCIL","PVRINOX","PFIZER","PPLPHARMA","POLYMED","POONAWALLA","PRAJIND","RRKABEL","RBLBANK","RHIM","RITES","RADICO","RAILTEL","RAINBOW","RKFORGE","RCF","REDINGTON","RELINFRA","RPOWER","SBFC","SKFINDIA","SAGILITY","SAILIFE","SAMMAANCAP","SAPPHIRE","SARDAEN","SAREGAMA","THELEELA","SCHNEIDER","SCI","SHYAMMETL","SIGNATURE","SOBHA","SONATSOFTW","STARHEALTH","SUMICHEM","SUNTV","SUNDRMFAST","SWANCORP","SYRMA","TBOTEK","TATACHEM","TTML","TECHNOE","TEJASNET","RAMCOCEM","TIMKEN","TITAGARH","TARIL","TRIDENT","TRIVENI","TRITURBINE","UTIAMC","USHAMART","VGUARD","DBREALTY","VTL","MANYAVAR","VENTIVE","VIJAYA","WELCORP","WELSPUNLIV","WHIRLPOOL","WOCKPHARMA","ZFCVINDIA","ZEEL","ZENTEC","ZENSARTECH","ECLERX",
]

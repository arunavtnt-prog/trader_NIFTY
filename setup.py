#!/usr/bin/env python
"""
Setup script for AI-Driven Paper Trading System
Handles environment setup, dependency installation, and initial configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("="*60)
    print("   AI-DRIVEN PAPER TRADING SYSTEM")
    print("   Setup and Installation Script")
    print("="*60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'charts', 'reports', 'data']
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    if not Path("venv").exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("  ✓ Virtual environment created")
    else:
        print("  ✓ Virtual environment already exists")
    
    # Provide activation instructions
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"\n  ⚠️  Please activate the virtual environment:")
    print(f"     {activate_cmd}")
    
    return activate_cmd

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies that must be installed
    core_deps = [
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "yfinance>=0.2.28",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "schedule>=1.2.0",
        "python-dotenv>=1.0.0"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        print(f"  Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                      capture_output=True, text=True)
    
    print("  ✓ Core dependencies installed")
    
    # Try to install ML dependencies
    print("\n  Installing ML dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "torch", "transformers", "scikit-learn"], 
                      capture_output=True, text=True)
        print("  ✓ ML dependencies installed")
    except Exception as e:
        print(f"  ⚠️  ML dependencies installation failed: {e}")
        print("     You may need to install PyTorch manually from https://pytorch.org")

def create_env_file():
    """Create .env file from template"""
    print("\n📝 Setting up configuration...")
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            with open(".env.example", "r") as src:
                with open(".env", "w") as dst:
                    dst.write(src.read())
            print("  ✓ Created .env file from template")
        else:
            # Create basic .env file
            env_content = """# News API Configuration
NEWS_API_KEY=your_newsapi_key_here

# Optional: LLM API Keys
OPENAI_API_KEY=
CLAUDE_API_KEY=
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print("  ✓ Created basic .env file")
    else:
        print("  ✓ .env file already exists")
    
    print("\n  ⚠️  Please edit .env file and add your API keys:")
    print("     - Get News API key from: https://newsapi.org/register")
    print("     - (Optional) OpenAI key from: https://platform.openai.com/")

def test_installation():
    """Test if all modules can be imported"""
    print("\n🧪 Testing installation...")
    
    modules_to_test = [
        "config",
        "data_fetcher",
        "ai_module",
        "technical_indicators",
        "paper_trading_engine",
        "visualization",
        "main"
    ]
    
    all_ok = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module} loaded successfully")
        except ImportError as e:
            print(f"  ❌ Failed to load {module}: {e}")
            all_ok = False
    
    return all_ok

def download_ml_models():
    """Pre-download ML models"""
    print("\n🤖 Downloading ML models (this may take a few minutes)...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "ProsusAI/finbert"
        print(f"  Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print("  ✓ FinBERT model downloaded successfully")
    except Exception as e:
        print(f"  ⚠️  Could not download ML model: {e}")
        print("     Model will be downloaded on first use")

def run_quick_test():
    """Run a quick test of the system"""
    print("\n🚀 Running quick system test...")
    
    try:
        # Test data fetching
        from data_fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher()
        
        print("  Testing market data fetch...")
        data = fetcher.get_nse_data("RELIANCE")
        if data:
            print(f"  ✓ Successfully fetched data for RELIANCE")
            print(f"    Current Price: ₹{data.get('current_price', 0):.2f}")
        else:
            print("  ⚠️  Could not fetch market data (market may be closed)")
        
        # Test technical indicators
        from technical_indicators import TechnicalIndicators
        print("  ✓ Technical indicators module loaded")
        
        # Test trading engine
        from paper_trading_engine import PaperTradingEngine
        engine = PaperTradingEngine()
        print(f"  ✓ Trading engine initialized with ₹{engine.portfolio.initial_capital:,.0f}")
        
        print("\n✅ System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1. Activate virtual environment (if not already activated):")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Configure API keys in .env file:")
    print("   Edit .env and add your News API key")
    
    print("\n3. Run the demo to see all features:")
    print("   python demo.py")
    
    print("\n4. Start paper trading:")
    print("   python main.py --run-once      # Single cycle")
    print("   python main.py --mode intraday # Continuous intraday")
    print("   python main.py --mode swing    # Swing trading")
    
    print("\n5. Run backtest on historical data:")
    print("   python main.py --backtest --start-date 2024-01-01")
    
    print("\n📚 For detailed documentation, see README.md")
    print("\n💡 Tips:")
    print("  - Start with demo.py to understand the system")
    print("  - Use --run-once flag for testing")
    print("  - Monitor logs/ directory for detailed information")
    print("  - Check charts/ directory for visualizations")
    
    print("\nHappy Trading! 📈")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Create virtual environment
    activate_cmd = create_virtual_environment()
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        print("\n✓ Virtual environment is active")
        
        # Install dependencies
        install_dependencies()
        
        # Create .env file
        create_env_file()
        
        # Test installation
        if test_installation():
            # Download ML models
            download_ml_models()
            
            # Run quick test
            run_quick_test()
            
            # Print next steps
            print_next_steps()
        else:
            print("\n❌ Installation test failed. Please check error messages above.")
    else:
        print("\n⚠️  Virtual environment is not active!")
        print(f"Please run: {activate_cmd}")
        print("Then run this setup script again: python setup.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
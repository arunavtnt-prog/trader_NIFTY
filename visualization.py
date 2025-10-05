"""
Visualization Module for Performance Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Create various performance visualizations"""
    
    def __init__(self, save_dir: str = "charts"):
        self.save_dir = save_dir
        
    def plot_portfolio_value(self, portfolio_history: List[Dict], 
                           benchmark_data: Optional[pd.Series] = None,
                           save: bool = True) -> None:
        """Plot portfolio value over time"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(portfolio_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'P&L'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add initial capital line
            fig.add_hline(
                y=df['initial_capital'].iloc[0],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
                row=1, col=1
            )
            
            # P&L
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['total_pnl'],
                    name='P&L',
                    marker_color=np.where(df['total_pnl'] >= 0, 'green', 'red')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Value (INR)", row=1, col=1)
            fig.update_yaxes(title_text="P&L (INR)", row=2, col=1)
            
            if save:
                fig.write_html(f"{self.save_dir}/portfolio_value.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting portfolio value: {e}")
    
    def plot_returns_distribution(self, returns: pd.Series, save: bool = True) -> None:
        """Plot returns distribution"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Histogram
            axes[0, 0].hist(returns, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.4f}')
            axes[0, 0].set_title('Returns Distribution')
            axes[0, 0].set_xlabel('Returns')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            
            # Cumulative returns
            cumulative = (1 + returns).cumprod()
            axes[1, 0].plot(cumulative.index, cumulative.values)
            axes[1, 0].set_title('Cumulative Returns')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Cumulative Return')
            axes[1, 0].grid(True)
            
            # Box plot
            axes[1, 1].boxplot(returns.values)
            axes[1, 1].set_title('Returns Box Plot')
            axes[1, 1].set_ylabel('Returns')
            
            plt.tight_layout()
            
            if save:
                plt.savefig(f"{self.save_dir}/returns_distribution.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
    
    def plot_drawdown(self, portfolio_history: List[Dict], save: bool = True) -> None:
        """Plot drawdown chart"""
        try:
            df = pd.DataFrame(portfolio_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate drawdown
            df['cumulative'] = df['total_value'] / df['total_value'].iloc[0]
            df['running_max'] = df['cumulative'].expanding().max()
            df['drawdown'] = (df['cumulative'] - df['running_max']) / df['running_max'] * 100
            
            fig = go.Figure()
            
            # Drawdown area
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['drawdown'],
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            # Max drawdown line
            max_dd = df['drawdown'].min()
            max_dd_date = df[df['drawdown'] == max_dd]['timestamp'].iloc[0]
            
            fig.add_hline(
                y=max_dd,
                line_dash="dash",
                line_color="darkred",
                annotation_text=f"Max Drawdown: {max_dd:.2f}%"
            )
            
            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=400,
                hovermode='x unified'
            )
            
            if save:
                fig.write_html(f"{self.save_dir}/drawdown.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
    
    def plot_trade_analysis(self, closed_positions: List[Dict], save: bool = True) -> None:
        """Analyze closed trades"""
        if not closed_positions:
            logger.warning("No closed positions to analyze")
            return
        
        try:
            df = pd.DataFrame(closed_positions)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Trade P&L Distribution',
                    'Win/Loss Ratio',
                    'Trade P&L Over Time',
                    'Holding Period Analysis'
                )
            )
            
            # P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=df['realized_pnl'],
                    nbinsx=30,
                    name='P&L Distribution',
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Win/Loss Pie Chart
            wins = len(df[df['realized_pnl'] > 0])
            losses = len(df[df['realized_pnl'] < 0])
            
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    marker=dict(colors=['green', 'red'])
                ),
                row=1, col=2
            )
            
            # P&L Over Time
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df_sorted = df.sort_values('exit_time')
            df_sorted['cumulative_pnl'] = df_sorted['realized_pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['exit_time'],
                    y=df_sorted['cumulative_pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Holding Period vs Returns
            fig.add_trace(
                go.Scatter(
                    x=df['holding_period'],
                    y=df['realized_pnl_pct'],
                    mode='markers',
                    name='Returns vs Holding Period',
                    marker=dict(
                        color=df['realized_pnl'],
                        colorscale='RdYlGn',
                        showscale=True,
                        size=10
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            fig.update_xaxes(title_text="P&L (INR)", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_xaxes(title_text="Holding Period (hours)", row=2, col=2)
            fig.update_yaxes(title_text="Cumulative P&L", row=2, col=1)
            fig.update_yaxes(title_text="Return (%)", row=2, col=2)
            
            if save:
                fig.write_html(f"{self.save_dir}/trade_analysis.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {e}")
    
    def plot_metrics_comparison(self, metrics: Dict, benchmark_metrics: Optional[Dict] = None,
                              save: bool = True) -> None:
        """Plot key metrics comparison"""
        try:
            fig = go.Figure()
            
            # Prepare data
            metric_names = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                          'Win Rate (%)', 'Profit Factor']
            
            strategy_values = [
                metrics.get('total_return_pct', 0),
                metrics.get('sharpe_ratio', 0),
                abs(metrics.get('max_drawdown', 0)),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0)
            ]
            
            x = np.arange(len(metric_names))
            
            fig.add_trace(go.Bar(
                name='Strategy',
                x=metric_names,
                y=strategy_values,
                marker_color='blue'
            ))
            
            if benchmark_metrics:
                benchmark_values = [
                    benchmark_metrics.get('total_return_pct', 0),
                    benchmark_metrics.get('sharpe_ratio', 0),
                    abs(benchmark_metrics.get('max_drawdown', 0)),
                    benchmark_metrics.get('win_rate', 0),
                    benchmark_metrics.get('profit_factor', 0)
                ]
                
                fig.add_trace(go.Bar(
                    name='Benchmark',
                    x=metric_names,
                    y=benchmark_values,
                    marker_color='gray'
                ))
            
            fig.update_layout(
                title='Performance Metrics Comparison',
                yaxis_title='Value',
                barmode='group',
                height=500
            )
            
            if save:
                fig.write_html(f"{self.save_dir}/metrics_comparison.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting metrics comparison: {e}")
    
    def create_dashboard(self, portfolio_history: List[Dict],
                        closed_positions: List[Dict],
                        metrics: Dict,
                        save: bool = True) -> None:
        """Create comprehensive dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Portfolio Value', 'Returns Distribution',
                    'Drawdown', 'Trade P&L',
                    'Key Metrics', 'Position Sizes'
                ),
                specs=[
                    [{'type': 'scatter'}, {'type': 'histogram'}],
                    [{'type': 'scatter'}, {'type': 'bar'}],
                    [{'type': 'table'}, {'type': 'pie'}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Convert data
            df_portfolio = pd.DataFrame(portfolio_history)
            df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])
            
            # 1. Portfolio Value
            fig.add_trace(
                go.Scatter(
                    x=df_portfolio['timestamp'],
                    y=df_portfolio['total_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 2. Returns Distribution
            if 'returns' in df_portfolio.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df_portfolio['returns'],
                        nbinsx=30,
                        name='Returns'
                    ),
                    row=1, col=2
                )
            
            # 3. Drawdown
            df_portfolio['cumulative'] = df_portfolio['total_value'] / df_portfolio['total_value'].iloc[0]
            df_portfolio['running_max'] = df_portfolio['cumulative'].expanding().max()
            df_portfolio['drawdown'] = (df_portfolio['cumulative'] - df_portfolio['running_max']) / df_portfolio['running_max']
            
            fig.add_trace(
                go.Scatter(
                    x=df_portfolio['timestamp'],
                    y=df_portfolio['drawdown'] * 100,
                    fill='tozeroy',
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # 4. Trade P&L
            if closed_positions:
                df_trades = pd.DataFrame(closed_positions)
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(df_trades))),
                        y=df_trades['realized_pnl'],
                        name='Trade P&L',
                        marker_color=np.where(df_trades['realized_pnl'] >= 0, 'green', 'red')
                    ),
                    row=2, col=2
                )
            
            # 5. Key Metrics Table
            metrics_table = go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(
                    values=[
                        ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 
                         'Win Rate', 'Total Trades'],
                        [f"{metrics.get('total_return_pct', 0):.2f}%",
                         f"{metrics.get('sharpe_ratio', 0):.2f}",
                         f"{metrics.get('max_drawdown', 0):.2f}%",
                         f"{metrics.get('win_rate', 0):.2f}%",
                         f"{metrics.get('total_trades', 0)}"]
                    ]
                )
            )
            fig.add_trace(metrics_table, row=3, col=1)
            
            # 6. Position Distribution (if available)
            if 'positions' in df_portfolio.columns and df_portfolio['positions'].iloc[-1]:
                positions = df_portfolio['positions'].iloc[-1]
                if positions:
                    symbols = list(positions.keys())
                    values = [pos['market_value'] for pos in positions.values()]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=symbols,
                            values=values,
                            name='Positions'
                        ),
                        row=3, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title='Trading Strategy Dashboard',
                height=1200,
                showlegend=False
            )
            
            if save:
                fig.write_html(f"{self.save_dir}/dashboard.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
    
    def plot_signals_on_chart(self, symbol: str, price_data: pd.DataFrame,
                             buy_signals: List[datetime], sell_signals: List[datetime],
                             indicators: Dict = None, save: bool = True) -> None:
        """Plot price chart with signals and indicators"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f'{symbol} Price & Signals', 'RSI', 'Volume')
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'SMA_short' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['SMA_short'],
                        mode='lines',
                        name='SMA Short',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_long' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['SMA_long'],
                        mode='lines',
                        name='SMA Long',
                        line=dict(color='purple', width=1)
                    ),
                    row=1, col=1
                )
            
            # Add buy signals
            if buy_signals:
                buy_prices = [price_data.loc[price_data.index == signal, 'Close'].iloc[0] 
                             for signal in buy_signals if signal in price_data.index]
                buy_times = [signal for signal in buy_signals if signal in price_data.index]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        )
                    ),
                    row=1, col=1
                )
            
            # Add sell signals
            if sell_signals:
                sell_prices = [price_data.loc[price_data.index == signal, 'Close'].iloc[0]
                              for signal in sell_signals if signal in price_data.index]
                sell_times = [signal for signal in sell_signals if signal in price_data.index]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        )
                    ),
                    row=1, col=1
                )
            
            # RSI
            if 'RSI' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            colors = ['red' if row['Close'] < row['Open'] else 'green' 
                     for _, row in price_data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Trading Signals',
                height=800,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=3, col=1)
            
            if save:
                fig.write_html(f"{self.save_dir}/{symbol}_signals.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting signals on chart: {e}")

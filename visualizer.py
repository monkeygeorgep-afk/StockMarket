import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    @staticmethod
    def plot_tech_main(df):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], line=dict(color='rgba(255,255,255,0.1)'), name="布林上"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], line=dict(color='rgba(255,255,255,0.1)'), fill='tonexty', name="布林下"), row=1, col=1)
        
        v_colors = ['#FF4D4D' if c >= o else '#2ECC71' for o, c in zip(df['Open'], df['Close'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['Volume'], marker_color=v_colors, name="成交量"), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        return fig

    @staticmethod
    def plot_broker_landscape(broker_df):
        if broker_df.empty: return None
        latest_date = broker_df['date'].max()
        df = broker_df[broker_df['date'] == latest_date].copy()
        df['net_buy'] = df['buy'] - df['sell']
        
        top_buyers = df.nlargest(10, 'net_buy').sort_values('net_buy', ascending=True)
        top_sellers = df.nsmallest(10, 'net_buy').sort_values('net_buy', ascending=True)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("🔥 買超分點", "❄️ 賣超分點"))
        fig.add_trace(go.Bar(x=top_buyers['net_buy'], y=top_buyers['broker_name'], orientation='h', marker=dict(color=top_buyers['net_buy'], colorscale='Reds')), row=1, col=1)
        fig.add_trace(go.Bar(x=top_sellers['net_buy'], y=top_sellers['broker_name'], orientation='h', marker=dict(color=top_sellers['net_buy'], colorscale='Greens_r')), row=1, col=2)
        fig.update_layout(height=450, template="plotly_dark", showlegend=False, title_text=f"⚖️ 分點資金對峙圖 ({latest_date})")
        return fig

    @staticmethod
    def plot_river(df):
        if 'ttm_eps' not in df.columns or df['ttm_eps'].isnull().all(): return None
        fig = go.Figure()
        pe_list = [10, 15, 20, 25, 30]
        for p in pe_list:
            fig.add_trace(go.Scatter(x=df['date'], y=df['ttm_eps']*p, name=f"{p}x PE", line_width=0.8))
        fig.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="收盤價", line=dict(color='white', width=2)))
        fig.update_layout(height=600, template="plotly_dark", title="本益比河流圖")
        return fig
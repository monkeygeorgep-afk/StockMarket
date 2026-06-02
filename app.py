import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import io

# --- 1. 系統配置 ---
ST_CONFIG = {
    "page_title": "台股 AI 智慧分析系統 (Messaging API 版)",
}

# --- 2. 原生技術指標計算 (免安裝額外套件) ---
def calculate_indicators(df):
    # 移動平均線 (MA)
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道 (Bollinger Bands)
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['20MA'] + (std * 2)
    df['BBL'] = df['20MA'] - (std * 2)
    
    # RSI 計算 (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 計算
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

# --- 3. 數據獲取模組 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(stock_id, start_date):
    dl = DataLoader()
    try:
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
        if df.empty: return pd.DataFrame()
        
        # 統一欄位名稱
        df = df.rename(columns={
            'max': 'High', 'min': 'Low', 'close': 'Close', 
            'open': 'Open', 'Trading_Volume': 'Volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = calculate_indicators(df)
        return df
    except Exception:
        return pd.DataFrame()

# --- 4. 監控演算法模組 ---
def scan_signals(df):
    signals = []
    if len(df) < 2: return signals
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # A. 量能異常
    avg_vol = df['Volume'].iloc[-6:-1].mean()
    if curr['Volume'] > avg_vol * 2.5:
        signals.append("🚀 爆量噴發：成交量大於5日均線 2.5 倍")
    
    # B. 價格突破
    recent_high = df['High'].iloc[-61:-1].max()
    if curr['Close'] > recent_high:
        signals.append("📈 強勢突破：創下 60 日新高")
        
    # C. 均線位階
    if prev['Close'] < prev['20MA'] and curr['Close'] > curr['20MA']:
        signals.append("📊 轉強訊號：收盤站上月線 (20MA)")
        
    return signals

# --- 5. Line Messaging API 通知模組 ---
def send_line_push(stock_id, signals):
    if not signals: return
    
    # 從 Streamlit Secrets 讀取 (部署時需設定)
    try:
        token = st.secrets["LINE_CHANNEL_ACCESS_TOKEN"]
        user_id = st.secrets["LINE_USER_ID"]
    except KeyError:
        st.error("請在 Secrets 中設定 LINE_CHANNEL_ACCESS_TOKEN 與 LINE_USER_ID")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    msg_text = f"🔔 【台股 AI 監控】\n標的：{stock_id}\n時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n偵測訊號：\n" + "\n".join(signals)
    
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": msg_text}]
    }
    
    res = requests.post(url, headers=headers, json=payload)
    return res.status_code

# --- 6. Streamlit UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    
    # 側邊欄控制區
    with st.sidebar:
        st.header("📊 參數設定")
        target_stock = st.text_input("股票代碼", value="2330")
        lookback = st.slider("顯示範圍 (天)", 60, 365, 120)
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback+60)).strftime('%Y-%m-%d')
        
        st.divider()
        if st.button("💬 測試 Line API 通知"):
            status = send_line_push(target_stock, ["測試訊息：系統連線成功！"])
            if status == 200:
                st.toast("通知已成功送出")
            else:
                st.error(f"發送失敗，代碼：{status}")

    st.title(f"台股智慧戰情室：{target_stock}")

    # 數據與運算
    df = fetch_stock_data(target_stock, start_date)
    
    if not df.empty:
        # 顯示即時訊號
        detected_signals = scan_signals(df)
        if detected_signals:
            for s in detected_signals:
                st.info(s)
        
        # --- 繪製多指標圖表 ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("K線與籌碼分佈", "MACD 指標", "RSI 強弱")
        )

        # K線、MA與布林
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['20MA'], name="20MA", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['60MA'], name="60MA", line=dict(color='cyan', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)

        # 分價量表疊加
        df_recent = df.tail(60)
        price_bins = np.linspace(df_recent['Low'].min(), df_recent['High'].max(), 15)
        vol_counts, _ = np.histogram(df_recent['Close'], bins=price_bins, weights=df_recent['Volume'])
        for i in range(len(vol_counts)):
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0.9, x1=0.9 + (vol_counts[i]/vol_counts.max()*0.1),
                          y0=price_bins[i], y1=price_bins[i+1], fillcolor="rgba(200,200,200,0.15)", line_width=0, row=1, col=1)

        # MACD (row 2)
        fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD Hist", marker_color='red'), row=2, col=1)

        # RSI (row 3)
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI", line=dict(color='yellow')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=30, b=5))
        st.plotly_chart(fig, use_container_width=True)
        
        # 數據表
        st.divider()
        st.subheader("📊 盤後詳細數據 (近5日)")
        st.dataframe(df[['date', 'Close', 'Volume', 'RSI', '20MA']].tail(5).sort_values(by='date', ascending=False), use_container_width=True)

    else:
        st.error("無法取得數據，請檢查代碼或稍後再試。")

if __name__ == "__main__":
    main()
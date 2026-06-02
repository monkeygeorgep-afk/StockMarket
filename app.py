import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import io

# --- 1. 系統配置與常數 ---
# 提醒：請將 line_token 換成你申請的 Token
    # 修改 app.py 裡的設定讀取方式
ST_CONFIG = {
    "page_title": "台股 AI 智慧分析系統",
    "line_token": st.secrets["LINE_TOKEN"], # 改用 secrets 讀取
}

# --- 2. 原生技術指標計算 (不依賴 pandas-ta) ---
def calculate_indicators(df):
    # 移動平均線 (MA)
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道 (Bollinger Bands)
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['20MA'] + (std * 2)
    df['BBL'] = df['20MA'] - (std * 2)
    
    # RSI 計算
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
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return pd.DataFrame()
    
    # 格式化欄位名稱
    df = df.rename(columns={
        'max': 'High', 'min': 'Low', 'close': 'Close', 
        'open': 'Open', 'Trading_Volume': 'Volume'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = calculate_indicators(df)
    return df

# --- 4. 監控演算法模組 ---
def scan_signals(df):
    signals = []
    if len(df) < 2: return signals
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # A. 量能異常 (大於5日平均2.5倍)
    avg_vol = df['Volume'].iloc[-6:-1].mean()
    if curr['Volume'] > avg_vol * 2.5:
        signals.append("🚀 爆量噴發 (量能大增)")
    
    # B. 價格突破 (突破60日高點)
    recent_high = df['High'].iloc[-61:-1].max()
    if curr['Close'] > recent_high:
        signals.append("📈 突破 60 日新高")
        
    # C. 站回月線
    if prev['Close'] < prev['20MA'] and curr['Close'] > curr['20MA']:
        signals.append("📊 站上 20MA 月線")
        
    return signals

# --- 5. Line 通知模組 ---
def send_line_alert(stock_id, signals, fig=None):
    if not signals: return
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {ST_CONFIG['line_token']}"}
    msg = f"\n🎯 訊號觸發：{stock_id}\n" + "\n".join(signals)
    
    payload = {"message": msg}
    files = None
    if fig:
        # 將 Plotly 圖表轉為圖片 (需要 kaleido)
        img_bytes = fig.to_image(format="png", width=800, height=500)
        files = {"imageFile": io.BytesIO(img_bytes)}
        
    try:
        requests.post(url, headers=headers, data=payload, files=files)
    except Exception as e:
        st.error(f"Line 發送失敗: {e}")

# --- 6. Streamlit 介面渲染 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    
    # 側邊欄
    with st.sidebar:
        st.header("🔍 股票篩選")
        target_stock = st.text_input("輸入台股代碼", value="2330")
        lookback = st.slider("顯示天數", 60, 365, 120)
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback+60)).strftime('%Y-%m-%d')
        
        st.divider()
        if st.button("🚀 執行 Line 通知測試"):
            df_test = fetch_stock_data(target_stock, start_date)
            sigs = scan_signals(df_test)
            # 若無訊號則傳送測試訊息
            send_line_alert(target_stock, sigs if sigs else ["系統連線測試正常"], None)
            st.toast("已嘗試發送通知")

    st.title(f"📈 {target_stock} 技術籌碼戰情室")

    # 讀取資料
    df = fetch_stock_data(target_stock, start_date)
    if df.empty:
        st.warning("請輸入正確的台股代碼，或檢查網路連線。")
        return

    # 執行訊號掃描
    detected_signals = scan_signals(df)
    if detected_signals:
        for s in detected_signals:
            st.info(s)

    # --- 繪製專業圖表 ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("K線與分價量表", "MACD 指標", "RSI 強弱指標")
    )

    # 1. K線圖
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name="K線"
    ), row=1, col=1)

    # 2. 均線與布林通道
    fig.add_trace(go.Scatter(x=df['date'], y=df['20MA'], name="20MA (月線)", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['60MA'], name="60MA (季線)", line=dict(color='cyan', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='gray', width=1, dash='dash')), row=1, col=1)

    # 3. 核心：分價量表 (Volume Profile) 疊加
    # 取最近 60 天資料計算籌碼分佈
    df_recent = df.tail(60)
    price_bins = np.linspace(df_recent['Low'].min(), df_recent['High'].max(), 20)
    vol_counts, _ = np.histogram(df_recent['Close'], bins=price_bins, weights=df_recent['Volume'])
    
    max_vol = vol_counts.max()
    for i in range(len(vol_counts)):
        fig.add_shape(
            type="rect", xref="paper", yref="y",
            x0=0.85, x1=0.85 + (vol_counts[i] / max_vol * 0.15),
            y0=price_bins[i], y1=price_bins[i+1],
            fillcolor="rgba(150, 150, 150, 0.2)", line_width=0, row=1, col=1
        )

    # 4. MACD 柱狀圖 (row 2)
    colors = ['red' if val >= 0 else 'green' for val in df['Hist']]
    fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD 柱狀體", marker_color=colors), row=2, col=1)

    # 5. RSI 強弱指標 (row 3)
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI (14)", line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # 圖表樣式優化 (適配手機顯示)
    fig.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=5, r=5, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Microsoft JhengHei, Arial", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 盤後資料表格
    st.divider()
    st.subheader("📋 近 5 日交易數據")
    st.dataframe(df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(5), use_container_width=True)

if __name__ == "__main__":
    main()
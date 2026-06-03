import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai

# --- 1. 系統與 AI 配置 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 3.1"}
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：均線、布林通道與技術指標 ---
def calculate_indicators(df):
    # 四大均線
    df['5MA'] = df['Close'].rolling(window=5).mean()
    df['10MA'] = df['Close'].rolling(window=10).mean()
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道 (Bollinger Bands)
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['20MA'] + (std * 2)  # 上軌
    df['BBL'] = df['20MA'] - (std * 2)  # 下軌
    
    # KD (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        new_k = (k[-1] * 2/3) + (rsv.iloc[i] * 1/3) if not np.isnan(rsv.iloc[i]) else k[-1]
        new_d = (d[-1] * 2/3) + (new_k * 1/3)
        k.append(new_k); d.append(new_d)
    df['K'], df['D'] = k, d
    
    # MACD & RSI 略 (邏輯同前版本，依按鈕顯示)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 支撐壓力
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return df, support, resistance

# --- 3. 數據獲取 ---
@st.cache_data(ttl=3600)
def fetch_all_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None, None, None
    
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)
    
    info = dl.taiwan_stock_info()
    stock_name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if not info[info['stock_id'] == stock_id].empty else "未知"
    
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=10)).strftime('%Y-%m-%d'))
    if not chip.empty: chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000
    
    return df, chip, stock_name, sup, res

# --- 4. UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    with st.sidebar:
        st.title("🛡️ 戰情控制台 3.1")
        target_stock = st.text_input("輸入股票代號", value="2330")
        
        st.subheader("🛠️ 指標開關")
        show_ma = st.multiselect("顯示均線", ["5MA", "10MA", "20MA", "60MA"], default=["5MA", "20MA", "60MA"])
        show_bb = st.toggle("開啟布林通道", value=True)
        show_kd = st.checkbox("顯示 KD", value=True)
        show_macd = st.checkbox("顯示 MACD", value=False)

    df, chip, s_name, sup, res = fetch_all_data(target_stock)

    if df is not None:
        st.title(f"📈 {target_stock} {s_name}")
        
        # 繪圖邏輯
        rows = 1 + show_kd + show_macd
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2][:rows])

        # 主圖：K線
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)

        # 疊加均線
        ma_colors = {"5MA": "white", "10MA": "yellow", "20MA": "orange", "60MA": "cyan"}
        for ma in show_ma:
            fig.add_trace(go.Scatter(x=df['date'], y=df[ma], name=ma, line=dict(color=ma_colors[ma], width=1)), row=1, col=1)

        # 疊加布林通道
        if show_bb:
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)

        # KD / MACD 子圖
        curr_row = 2
        if show_kd:
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=curr_row, col=1)
            curr_row += 1
        if show_macd:
            colors = ['red' if v >= 0 else 'green' for v in df['Hist']]
            fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD", marker_color=colors), row=curr_row, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 法人看板
        st.subheader("🏦 三大法人當日動態 (張)")
        if not chip.empty:
            st.dataframe(chip.tail(3))

if __name__ == "__main__":
    main()
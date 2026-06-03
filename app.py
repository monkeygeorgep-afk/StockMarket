import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests

# --- 1. 系統配置 ---
ST_CONFIG = {"page_title": "台股 AI 全自動戰情室 2.1"}

# --- 2. 核心運算：技術指標 (含 KD) ---
def calculate_indicators(df):
    # MA
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # KD 指標 (標準 9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    # 初始化 K, D 均為 50
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        new_k = (k[-1] * 2/3) + (rsv.iloc[i] * 1/3)
        new_d = (d[-1] * 2/3) + (new_k * 1/3)
        k.append(new_k)
        d.append(new_d)
    df['K'], df['D'] = k, d
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    return df

# --- 3. 籌碼獲取與統計 ---
def fetch_chip_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    try:
        # 獲取三大法人買賣超
        chip_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
        if chip_df.empty: return pd.DataFrame()
        
        # 整理數據：將外資、投信、自營商合併
        chip_pivot = chip_df.pivot_table(index='date', columns='name', values='buy', aggfunc='sum')
        # 這裡簡化為買進金額或張數，具體視 API 回傳而定，通常 FinMind 回傳的是張數
        return chip_df
    except:
        return pd.DataFrame()

def get_chip_summary(chip_df):
    if chip_df.empty: return None
    # 取得最近 5 日
    latest_days = sorted(chip_df['date'].unique(), reverse=True)[:5]
    summary = []
    for day_count in [1, 3, 5]:
        target_days = latest_days[:day_count]
        temp = chip_df[chip_df['date'].isin(target_days)]
        stats = temp.groupby('name')['buy'].sum() - temp.groupby('name')['sell'].sum()
        summary.append({
            "期間": f"近 {day_count} 日",
            "外資": int(stats.get('Foreign_Investor', 0)),
            "投信": int(stats.get('Investment_Trust', 0)),
            "自營商": int(stats.get('Dealer', 0))
        })
    return pd.DataFrame(summary)

# --- 4. 數據下載 (主功能) ---
@st.cache_data(ttl=3600)
def fetch_all_info(stock_id):
    df = DataLoader().taiwan_stock_daily(stock_id=stock_id, start_date='2026-01-01')
    if df.empty: return None, None
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = calculate_indicators(df)
    chip = fetch_chip_data(stock_id)
    return df, chip

# --- 5. UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    
    with st.sidebar:
        st.title("🛡️ 智慧戰情室 2.1")
        target = st.text_input("輸入台股代碼", value="2330")
        st.divider()
        st.info("💡 KD < 20 為超賣區，> 80 為超買區。法人連續買超通常代表趨勢成形。")

    df, chip = fetch_all_info(target)

    if df is not None:
        st.title(f"📈 {target} 技術與籌碼分析")
        
        # --- A. 籌碼看板 ---
        st.subheader("🏦 三大法人買賣超統計 (張)")
        chip_sum = get_chip_summary(chip)
        if chip_sum is not None:
            cols = st.columns(3)
            for i, row in chip_sum.iterrows():
                with cols[i]:
                    st.metric(row['期間'], f"外資 {row['外資']}", f"投信 {row['投信']}")
            st.table(chip_sum.set_index('期間'))

        # --- B. 圖表展示 (含 KD) ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("K線與分價量表", "KD 指標", "成交量")
        )

        # K線
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        
        # KD (Row 2)
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K值", line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D值", line=dict(color='cyan')), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)

        # Volume (Row 3)
        fig.add_trace(go.Bar(x=df['date'], y=df['Volume'], name="成交量", marker_color='gray'), row=3, col=1)

        fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
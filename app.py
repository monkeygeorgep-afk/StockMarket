import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai

# --- 1. 系統與 AI 配置 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 3.2"}

# 設定 Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：技術指標與支撐壓力 ---
def calculate_indicators(df):
    # 四大均線
    df['5MA'] = df['Close'].rolling(window=5).mean()
    df['10MA'] = df['Close'].rolling(window=10).mean()
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['20MA'] + (std * 2)
    df['BBL'] = df['20MA'] - (std * 2)
    
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
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 支撐與壓力 (取 60 日高低點)
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return df, support, resistance

# --- 3. 數據獲取 ---
@st.cache_data(ttl=3600)
def fetch_comprehensive_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 股價資料
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None, None, None, None, None, None
    
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)
    
    # 股票名稱
    info = dl.taiwan_stock_info()
    s_name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if not info[info['stock_id'] == stock_id].empty else "未知"
    
    # 法人籌碼 (校正為張數)
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if not chip.empty:
        chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000

    # 新聞與財報
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d'))
    financial = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    
    return df, chip, news, financial, s_name, sup, res

# --- 4. AI 語義分析 ---
def run_ai_analysis(news_df):
    if news_df.empty or "GEMINI_API_KEY" not in st.secrets:
        return "無法生成 AI 報告：缺新聞數據或 API 金鑰。"
    
    titles = "\n".join(news_df['title'].tail(8).tolist())
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"請擔任台股分析師，針對以下新聞標題總結出該股的『看多理由』、『看空理由』及『整體評估』：\n{titles}"
    try:
        return model.generate_content(prompt).text
    except:
        return "AI 繁忙中，請稍後再試。"

# --- 5. UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    # 自選股清單初始化
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ 終極戰情室 3.2")
        target_id = st.text_input("輸入股票代號", value="2330")
        
        st.divider()
        st.subheader("🛠️ 技術圖表設定")
        sel_ma = st.multiselect("顯示均線", ["5MA", "10MA", "20MA", "60MA"], default=["5MA", "20MA", "60MA"])
        use_bb = st.toggle("顯示布林通道", value=True)
        use_kd = st.checkbox("顯示 KD", value=True)
        use_macd = st.checkbox("顯示 MACD", value=False)
        use_rsi = st.checkbox("顯示 RSI", value=False)

        st.divider()
        st.subheader("📋 自選股管理")
        add_s = st.text_input("新增代碼到清單")
        if st.button("新增"):
            if add_s and add_s not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_s)
                st.rerun()
        st.write("目前追蹤:", ", ".join(st.session_state.watchlist))
        if st.button("清空追蹤"):
            st.session_state.watchlist = []; st.rerun()

    # 獲取資料
    df, chip, news, fin, s_name, sup, res = fetch_comprehensive_data(target_id)

    if df is not None:
        st.title(f"📈 {target_id} {s_name}")
        
        # 支撐壓力看板
        m1, m2, m3 = st.columns(3)
        m1.metric("當前股價", f"{df['Close'].iloc[-1]}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.1f}")
        m2.error(f"🚩 壓力位：{res:.2f}")
        m3.success(f"⚓ 支撐位：{sup:.2f}")

        # AI 報告區
        with st.expander("🤖 AI 多空語義分析 (Gemini)", expanded=True):
            if st.button("🔍 開始 AI 分析"):
                with st.spinner("AI 正在閱讀新聞..."):
                    st.write(run_ai_analysis(news))
            else:
                st.write("點擊按鈕分析近期新聞動態。")

        # 法人看板 (張數)
        st.subheader("🏦 三大法人近期買賣超 (張)")
        if not chip.empty:
            latest_date = chip['date'].max()
            today_chip = chip[chip['date'] == latest_date]
            # 合併統計今日張數
            fi = today_chip[today_chip['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            it = today_chip[today_chip['name']=='Investment_Trust'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            dl = today_chip[today_chip['name']=='Dealer'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("外資今日", f"{int(fi)} 張")
            c2.metric("投信今日", f"{int(it)} 張")
            c3.metric("自營商今日", f"{int(dl)} 張")

        # --- 繪圖區 ---
        rows = 1 + use_kd + use_macd + use_rsi
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5] + [0.15]*(rows-1))

        # 1. K線 + 均線 + 布林
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        # 均線顏色
        ma_colors = {"5MA":"#FFFFFF", "10MA":"#FFFF00", "20MA":"#FF8C00", "60MA":"#00FFFF"}
        for ma in sel_ma:
            fig.add_trace(go.Scatter(x=df['date'], y=df[ma], name=ma, line=dict(color=ma_colors[ma], width=1.5)), row=1, col=1)
        # 布林通道
        if use_bb:
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
        # 支撐壓力線
        fig.add_hline(y=res, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="green", row=1, col=1)

        # 2. 技術指標子圖
        curr_row = 2
        if use_kd:
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=curr_row, col=1)
            curr_row += 1
        if use_macd:
            colors = ['red' if v >= 0 else 'green' for v in df['Hist']]
            fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD"), row=curr_row, col=1)
            curr_row += 1
        if use_rsi:
            fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=curr_row, col=1)

        fig.update_layout(height=400 + 150*rows, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 基本面 & 新聞表格
        t1, t2 = st.tabs(["📊 財報數據", "📰 最新新聞"])
        with t1: st.dataframe(fin.tail(10))
        with t2: st.dataframe(news[['date', 'title', 'link']].tail(15))

    else:
        st.error("代碼錯誤或資料庫連線中斷。")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai

# --- 1. 系統與 AI 配置 ---
ST_CONFIG = {
    "page_title": "台股 AI 終極戰情室 3.0",
}

# 若有 Gemini API Key，請在 Secrets 設定 GEMINI_API_KEY
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：技術分析與支撐壓力 ---
def calculate_indicators(df):
    # MA & Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
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
    
    # 支撐與壓力演算法 (取近期 60 日)
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return df, support, resistance

# --- 3. 數據獲取 (股價、法人、新聞、財報) ---
@st.cache_data(ttl=3600)
def fetch_all_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 1. 股價與名稱
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    info = dl.taiwan_stock_info()
    stock_name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if not info[info['stock_id'] == stock_id].empty else "未知"
    
    if df.empty: return None, None, None, None, None
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)

    # 2. 法人 (換算為張數)
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=10)).strftime('%Y-%m-%d'))
    if not chip.empty:
        chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000 # 股轉張

    # 3. 新聞
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=3)).strftime('%Y-%m-%d'))
    
    # 4. 基本面 (損益表)
    financial = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    
    return df, chip, news, financial, stock_name, sup, res

# --- 4. AI 語義分析 ---
def get_ai_insight(news_df):
    if news_df.empty or "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ 未偵測到新聞或 AI API Key 未設定。"
    
    titles = "\n".join(news_df['title'].tail(5).tolist())
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"你是專業台股分析師，請根據以下新聞摘要該個股的『多空觀點』與『潛在風險』，用簡短清單呈現：\n{titles}"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI 分析暫時無法使用。"

# --- 5. UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    # 初始化自選股
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['2330', '2317', '2454']

    # --- 側邊欄 ---
    with st.sidebar:
        st.title("🛡️ 戰情控制台")
        target_stock = st.text_input("輸入股票代號", value="2330")
        
        st.subheader("🛠️ 技術指標顯示")
        show_kd = st.checkbox("顯示 KD", value=True)
        show_macd = st.checkbox("顯示 MACD", value=False)
        show_rsi = st.checkbox("顯示 RSI", value=False)

        st.divider()
        st.subheader("📋 自選股管理")
        new_s = st.text_input("新增代碼")
        if st.button("加入"):
            st.session_state.watchlist.append(new_s)
            st.rerun()
        st.write("目前追蹤:", st.session_state.watchlist)

    # --- 主畫面數據讀取 ---
    df, chip, news, financial, s_name, sup, res = fetch_all_data(target_stock)

    if df is not None:
        st.title(f"📈 {target_stock} {s_name}")
        
        # 支撐壓力呈現
        c1, c2, c3 = st.columns(3)
        c1.metric("當前股價", f"{df['Close'].iloc[-1]}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.1f}")
        c2.info(f"🧱 壓力價位：{res:.2f}")
        c3.success(f"⚓ 支撐價位：{sup:.2f}")

        # --- A. AI 新聞輿情 ---
        with st.expander("🤖 AI 語義分析報告 (Gemini)", expanded=True):
            if st.button("生成 AI 多空分析"):
                st.write(get_ai_insight(news))
            else:
                st.write("點擊按鈕分析近期 3 日新聞...")

        # --- B. 法人動態 (張數) ---
        st.subheader("🏦 三大法人買賣超統計 (張)")
        if not chip.empty:
            latest_date = chip['date'].max()
            today_chip = chip[chip['date'] == latest_date]
            stats = today_chip.groupby('name').apply(lambda x: x['buy'].sum() - x['sell'].sum())
            cols = st.columns(3)
            cols[0].metric("外資", f"{int(stats.get('Foreign_Investor', 0))}")
            cols[1].metric("投信", f"{int(stats.get('Investment_Trust', 0))}")
            cols[2].metric("自營商", f"{int(stats.get('Dealer', 0))}")

        # --- C. 綜合圖表 ---
        rows = 1 + show_kd + show_macd + show_rsi
        row_heights = [0.5] + [0.15] * (rows - 1)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

        # 主圖 (K線 + 支撐壓力線)
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_hline(y=res, line_dash="dash", line_color="red", annotation_text="壓力", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="green", annotation_text="支撐", row=1, col=1)

        curr_row = 2
        if show_kd:
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=curr_row, col=1)
            curr_row += 1
        if show_macd:
            fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD"), row=curr_row, col=1)
            curr_row += 1
        if show_rsi:
            fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI"), row=curr_row, col=1)

        fig.update_layout(height=400 + 150*rows, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- D. 基本面 ---
        with st.expander("📊 基本面財報數據"):
            st.dataframe(financial.tail(10))

    else:
        st.error("查無資料，請確認代號是否正確。")

if __name__ == "__main__":
    main()
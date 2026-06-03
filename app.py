import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai
import io

# --- 1. 系統與 AI 初始化配置 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 3.5"}

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心技術指標運算 (包含均線、布林、KD、RSI、MACD、支撐壓力) ---
def calculate_all_indicators(df):
    # 均線系統
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
        val = rsv.iloc[i] if not np.isnan(rsv.iloc[i]) else 50.0
        new_k = (k[-1] * 2/3) + (val * 1/3)
        new_d = (d[-1] * 2/3) + (new_k * 1/3)
        k.append(new_k); d.append(new_d)
    df['K'], df['D'] = k, d
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 支撐壓力 (近 60 日)
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return df, support, resistance

# --- 3. 數據獲取與法人張數校正 ---
@st.cache_data(ttl=3600)
def fetch_stock_master(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 基礎價量
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None, None, None, None, None, None
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_all_indicators(df)
    
    # 股票名稱
    info = dl.taiwan_stock_info()
    name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if not info[info['stock_id'] == stock_id].empty else "未知"
    
    # 法人籌碼 (股轉張)
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if not chip.empty: chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000

    # 新聞與財報
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d'))
    fin = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    
    return df, chip, news, fin, name, sup, res

# --- 4. 一鍵巡邏掃描演算法 (Line 通知) ---
def run_full_scan(watchlist):
    reports = []
    for sid in watchlist:
        df, chip, _, _, _, _, _ = fetch_stock_master(sid)
        if df is None: continue
        curr, prev = df.iloc[-1], df.iloc[-2]
        sigs = []
        # 技術面訊號
        if prev['K'] < prev['D'] and curr['K'] > curr['D'] and curr['K'] < 30: sigs.append("⚡KD低檔金叉")
        if curr['Close'] > curr['BBU'] and prev['Close'] < prev['BBU']: sigs.append("💥突破布林上軌")
        # 籌碼面訊號
        if not chip.empty:
            today = chip[chip['date'] == chip['date'].max()]
            fi_net = today[today['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            if fi_net > 1000: sigs.append("🏦外資大買千張")
        
        if sigs: reports.append(f"【{sid}】\n" + "\n".join(sigs))
    
    if reports:
        token = st.secrets["LINE_CHANNEL_ACCESS_TOKEN"]
        uid = st.secrets["LINE_USER_ID"]
        msg = f"🛡️ 全自動巡邏報告\n\n" + "\n\n".join(reports)
        requests.post("https://api.line.me/v2/bot/message/push", 
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                      json={"to": uid, "messages": [{"type": "text", "text": msg}]})
        return True
    return False

# --- 5. 主程式 UI ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ AI 戰情室 3.5")
        target_id = st.text_input("即時分析代碼", value="2330")
        
        st.divider()
        st.subheader("🛠️ 技術圖表設定")
        sel_ma = st.multiselect("均線選擇", ["5MA", "10MA", "20MA", "60MA"], default=["5MA", "20MA", "60MA"])
        use_bb = st.toggle("顯示布林通道", value=True)
        use_kd = st.checkbox("KD 指標", value=True)
        use_macd = st.checkbox("MACD 指標", value=False)
        use_rsi = st.checkbox("RSI 指標", value=False)

        st.divider()
        st.subheader("📡 全自動巡邏")
        if st.button("🔴 啟動 Full Scan (發送 Line)"):
            if run_full_scan(st.session_state.watchlist): st.success("訊號已送達 Line")
            else: st.info("目前無顯著訊號")

        st.divider()
        st.subheader("📋 自選清單")
        new_id = st.text_input("新增代碼")
        if st.button("加入"):
            st.session_state.watchlist.append(new_id); st.rerun()
        st.write("目前監控:", ", ".join(st.session_state.watchlist))

    # --- 主畫面資料載入 ---
    df, chip, news, fin, s_name, sup, res = fetch_stock_master(target_id)

    if df is not None:
        st.title(f"📈 {target_id} {s_name}")
        
        # 1. 核心看板
        c1, c2, c3 = st.columns(3)
        c1.metric("今日收盤", f"{df['Close'].iloc[-1]}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.1f}")
        c2.error(f"🚩 壓力位：{res:.2f}")
        c3.success(f"⚓ 支撐位：{sup:.2f}")

        # 2. AI 語義分析
        with st.expander("🤖 AI 多空觀點摘要 (Gemini)", expanded=True):
            if st.button("生成 AI 分析報告"):
                if "GEMINI_API_KEY" in st.secrets:
                    model = genai.GenerativeModel('gemini-pro')
                    titles = "\n".join(news['title'].tail(5).tolist())
                    response = model.generate_content(f"請分析以下新聞對{s_name}的多空影響：\n{titles}")
                    st.write(response.text)
                else: st.warning("請先設定 Gemini API Key")

        # 3. 三大法人看板 (張)
        st.subheader("🏦 三大法人今日買賣 (張)")
        if not chip.empty:
            last_chip = chip[chip['date'] == chip['date'].max()]
            f_col, i_col, d_col = st.columns(3)
            f_net = last_chip[last_chip['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            i_net = last_chip[last_chip['name']=='Investment_Trust'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            d_net = last_chip[last_chip['name']=='Dealer'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            f_col.metric("外資", f"{int(f_net)}")
            i_col.metric("投信", f"{int(i_net)}")
            d_col.metric("自營商", f"{int(d_net)}")

        # 4. 技術圖表
        rows = 1 + use_kd + use_macd + use_rsi
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6]+[0.15]*(rows-1))
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        
        # 均線顏色對應
        ma_map = {"5MA":"white", "10MA":"yellow", "20MA":"orange", "60MA":"cyan"}
        for ma in sel_ma:
            fig.add_trace(go.Scatter(x=df['date'], y=df[ma], name=ma, line=dict(color=ma_map[ma], width=1.5)), row=1, col=1)
        
        if use_bb:
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="上軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="下軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot'), fill='tonexty'), row=1, col=1)
        
        fig.add_hline(y=res, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="green", row=1, col=1)

        curr_r = 2
        if use_kd:
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=curr_r, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=curr_r, col=1)
            curr_r += 1
        if use_macd:
            fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD"), row=curr_r, col=1)
            curr_r += 1
        if use_rsi:
            fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=curr_r, col=1)

        fig.update_layout(height=400+150*rows, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 5. 基本面與新聞
        t1, t2 = st.tabs(["📊 財報概況", "📰 近期新聞原文"])
        t1.dataframe(fin.tail(10))
        t2.dataframe(news[['date', 'title', 'link']].tail(10))

if __name__ == "__main__":
    main()
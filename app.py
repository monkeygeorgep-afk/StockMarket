import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai

# --- 1. 系統配置 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 3.6"}
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：指標與支撐壓力 ---
def calculate_indicators(df):
    # 均線
    for m in [5, 10, 20, 60]:
        df[f'{m}MA'] = df['Close'].rolling(window=m).mean()
    # 布林通道
    std = df['Close'].rolling(window=20).std()
    df['BBU'], df['BBL'] = df['20MA'] + (std * 2), df['20MA'] - (std * 2)
    # KD (9, 3, 3)
    l9, h9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - l9) / (h9 - l9)
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        v = rsv.iloc[i] if not np.isnan(rsv.iloc[i]) else 50.0
        nk = (k[-1] * 2/3) + (v * 1/3)
        nd = (d[-1] * 2/3) + (nk * 1/3)
        k.append(nk); d.append(nd)
    df['K'], df['D'] = k, d
    # 支撐壓力 (60日)
    sup, res = df['Low'].tail(60).min(), df['High'].tail(60).max()
    return df, sup, res

# --- 3. 數據獲取 ---
@st.cache_data(ttl=3600)
def fetch_master_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None, None, None, None, None, None
    
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)
    
    name = dl.taiwan_stock_info()
    name = name[name['stock_id']==stock_id]['stock_name'].values[0] if stock_id in name['stock_id'].values else "未知"
    
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now()-pd.Timedelta(days=10)).strftime('%Y-%m-%d'))
    if not chip.empty: chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000
    
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now()-pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
    fin = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now()-pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    return df, chip, news, fin, name, sup, res

# --- 4. 優化後的 AI 語義分析 (分析師口吻) ---
def get_ai_insight(s_name, sid, news_df):
    if news_df.empty or len(news_df) < 2:
        return "📉 **分析簡報：** 近期該股新聞量較少，AI 建議優先參考技術面支撐位與法人籌碼動態。"
    
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        titles = "\n".join(news_df['title'].tail(8).tolist())
        prompt = f"""
        你是一位擁有 20 年經驗的台股首席策略分析師。請針對 {s_name}({sid}) 的近期新聞進行深度評估：
        1. 【核心多空因素】：提取 3 個關鍵點。
        2. 【影響力評級】：針對整體新聞給予股價影響力評分（0-100）。
        3. 【專業投資摘要】：給予投資者的一句話核心建議。
        
        新聞標題如下：
        {titles}
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI 分析暫時無法連線: {e}"

# --- 5. Line 通知與自動巡邏 ---
def send_line_push(msg):
    try:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {st.secrets['LINE_CHANNEL_ACCESS_TOKEN']}", "Content-Type": "application/json"}
        payload = {"to": st.secrets['LINE_USER_ID'], "messages": [{"type": "text", "text": msg}]}
        requests.post(url, headers=headers, json=payload)
    except: pass

# --- 6. UI 渲染 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    if 'watchlist' not in st.session_state: st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ AI 終極戰情室 3.6")
        target_id = st.text_input("輸入股票代號", value="2330")
        
        st.divider()
        st.subheader("📡 全自動巡邏 (Line)")
        if st.button("🔴 啟動 Full Scan"):
            reports = []
            for sid in st.session_state.watchlist:
                d, c, _, _, _, s, r = fetch_master_data(sid)
                if d is None: continue
                price = d['Close'].iloc[-1]
                # 支撐壓力警示邏輯
                if price <= s * 1.02: reports.append(f"【{sid}】⚓ 接近支撐位！")
                if price >= r * 0.98: reports.append(f"【{sid}】🚩 接近壓力位！")
                if d['K'].iloc[-1] > d['D'].iloc[-1] and d['K'].iloc[-2] < d['D'].iloc[-2]: reports.append(f"【{sid}】⚡ KD 金叉")
            
            if reports:
                send_line_push("🔔 盤後巡邏警報：\n\n" + "\n".join(reports))
                st.success("已發送巡邏報告")
            else: st.info("目前無觸發警訊")

    df, chip, news, fin, name, sup, res = fetch_master_data(target_id)

    if df is not None:
        st.title(f"📈 {target_id} {name}")
        
        # 支撐壓力與股價看板
        curr_p = df['Close'].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("今日收盤", f"{curr_p}", f"{curr_p - df['Close'].iloc[-2]:.1f}")
        c2.error(f"🚩 壓力價位：{res:.2f} (距離 {(res/curr_p-1)*100:.1f}%)")
        c3.success(f"⚓ 支撐價位：{sup:.2f} (距離 {(curr_p/sup-1)*100:.1f}%)")

        # AI 語義分析
        with st.expander("🤖 AI 首席分析師簡報", expanded=True):
            if st.button("🔍 生成深度多空分析"):
                with st.spinner("AI 分析中..."):
                    st.write(get_ai_insight(name, target_id, news))

        # 技術圖表 (整合均線與布林)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        
        # 繪製支撐壓力線
        fig.add_hline(y=res, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="green", row=1, col=1)

        # 均線
        for m, clr in zip(['5MA', '20MA', '60MA'], ['white', 'orange', 'cyan']):
            fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=ma, line=dict(color=clr, width=1)), row=1, col=1)
        
        # KD
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 籌碼與數據
        t1, t2 = st.tabs(["🏦 法人籌碼 (張)", "📊 財報與新聞"])
        with t1: st.dataframe(chip.tail(10))
        with t2: st.dataframe(news[['date','title']].tail(10))

if __name__ == "__main__": main()
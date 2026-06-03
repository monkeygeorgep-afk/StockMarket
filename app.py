import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai

# --- 1. 系統與 AI 配置 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 3.7"}

# 設定 Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：技術指標、布林與支撐壓力 ---
def calculate_indicators(df):
    # 四大均線
    for m in [5, 10, 20, 60]:
        df[f'{m}MA'] = df['Close'].rolling(window=m).mean()
    
    # 布林通道 (20MA, 2倍標準差)
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['20MA'] + (std * 2)
    df['BBL'] = df['20MA'] - (std * 2)
    
    # KD (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        v = rsv.iloc[i] if not np.isnan(rsv.iloc[i]) else 50.0
        nk = (k[-1] * 2/3) + (v * 1/3)
        nd = (d[-1] * 2/3) + (nk * 1/3)
        k.append(nk); d.append(nd)
    df['K'], df['D'] = k, d
    
    # 支撐與壓力 (取近 60 日高低點)
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return df, support, resistance

# --- 3. 數據獲取模組 ---
@st.cache_data(ttl=3600)
def fetch_master_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 股價與技術指標
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None, None, None, None, None, None
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)
    
    # 股票名稱
    info = dl.taiwan_stock_info()
    name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if not info[info['stock_id'] == stock_id].empty else "未知"
    
    # 法人籌碼 (股轉張)
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if not chip.empty:
        chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000

    # 新聞與財報
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
    fin = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    
    return df, chip, news, fin, name, sup, res

# --- 4. AI 首席分析師簡報模組 ---
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
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI 服務暫時無法連線: {e}"

# --- 5. Line 通知與自動巡邏 ---
def send_line_push(msg):
    try:
        url = "https://api.line.me/v2/bot/message/push"
        token = st.secrets["LINE_CHANNEL_ACCESS_TOKEN"]
        uid = st.secrets["LINE_USER_ID"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"to": uid, "messages": [{"type": "text", "text": msg}]}
        requests.post(url, headers=headers, json=payload)
    except:
        pass

# --- 6. UI 主程式 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    # 初始化自選股
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ AI 終極戰情室 3.7")
        target_id = st.text_input("輸入股票代號", value="2330")
        
        st.divider()
        st.subheader("📡 全自動巡邏系統")
        if st.button("🔴 啟動 Full Scan (Line 通知)"):
            with st.spinner("掃描中..."):
                reports = []
                for sid in st.session_state.watchlist:
                    d, c, _, _, _, s, r = fetch_master_data(sid)
                    if d is None: continue
                    price = d['Close'].iloc[-1]
                    # 警示邏輯：接近支撐壓力或 KD 金叉
                    if price <= s * 1.02: reports.append(f"【{sid}】⚓ 接近支撐位！")
                    if price >= r * 0.98: reports.append(f"【{sid}】🚩 接近壓力位！")
                    if d['K'].iloc[-1] > d['D'].iloc[-1] and d['K'].iloc[-2] < d['D'].iloc[-2]: reports.append(f"【{sid}】⚡ KD 金叉")
                
                if reports:
                    send_line_push("🔔 盤後巡邏警報：\n\n" + "\n".join(reports))
                    st.success("巡邏完成，已發送至 Line。")
                else:
                    st.info("目前無觸發警訊。")

        st.divider()
        st.subheader("📋 自選清單管理")
        add_sid = st.text_input("新增代碼")
        if st.button("加入清單"):
            if add_sid and add_sid not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_sid)
                st.rerun()
        st.write("目前追蹤:", ", ".join(st.session_state.watchlist))
        if st.button("清空重置"):
            st.session_state.watchlist = ['2330']; st.rerun()

    # --- 數據展示區 ---
    df, chip, news, fin, name, sup, res = fetch_master_data(target_id)

    if df is not None:
        st.title(f"📈 {target_id} {name}")
        
        # 1. 支撐壓力與價格看板
        curr_p = df['Close'].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("今日收盤", f"{curr_p}", f"{curr_p - df['Close'].iloc[-2]:.1f}")
        c2.error(f"🚩 壓力位：{res:.2f} ({(res/curr_p-1)*100:.1f}%)")
        c3.success(f"⚓ 支撐位：{sup:.2f} ({(curr_p/sup-1)*100:.1f}%)")

        # 2. AI 語義分析
        with st.expander("🤖 AI 首席分析師簡報", expanded=True):
            if st.button("🔍 生成深度分析"):
                if "GEMINI_API_KEY" in st.secrets:
                    with st.spinner("AI 分析中..."):
                        st.write(get_ai_insight(name, target_id, news))
                else:
                    st.warning("請先設定 GEMINI_API_KEY")

        # 3. 三大法人看板 (張)
        st.subheader("🏦 三大法人今日買賣 (張)")
        if not chip.empty:
            last_date = chip['date'].max()
            t_chip = chip[chip['date'] == last_date]
            f_net = t_chip[t_chip['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            i_net = t_chip[t_chip['name']=='Investment_Trust'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            d_net = t_chip[t_chip['name']=='Dealer'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
            
            f_col, i_col, d_col = st.columns(3)
            f_col.metric("外資", f"{int(f_net)}")
            i_col.metric("投信", f"{int(i_net)}")
            d_col.metric("自營商", f"{int(d_net)}")

        # 4. 技術分析圖表
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # 主圖: K線 + 均線 + 布林
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        
        # 修正後的均線繪製 (使用 m 變數)
        ma_colors = {"5MA": "white", "20MA": "orange", "60MA": "cyan"}
        for m, clr in zip(['5MA', '20MA', '60MA'], ['white', 'orange', 'cyan']):
            fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(color=clr, width=1)), row=1, col=1)
        
        # 布林通道與支撐壓力線
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='rgba(255,255,255,0.2)', dash='dot'), fill='tonexty'), row=1, col=1)
        fig.add_hline(y=res, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="green", row=1, col=1)

        # KD 指標 (子圖)
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 5. 基本面與新聞表格
        t1, t2 = st.tabs(["📊 財報概況", "📰 近期新聞"])
        with t1: st.dataframe(fin.tail(10))
        with t2: st.dataframe(news[['date', 'title', 'link']].tail(10))

    else:
        st.error("查無資料，請確認股票代號。")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import google.generativeai as genai
from datetime import datetime

# --- 1. 系統配置與 AI 初始化 ---
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 6.6"}
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算模組 ---
def calculate_indicators(df):
    for m in [5, 10, 20, 60]:
        df[f'{m}MA'] = df['Close'].rolling(window=m).mean()
    std = df['Close'].rolling(window=20).std()
    df['BBU'], df['BBL'] = df['20MA'] + (std * 2), df['20MA'] - (std * 2)
    l9, h9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - l9) / (h9 - l9)
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        v = rsv.iloc[i] if not np.isnan(rsv.iloc[i]) else 50.0
        nk = (k[-1] * 2/3) + (v * 1/3)
        nd = (d[-1] * 2/3) + (nk * 1/3)
        k.append(nk); d.append(nd)
    df['K'], df['D'] = k, d
    sup, res = df['Low'].tail(60).min(), df['High'].tail(60).max()
    return df, sup, res

def calculate_ai_score(df, chip):
    if df.empty: return 50, []
    curr = df.iloc[-1]
    score = 50 
    details = []
    if curr['Close'] > curr['20MA']: 
        score += 10; details.append("股價站上月線 (多)")
    if curr['K'] > curr['D']: 
        score += 10; details.append("KD金叉 (多)")
    if curr['Close'] > curr['BBU']: 
        score += 10; details.append("布林強勢突破 (多)")
    if not chip.empty:
        last_chip = chip[chip['date'] == chip['date'].max()]
        fi = last_chip[last_chip['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
        it = last_chip[last_chip['name']=='Investment_Trust'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
        if fi > 0 and it > 0: 
            score += 20; details.append("外資投信同步買超 (強多)")
        elif it > 0: 
            score += 10; details.append("投信單獨鎖碼 (多)")
    return min(score, 100), details

def estimate_volume(current_volume):
    now = datetime.now()
    start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
    if now < start_time: return current_volume, 0
    if now > end_time: return current_volume, 100
    elapsed_minutes = (now - start_time).seconds / 60
    total_minutes = 270
    ratio = max(elapsed_minutes / total_minutes, 0.01)
    if elapsed_minutes <= 60: ratio *= 1.5 
    estimated_vol = current_volume / min(ratio, 0.99)
    progress = min((elapsed_minutes / total_minutes) * 100, 100)
    return int(estimated_vol), int(progress)

# --- 3. 數據綜合獲取 (修正修正 API 名稱與合併邏輯) ---
@st.cache_data(ttl=3600)
def fetch_master_data(stock_id):
    # 修正 DataLoader 初始化方式
    token = st.secrets.get("FINMIND_TOKEN", None)
    dl = DataLoader(token=token) if token else DataLoader()
        
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return [None]*8
    
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df, sup, res = calculate_indicators(df)
    
    info = dl.taiwan_stock_info()
    name = info[info['stock_id'] == stock_id]['stock_name'].values[0] if stock_id in info['stock_id'].values else "未知"
    
    chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if not chip.empty: chip[['buy', 'sell']] = chip[['buy', 'sell']] / 1000

    fin = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d'))
    news = dl.taiwan_stock_news(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
    
    # 修正持股 API 名稱與異常處理
    share_hold = pd.DataFrame()
    try:
        share_hold = dl.taiwan_stock_holding_shares_per(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d'))
    except: pass
    
    # 修正本益比河流圖合併邏輯
    df['ttm_eps'] = np.nan
    eps_df = fin[fin['type'] == 'EPS'].copy()
    if not eps_df.empty:
        try:
            eps_df['date'] = pd.to_datetime(eps_df['date'])
            eps_df = eps_df.sort_values('date')
            eps_df['ttm_eps'] = eps_df['value'].rolling(window=4).sum()
            df = df.sort_values('date')
            df = pd.merge_asof(df, eps_df[['date', 'ttm_eps']], on='date', direction='backward')
        except: pass
    
    return df, chip, news, fin, name, sup, res, share_hold

# --- 4. Line Flex Card 發送 ---
def send_line_flex_card(sid, name, price, change, ai_score, signals):
    try:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {st.secrets['LINE_CHANNEL_ACCESS_TOKEN']}", "Content-Type": "application/json"}
        color_trend = "#FF4D4D" if change >= 0 else "#2ECC71"
        arrow = "▲" if change >= 0 else "▼"
        ai_color = "#FF4D4D" if ai_score >= 70 else "#2ECC71" if ai_score <= 45 else "#7F8C8D"
        
        flex_contents = {
          "type": "bubble",
          "header": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": "🛡️ AI 戰情室巡邏", "color": "#FFFFFF", "weight": "bold"}]},
          "styles": {"header": {"backgroundColor": "#1F2421"}},
          "body": {"type": "box", "layout": "vertical", "contents": [
              {"type": "text", "text": f"{name} {sid}", "weight": "bold", "size": "xl"},
              {"type": "separator", "margin": "md"},
              {"type": "box", "layout": "horizontal", "margin": "md", "contents": [
                  {"type": "text", "text": f"股價: {price}", "weight": "bold", "size": "md"},
                  {"type": "text", "text": f"{arrow}{abs(change):.1f}", "color": color_trend, "align": "end", "weight": "bold"}
              ]},
              {"type": "text", "text": " / ".join(signals), "size": "sm", "color": "#555555", "margin": "md", "wrap": True}
          ]},
          "footer": {"type": "box", "layout": "vertical", "contents": [{"type": "button", "style": "primary", "color": ai_color, "action": {"type": "uri", "label": f"AI 評分: {ai_score}", "uri": "https://line.me"}}]}
        }
        payload = {"to": st.secrets["LINE_USER_ID"], "messages": [{"type": "flex", "altText": f"戰報: {name}", "contents": flex_contents}]}
        requests.post(url, headers=headers, json=payload)
    except: pass

# --- 5. UI 主程式 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    if 'watchlist' not in st.session_state: st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ AI 終極戰情室 6.6")
        target_id = st.text_input("輸入股票代號", value="2330")
        st.divider()
        mode = st.radio("主分析視角", ["1. 技術與 AI 報告", "2. 本益比河流圖", "3. 大戶籌碼趨勢"])
        if st.button("🔴 啟動 Full Scan (手機接收)"):
            with st.spinner("巡邏中..."):
                for sid in st.session_state.watchlist:
                    d, c, _, _, n, s, r, _ = fetch_master_data(sid)
                    if d is None: continue
                    p, score = d['Close'].iloc[-1], calculate_ai_score(d, c)[0]
                    sigs = []
                    if p <= s * 1.02: sigs.append("⚓支撐")
                    if p >= r * 0.98: sigs.append("🚩壓力")
                    if d['K'].iloc[-1] > d['D'].iloc[-1] and d['K'].iloc[-2] < d['D'].iloc[-2]: sigs.append("⚡KD金叉")
                    if sigs: send_line_flex_card(sid, n, p, p-d['Close'].iloc[-2], score, sigs)
                st.success("已發送至 Line")

    df, chip, news, fin, name, sup, res, share_hold = fetch_master_data(target_id)
    if df is not None:
        ai_score, ai_details = calculate_ai_score(df, chip)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("今日收盤", f"{df['Close'].iloc[-1]}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.1f}")
        c2.metric("AI 綜合評分", f"{ai_score} 分")
        c3.error(f"🚩 壓力位：{res:.2f}")
        c4.success(f"⚓ 支撐位：{sup:.2f}")

        if "1." in mode:
            with st.expander("🤖 AI 策略報告", expanded=True):
                if st.button("🔍 執行 AI 分析"):
                    model = genai.GenerativeModel('models/gemini-2.5-flash')
                    prompt = f"{name}({target_id}) AI評分{ai_score}分，指標：{ai_details}。請給予短中線建議。"
                    st.write(model.generate_content(prompt).text)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K"), row=1, col=1)
            for m, clr in zip(['5MA', '20MA', '60MA'], ['white', 'orange', 'cyan']):
                fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(color=clr, width=1)), row=1, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        elif "2." in mode:
            if 'ttm_eps' in df.columns and not df['ttm_eps'].isnull().all():
                fig_river = go.Figure()
                for m, color in zip([10, 15, 20, 25, 30], ['rgba(0,255,0,0.1)', 'rgba(0,255,0,0.2)', 'rgba(255,255,0,0.2)', 'rgba(255,100,0,0.2)', 'rgba(255,0,0,0.2)']):
                    fig_river.add_trace(go.Scatter(x=df['date'], y=df['ttm_eps']*m, fill='tonexty' if m>10 else None, name=f"{m}x PE", line_width=0))
                fig_river.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="股價", line=dict(color='white')))
                st.plotly_chart(fig_river, use_container_width=True)
            else: st.warning("財報數據不全，無法繪製河流圖。")
        elif "3." in mode:
            if not share_hold.empty:
                big = share_hold[share_hold['Holding_class'] == '1000張以上']
                fig_c = make_subplots(specs=[[{"secondary_y": True}]])
                fig_c.add_trace(go.Scatter(x=big['date'], y=big['percent'], name="1000張大戶(%)"), secondary_y=False)
                fig_c.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="股價", line=dict(dash='dot')), secondary_y=True)
                st.plotly_chart(fig_c, use_container_width=True)
            else: st.warning("無法載入籌碼數據。")
    else: st.error("查無資料。")

if __name__ == "__main__": main()
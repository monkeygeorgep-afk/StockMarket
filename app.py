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
ST_CONFIG = {"page_title": "台股 AI 終極戰情室 6.0"}
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- 2. 核心運算：技術指標、河流圖與 AI 量化評分 ---
def calculate_indicators(df):
    # 四大均線
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

def calculate_ai_score(df, chip):
    """多因子量化評分引擎 (0-100)"""
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
    if curr['20MA'] > curr['60MA']: 
        score += 10; details.append("月季線多頭排列")
    return min(score, 100), details

def estimate_volume(current_volume):
    """盤中成交量預估演算法"""
    now = datetime.now()
    start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
    if now < start_time: return current_volume, 0
    if now > end_time: return current_volume, 100
    elapsed_minutes = (now - start_time).seconds / 60
    total_minutes = 270
    if elapsed_minutes <= 60:
        ratio = (elapsed_minutes / total_minutes) * 1.5 
    elif elapsed_minutes >= 210:
        ratio = (elapsed_minutes / total_minutes) * 1.2
    else:
        ratio = (elapsed_minutes / total_minutes)
    estimated_vol = current_volume / min(ratio, 0.99)
    progress = (elapsed_minutes / total_minutes) * 100
    return int(estimated_vol), int(progress)

# --- 3. 數據綜合獲取與持久化處理 ---
@st.cache_data(ttl=3600)
def fetch_master_data(stock_id):
    dl = DataLoader()
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
    share_hold = dl.taiwan_stock_holding_shares_per(stock_id=stock_id, start_date=(pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'))
    
    # 計算 TTM EPS (滾動四季)
    eps_df = fin[fin['type'] == 'EPS'].sort_values('date')
    eps_df['ttm_eps'] = eps_df['value'].rolling(window=4).sum()
    df = pd.merge_asof(df.sort_values('date'), eps_df[['date', 'ttm_eps']], on='date', direction='backward')
    
    return df, chip, news, fin, name, sup, res, share_hold

# --- 4. 介面優化：發送手機端 Flex Message 戰情小卡 ---
def send_line_flex_card(sid, name, price, change, ai_score, signals):
    try:
        url = "https://api.line.me/v2/bot/message/push"
        token = st.secrets["LINE_CHANNEL_ACCESS_TOKEN"]
        uid = st.secrets["LINE_USER_ID"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        color_trend = "#FF4D4D" if change >= 0 else "#2ECC71"
        arrow = "▲" if change >= 0 else "▼"
        change_str = f"{arrow} {abs(change):.1f}" if change != 0 else "0.0"
        
        if ai_score >= 70: ai_status, ai_color = "強勢看多", "#FF4D4D"
        elif ai_score <= 45: ai_status, ai_color = "保守看空", "#2ECC71"
        else: ai_status, ai_color = "中立盤整", "#7F8C8D"

        signal_text = " / ".join(signals) if signals else "暫無顯著訊號"

        flex_contents = {
          "type": "bubble",
          "styles": {"header": {"backgroundColor": "#1F2421"}, "footer": {"backgroundColor": "#F8F9FA"}},
          "header": {
            "type": "box", "layout": "vertical",
            "contents": [
              {"type": "text", "text": "🛡️ AI 戰情室巡邏廣播", "textColor": "#FFFFFF", "weight": "bold", "size": "md"},
              {"type": "text", "text": f"發送時間: {pd.Timestamp.now().strftime('%m/%d %H:%M')}", "textColor": "#A0A0A0", "size": "xs", "margin": "xs"}
            ]
          },
          "body": {
            "type": "box", "layout": "vertical",
            "contents": [
              {"type": "box", "layout": "horizontal", "contents": [
                  {"type": "text", "text": f"{name}", "weight": "bold", "size": "xl", "flex": 4},
                  {"type": "text", "text": f"{sid}", "textColor": "#7F8C8D", "size": "md", "align": "end", "gravity": "center", "flex": 2}
              ]},
              {"type": "separator", "margin": "md"},
              {"type": "box", "layout": "horizontal", "margin": "lg", "contents": [
                  {"type": "box", "layout": "vertical", "contents": [
                      {"type": "text", "text": "當前股價", "size": "xs", "textColor": "#888888"},
                      {"type": "text", "text": f"{price}", "size": "xxl", "weight": "bold"}
                  ]},
                  {"type": "box", "layout": "vertical", "align": "end", "contents": [
                      {"type": "text", "text": "今日漲跌", "size": "xs", "textColor": "#888888", "align": "end"},
                      {"type": "text", "text": f"{change_str}", "size": "xl", "weight": "bold", "textColor": color_trend, "align": "end"}
                  ]}
              ]},
              {"type": "box", "layout": "vertical", "margin": "md", "backgroundColor": "#F4F6F6", "paddingAll": "md", "cornerRadius": "md", "contents": [
                  {"type": "text", "text": "📡 觸發戰情訊號", "size": "xs", "textColor": "#7F8C8D", "weight": "bold"},
                  {"type": "text", "text": f"{signal_text}", "size": "sm", "color": "#2C3E50", "margin": "xs", "wrap": True}
              ]}
            ]
          },
          "footer": {
            "type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
              {"type": "box", "layout": "horizontal", "backgroundColor": ai_color, "cornerRadius": "md", "paddingAll": "sm", "justifyContent": "center", "contents": [
                  {"type": "text", "text": f"AI 評分: {ai_score} 分 ({ai_status})", "color": "#FFFFFF", "weight": "bold", "size": "sm", "align": "center"}
              ]}
            ]
          }
        }

        payload = {"to": uid, "messages": [{"type": "flex", "altText": f"【戰情警報】{name} 觸發訊號！", "contents": flex_contents}]}
        return requests.post(url, headers=headers, json=payload).status_code == 200
    except:
        return False

# --- 5. UI 主程式 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['2330', '2317', '2454']

    with st.sidebar:
        st.title("🛡️ AI 終極戰情室 6.0")
        target_id = st.text_input("輸入股票代號", value="2330")
        st.divider()
        mode = st.radio("切換主分析視角", ["1. 技術指標與 AI 報告", "2. 本益比河流圖", "3. 大戶籌碼趨勢"])
        st.divider()
        
        # 全自動巡邏系統 (已加入爆量因子與 Flex Message 戰情小卡)
        st.subheader("📡 全自動巡邏系統")
        if st.button("🔴 啟動 Full Scan (手機接收)"):
            with st.spinner("多因子後台巡邏中..."):
                scan_triggered = False
                for sid in st.session_state.watchlist:
                    d, c, _, _, name, s, r, _ = fetch_master_data(sid)
                    if d is None: continue
                    price = d['Close'].iloc[-1]
                    change = price - d['Close'].iloc[-2]
                    ai_score, ai_details = calculate_ai_score(d, c)
                    
                    # 警示觸發判定
                    signals = []
                    if price <= s * 1.02: signals.append("⚓ 接近支撐")
                    if price >= r * 0.98: signals.append("🚩 接近壓力")
                    if d['K'].iloc[-1] > d['D'].iloc[-1] and d['K'].iloc[-2] < d['D'].iloc[-2]: signals.append("⚡ KD金叉")
                    
                    # 盤中預估爆量判定
                    curr_vol = d['Volume'].iloc[-1] / 1000
                    avg_vol_5d = d['Volume'].iloc[-6:-1].mean() / 1000
                    est_v, _ = estimate_volume(curr_vol)
                    if avg_vol_5d > 0 and (est_v / avg_vol_5d) >= 1.5:
                        signals.append("🚀 異常爆量")
                    
                    if signals:
                        send_line_flex_card(sid, name, price, change, ai_score, signals)
                        scan_triggered = True
                
                if scan_triggered: st.success("🎯 巡邏完成！戰情小卡已發送至您的 Line。")
                else: st.info("盤面平穩，未偵測到特殊訊號。")

        st.divider()
        st.subheader("📋 自選清單管理")
        add_sid = st.text_input("新增代碼")
        if st.button("加入清單"):
            if add_sid and add_sid not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_sid)
                st.rerun()
        st.write("追蹤中:", ", ".join(st.session_state.watchlist))

    # --- 核心數據展示區 ---
    df, chip, news, fin, name, sup, res, share_hold = fetch_master_data(target_id)

    if df is not None:
        ai_score, ai_details = calculate_ai_score(df, chip)
        curr_vol = df['Volume'].iloc[-1] / 1000
        avg_vol_5d = df['Volume'].iloc[-6:-1].mean() / 1000
        est_v, progress = estimate_volume(curr_vol)
        vol_ratio = est_v / avg_vol_5d if avg_vol_5d > 0 else 0

        # 頂部戰術儀表板
        curr_p = df['Close'].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("今日收盤", f"{curr_p}", f"{curr_p - df['Close'].iloc[-2]:.1f}")
        c2.metric("AI 綜合量化評分", f"{ai_score} 分")
        c3.error(f"🚩 壓力位：{res:.2f} ({(res/curr_p-1)*100:.1f}%)")
        c4.success(f"⚓ 支撐位：{sup:.2f} ({(curr_p/sup-1)*100:.1f}%)")

        # 視角分流展示
        if "1. 技術指標與 AI 報告" in mode:
            # 盤中爆量即時警示看板
            if vol_ratio >= 1.5:
                st.error(f"🚨 【量能爆發警示】當前預估成交量 ({est_v}張) 已達 5日均量之 {vol_ratio:.1f} 倍！進度: {progress}%")

            with st.expander("🤖 AI 首席策略報告 (多因子診斷)", expanded=True):
                if st.button("🔍 運行 Gemini 智慧解讀"):
                    if "GEMINI_API_KEY" in st.secrets:
                        with st.spinner("AI 分析中..."):
                            model = genai.GenerativeModel('models/gemini-2.5-flash')
                            tech_summary = ", ".join(ai_details)
                            news_titles = "\n".join(news['title'].tail(5).tolist())
                            prompt = f"""
                            你是台股首席資深操盤手。請針對 {name}({target_id}) 進行診斷：
                            數據指標：
                            - 多因子量化總評：{ai_score}分。
                            - 觸發之核心指標：{tech_summary}。
                            - 盤中預估量能比：{vol_ratio:.1f}倍。
                            - 近期新聞標題：\n{news_titles}。
                            
                            請具體產出：
                            1. 【多空診斷】：判斷當前主流資金與技術面表態方向。
                            2. 【操作戰術】：結合支撐位 ({sup}) 與壓力位 ({res}) 給予具體交易點位建議。
                            3. 【風險雷達】：目前盤面上最需要警惕的利空因子。
                            """
                            st.write(model.generate_content(prompt).text)
                    else: st.warning("請先設定 GEMINI_API_KEY")

            # 主圖表 (K線 + 均線 + 布林)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            
            # 均線繪製 (修正 name=m 錯誤)
            for m, clr in zip(['5MA', '20MA', '60MA'], ['white', 'orange', 'cyan']):
                fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(color=clr, width=1.2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name="布林上軌", line=dict(color='rgba(255,255,255,0.15)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name="布林下軌", line=dict(color='rgba(255,255,255,0.15)', dash='dot'), fill='tonexty'), row=1, col=1)
            fig.add_hline(y=res, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=sup, line_dash="dash", line_color="green", row=1, col=1)
            
            # 子圖 KD
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=2, col=1)
            fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "2. 本益比河流圖" in mode:
            st.subheader("🌌 價值投資：本益比河流圖 (P/E Band)")
            fig_river = go.Figure()
            pe_multipliers = [10, 15, 20, 25, 30]
            pe_colors = ['rgba(0,255,0,0.08)', 'rgba(100,255,0,0.12)', 'rgba(255,255,0,0.15)', 'rgba(255,100,0,0.18)', 'rgba(255,0,0,0.2)']
            
            for i in range(len(pe_multipliers)-1):
                fig_river.add_trace(go.Scatter(
                    x=df['date'].tolist() + df['date'].tolist()[::-1],
                    y=(df['ttm_eps']*pe_multipliers[i+1]).tolist() + (df['ttm_eps']*pe_multipliers[i]).tolist()[::-1],
                    fill='toself', fillcolor=pe_colors[i], line_color='rgba(255,255,255,0)',
                    name=f"{pe_multipliers[i]}x-{pe_multipliers[i+1]}x PE"
                ))
            fig_river.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="目前股價", line=dict(color='white', width=2)))
            fig_river.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig_river, use_container_width=True)
            st.info("💡 綠色區為歷史便宜價、黃色為合理價、紅色為昂貴價。河流向上代表基本面成長。")

        elif "3. 大戶籌碼趨勢" in mode:
            st.subheader("🕵️ 籌碼核心：1000張大戶持股比率 vs 股價")
            big_1000 = share_hold[share_hold['Holding_class'] == '1000張以上'].copy()
            if not big_1000.empty:
                fig_chip = make_subplots(specs=[[{"secondary_y": True}]])
                fig_chip.add_trace(go.Scatter(x=big_1000['date'], y=big_1000['percent'], name="1000張大戶比例(%)", line=dict(color='#FF4D4D', width=2.5)), secondary_y=False)
                fig_chip.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="收盤價", line=dict(color='white', width=1, dash='dot')), secondary_y=True)
                fig_chip.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig_chip, use_container_width=True)
            else:
                st.warning("暫無該股的大戶持股比率資料")

        # 底部數據標籤頁 (籌碼與財報明細)
        t1, t2 = st.tabs(["🏦 法人每日買賣 (張)", "📰 相關數據源明細"])
        with t1:
            if not chip.empty:
                last_date = chip['date'].max()
                t_chip = chip[chip['date'] == last_date]
                f_net = t_chip[t_chip['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
                i_net = t_chip[t_chip['name']=='Investment_Trust'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
                d_net = t_chip[t_chip['name']=='Dealer'].apply(lambda x: x['buy']-x['sell'], axis=1).sum()
                col_f, col_i, col_d = st.columns(3)
                col_f.metric("外資買賣超", f"{int(f_net)} 張", delta=f"{int(f_net)}")
                col_i.metric("投信買賣超", f"{int(i_net)} 張", delta=f"{int(i_net)}")
                col_d.metric("自營商買賣超", f"{int(d_net)} 張", delta=f"{int(d_net)}")
                st.dataframe(chip.tail(10), use_container_width=True)
        with t2:
            st.subheader("近期新聞")
            st.dataframe(news[['date', 'title', 'link']].tail(8), use_container_width=True)
            st.subheader("季度財報摘要")
            st.dataframe(fin.tail(10), use_container_width=True)
    else:
        st.error("查無相關資料，請檢查股票代號是否輸入正確。")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import requests
import io

# --- 1. 系統配置 ---
ST_CONFIG = {
    "page_title": "台股 AI 全自動戰情室 2.2",
    "watchlist": ['2330', '2317', '2454', '2308', '2603', '2382'] # 預設監控清單
}

# --- 2. 技術指標運算 (含 KD, MA, RSI, MACD) ---
def calculate_indicators(df):
    # 移動平均線
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # KD 指標 (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        # 若當前 RSV 為 NaN (初期資料不足), 則維持前值
        current_rsv = rsv.iloc[i] if not np.isnan(rsv.iloc[i]) else 50.0
        new_k = (k[-1] * 2/3) + (current_rsv * 1/3)
        new_d = (d[-1] * 2/3) + (new_k * 1/3)
        k.append(new_k)
        d.append(new_d)
    df['K'], df['D'] = k, d
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

# --- 3. 籌碼數據獲取與統計 ---
def fetch_chip_data(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
    try:
        chip_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
        if chip_df.empty: return pd.DataFrame()
        # 轉換為張數 (FinMind 回傳通常是張數或股數，這裡假設為張數)
        return chip_df
    except:
        return pd.DataFrame()

def analyze_chip_flow(chip_df):
    if chip_df.empty: return None
    latest_days = sorted(chip_df['date'].unique(), reverse=True)[:5]
    summary = []
    for count in [1, 3, 5]:
        target_days = latest_days[:count]
        temp = chip_df[chip_df['date'].isin(target_days)]
        # 計算買賣差額
        diff = temp.groupby('name')['buy'].sum() - temp.groupby('name')['sell'].sum()
        summary.append({
            "期間": f"近 {count} 日",
            "外資": int(diff.get('Foreign_Investor', 0)),
            "投信": int(diff.get('Investment_Trust', 0)),
            "自營商": int(diff.get('Dealer', 0))
        })
    return pd.DataFrame(summary)

# --- 4. 數據整合與緩存 ---
@st.cache_data(ttl=3600)
def get_full_stock_info(stock_id):
    dl = DataLoader()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    # 抓取日股價
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    if df.empty: return None, None
    df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = calculate_indicators(df)
    # 抓取法人
    chip = fetch_chip_data(stock_id)
    return df, chip

# --- 5. 綜合策略掃描演算法 ---
def scan_advanced_signals(df, chip_df):
    sigs = []
    if df is None or len(df) < 5: return sigs
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 技術面: KD 黃金交叉
    if prev['K'] < prev['D'] and curr['K'] > curr['D'] and curr['K'] < 30:
        sigs.append("⚡ KD 低檔黃金交叉")
    
    # 技術面: 爆量突破
    avg_vol = df['Volume'].iloc[-6:-1].mean()
    if curr['Volume'] > avg_vol * 2 and curr['Close'] > prev['High']:
        sigs.append("🚀 帶量突破前高")

    # 籌碼面: 法人同步大買
    if chip_df is not None and not chip_df.empty:
        today = sorted(chip_df['date'].unique())[-1]
        t_chip = chip_df[chip_df['date'] == today]
        fi = t_chip[t_chip['name']=='Foreign_Investor']['buy'].sum() - t_chip[t_chip['name']=='Foreign_Investor']['sell'].sum()
        it = t_chip[t_chip['name']=='Investment_Trust']['buy'].sum() - t_chip[t_chip['name']=='Investment_Trust']['sell'].sum()
        if fi > 500 and it > 500:
            sigs.append("🏦 法人雙買 (外資投信同步大買)")
            
    return sigs

# --- 6. Line 推送 (Messaging API) ---
def send_line_alert(msg):
    try:
        token = st.secrets["LINE_CHANNEL_ACCESS_TOKEN"]
        uid = st.secrets["LINE_USER_ID"]
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"to": uid, "messages": [{"type": "text", "text": msg}]}
        res = requests.post(url, headers=headers, json=payload)
        return res.status_code
    except:
        return 500

# --- 7. Streamlit UI 介面 ---
def main():
    st.set_page_config(layout="wide", page_title=ST_CONFIG["page_title"])
    
    # 側邊欄：自選監控
    with st.sidebar:
        st.title("🛡️ 監控控制台")
        target_stock = st.text_input("查看股票代碼", value="2330")
        
        st.divider()
        st.subheader("📡 全自動巡邏")
        if st.button("執行全清單掃描"):
            with st.spinner("掃描中..."):
                reports = []
                for s in ST_CONFIG["watchlist"]:
                    d, c = get_full_stock_info(s)
                    s_sigs = scan_advanced_signals(d, c)
                    if s_sigs:
                        reports.append(f"【{s}】\n" + "\n".join(s_sigs))
                
                if reports:
                    final_msg = "🚨 盤後強勢股掃描報告：\n\n" + "\n\n".join(reports)
                    send_line_alert(final_msg)
                    st.success("已發送 Line 通知！")
                else:
                    st.info("目前清單中無特殊訊號。")

    # 主畫面展示
    df, chip = get_full_stock_info(target_stock)
    
    if df is not None:
        st.title(f"台股智慧分析：{target_stock}")
        
        # 1. 籌碼看板
        st.subheader("🏦 三大法人買賣超動態 (張)")
        chip_summary = analyze_chip_flow(chip)
        if chip_summary is not None:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("今日外資", f"{chip_summary.iloc[0]['外資']:,}")
            with c2: st.metric("今日投信", f"{chip_summary.iloc[0]['投信']:,}")
            with c3: st.metric("今日自營商", f"{chip_summary.iloc[0]['自營商']:,}")
            st.table(chip_summary.set_index("期間"))

        # 2. 技術指標圖表
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("K線與分價量表", "KD 強弱指標", "MACD 指標")
        )

        # K線與分價量
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['20MA'], name="20MA", line=dict(color='orange')), row=1, col=1)
        
        # 簡易分價量表
        p_bins = np.linspace(df['Low'].min(), df['High'].max(), 15)
        v_dist, _ = np.histogram(df['Close'], bins=p_bins, weights=df['Volume'])
        for i in range(len(v_dist)):
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0.9, x1=0.9+(v_dist[i]/v_dist.max()*0.1),
                          y0=p_bins[i], y1=p_bins[i+1], fillcolor="rgba(200,200,200,0.15)", line_width=0, row=1, col=1)

        # KD (Row 2)
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K值", line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D值", line=dict(color='cyan')), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)

        # MACD (Row 3)
        colors = ['red' if val >= 0 else 'green' for val in df['Hist']]
        fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name="MACD", marker_color=colors), row=3, col=1)

        fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("查無資料，請確認股票代碼是否正確。")

if __name__ == "__main__":
    main()
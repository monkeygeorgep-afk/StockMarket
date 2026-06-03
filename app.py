import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import google.generativeai as genai
from datetime import datetime, timedelta
import requests

# ==========================================
# 模組一：數據中心 (Data Center)
# ==========================================
class StockDataCenter:
    def __init__(self):
        token = st.secrets.get("FINMIND_TOKEN", "")
        self.dl = DataLoader(token=token) if token else DataLoader()
        self.stock_info = self.dl.taiwan_stock_info()

    @st.cache_data(ttl=3600)
    def get_full_analysis_data(_self, stock_id):
        try:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            df = _self.dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty: return None
            df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            
            # 指標與基本資訊
            df = _self._add_indicators(df)
            info = _self.stock_info[_self.stock_info['stock_id'] == stock_id]
            name = info['stock_name'].values[0] if not info.empty else "未知"
            industry = info['industry_category'].values[0] if not info.empty else "未知"
            
            # 籌碼與財報 (Fail-safe)
            chip = _self.dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            fin = _self.dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
            
            # 處理 TTM EPS (河流圖用)
            df['ttm_eps'] = np.nan
            eps_df = fin[fin['type'] == 'EPS'].copy()
            if not eps_df.empty:
                eps_df['date'] = pd.to_datetime(eps_df['date'])
                eps_df = eps_df.sort_values('date')
                eps_df['ttm_eps'] = eps_df['value'].rolling(window=4).sum()
                df = pd.merge_asof(df.sort_values('date'), eps_df[['date', 'ttm_eps']], on='date', direction='backward')
            
            return {"df": df, "name": name, "industry": industry, "chip": chip, "stock_id": stock_id}
        except: return None

    def _add_indicators(self, df):
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['VMA5'] = df['Volume'].rolling(window=5).mean()
        std = df['Close'].rolling(window=20).std()
        df['BBU'], df['BBL'] = df['20MA'] + (std * 2), df['20MA'] - (std * 2)
        l9, h9 = df['Low'].rolling(window=9).min(), df['High'].rolling(window=9).max()
        rsv = 100 * (df['Close'] - l9) / (h9 - l9 + 1e-9)
        k, d = [50.0], [50.0]
        for i in range(1, len(rsv)):
            nk = (k[-1] * 2/3) + (rsv.iloc[i] * 1/3)
            nd = (d[-1] * 2/3) + (nk * 1/3)
            k.append(nk); d.append(nd)
        df['K'], df['D'] = k, d
        return df

# ==========================================
# 模組二：量化引擎 (Quant Engine)
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_ai_score(data):
        df, chip = data['df'], data['chip']
        curr = df.iloc[-1]
        score = 50
        if curr['Close'] > curr['20MA']: score += 15
        if curr['K'] > curr['D']: score += 10
        if not chip.empty:
            last = chip[chip['date'] == chip['date'].max()]
            if last[last['name']=='Foreign_Investor'].apply(lambda x: x['buy']-x['sell'], axis=1).sum() > 0: score += 15
        return min(score, 100)

    @staticmethod
    def run_backtest(df):
        res = []
        for i in range(60, len(df) - 5):
            if df.iloc[i]['K'] > df.iloc[i]['D'] and df.iloc[i]['Close'] > df.iloc[i]['20MA']:
                res.append((df.iloc[i+5]['Close'] - df.iloc[i+1]['Open']) / df.iloc[i+1]['Open'])
        return (len(res), np.mean(res)*100 if res else 0, len([r for r in res if r>0])/len(res)*100 if res else 0)

# ==========================================
# 模組三：通知中心 (Notification)
# ==========================================
class Notifier:
    @staticmethod
    def send_flex_card(data, ai_score, signals):
        try:
            url = "https://api.line.me/v2/bot/message/push"
            headers = {"Authorization": f"Bearer {st.secrets['LINE_CHANNEL_ACCESS_TOKEN']}", "Content-Type": "application/json"}
            payload = {
                "to": st.secrets["LINE_USER_ID"],
                "messages": [{
                    "type": "flex", "altText": f"戰情警報:{data['name']}",
                    "contents": {
                        "type": "bubble", "header": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": "🛡️ AI 終極戰情巡邏", "color": "#FFFFFF", "weight": "bold"}]},
                        "styles": {"header": {"backgroundColor": "#1F2421"}},
                        "body": {"type": "box", "layout": "vertical", "contents": [
                            {"type": "text", "text": f"{data['name']} ({data['stock_id']})", "weight": "bold", "size": "xl"},
                            {"type": "text", "text": f"AI 評分: {ai_score} 分", "margin": "md", "color": "#FF4D4D" if ai_score > 70 else "#7F8C8D"},
                            {"type": "text", "text": f"觸發訊號: {' / '.join(signals)}", "size": "sm", "wrap": True, "margin": "md"}
                        ]}
                    }
                }]
            }
            requests.post(url, headers=headers, json=payload)
        except: pass

# ==========================================
# 模組四：主程式 UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="台股 AI 終極戰情室 8.0")
    if "GEMINI_API_KEY" in st.secrets: genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    dc = StockDataCenter()

    with st.sidebar:
        st.title("🛡️ 終極戰情室 8.0")
        stock_input = st.text_input("多股監控 (代號逗號隔開)", value="2330, 2317, 2454")
        stock_list = [s.strip() for s in stock_input.split(",")]
        st.divider()
        if st.button("🔴 啟動全方位巡邏 (Line)"):
            with st.spinner("巡邏中..."):
                for sid in stock_list:
                    data = dc.get_full_analysis_data(sid)
                    if data:
                        score = QuantEngine.calculate_ai_score(data)
                        if score >= 70:
                            Notifier.send_flex_card(data, score, ["AI 高分", "趨勢轉強"])
                st.success("巡邏完畢，高分股已發送通知。")

    tab1, tab2 = st.tabs(["🎯 多股對比熱力圖", "📊 單股深度診斷"])

    with tab1:
        comparison = []
        for sid in stock_list:
            data = dc.get_full_analysis_data(sid)
            if data:
                score = QuantEngine.calculate_ai_score(data)
                count, ret, win = QuantEngine.run_backtest(data['df'])
                comparison.append({"代號": sid, "名稱": data['name'], "AI評分": score, "回測勝率": f"{win:.1f}%", "預期報酬": f"{ret:.2f}%"})
        
        if comparison:
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df.style.background_gradient(subset=['AI評分'], cmap='RdYlGn'), use_container_width=True)
            fig_heat = go.Figure(data=go.Heatmap(z=[comp_df['AI評分']], x=comp_df['名稱'], colorscale='Viridis'))
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        target_sid = st.selectbox("選擇診斷代號", stock_list)
        data = dc.get_full_analysis_data(target_sid)
        if data:
            df = data['df']
            col_l, col_r = st.columns([3, 1])
            with col_l:
                # 繪製 K線 + 布林 + 成交量
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], line=dict(color='rgba(255,255,255,0.2)'), name="布林上軌"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="布林下軌"), row=1, col=1)
                fig.add_trace(go.Bar(x=df['date'], y=df['Volume'], name="成交量", marker_color='gray'), row=2, col=1)
                fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                score = QuantEngine.calculate_ai_score(data)
                st.metric("AI 綜合評分", f"{score} 分")
                if st.button("🔍 生成 AI 深度報告"):
                    model = genai.GenerativeModel('models/gemini-2.5-flash')
                    prompt = f"你是操盤手。{data['name']} AI 評分 {score}。請結合技術面給予建議。"
                    st.write(model.generate_content(prompt).text)

if __name__ == "__main__": main()
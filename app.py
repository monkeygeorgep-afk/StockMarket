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
            
            # 1. 技術指標
            df = _self._add_indicators(df)
            
            # 2. 產業與名稱
            info = _self.stock_info[_self.stock_info['stock_id'] == stock_id]
            name = info['stock_name'].values[0] if not info.empty else "未知"
            industry = info['industry_category'].values[0] if not info.empty else "未知"
            
            # 3. 籌碼面 (三大法人、分點)
            chip = _self.dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            broker = _self.dl.taiwan_stock_daily_collect(stock_id=stock_id, start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            
            # 4. 財報 (河流圖用)
            fin = _self.dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
            df['ttm_eps'] = np.nan
            eps_df = fin[fin['type'] == 'EPS'].copy()
            if not eps_df.empty:
                eps_df['date'] = pd.to_datetime(eps_df['date'])
                eps_df = eps_df.sort_values('date')
                eps_df['ttm_eps'] = eps_df['value'].rolling(window=4).sum()
                df = pd.merge_asof(df.sort_values('date'), eps_df[['date', 'ttm_eps']], on='date', direction='backward')
            
            return {"df": df, "name": name, "industry": industry, "chip": chip, "broker": broker, "stock_id": stock_id}
        except Exception: return None

    def _add_indicators(self, df):
        # 均線
        for m in [5, 10, 20, 60]:
            df[f'{m}MA'] = df['Close'].rolling(window=m).mean()
        # 布林通道 (20MA, 2SD)
        std = df['Close'].rolling(window=20).std()
        df['BBU'], df['BBL'] = df['20MA'] + (std * 2), df['20MA'] - (std * 2)
        # 成交量均線
        df['VMA5'] = df['Volume'].rolling(window=5).mean()
        # KD (9, 3, 3)
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
# 模組二：視覺化中心 (Visualizer)
# ==========================================
class Visualizer:
    @staticmethod
    def plot_technical_master(data):
        df = data['df']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3])
        # 1. K線、均線、布林
        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], line=dict(color='rgba(255,255,255,0.2)'), name="布林上軌"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="布林下軌"), row=1, col=1)
        for m, c in zip(['5MA', '20MA', '60MA'], ['white', 'orange', 'cyan']):
            fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(color=c, width=1)), row=1, col=1)
        
        # 2. 成交量圖
        v_colors = ['red' if c >= o else 'green' for o, c in zip(df['Open'], df['Close'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['Volume'], marker_color=v_colors, name="成交量"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['VMA5'], line=dict(color='yellow'), name="5日量均"), row=2, col=1)
        
        # 3. KD 指標
        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name="K", line=dict(color='yellow')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name="D", line=dict(color='cyan')), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        return fig

    @staticmethod
    def plot_river_chart(data):
        df = data['df']
        if 'ttm_eps' not in df.columns or df['ttm_eps'].isnull().all(): return None
        fig = go.Figure()
        pe_bands = [10, 15, 20, 25, 30]
        colors = ['rgba(0,255,0,0.1)', 'rgba(127,255,0,0.15)', 'rgba(255,255,0,0.2)', 'rgba(255,127,0,0.25)', 'rgba(255,0,0,0.3)']
        for i in range(len(pe_bands)-1):
            fig.add_trace(go.Scatter(
                x=df['date'].tolist() + df['date'].tolist()[::-1],
                y=(df['ttm_eps']*pe_bands[i+1]).tolist() + (df['ttm_eps']*pe_bands[i]).tolist()[::-1],
                fill='toself', fillcolor=colors[i], line_color='rgba(0,0,0,0)', name=f"{pe_bands[i]}x-{pe_bands[i+1]}x"
            ))
        fig.add_trace(go.Scatter(x=df['date'], y=df['Close'], line=dict(color='white', width=2), name="收盤價"))
        fig.update_layout(height=500, template="plotly_dark", title="本益比河流圖 (TTM)")
        return fig

# ==========================================
# 模組三：量化與 AI 引擎 (Quant Engine)
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_metrics(data):
        df = data['df']
        curr = df.iloc[-1]
        score = 50
        details = []
        if curr['Close'] > curr['20MA']: score += 10; details.append("站上月線")
        if curr['K'] > curr['D']: score += 10; details.append("KD金叉")
        if curr['Volume'] > curr['VMA5']: score += 10; details.append("量能升溫")
        
        # 回測勝率
        wins = []
        for i in range(60, len(df)-5):
            if df.iloc[i]['Close'] > df.iloc[i]['20MA'] and df.iloc[i]['K'] > df.iloc[i]['D']:
                wins.append((df.iloc[i+5]['Close'] - df.iloc[i+1]['Open']) > 0)
        win_rate = (sum(wins)/len(wins)*100) if wins else 0
        
        return min(score, 100), details, win_rate

# ==========================================
# 模組四：主程式 UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="台股 AI 終極戰情室 8.2")
    if "GEMINI_API_KEY" in st.secrets: genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    dc = StockDataCenter()

    with st.sidebar:
        st.title("🛡️ 戰情室 8.2")
        input_ids = st.text_input("代號監控 (逗號隔開)", value="2330, 2454, 2317")
        stock_list = [s.strip() for s in input_ids.split(",")]
        mode = st.radio("導航中心", ["📈 多股動能對比", "🔍 單股深度診斷"])

    if mode == "📈 多股動能對比":
        st.subheader("🔥 族群資金動能與回測勝率對照")
        comp_results = []
        for sid in stock_list:
            d = dc.get_full_analysis_data(sid)
            if d:
                score, _, win_rate = QuantEngine.calculate_metrics(d)
                comp_results.append({
                    "名稱": d['name'], "代號": sid, "產業": d['industry'],
                    "AI評分": score, "回測勝率": f"{win_rate:.1f}%", "收盤價": d['df']['Close'].iloc[-1]
                })
        if comp_results:
            cdf = pd.DataFrame(comp_results)
            st.dataframe(cdf.style.background_gradient(subset=['AI評分'], cmap='RdYlGn'), use_container_width=True)
            # 熱力圖
            fig_h = go.Figure(data=go.Heatmap(z=[[r['AI評分'] for r in comp_results]], x=[r['名稱'] for r in comp_results], colorscale='Viridis'))
            st.plotly_chart(fig_h, use_container_width=True)

    else:
        target_sid = st.selectbox("選擇診斷代號", stock_list)
        data = dc.get_full_analysis_data(target_sid)
        if data:
            score, details, win_rate = QuantEngine.calculate_metrics(data)
            st.title(f"📈 {data['name']} ({target_sid}) | {data['industry']}")
            
            # 指標儀表板
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("今日收盤", data['df']['Close'].iloc[-1])
            c2.metric("AI 綜合評分", f"{score} 分")
            c3.metric("歷史回測勝率", f"{win_rate:.1f}%")
            c4.write(f"✅ 觸發訊號：\n{', '.join(details)}")

            # 功能標籤頁
            tab1, tab2, tab3, tab4 = st.tabs(["📊 技術面 (含布林成交量)", "🌊 本益比河流圖", "🕵️ 關鍵分點", "🤖 AI 深度報告"])
            
            with tab1:
                st.plotly_chart(Visualizer.plot_technical_master(data), use_container_width=True)
            
            with tab2:
                fig_r = Visualizer.plot_river_chart(data)
                if fig_r: st.plotly_chart(fig_r, use_container_width=True)
                else: st.warning("財報數據不足。")
            
            with tab3:
                # 簡單分點展示邏輯
                if not data['broker'].empty:
                    b_data = data['broker'][data['broker']['date'] == data['broker']['date'].max()].copy()
                    b_data['net'] = b_data['buy'] - b_data['sell']
                    fig_b = go.Figure(go.Bar(x=b_data.nlargest(10, 'net')['net'], y=b_data.nlargest(10, 'net')['broker_name'], orientation='h', marker_color='red'))
                    st.plotly_chart(fig_b, use_container_width=True)

            with tab4:
                if st.button("🔍 生成 AI 量化分析報告"):
                    model = genai.GenerativeModel('models/gemini-2.5-flash')
                    prompt = f"分析 {data['name']}({target_sid})。AI分:{score}, 勝率:{win_rate}%, 產業:{data['industry']}。請給予實戰策略。"
                    st.write(model.generate_content(prompt).text)

if __name__ == "__main__": main()
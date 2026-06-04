import streamlit as st
from data_center import StockDataCenter
from quant_engine import QuantEngine
from visualizer import Visualizer
import google.generativeai as genai

# 初始化模組
dc = StockDataCenter()
qe = QuantEngine()
vz = Visualizer()

def main():
    st.set_page_config(layout="wide", page_title="台股 AI 終極戰情室 8.2")
    
    with st.sidebar:
        st.title("🛡️ 旗艦戰情室 8.2")
        stock_input = st.text_input("多股監控 (代號以逗號隔開)", value="2330, 2454, 2317")
        stock_list = [s.strip() for s in stock_input.split(",")]
        target_sid = st.selectbox("核心診斷對象", stock_list)
        st.divider()
        sl_val = st.slider("停損百分比設定", -10.0, -2.0, -5.0)

    # 1. 資料抓取與計算
    data = dc.fetch_master_data(target_sid)
    if data:
        df = qe.add_indicators(data['df'])
        chip, broker, news = dc.fetch_chip_data(target_sid)
        
        st.title(f"📈 {data['name']} ({target_sid}) | {data['industry']}")
        
        tab1, tab2, tab3 = st.tabs(["📊 技術量價", "🌌 價值河流", "🕵️ 籌碼與 AI 報告"])
        
        with tab1:
            est_v, prog = qe.estimate_vol(df['Volume'].iloc[-1]/1000)
            avg_v = df['Volume'].iloc[-6:-1].mean()/1000
            if est_v / avg_v >= 1.5: st.error(f"🚨 爆量警示：預估成交量達均量 {est_v/avg_v:.1f} 倍！")
            st.plotly_chart(vz.plot_tech_main(df), use_container_width=True)
            
        with tab2:
            fig_river = vz.plot_river(df)
            if fig_river: st.plotly_chart(fig_river, use_container_width=True)
            
        with tab3:
            bt = qe.run_backtest(df, stop_loss_pct=sl_val)
            geo_brokers = qe.detect_geographic_brokers(broker)
            
            # 分點地圖繪製
            fig_broker = vz.plot_broker_landscape(broker)
            if fig_broker: st.plotly_chart(fig_broker, use_container_width=True)
            
            if st.button("🔍 生成 AI 籌碼深度報告"):
                if "GEMINI_API_KEY" in st.secrets:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel('models/gemini-2.5-flash')
                    prompt = f"""
                    你是台股分析師。分析 {data['name']}({target_sid})：
                    - 地緣分點跡象：{geo_brokers}
                    - 5日回測勝率：{bt['win_rate']:.1f}%，平均報酬：{bt['avg_return']:.2f}%
                    - 停損設定：{sl_val}%，歷史觸發停損次數：{bt['sl_triggered']}
                    請給予結合籌碼集中度與地緣性的操作建議。
                    """
                    st.write(model.generate_content(prompt).text)
    else: st.error("查無資料，請檢查代號或 API 狀態。")

if __name__ == "__main__": main()
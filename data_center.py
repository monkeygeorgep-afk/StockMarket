import streamlit as st
import pandas as pd
from FinMind.data import DataLoader
from datetime import datetime, timedelta

class StockDataCenter:
    def __init__(self):
        # 優先從 secrets 讀取 Token，若無則以訪客身份執行
        token = st.secrets.get("FINMIND_TOKEN", "")
        self.dl = DataLoader(token=token) if token else DataLoader()
        self.stock_info = self.dl.taiwan_stock_info()

    @st.cache_data(ttl=3600)
    def fetch_master_data(_self, stock_id):
        """抓取核心日線、財報與基本資訊"""
        try:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            df = _self.dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty: return None
            
            df = df.rename(columns={'max':'High','min':'Low','close':'Close','open':'Open','Trading_Volume':'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            
            info = _self.stock_info[_self.stock_info['stock_id'] == stock_id]
            name = info['stock_name'].values[0] if not info.empty else "未知"
            industry = info['industry_category'].values[0] if not info.empty else "未知"
            
            # 抓取財報進行 TTM EPS 計算 (河流圖基礎)
            fin = _self.dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
            eps_df = fin[fin['type'] == 'EPS'].copy()
            if not eps_df.empty:
                eps_df['date'] = pd.to_datetime(eps_df['date'])
                eps_df = eps_df.sort_values('date')
                eps_df['ttm_eps'] = eps_df['value'].rolling(window=4).sum()
                df = pd.merge_asof(df.sort_values('date'), eps_df[['date', 'ttm_eps']], on='date', direction='backward')
            
            return {"df": df, "name": name, "industry": industry}
        except Exception as e:
            return None

    def fetch_chip_data(self, stock_id):
        """抓取法人、分點與新聞"""
        # 1. 抓取三大法人資料
        chip = self.dl.taiwan_stock_institutional_investors(
            stock_id=stock_id, 
            start_date=(datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
        )
        
        # 2. 抓取分點資料 (加入錯誤防護)
        broker = pd.DataFrame()
        try:
            # 嘗試使用正確的 API 名稱
            # 註：部分版本可能更名為 taiwan_stock_daily_collect 或需透過其他方式呼叫
            if hasattr(self.dl, 'taiwan_stock_daily_collect'):
                broker = self.dl.taiwan_stock_daily_collect(
                    stock_id=stock_id, 
                    start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                )
            else:
                # 如果找不到方法，回傳空表避免當機
                st.warning(f"目前 FinMind 版本不支援分點資料抓取，請檢查套件版本。")
        except Exception as e:
            st.error(f"分點資料抓取失敗: {e}")

        # 3. 抓取新聞
        news = self.dl.taiwan_stock_news(
            stock_id=stock_id, 
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        )
        
        return chip, broker, news
import numpy as np
from datetime import datetime

class QuantEngine:
    @staticmethod
    def add_indicators(df):
        """計算技術指標：均線、布林、KD"""
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

    @staticmethod
    def run_backtest(df, hold_days=5, stop_loss_pct=-5.0):
        """具備停損機制的深度回測"""
        results = []
        sl_count = 0
        for i in range(60, len(df) - hold_days):
            # 進場條件：KD金叉且站上月線
            if df.iloc[i]['K'] > df.iloc[i]['D'] and df.iloc[i]['Close'] > df.iloc[i]['20MA']:
                entry_p = df.iloc[i+1]['Open']
                triggered_sl = False
                for d in range(1, hold_days + 1):
                    low_p = df.iloc[i+d]['Low']
                    if ((low_p - entry_p) / entry_p * 100) <= stop_loss_pct:
                        results.append(stop_loss_pct)
                        sl_count += 1
                        triggered_sl = True
                        break
                if not triggered_sl:
                    exit_p = df.iloc[i + hold_days]['Close']
                    results.append((exit_p - entry_p) / entry_p * 100)
        if not results: return None
        return {"win_rate": len([r for r in results if r > 0]) / len(results) * 100,
                "avg_return": np.mean(results), "count": len(results), "sl_triggered": sl_count}

    @staticmethod
    def detect_geographic_brokers(broker_df):
        """偵測特定熱點分點"""
        geo_keywords = ["新竹", "竹科", "台中", "台南", "高雄", "內湖", "敦南", "桃園"]
        if broker_df.empty: return []
        latest_date = broker_df['date'].max()
        top_buyers = broker_df[broker_df['date'] == latest_date].nlargest(10, 'buy')
        detected = [b for b in top_buyers['broker_name'] if any(k in b for k in geo_keywords)]
        return list(set(detected))

    @staticmethod
    def estimate_vol(current_vol):
        now = datetime.now()
        start = now.replace(hour=9, minute=0, second=0)
        elapsed = min((now - start).seconds / 16200, 1.0)
        return int(current_vol / max(elapsed, 0.01)), int(elapsed * 100)
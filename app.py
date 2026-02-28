import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
import traceback
import os
import threading
import gradio as gr
from dotenv import load_dotenv
import hmac
import hashlib
import json
import base64
import math

# ==============================================================================
# ========== 1. CẤU HÌNH & BIẾN TOÀN CỤC ==========
# ==============================================================================
if os.path.exists(".env"):
    load_dotenv(".env")

OKX_API_KEY = os.environ.get("OKX_API_KEY")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE")
OKX_BASE_URL = "https://www.okx.com"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

GLOBAL_RUNNING = False
TRADE_AMOUNT_USDT = 10.0  
GLOBAL_LEVERAGE = 25       
TIMEFRAME = "5m"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LAST_PROCESSED_MINUTE = -1 

# SETTING THEO YÊU CẦU
LOOKBACK_CANDLES = 100 
MAX_OPEN_POSITIONS = 6
BUFFER_PERCENT = 0.15 

# DANH SÁCH 50 CẶP COIN
SYMBOL_CONFIGS = {
    "BTC-USDT-SWAP": {"X": 0.15, "Y": 0.05, "Active": True},
    "ETH-USDT-SWAP": {"X": 0.2, "Y": 0.05, "Active": True},
    "SOL-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "BNB-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "XRP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ADA-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOGE-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "AVAX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOT-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "LINK-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "MATIC-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "UNI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "LTC-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "NEAR-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ATOM-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ETC-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "BCH-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "FIL-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "APT-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "OP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ARB-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "INJ-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "SUI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "TIA-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "SEI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ORDI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "RNDR-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "PEPE-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "SHIB-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "TRX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "STX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ICP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "IMX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "KAS-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "GRT-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "AAVE-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "FTM-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "GALA-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "RUNE-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "DYDX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "JUP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "PYTH-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "WLD-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "BONK-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "FLOKI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "LDO-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "FET-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "AGIX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "OCEAN-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ARKM-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
}

MARKET_DATA_CACHE = {}

# ==============================================================================
# ========== 2. HÀM TIỆN ÍCH API ==========
# ==============================================================================

def okx_request(method, endpoint, body=None):
    try:
        ts = datetime.now(UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        body_str = json.dumps(body) if body else ""
        message = ts + method + endpoint + body_str
        mac = hmac.new(bytes(OKX_SECRET_KEY, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        headers = {
            'OK-ACCESS-KEY': OKX_API_KEY,
            'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': ts,
            'OK-ACCESS-PASSPHRASE': OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        res = requests.request(method, OKX_BASE_URL + endpoint, headers=headers, data=body_str, timeout=10)
        return res.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def get_market_rules(symbol):
    if symbol in MARKET_DATA_CACHE: return MARKET_DATA_CACHE[symbol]
    try:
        url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP&instId={symbol}"
        res = requests.get(url, timeout=10).json()
        if res.get('code') == '0' and res.get('data'):
            inst = res['data'][0]
            data = {
                "lotSz": float(inst['lotSz']),
                "tickSz": float(inst['tickSz']),
                "prec": len(inst['tickSz'].split('.')[-1]) if '.' in inst['tickSz'] else 0,
                "minSz": float(inst['minSz']),
                "ctVal": float(inst['ctVal'])
            }
            MARKET_DATA_CACHE[symbol] = data
            return data
    except Exception:
        return None

def count_open_positions():
    res = okx_request("GET", "/api/v5/account/positions")
    if res and res.get('code') == '0' and res.get('data'):
        return len([p for p in res['data'] if p['pos'] != '0'])
    return 0

def check_existing_position(symbol):
    res = okx_request("GET", f"/api/v5/account/positions?instId={symbol}")
    if res and res.get('code') == '0' and res.get('data'):
        for pos in res['data']:
            if pos['pos'] != '0': return pos['posSide']
    return None

# ==============================================================================
# ========== 3. LOGIC SWING HIGH/LOW (5-5) & KHÁNG CỰ HỖ TRỢ ==========
# ==============================================================================

def find_confirmed_swings(df, lookback=100):
    sub_df = df.iloc[-(lookback + 11):-1].reset_index(drop=True)
    
    swing_highs = []
    swing_lows = []

    for i in range(5, len(sub_df) - 5):
        current_h = sub_df.iloc[i]['h']
        current_l = sub_df.iloc[i]['l']
        
        # Check Swing High (Đỉnh)
        if all(current_h > sub_df.iloc[i-j]['h'] for j in range(1, 6)) and \
           all(current_h > sub_df.iloc[i+j]['h'] for j in range(1, 6)):
            swing_highs.append(current_h)

        # Check Swing Low (Đáy)
        if all(current_l < sub_df.iloc[i-j]['l'] for j in range(1, 6)) and \
           all(current_l < sub_df.iloc[i+j]['l'] for j in range(1, 6)):
            swing_lows.append(current_l)
            
    return swing_highs, swing_lows

def is_near_resistance(df, side):
    """Kiểm tra giá hiện tại có đâm vào vùng Đỉnh/Đáy Swing 5-5 hay không"""
    current_close = df.iloc[-2]['c']
    sh, sl = find_confirmed_swings(df, LOOKBACK_CANDLES)
    
    buffer = current_close * (BUFFER_PERCENT / 100)
    
    if side == "buy" and sh:
        max_res = max(sh)
        if current_close >= (max_res - buffer):
            return True, f"Gần đỉnh Swing High 5-5 ({max_res})"
            
    elif side == "sell" and sl:
        min_sup = min(sl)
        if current_close <= (min_sup + buffer):
            return True, f"Gần đáy Swing Low 5-5 ({min_sup})"
            
    return False, ""

# ==============================================================================
# ========== 4. THỰC THI VÀO LỆNH & TRAILING SL ==========
# ==============================================================================

def execute_smart_trade(symbol, side, entry_price, low, high):
    try:
        if check_existing_position(symbol):
            return None, "0", 0, 0, "Đã có vị thế"

        rules = get_market_rules(symbol)
        if not rules: return None, "0", 0, 0, "Lỗi rules"

        total_vol = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE
        raw_sz = total_vol / (entry_price * rules['ctVal'])
        size = math.floor(raw_sz / rules['lotSz']) * rules['lotSz']
        if size < rules['minSz']: size = rules['minSz']
        sz_str = format(size, 'f').rstrip('0').rstrip('.')

        pos_side = "long" if side == "buy" else "short"
        
        # Stop Loss Offset 0.2% từ râu nến
        if side == "buy":
            sl = round(low * (1 - 0.002), rules['prec'])
        else:
            sl = round(high * (1 + 0.002), rules['prec'])

        risk = abs(entry_price - sl)
        tp = round(entry_price + (risk * 2), rules['prec']) if side == "buy" else round(entry_price - (risk * 2), rules['prec'])

        # Cài đặt đòn bẩy
        okx_request("POST", "/api/v5/account/set-leverage", {
            "instId": symbol, "lever": str(GLOBAL_LEVERAGE), "mgnMode": "isolated", "posSide": pos_side
        })

        # Đặt lệnh Market + TP/SL kèm theo
        body = {
            "instId": symbol, "tdMode": "isolated", "side": side, "posSide": pos_side,
            "ordType": "market", "sz": sz_str,
            "attachAlgoOrds": [
                {"attachAlgoOrdType": "sl", "slTriggerPx": str(sl), "slOrdPx": "-1"},
                {"attachAlgoOrdType": "tp", "tpTriggerPx": str(tp), "tpOrdPx": "-1"}
            ]
        }
        res = okx_request("POST", "/api/v5/trade/order", body)
        return res, sz_str, sl, tp, res.get('msg') if res and res.get('code') != '0' else ""
    except Exception as e:
        return None, "0", 0, 0, str(e)

def manage_trailing_sl():
    """Tự động dời SL về Entry (hòa vốn) hoặc RR1 khi giá chạy tốt"""
    try:
        pos_res = okx_request("GET", "/api/v5/account/positions")
        if not pos_res or pos_res.get('code') != '0': return
        for pos in pos_res.get('data', []):
            if pos['pos'] == '0': continue
            sym, entry_px, pos_side = pos['instId'], float(pos['avgPx']), pos['posSide']
            
            c_res = requests.get(f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={sym}&bar={TIMEFRAME}&limit=5").json()
            if not c_res.get('data'): continue
            last_close = float(c_res['data'][1][4])

            algo_res = okx_request("GET", f"/api/v5/trade/orders-algo?instId={sym}&ordType=conditional")
            current_sl, algo_id = 0, ""
            for algo in algo_res.get('data', []):
                if algo.get('slTriggerPx'):
                    current_sl, algo_id = float(algo['slTriggerPx']), algo['algoId']
                    break
            
            if not algo_id: continue
            risk = abs(entry_px - current_sl)
            rr1 = entry_px + risk if pos_side == 'long' else entry_px - risk
            rr2 = entry_px + (risk * 2) if pos_side == 'long' else entry_px - (risk * 2)
            
            rules = get_market_rules(sym)
            if not rules: continue
            prec = rules['prec']

            new_sl = None
            if pos_side == 'long':
                if last_close >= rr2 and current_sl < rr1: new_sl = round(rr1, prec)
                elif last_close >= rr1 and current_sl < entry_px: new_sl = round(entry_px, prec)
            else:
                if last_close <= rr2 and current_sl > rr1: new_sl = round(rr1, prec)
                elif last_close <= rr1 and current_sl > entry_px: new_sl = round(entry_px, prec)

            if new_sl:
                okx_request("POST", "/api/v5/trade/amend-algos", {
                    "instId": sym, "algoId": algo_id, "newSlTriggerPx": str(new_sl)
                })
                print(f"🛡️ {sym} Trail SL -> {new_sl}")
    except Exception:
        pass

# ==============================================================================
# ========== 5. QUÉT THỊ TRƯỜNG & LOG HỆ THỐNG ==========
# ==============================================================================

def run_market_scan():
    now_vn = datetime.now(VIETNAM_TZ).strftime("%H:%M:%S")
    print(f"\n🚀 [{now_vn}] --- BẮT ĐẦU CHU KỲ QUÉT {TIMEFRAME} ---")
    
    open_count = count_open_positions()
    if open_count >= MAX_OPEN_POSITIONS:
        print(f"🛑 Đã đạt giới hạn {MAX_OPEN_POSITIONS} lệnh. Dừng quét.")
        return

    coins_checked = 0
    signals_found = 0

    for sym, cfg in SYMBOL_CONFIGS.items():
        if not cfg.get("Active"): continue
        try:
            url = f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={sym}&bar={TIMEFRAME}&limit=150"
            resp = requests.get(url, timeout=10).json()
            data = resp.get('data', [])
            if not data: continue
            
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            df[['o','h','l','c']] = df[['o','h','l','c']].astype(float)
            df = df.sort_values('ts').reset_index(drop=True)
            df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
            
            s = df.iloc[-2] # Nến tín hiệu vừa đóng

            max_oc, min_oc = max(s['o'], s['c']), min(s['o'], s['c'])
            up_wick = ((s['h'] - max_oc) / max_oc) * 100
            lo_wick = ((min_oc - s['l']) / min_oc) * 100
            
            side = None
            if (s['c'] > s['o']) and (s['c'] > s['ema20']) and (lo_wick >= cfg['X']) and (up_wick <= cfg['Y']): 
                side = "buy"
            elif (s['c'] < s['o']) and (s['c'] < s['ema20']) and (up_wick >= cfg['X']) and (lo_wick <= cfg['Y']): 
                side = "sell"

            if side:
                signals_found += 1
                # Kiểm tra bộ lọc Swing 5-5 (100 nến)
                is_blocked, reason = is_near_resistance(df, side)
                if is_blocked:
                    msg = f"⚠️ {sym}: Tín hiệu đẹp nhưng bị chặn bởi Swing. Lý do: {reason}"
                    print(f"   {msg}")
                    # GỬI CẢNH BÁO SLACK KHI BỊ CHẶN BỞI SWING
                    if SLACK_WEBHOOK_URL:
                        requests.post(SLACK_WEBHOOK_URL, json={"text": msg})
                    continue

                # Vào lệnh
                res, sz, sl, tp, err = execute_smart_trade(sym, side, s['c'], s['l'], s['h'])
                
                status_msg = ""
                if res and res.get('code') == '0':
                    status_msg = f"✅ KHỚP LỆNH: {side.upper()} {sym} | Size: {sz} | SL: {sl} | TP: {tp}"
                    open_count += 1
                else:
                    status_msg = f"❌ LỖI VÀO LỆNH {sym}: {err if err else 'Fail'}"
                
                print(f"   {status_msg}")
                if SLACK_WEBHOOK_URL:
                    requests.post(SLACK_WEBHOOK_URL, json={"text": status_msg})
                
                if open_count >= MAX_OPEN_POSITIONS:
                    print("🛑 Đã đạt giới hạn lệnh tối đa trong chu kỳ này.")
                    break
            
            coins_checked += 1
        except Exception as e:
            print(f"❌ Lỗi quét {sym}: {e}")
            
    print(f"🏁 [{now_vn}] Kết thúc quét. Đã check {coins_checked} coin. Tìm thấy {signals_found} tín hiệu.")

# ==============================================================================
# ========== 6. LUỒNG CHẠY NGẦM & GRADIO UI ==========
# ==============================================================================

def main_loop():
    global LAST_PROCESSED_MINUTE
    print("🤖 Bot đang ở chế độ chờ kích hoạt...")
    while True:
        if GLOBAL_RUNNING:
            now = datetime.now(VIETNAM_TZ)
            if now.minute % 5 == 0 and now.minute != LAST_PROCESSED_MINUTE:
                # Đợi 5 giây để nến sàn đóng hẳn
                time.sleep(5)
                run_market_scan()
                manage_trailing_sl()
                LAST_PROCESSED_MINUTE = now.minute
        time.sleep(1)

# Chạy main_loop trong một Thread riêng
threading.Thread(target=main_loop, daemon=True).start()

def update_settings(amt, lev, run):
    global TRADE_AMOUNT_USDT, GLOBAL_LEVERAGE, GLOBAL_RUNNING
    TRADE_AMOUNT_USDT = float(amt)
    GLOBAL_LEVERAGE = int(lev)
    GLOBAL_RUNNING = run
    
    mode = "🟢 ĐANG CHẠY" if run else "🔴 ĐANG DỪNG"
    return f"{mode} | Vốn: {amt} USDT | Lever: x{lev} | Max lệnh: {MAX_OPEN_POSITIONS} | Swing: 5-5"

with gr.Blocks(title="OKX Master Bot V6") as demo:
    gr.Markdown("# 🤖 OKX Master Bot (50 Coins - Swing 5/5 - Range Filter)")
    gr.Markdown("Bot quét 50 cặp coin mỗi 5 phút. Vào lệnh dựa trên râu nến và EMA20, báo cáo cả trường hợp Swing chặn lệnh.")
    
    with gr.Row():
        num_amt = gr.Number(label="Số tiền vào mỗi lệnh (USDT)", value=10)
        num_lev = gr.Number(label="Mức đòn bẩy", value=25)
        chk_run = gr.Checkbox(label="KÍCH HOẠT BOT")
        
    btn = gr.Button("LƯU CẤU HÌNH & CHẠY", variant="primary")
    out = gr.Textbox(label="Trạng thái hệ thống", interactive=False)
    
    btn.click(update_settings, [num_amt, num_lev, chk_run], out)

if __name__ == "__main__":
    # Launch Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860)

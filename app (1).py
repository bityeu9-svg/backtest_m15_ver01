import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
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
# ========== CẤU HÌNH & BIẾN TOÀN CỤC ==========
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
# ========== HÀM TIỆN ÍCH API ==========
# ==============================================================================

def okx_request(method, endpoint, body=None):
    try:
        ts = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        body_str = json.dumps(body) if body else ""
        message = ts + method + endpoint + body_str
        mac = hmac.new(bytes(OKX_SECRET_KEY, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        headers = {
            'OK-ACCESS-KEY': OKX_API_KEY, 'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': ts, 'OK-ACCESS-PASSPHRASE': OKX_PASSPHRASE,
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
    except Exception as e:
        print(f"⚠️ Rules Error {symbol}: {e}")
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
# ========== LOGIC SWING HIGH/LOW & KHÁNG CỰ HỖ TRỢ ==========
# ==============================================================================

def find_confirmed_swings(df, lookback=100):
    """Tìm Swing High/Low với 5 nến trái và 5 nến phải trong cửa sổ Lookback"""
    # Lấy dữ liệu lịch sử (không bao gồm nến tín hiệu và nến đang chạy)
    # Cần tối thiểu lookback + 10 nến để quét
    sub_df = df.iloc[-(lookback + 10):-1].reset_index(drop=True)
    
    swing_highs = []
    swing_lows = []

    # Quét từ index 5 đến len-5 để đảm bảo có đủ 5 nến 2 bên
    for i in range(5, len(sub_df) - 5):
        current_h = sub_df.iloc[i]['h']
        current_l = sub_df.iloc[i]['l']
        
        # Check Swing High (Đỉnh)
        is_high = True
        for j in range(1, 6):
            if current_h <= sub_df.iloc[i-j]['h'] or current_h <= sub_df.iloc[i+j]['h']:
                is_high = False
                break
        if is_high: swing_highs.append(current_h)

        # Check Swing Low (Đáy)
        is_low = True
        for j in range(1, 6):
            if current_l >= sub_df.iloc[i-j]['l'] or current_l >= sub_df.iloc[i+j]['l']:
                is_low = False
                break
        if is_low: swing_lows.append(current_l)
            
    return swing_highs, swing_lows

def is_near_resistance(df, side):
    """Kiểm tra xem giá đóng nến có đang đâm vào vùng Đỉnh/Đáy Swing 5-5 không"""
    current_close = df.iloc[-2]['c']
    swing_highs, swing_lows = find_confirmed_swings(df, LOOKBACK_CANDLES)
    
    buffer = current_close * (BUFFER_PERCENT / 100)
    
    if side == "buy" and swing_highs:
        max_resistance = max(swing_highs)
        if current_close >= (max_resistance - buffer):
            return True, f"Gần đỉnh xác nhận (Swing High 5-5): {max_resistance}"
            
    elif side == "sell" and swing_lows:
        min_support = min(swing_lows)
        if current_close <= (min_support + buffer):
            return True, f"Gần đáy xác nhận (Swing Low 5-5): {min_support}"
            
    return False, ""

# ==============================================================================
# ========== VÀO LỆNH & QUẢN LÝ LỆNH ==========
# ==============================================================================

def execute_smart_trade(symbol, side, entry_price, low, high):
    try:
        existing_pos = check_existing_position(symbol)
        if existing_pos:
            return None, "0", 0, 0, f"Đã có vị thế {existing_pos}"

        rules = get_market_rules(symbol)
        if not rules: return None, "0", 0, 0, "Không lấy được rules sàn"

        total_notional_usdt = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE
        raw_sz = total_notional_usdt / (entry_price * rules['ctVal'])
        size = math.floor(raw_sz / rules['lotSz']) * rules['lotSz']
        if size < rules['minSz']: size = rules['minSz']
        sz_str = format(size, 'f').rstrip('0').rstrip('.')

        pos_side = "long" if side == "buy" else "short"
        
        # Stop Loss Offset 0.2%
        if side == "buy":
            sl = round(low * (1 - 0.002), rules['prec'])
        else:
            sl = round(high * (1 + 0.002), rules['prec'])

        risk = abs(entry_price - sl)
        tp = round(entry_price + (risk * 2), rules['prec']) if side == "buy" else round(entry_price - (risk * 2), rules['prec'])

        # Set Leverage
        okx_request("POST", "/api/v5/account/set-leverage", {
            "instId": symbol, "lever": str(GLOBAL_LEVERAGE), "mgnMode": "isolated", "posSide": pos_side
        })

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
            rr2 = entry_px + risk*2 if pos_side == 'long' else entry_px - risk*2
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
                okx_request("POST", "/api/v5/trade/amend-algos", {"instId": sym, "algoId": algo_id, "newSlTriggerPx": str(new_sl)})
    except: pass

# ==============================================================================
# ========== VÒNG LẶP CHÍNH (SCANNER) ==========
# ==============================================================================

def run_market_scan():
    # 1. KIỂM TRA GIỚI HẠN 6 LỆNH
    open_count = count_open_positions()
    if open_count >= MAX_OPEN_POSITIONS:
        print(f"🛑 Đã đạt giới hạn {MAX_OPEN_POSITIONS} vị thế. Dừng quét.")
        return

    for sym, cfg in SYMBOL_CONFIGS.items():
        if not cfg.get("Active"): continue
        try:
            # Lấy 150 nến để đủ dữ liệu Swing 5-5 + Lookback 100
            url = f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={sym}&bar={TIMEFRAME}&limit=150"
            resp = requests.get(url, timeout=10).json()
            data = resp.get('data', [])
            if not data: continue
            
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQuote', 'confirm'])
            df[['o', 'h', 'l', 'c']] = df[['o', 'h', 'l', 'c']].astype(float)
            df = df.sort_values('ts').reset_index(drop=True)
            df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
            
            s = df.iloc[-2]      # Nến tín hiệu vừa đóng
            prev_s = df.iloc[-3]  # Nến trước đó
            
            # CHIỀU DÀI NẾN (Range)
            current_range = s['h'] - s['l']
            prev_range = prev_s['h'] - prev_s['l']
            
            max_oc, min_oc = max(s['o'], s['c']), min(s['o'], s['c'])
            up_wick, lo_wick = ((s['h'] - max_oc) / max_oc) * 100, ((min_oc - s['l']) / min_oc) * 100
            
            side = None
            # Điều kiện: Biên độ nến hiện tại > nến trước
            if current_range > prev_range:
                if (s['c'] > s['o']) and (s['c'] > s['ema20']) and (lo_wick >= cfg['X']) and (up_wick <= cfg['Y']): 
                    side = "buy"
                elif (s['c'] < s['o']) and (s['c'] < s['ema20']) and (up_wick >= cfg['X']) and (lo_wick <= cfg['Y']): 
                    side = "sell"

            if side:
                # 2. KIỂM TRA KHÁNG CỰ HỖ TRỢ SWING 5-5
                is_blocked, reason = is_near_resistance(df, side)
                if is_blocked:
                    print(f"⚠️ {sym}: {reason}")
                    continue

                res, sz, sl, tp, err = execute_smart_trade(sym, side, s['c'], s['l'], s['h'])
                
                total_vol = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE
                if res and res.get('code') == '0':
                    msg = f"✅ KHỚP LỆNH | {side.upper()} {sym}\nVol: {total_vol} USDT | SL: {sl} | TP: {tp}"
                else:
                    msg = f"❌ THẤT BẠI: {err if err else 'Fail'} | {side.upper()} {sym}\nSize: {sz} | SL: {sl} | TP: {tp}"
                
                print(msg)
                if SLACK_WEBHOOK_URL:
                    requests.post(SLACK_WEBHOOK_URL, json={"text": msg})
                
                # Sau khi vào 1 lệnh thành công, update lại số lượng để tránh vượt limit trong cùng 1 vòng lặp
                open_count += 1
                if open_count >= MAX_OPEN_POSITIONS:
                    print("🛑 Đã đạt giới hạn tối đa sau lệnh này.")
                    break
        except Exception as e:
            print(f"Lỗi scan {sym}: {e}")

def main_loop():
    global LAST_PROCESSED_MINUTE
    while True:
        if GLOBAL_RUNNING:
            now = datetime.now(VIETNAM_TZ)
            if now.minute % 5 == 0 and now.minute != LAST_PROCESSED_MINUTE:
                time.sleep(5)
                run_market_scan()
                manage_trailing_sl()
                LAST_PROCESSED_MINUTE = now.minute
        time.sleep(1)

threading.Thread(target=main_loop, daemon=True).start()

# ==============================================================================
# ========== UI GRADIO ==========
# ==============================================================================

def update_settings(amt, lev, run):
    global TRADE_AMOUNT_USDT, GLOBAL_LEVERAGE, GLOBAL_RUNNING
    TRADE_AMOUNT_USDT, GLOBAL_LEVERAGE, GLOBAL_RUNNING = float(amt), int(lev), run
    status = "🟢 ĐANG CHẠY" if run else "🔴 ĐANG DỪNG"
    return f"{status} | Tổng coin: 50 | Max lệnh: {MAX_OPEN_POSITIONS} | Lookback: 100 nến | Swing: 5-5"

with gr.Blocks(title="OKX Pro Bot V6") as demo:
    gr.Markdown("# 🤖 OKX Pro Bot (50 Coins + Swing 5-5 + Range Filter)")
    with gr.Row():
        num_amt = gr.Number(label="Vốn mỗi lệnh (USDT)", value=10)
        num_lev = gr.Number(label="Đòn bẩy", value=25)
        chk_run = gr.Checkbox(label="Kích hoạt Bot")
    
    btn = gr.Button("LƯU & KÍCH HOẠT", variant="primary")
    out = gr.Textbox(label="Trạng thái hệ thống", interactive=False)
    
    btn.click(update_settings, [num_amt, num_lev, chk_run], out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

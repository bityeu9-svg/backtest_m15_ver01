import requests
import pandas as pd
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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
# ========== CẤU HÌNH & DANH SÁCH COIN (FULL 68 MÃ) ==========
# ==============================================================================
if os.path.exists(".env"):
    load_dotenv(".env")

OKX_API_KEY = os.environ.get("OKX_API_KEY")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE")
OKX_BASE_URL = "https://www.okx.com"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

BTC_ETH_GOLD = ["BTC-USDT-SWAP", "XAU-USDT"]
ALTS_STANDARD = ["ETH-USDT-SWAP", "SOL-USDT-SWAP", "BNB-USDT-SWAP", "OKB-USDT-SWAP", "ADA-USDT-SWAP", "XRP-USDT-SWAP", "DOT-USDT-SWAP", "AVAX-USDT-SWAP", "TRX-USDT-SWAP", "LTC-USDT-SWAP", "LINK-USDT-SWAP", "NEAR-USDT-SWAP", "APT-USDT-SWAP", "SUI-USDT-SWAP", "ALGO-USDT-SWAP", "ATOM-USDT-SWAP", "BCH-USDT-SWAP"]

ALTS_HIGH_VOL = [
    "OP-USDT-SWAP", "ARB-USDT-SWAP", "METIS-USDT-SWAP", "IMX-USDT-SWAP", "MANTA-USDT-SWAP", 
    "STRK-USDT-SWAP", "TIA-USDT-SWAP", "SEI-USDT-SWAP", "PEPE-USDT-SWAP", "DOGE-USDT-SWAP", 
    "SHIB-USDT-SWAP", "FLOKI-USDT-SWAP", "BONK-USDT-SWAP", "FET-USDT-SWAP", "RNDR-USDT-SWAP", 
    "GRT-USDT-SWAP", "ORDI-USDT-SWAP", "STX-USDT-SWAP", "FIL-USDT-SWAP", "LDO-USDT-SWAP",
    "WIF-USDT-SWAP", "JUP-USDT-SWAP", "DYM-USDT-SWAP", "PYTH-USDT-SWAP", "MEME-USDT-SWAP",
    "PIXEL-USDT-SWAP", "ALT-USDT-SWAP", "AXL-USDT-SWAP", "MYRO-USDT-SWAP", "1000SATS-USDT-SWAP",
    "ENS-USDT-SWAP", "PENDLE-USDT-SWAP", "MAV-USDT-SWAP", "MAGIC-USDT-SWAP", "GALA-USDT-SWAP",
    "MKR-USDT-SWAP", "COMP-USDT-SWAP", "AAVE-USDT-SWAP", "RUNE-USDT-SWAP", "EGLD-USDT-SWAP",
    "ICP-USDT-SWAP", "FLOW-USDT-SWAP", "THETA-USDT-SWAP", "SAND-USDT-SWAP", "MANA-USDT-SWAP",
    "AXS-USDT-SWAP", "CHZ-USDT-SWAP", "DYDX-USDT-SWAP", "ZIL-USDT-SWAP", "KAVA-USDT-SWAP",
    "1INCH-USDT-SWAP", "CRV-USDT-SWAP", "SNX-USDT-SWAP", "GMX-USDT-SWAP", "LRC-USDT-SWAP",
    "ZRX-USDT-SWAP", "BTT-USDT-SWAP", "HOT-USDT-SWAP", "ENJ-USDT-SWAP", "IOTA-USDT-SWAP",
    "ANKR-USDT-SWAP", "T-USDT-SWAP", "FLM-USDT-SWAP", "ONT-USDT-SWAP", "IOST-USDT-SWAP",
    "WAXP-USDT-SWAP", "GLMR-USDT-SWAP", "MOVR-USDT-SWAP", "SATS-USDT-SWAP", "ZK-USDT-SWAP"
]

TIMEFRAME = "5m"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

CONFIG = {
    "RUNNING": False,
    "AMOUNT": 10.0,
    "LEV_MAIN": 25,
    "LEV_ALTS": 15,
    "LEV_HIGHVOL": 10,
    "WICK_MAIN": 0.13,
    "WICK_ALTS": 0.25,
    "WICK_NEW": 0.4,
    "WICK_SUB": 0.05,
    "RR_RATIO": 1.5,
    "SL_BUFFER_PCT": 0.15,
    "MAX_POSITIONS": 5,
    "LAST_PROCESSED_MIN": -1
}

MARKET_DATA_CACHE = {}

# ==============================================================================
# ========== HÀM API HỖ TRỢ ==========
# ==============================================================================

def send_slack(msg):
    if SLACK_WEBHOOK_URL:
        try: requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=5)
        except: pass

def okx_request(method, endpoint, body=None):
    try:
        ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z")
        body_str = json.dumps(body) if body else ""
        message = ts + method + endpoint + body_str
        mac = hmac.new(bytes(OKX_SECRET_KEY, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        headers = {
            'OK-ACCESS-KEY': OKX_API_KEY, 'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': ts, 'OK-ACCESS-PASSPHRASE': OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        res = requests.request(method, OKX_BASE_URL + endpoint, headers=headers, data=body_str, timeout=15)
        return res.json()
    except Exception as e: return {"code": "-1", "msg": str(e)}

def get_all_open_positions():
    res = okx_request("GET", "/api/v5/account/positions")
    if res and res.get('code') == '0':
        return [p['instId'] for p in res.get('data', []) if float(p['pos']) != 0]
    return []

def get_market_rules(symbol):
    if symbol in MARKET_DATA_CACHE: return MARKET_DATA_CACHE[symbol]
    try:
        url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP&instId={symbol}"
        res = requests.get(url, timeout=10).json()
        if res.get('code') == '0' and res['data']:
            inst = res['data'][0]
            prec = len(inst['tickSz'].split('.')[-1]) if '.' in inst['tickSz'] else 0
            data = {"lotSz": float(inst['lotSz']), "tickSz": float(inst['tickSz']), 
                    "prec": prec, "ctVal": float(inst.get('ctVal', 1)), "minSz": float(inst['minSz'])}
            MARKET_DATA_CACHE[symbol] = data
            return data
    except: return None

# ==============================================================================
# ========== CORE LOGIC (VẬN HÀNH THEO APP.PY) ==========
# ==============================================================================

def calculate_wick_pct(row):
    max_oc, min_oc = max(row['o'], row['c']), min(row['o'], row['c'])
    up_wick = ((row['h'] - max_oc) / max_oc) * 100 if max_oc > 0 else 0
    lo_wick = ((min_oc - row['l']) / min_oc) * 100 if min_oc > 0 else 0
    return up_wick, lo_wick, row['c'] > row['o'], row['c'] < row['o']

def execute_trade(symbol, side, n0_data, n_curr_data):
    try:
        rules = get_market_rules(symbol)
        if not rules: return
        
        # 1. Chọn đòn bẩy theo nhóm
        if symbol in BTC_ETH_GOLD: target_lev = int(CONFIG["LEV_MAIN"])
        elif symbol in ALTS_STANDARD: target_lev = int(CONFIG["LEV_ALTS"])
        else: target_lev = int(CONFIG["LEV_HIGHVOL"])
        
        okx_request("POST", "/api/v5/account/set-leverage", {
            "instId": symbol, "lever": str(target_lev), "mgnMode": "isolated", "posSide": side
        })
        
        # 2. Logic Entry khớp 100% app.py gốc
        curr_close, curr_high, curr_low = n_curr_data['c'], n_curr_data['h'], n_curr_data['l']
        if side == "long":
            entry = curr_close if curr_close >= curr_high else min(n0_data['o'], n0_data['c'])
        else:
            entry = curr_close if curr_close <= curr_low else max(n0_data['o'], n0_data['c'])
        
        # 3. Tính toán SL/TP với độ chính xác chuẩn sàn
        buffer = CONFIG["SL_BUFFER_PCT"] / 100
        sl_price = n0_data['l'] * (1 - buffer) if side == "long" else n0_data['h'] * (1 + buffer)
        risk = abs(entry - sl_price)
        if risk == 0: return
        
        tp_price = (entry + risk * CONFIG["RR_RATIO"]) if side == "long" else (entry - risk * CONFIG["RR_RATIO"])
        
        # Format string tránh lỗi Code 1
        fmt_entry = "{:.{}f}".format(entry, rules['prec'])
        fmt_sl = "{:.{}f}".format(sl_price, rules['prec'])
        fmt_tp = "{:.{}f}".format(tp_price, rules['prec'])

        # 4. Tính Size
        size = math.floor((CONFIG["AMOUNT"] * target_lev / (entry * rules['ctVal'])) / rules['lotSz']) * rules['lotSz']
        if size < rules['minSz']: return

        body = {
            "instId": symbol, "tdMode": "isolated", "side": "buy" if side == "long" else "sell", 
            "posSide": side, "ordType": "limit", "px": fmt_entry, "sz": str(size),
            "attachAlgoOrds": [
                {"attachAlgoOrdType": "sl", "slTriggerPx": fmt_sl, "slOrdPx": "-1"},
                {"attachAlgoOrdType": "tp", "tpTriggerPx": fmt_tp, "tpOrdPx": "-1"}
            ]
        }
        res = okx_request("POST", "/api/v5/trade/order", body)
        
        if res and res.get('code') == '0':
            msg = f"🚀 ✅ {side.upper()} {symbol} (x{target_lev}) | Entry: {fmt_entry} | SL: {fmt_sl}"
            send_slack(msg)
            print(f"   └─ ✅ {msg}", flush=True)
        else:
            err_msg = res.get('msg', 'Unknown Error')
            msg = f"❌ ĐẶT LỆNH LỖI: {symbol} | {side.upper()} | {err_msg} (Code: {res.get('code')})"
            send_slack(msg)
            print(f"   └─ {msg}", flush=True)

    except Exception as e:
        send_slack(f"🚨 CRITICAL ERROR: {symbol} | {str(e)}")

def scan_symbol(symbol, X, Y, open_symbols, current_count):
    try:
        url = f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={symbol}&bar={TIMEFRAME}&limit=5"
        resp = requests.get(url, timeout=10).json()
        if not resp or not resp.get('data'): return current_count

        df = pd.DataFrame(resp['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df[['o','h','l','c']] = df[['o','h','l','c']].astype(float)
        
        n_curr, n0, n1 = df.iloc[0], df.iloc[1], df.iloc[2]
        up0, lo0, g0, r0 = calculate_wick_pct(n0)
        up1, lo1, g1, r1 = calculate_wick_pct(n1)

        l_pass = (lo0 >= X and up0 <= Y) and (lo1 >= X and up1 <= Y) and (g0 or g1)
        s_pass = (up0 >= X and lo0 <= Y) and (up1 >= X and lo1 <= Y) and (r0 or r1)

        status_icon = "🔵" if l_pass else "🔴" if s_pass else "⚪"
        print(f"{status_icon} {symbol:15} | N0: U{up0:.2f}% L{lo0:.2f}% | N1: U{up1:.2f}% L{lo1:.2f}%", flush=True)

        if l_pass or s_pass:
            if "XAU-USDT" in symbol: return current_count
            if current_count >= CONFIG["MAX_POSITIONS"] and symbol not in open_symbols: return current_count
            
            execute_trade(symbol, "long" if l_pass else "short", n0, n_curr)
            if symbol not in open_symbols:
                return current_count + 1
        return current_count
    except: return current_count

def run_market_scan():
    open_symbols = get_all_open_positions()
    current_count = len(open_symbols)
    Y = CONFIG["WICK_SUB"]
    now_str = datetime.now(VIETNAM_TZ).strftime("%H:%M:%S")
    print(f"\n🕒 SCAN TUẦN TỰ {now_str} | Open: {current_count}/{CONFIG['MAX_POSITIONS']}")

    for s in BTC_ETH_GOLD:
        current_count = scan_symbol(s, CONFIG["WICK_MAIN"], Y, open_symbols, current_count)
    for s in ALTS_STANDARD:
        current_count = scan_symbol(s, CONFIG["WICK_ALTS"], Y, open_symbols, current_count)
    for s in ALTS_HIGH_VOL:
        current_count = scan_symbol(s, CONFIG["WICK_NEW"], Y, open_symbols, current_count)

# ==============================================================================
# ========== UI & LOOP ==========
# ==============================================================================

def update_ui(amt, lev_m, lev_a, lev_h, m_main, m_alt, m_new, sub, rr, buf, run):
    CONFIG.update({
        "AMOUNT": amt, "LEV_MAIN": lev_m, "LEV_ALTS": lev_a, "LEV_HIGHVOL": lev_h,
        "WICK_MAIN": m_main, "WICK_ALTS": m_alt, "WICK_NEW": m_new, "WICK_SUB": sub, 
        "RR_RATIO": rr, "SL_BUFFER_PCT": buf, "RUNNING": run
    })
    return f"Bot: {'ON' if run else 'OFF'} | Logic: app.py (Synced)"

def main_loop():
    while True:
        if CONFIG["RUNNING"]:
            now = datetime.now(VIETNAM_TZ)
            if now.minute % 5 == 0 and now.minute != CONFIG["LAST_PROCESSED_MIN"]:
                time.sleep(7)
                run_market_scan()
                CONFIG["LAST_PROCESSED_MIN"] = now.minute
        time.sleep(1)

threading.Thread(target=main_loop, daemon=True).start()

with gr.Blocks(title="OKX Bot V18.0 Professional") as demo:
    gr.Markdown("# 🤖 OKX Bot V18.0 - app.py Logic Synced")
    with gr.Tab("Cấu hình Giao dịch"):
        with gr.Row():
            n_amt = gr.Number(label="Vốn/Lệnh (USDT)", value=10)
            n_rr = gr.Number(label="R:R Ratio", value=1.5)
            n_buf = gr.Number(label="SL Buffer (%)", value=0.15)
        with gr.Row():
            lev_main = gr.Slider(1, 100, 25, step=1, label="Leverage BTC/ETH Group")
            lev_alts = gr.Slider(1, 100, 15, step=1, label="Leverage Alts Top Group")
            lev_highvol = gr.Slider(1, 100, 10, step=1, label="Leverage High Vol Group")
    with gr.Tab("Cấu hình Râu"):
        with gr.Row():
            n_main = gr.Number(label="Râu BTC/ETH", value=0.13)
            n_alt = gr.Number(label="Râu Alt Top", value=0.25)
            n_new = gr.Number(label="Râu High Vol", value=0.4)
            n_sub = gr.Number(label="Râu Phụ Max", value=0.05)
    c_run = gr.Checkbox(label="KÍCH HOẠT BOT")
    btn = gr.Button("LƯU & CHẠY", variant="primary")
    out = gr.Textbox(label="Status")
    btn.click(update_ui, [n_amt, lev_main, lev_alts, lev_highvol, n_main, n_alt, n_new, n_sub, n_rr, n_buf, c_run], out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

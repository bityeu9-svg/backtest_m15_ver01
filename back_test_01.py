import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import itertools
import gradio as gr

# --- CẤU HÌNH CỐ ĐỊNH ---
# SYMBOL mặc định sẽ được lấy từ UI
TIMEFRAME = "15m"
TAKER_FEE_RATE = 0.05 / 100

# ----------------- CÁC HÀM TIỆN ÍCH -----------------
def fetch_okx_data(symbol, timeframe, days):
    """
    Tải dữ liệu nến từ OKX và lưu cache dưới dạng CSV.
    Sử dụng CSV để tránh lỗi pyarrow/parquet trên local.
    """
    # Làm sạch tên symbol để dùng làm tên file (tránh lỗi ký tự đặc biệt nếu có)
    safe_symbol = symbol.replace("-", "_")
    cache_filename = f"{safe_symbol}_{timeframe}_{days}d_candles_v4.csv"
    
    # 1. KIỂM TRA CACHE
    if os.path.exists(cache_filename):
        print(f"✅ [CACHE] Tìm thấy cache! Đang đọc file: {cache_filename}")
        try:
            df = pd.read_csv(cache_filename)
            # Convert timestamp từ string/số sang datetime chuẩn
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Xử lý timezone (nếu mất timezone khi lưu CSV)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Ho_Chi_Minh')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Ho_Chi_Minh')
            
            # --- LOG DATA CONTENT ---
            print(f"📊 [LOG] Dữ liệu Cache ({symbol}): {len(df)} nến. Từ {df.iloc[0]['timestamp']} đến {df.iloc[-1]['timestamp']}")
            print("-" * 30)
            return df
        except Exception as e:
            print(f"⚠️ Lỗi đọc cache ({e}), sẽ tải lại từ đầu.")

    # 2. TẢI MỚI TỪ API
    print(f"🌐 [OKX API] Đang tải dữ liệu {days} ngày cho {symbol} ({timeframe})...")
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    limit = 100
    
    tf_map = {'1m': 1, '5m': 5, '15m': 15, '1H': 60, '4H': 240, '1D': 1440}
    timeframe_minutes = tf_map.get(timeframe, 15)
    
    total_candles_needed = int(days * 24 * (60 // timeframe_minutes))
    # Tăng buffer loop để đảm bảo lấy đủ dữ liệu
    iterations = (total_candles_needed // limit) + 10
    
    for i in range(iterations):
        try:
            url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&limit={limit}&after={end_time}"
            response = requests.get(url, timeout=10)
            data = response.json().get('data', [])
            
            if not data: 
                print("   ⚠️ Sàn không trả về thêm dữ liệu.")
                break
                
            all_data.extend(data)
            end_time = data[-1][0] # Cập nhật thời gian cho lần gọi sau
            
            if len(all_data) >= total_candles_needed: 
                print("   ✅ Đã tải đủ số lượng nến yêu cầu.")
                break
            
            # Nghỉ ngắn để tránh bị block IP
            time.sleep(0.05) 
        except Exception as e: 
            print(f"   ❌ Lỗi kết nối API: {e}")
            break
        
    if not all_data: 
        print(f"❌ Không tải được dữ liệu nào cho {symbol}.")
        return pd.DataFrame()
        
    # 3. XỬ LÝ DỮ LIỆU
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Ép kiểu dữ liệu sang số (Tránh lỗi FutureWarning và lỗi tính toán)
    cols_to_numeric = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col])
    
    # Convert Timestamp sang Datetime (Fix lỗi unit='ms' với string)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Ho_Chi_Minh')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # --- LOG DATA CONTENT ---
    print(f"📊 [LOG] Dữ liệu Mới từ OKX ({symbol}): {len(df)} nến")
    print("-" * 30)
    
    # Lưu file CSV (index=False để gọn, không cần pyarrow)
    df.to_csv(cache_filename, index=False)
    print(f"✅ Đã lưu dữ liệu vào: {cache_filename}")
    
    return df

# ----------------- CHIẾN LƯỢC CỐT LÕI -----------------
def backtest_strategy(df, params, position_size_usdt, leverage, candles_to_wait):
    """
    Hàm backtest logic:
    - X, Y: Râu nến
    - Z: Volume Ratio
    - K: Breakout Offset (%)
    - Exit: 1R (50%, dời SL về Entry), 2R (100%).
    """
    X = params['X_WICK_MAIN']
    Y = params['Y_WICK_OPPOSITE']
    Z = params['Z_VOL_RATIO']
    K = params['K_PRICE_OFFSET'] 
    
    if df.empty: return []
    trades = []
    
    # --- TÍNH TOÁN INDICATOR (VECTORIZATION) ---
    df = df.copy() # Avoid SettingWithCopyWarning
    df['prev_volume'] = df['volume'].shift(1)
    df['vol_ratio'] = df['volume'] / df['prev_volume']
    
    df['range'] = df['high'] - df['low']
    
    # Râu trên = High - Max(Open, Close)
    df['upper_wick_len'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Râu dưới = Min(Open, Close) - Low
    df['lower_wick_len'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # % Râu so với Range
    df['upper_wick_pct'] = np.where(df['range'] > 0, df['upper_wick_len'] / df['range'], 0)
    df['lower_wick_pct'] = np.where(df['range'] > 0, df['lower_wick_len'] / df['range'], 0)
    
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']

    # --- LOOP QUA TỪNG NẾN ---
    for i in range(1, len(df) - candles_to_wait - 1):
        
        # 1. TÌM TÍN HIỆU (SETUP)
        signal_type = None
        if df.iloc[i]['vol_ratio'] < Z: continue
        if df.iloc[i]['range'] == 0: continue

        # Long Setup
        if df.iloc[i]['is_green']:
            if (df.iloc[i]['lower_wick_pct'] >= X) and (df.iloc[i]['upper_wick_pct'] <= Y):
                signal_type = 'LONG'
        # Short Setup
        elif df.iloc[i]['is_red']:
            if (df.iloc[i]['upper_wick_pct'] >= X) and (df.iloc[i]['lower_wick_pct'] <= Y):
                signal_type = 'SHORT'
        
        if not signal_type: continue

        # 2. XÁC ĐỊNH ENTRY & SL (PLANNING)
        signal_high = df.iloc[i]['high']
        signal_low = df.iloc[i]['low']
        
        entry_trigger_price = 0
        stop_loss = 0
        
        if signal_type == 'LONG':
            # Buy Stop tại High + K%
            entry_trigger_price = signal_high * (1 + K/100)
            stop_loss = signal_low
            risk_per_unit = entry_trigger_price - stop_loss
        else: # SHORT
            # Sell Stop tại Low - K%
            entry_trigger_price = signal_low * (1 - K/100)
            stop_loss = signal_high
            risk_per_unit = stop_loss - entry_trigger_price
            
        if risk_per_unit <= 0: continue

        # 3. CHỜ KHỚP LỆNH (WAIT & VALIDATION)
        order_filled = False
        entry_idx = -1
        
        # Loop qua 5 cây nến tiếp theo để xem có khớp không
        for w in range(1, candles_to_wait + 1):
            idx_check = i + w
            if idx_check >= len(df): break
            
            curr_h = df.iloc[idx_check]['high']
            curr_l = df.iloc[idx_check]['low']
            
            # A. KIỂM TRA HUỶ LỆNH (INVALIDATION - SL Hit First)
            if signal_type == 'LONG' and curr_l <= stop_loss:
                order_filled = False; break
            if signal_type == 'SHORT' and curr_h >= stop_loss:
                order_filled = False; break
            
            # B. KIỂM TRA KHỚP LỆNH (TRIGGER)
            is_triggered = False
            if signal_type == 'LONG' and curr_h >= entry_trigger_price: is_triggered = True
            if signal_type == 'SHORT' and curr_l <= entry_trigger_price: is_triggered = True
            
            if is_triggered:
                order_filled = True
                entry_idx = idx_check
                break
        
        if not order_filled: continue

        # 4. QUẢN LÝ LỆNH (EXECUTION)
        tp1_price = entry_trigger_price + (1 * risk_per_unit) if signal_type == 'LONG' else entry_trigger_price - (1 * risk_per_unit)
        tp2_price = entry_trigger_price + (2 * risk_per_unit) if signal_type == 'LONG' else entry_trigger_price - (2 * risk_per_unit)
        
        position_remaining = 1.0 # 100% Volume
        tp1_hit = False
        pnl_accumulated = 0
        trade_result = 'UNKNOWN'
        
        current_sl = stop_loss
        exit_time = None

        for j in range(entry_idx, len(df)):
            curr_h = df.iloc[j]['high']
            curr_l = df.iloc[j]['low']
            
            # Check SL
            is_sl_hit = False
            if signal_type == 'LONG' and curr_l <= current_sl: is_sl_hit = True
            if signal_type == 'SHORT' and curr_h >= current_sl: is_sl_hit = True
            
            if is_sl_hit:
                # Lỗ/Hoà vốn phần còn lại
                diff = (current_sl - entry_trigger_price) if signal_type == 'LONG' else (entry_trigger_price - current_sl)
                pnl_accumulated += diff * position_remaining
                
                trade_result = 'LOSS' if not tp1_hit else 'WIN_PARTIAL'
                exit_time = df.iloc[j]['timestamp']
                break

            # Check TP1 (1R)
            if not tp1_hit:
                hit_tp1_cond = (signal_type == 'LONG' and curr_h >= tp1_price) or \
                               (signal_type == 'SHORT' and curr_l <= tp1_price)
                if hit_tp1_cond:
                    tp1_hit = True
                    pnl_accumulated += (1 * abs(risk_per_unit)) * 0.5
                    position_remaining = 0.5
                    current_sl = entry_trigger_price # Breakeven

            # Check TP2 (2R)
            if position_remaining > 0:
                hit_tp2_cond = (signal_type == 'LONG' and curr_h >= tp2_price) or \
                               (signal_type == 'SHORT' and curr_l <= tp2_price)
                if hit_tp2_cond:
                    pnl_accumulated += (2 * abs(risk_per_unit)) * 0.5
                    position_remaining = 0
                    trade_result = 'WIN_FULL'
                    exit_time = df.iloc[j]['timestamp']
                    break
        
        # Nếu hết dữ liệu
        if position_remaining > 0 and trade_result == 'UNKNOWN':
            last_close = df.iloc[-1]['close']
            diff = (last_close - entry_trigger_price) if signal_type == 'LONG' else (entry_trigger_price - last_close)
            pnl_accumulated += diff * position_remaining
            trade_result = 'ONGOING'
            exit_time = df.iloc[-1]['timestamp']

        # 5. TÍNH PNL RA USDT
        pos_qty = (position_size_usdt * leverage) / entry_trigger_price
        pnl_usdt_gross = pnl_accumulated * pos_qty
        
        # Phí (Vào + Ra)
        total_volume_usdt = position_size_usdt * leverage
        fee = total_volume_usdt * TAKER_FEE_RATE * 2
        
        net_pnl = pnl_usdt_gross - fee
        
        trades.append({
            'entry_time': df.iloc[entry_idx]['timestamp'],
            'exit_time': exit_time,
            'type': signal_type,
            'entry_price': entry_trigger_price,
            'risk_percent': (abs(risk_per_unit)/entry_trigger_price)*100,
            'result': trade_result,
            'pnl_usdt': net_pnl,
            'tp1_hit': tp1_hit,
            'params_used': f"X={params['X_WICK_MAIN']}, Y={params['Y_WICK_OPPOSITE']}, Z={params['Z_VOL_RATIO']}, K={params['K_PRICE_OFFSET']}"
        })

    return trades

# ----------------- HÀM TÌM PARAMS TỐT NHẤT -----------------
def find_best_params(train_df, position_size, leverage):
    """
    Chạy Grid Search trên tập dữ liệu train để tìm bộ tham số có PnL cao nhất.
    """
    if train_df.empty: return None

    # GRID NHỎ (Tùy chỉnh để chạy nhanh/chậm)
    X_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    Y_values = [0.03, 0.05, 0.07, 0.1]
    Z_values = [1.5, 2.0, 3.0]
    K_values = [0.02, 0.05]
    
    param_grid = {
        'X_WICK_MAIN': X_values,
        'Y_WICK_OPPOSITE': Y_values,
        'Z_VOL_RATIO': Z_values,
        'K_PRICE_OFFSET': K_values
    }
    
    combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    best_pnl = -np.inf
    best_param = None
    
    # Pre-calc indicators for speed optimization
    train_df = train_df.copy()
    train_df['range'] = train_df['high'] - train_df['low']
    train_df['prev_volume'] = train_df['volume'].shift(1)
    train_df['vol_ratio'] = train_df['volume'] / train_df['prev_volume']
    
    for p in combinations:
        trades = backtest_strategy(train_df, p, position_size, leverage, candles_to_wait=5)
        
        if not trades:
            current_pnl = 0
        else:
            current_pnl = sum(t['pnl_usdt'] for t in trades)
        
        if current_pnl > best_pnl:
            best_pnl = current_pnl
            best_param = p
            
    return best_param

# ----------------- WALK-FORWARD ALGORITHM (BLOCK) -----------------
def run_walk_forward(symbol, total_days_load, block_days, capital, leverage):
    """
    Block Walk-Forward Logic:
    1. Chia dữ liệu thành các Block có độ dài 'block_days'.
    2. Block i: Train -> Tìm Params.
    3. Block i+1: Test -> Dùng Params của Block i để trade.
    """
    # 1. Load Data
    df = fetch_okx_data(symbol, TIMEFRAME, int(total_days_load))
    if df.empty:
        yield "❌ Không có dữ liệu.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return

    # Lấy danh sách các ngày unique
    df['date'] = df['timestamp'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    block_size = int(block_days)
    
    # Cần tối thiểu 2 Blocks
    if len(unique_dates) < block_size * 2:
        yield f"❌ Dữ liệu không đủ. Cần ít nhất {block_size * 2} ngày (2 Blocks).", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return

    all_walk_forward_trades = []
    block_logs = []
    
    start_time = time.time()
    
    # Logic lặp theo từng Block (bước nhảy = block_size)
    # i bắt đầu từ block_size (Ví dụ 15).
    # i=15 -> Train [0:15] -> Test [15:30]
    # i=30 -> Train [15:30] -> Test [30:45]
    for i in range(block_size, len(unique_dates), block_size):
        
        # --- 1. XÁC ĐỊNH TRAINING BLOCK ---
        train_start_idx = i - block_size
        train_end_idx = i - 1
        
        train_start_date = unique_dates[train_start_idx]
        train_end_date = unique_dates[train_end_idx]
        
        # --- 2. XÁC ĐỊNH TESTING BLOCK ---
        test_start_idx = i
        test_end_idx = min(i + block_size - 1, len(unique_dates) - 1)
        
        test_start_date = unique_dates[test_start_idx]
        test_end_date = unique_dates[test_end_idx]
        
        # Nếu Test Block quá ngắn thì break
        if test_start_date > test_end_date:
            break

        yield f"🔄 Block {int(i/block_size)}: Train [{train_start_date} -> {train_end_date}] | Trade [{test_start_date} -> {test_end_date}]...", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # --- 3. OPTIMIZE TRÊN TRAIN BLOCK ---
        mask_train = (df['date'] >= train_start_date) & (df['date'] <= train_end_date)
        train_df = df.loc[mask_train].copy().reset_index(drop=True)
        
        best_params = find_best_params(train_df, capital, leverage)
        
        if not best_params:
            best_params = {'X_WICK_MAIN': 0.35, 'Y_WICK_OPPOSITE': 0.07, 'Z_VOL_RATIO': 2.0, 'K_PRICE_OFFSET': 0.02}
            note = "(Mặc định)"
        else:
            note = ""
        
        # --- 4. BACKTEST TRÊN TEST BLOCK ---
        mask_test = (df['date'] >= test_start_date) & (df['date'] <= test_end_date)
        test_df = df.loc[mask_test].copy().reset_index(drop=True)
        
        trades_block = backtest_strategy(test_df, best_params, capital, leverage, candles_to_wait=5)
        
        pnl_block = sum(t['pnl_usdt'] for t in trades_block)
        all_walk_forward_trades.extend(trades_block)
        
        block_logs.append({
            'Giai Đoạn Trade': f"{test_start_date} -> {test_end_date}",
            'Params Dùng (Từ Block Trước)': str(best_params).replace("'", "").replace("{", "").replace("}", "") + f" {note}",
            'Số Lệnh': len(trades_block),
            'PnL ($)': round(pnl_block, 2)
        })

    # 3. Tổng hợp kết quả
    total_pnl = sum(t['pnl_usdt'] for t in all_walk_forward_trades)
    final_capital = capital + total_pnl
    roi = (total_pnl / capital) * 100
    
    summary_data = [{
        'Vốn Ban Đầu': f"${capital}",
        'Vốn Cuối Cùng': f"${final_capital:.2f}",
        'Tổng Lợi Nhuận': f"${total_pnl:.2f}",
        'ROI (%)': f"{roi:.2f}%",
        'Tổng Lệnh': len(all_walk_forward_trades)
    }]
    summary_df = pd.DataFrame(summary_data)
    block_log_df = pd.DataFrame(block_logs)
    monthly_df, detailed_df = process_detailed_results(all_walk_forward_trades)
    
    elapsed = time.time() - start_time
    yield f"✅ Hoàn tất Block Walk-Forward! ({elapsed:.1f}s)", summary_df, block_log_df, detailed_df


# ----------------- HÀM CHO TAB 1 & 2 -----------------
def run_optimization_process(symbol, days, position_size, leverage):
    yield "🔄 Đang tải dữ liệu...", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    df = fetch_okx_data(symbol, TIMEFRAME, int(days))
    if df.empty:
        yield "❌ Lỗi Data", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
        return

    # Params Grid (Rút gọn)
    param_grid = {
        'X_WICK_MAIN': [0.1, 0.15, 0.2, 0.25, 0.3,0.35, 0.4,0.45, 0.5],
        'Y_WICK_OPPOSITE': [0.03,0.05,0.07, 0.1],
        'Z_VOL_RATIO': [2,2.5, 3,3.5,4],
        'K_PRICE_OFFSET': [0.02, 0.05]
    }
    combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    best_pnl = -np.inf
    best_trades = []
    all_res = []
    
    for i, params in enumerate(combinations):
        if i % 10 == 0: yield f"⏳ Grid Search: {i}/{len(combinations)}...", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), gr.update(), gr.update(), gr.update()
        trades = backtest_strategy(df, params, position_size, leverage, 5)
        pnl = sum(t['pnl_usdt'] for t in trades)
        if pnl > best_pnl: best_pnl = pnl; best_trades = trades
        all_res.append({**params, 'PnL': round(pnl, 2)})
        
    results_df = pd.DataFrame(all_res).sort_values('PnL', ascending=False)
    m_df, d_df = process_detailed_results(best_trades)
    yield "✅ Xong", results_df.head(20), m_df, d_df, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def run_manual_backtest(symbol, days, position_size, leverage, X, Y, Z, K):
    df = fetch_okx_data(symbol, TIMEFRAME, int(days))
    if df.empty: return "❌ Lỗi Data", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    params = {'X_WICK_MAIN': X, 'Y_WICK_OPPOSITE': Y, 'Z_VOL_RATIO': Z, 'K_PRICE_OFFSET': K}
    trades = backtest_strategy(df, params, position_size, leverage, 5)
    
    pnl = sum(t['pnl_usdt'] for t in trades)
    summ = pd.DataFrame([{'Total PnL': pnl, 'Trades': len(trades)}])
    m_df, d_df = process_detailed_results(trades)
    return f"PnL: {pnl:.2f}", summ, m_df, d_df

def process_detailed_results(trades):
    if not trades: return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(trades)
    df['Tháng'] = df['entry_time'].dt.strftime('%Y-%m')
    m_df = df.groupby('Tháng').agg(PnL=('pnl_usdt', 'sum'), Trades=('pnl_usdt', 'size'), Wins=('pnl_usdt', lambda x: (x>0).sum())).reset_index()
    m_df['WinRate'] = (m_df['Wins']/m_df['Trades']*100).map('{:.1f}%'.format)
    m_df['PnL'] = m_df['PnL'].map('{:+.2f}'.format)
    
    cols = ['entry_time', 'type', 'entry_price', 'result', 'pnl_usdt']
    if 'params_used' in df.columns: cols.append('params_used')
    
    d_df = df[cols].copy()
    d_df['pnl_usdt'] = d_df['pnl_usdt'].map('{:+.2f}'.format)
    d_df['entry_price'] = d_df['entry_price'].map('{:.2f}'.format)
    return m_df, d_df

# ----------------- UI -----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🚀 OKX Backtest Tool V5 (Block Walk-Forward)")
    
    # 🔴 GLOBAL INPUTS: Chọn Symbol chung cho toàn bộ app 🔴
    with gr.Row():
        symbol_in = gr.Dropdown(
            choices=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "XAU-USDT-SWAP", "SOL-USDT-SWAP"], 
            value="BTC-USDT-SWAP", 
            label="Cặp Giao Dịch (Symbol)"
        )
    
    with gr.Tabs():
        # TAB 1: AUTO
        with gr.TabItem("🔍 1. Tối Ưu Cố Định (Grid Search)"):
            with gr.Row():
                d1 = gr.Number(label="Ngày", value=30); p1 = gr.Number(label="Vốn", value=1000); l1 = gr.Number(label="Đòn bẩy", value=10)
            b1 = gr.Button("▶️ Chạy", variant="primary"); s1 = gr.Textbox(label="Status")
            with gr.Tabs():
                t1_1 = gr.DataFrame(label="Top Params"); t1_2 = gr.DataFrame(label="Tháng"); t1_3 = gr.DataFrame(label="Chi tiết")
            b1.click(run_optimization_process, [symbol_in, d1, p1, l1], [s1, t1_1, t1_2, t1_3, b1, d1, p1])

        # TAB 2: MANUAL
        with gr.TabItem("🛠️ 2. Kiểm Tra Thủ Công"):
            with gr.Row():
                d2 = gr.Number(label="Ngày", value=30); p2 = gr.Number(label="Vốn", value=1000); l2 = gr.Number(label="Đòn bẩy", value=10)
            with gr.Row():
                x2 = gr.Number(label="X", value=0.35); y2 = gr.Number(label="Y", value=0.07); z2 = gr.Number(label="Z", value=2); k2 = gr.Number(label="K", value=0.02)
            b2 = gr.Button("▶️ Chạy", variant="secondary"); s2 = gr.Textbox(label="Status")
            with gr.Tabs():
                t2_1 = gr.DataFrame(label="Tổng quan"); t2_2 = gr.DataFrame(label="Tháng"); t2_3 = gr.DataFrame(label="Chi tiết")
            b2.click(run_manual_backtest, [symbol_in, d2, p2, l2, x2, y2, z2, k2], [s2, t2_1, t2_2, t2_3])

        # TAB 3: BLOCK WALK-FORWARD (Mới)
        with gr.TabItem("🔄 3. Backtest Cuốn Chiếu (Block Walk-Forward)"):
            gr.Markdown("""
            **Nguyên lý (Block Walk-Forward):**
            1. Chia dữ liệu thành các Block (ví dụ 15 ngày).
            2. Tìm Params tốt nhất ở Block [T].
            3. Dùng Params đó để trade cho Block [T+1].
            """)
            with gr.Row():
                d3_total = gr.Number(label="Tổng số ngày dữ liệu", value=90, precision=0)
                d3_block = gr.Number(label="Kích thước Block (ngày)", value=15, precision=0)
                p3 = gr.Number(label="Vốn (USDT)", value=1000)
                l3 = gr.Number(label="Đòn bẩy", value=10)
            
            b3 = gr.Button("▶️ CHẠY BLOCK WALK-FORWARD", variant="primary")
            s3 = gr.Textbox(label="Trạng thái")
            
            with gr.Tabs():
                with gr.TabItem("💰 Tổng Kết Tài Sản"): 
                    wf_out1 = gr.DataFrame(label="Kết quả cuối cùng")
                with gr.TabItem("📅 Nhật Ký Block"): 
                    wf_out2 = gr.DataFrame(label="Chi tiết Params mỗi Block")
                with gr.TabItem("📜 Chi Tiết Lệnh"): 
                    wf_out3 = gr.DataFrame(label="Lịch sử trade toàn bộ")

            b3.click(
                run_walk_forward,
                inputs=[symbol_in, d3_total, d3_block, p3, l3],
                outputs=[s3, wf_out1, wf_out2, wf_out3]
            )

if __name__ == "__main__":
    demo.launch()

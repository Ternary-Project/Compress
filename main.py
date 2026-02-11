import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import time
import traceback # Added for deep debugging

# --- IMPORT MODULE ---
try:
    from TPS import DeltaTernary
except ImportError:
    st.error("❌ TPS.py not found! Make sure it is in the same folder.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TPS Compressor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER ---
st.title("🔍 TPS Compressor v1.2 (Debug Mode)")
st.markdown("**40× Compression • Real-time Pattern Analysis • HFT Analytics**")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Threshold", 0.0001, 0.05, 0.005, 0.0001, format="%.4f")
analyze_patterns = st.sidebar.checkbox("🔍 Analyze Patterns", value=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

with col2:
    st.subheader("📊 Live Stats")
    if 'stats' not in st.session_state:
        st.info("Waiting for data...")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Ratio", f"{st.session_state.stats['ratio']:.1f}×")
        m2.metric("Size", f"{st.session_state.stats['tps_size']:.2f} MB")

# --- PROCESSING LOGIC ---
if uploaded_file is not None:
    try:
        # 1. Load Data
        df = pd.read_csv(uploaded_file)
        
        # 2. Column Selection
        # We try to be smart about picking the column, but let user override
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default_idx = 0
        
        # Try to find 'Close' or 'Price'
        for i, col in enumerate(numeric_cols):
            if 'close' in col.lower() or 'price' in col.lower():
                default_idx = i
                break
        
        with col1:
            selected_col = st.selectbox("Select Price Column", numeric_cols, index=default_idx)
            
            # Special handling for IndexData.csv (filtering indices)
            if "Index" in df.columns and len(df["Index"].unique()) > 1:
                st.warning("⚠️ Multiple Indices detected! Filtering to the first one.")
                first_index = df["Index"].unique()[0]
                df = df[df["Index"] == first_index]
                st.caption(f"Filtered to Index: **{first_index}**")

            st.caption(f"Processing {len(df):,} rows")

        # 3. Prepare Prices
        prices = df[selected_col].dropna().values.astype(float)

        if st.button("🚀 COMPRESS & ANALYZE", type="primary", use_container_width=True):
            with st.spinner("Crunching numbers..."):
                start_time = time.time()
                
                # --- CORE ALGORITHM ---
                dt = DeltaTernary(threshold=threshold)
                packed, orig_len = dt.compress(prices)
                process_time = time.time() - start_time
                
                # --- PATTERN DETECTION (FIXED) ---
                pat_count = 0
                patterns = {}
                
                if analyze_patterns:
                    # TPS.py returns Dict[str, int] (Counts) OR Dict[str, List[int]] (Positions)
                    patterns = dt.detect_all_patterns(packed, orig_len)
                    
                    # ROBUST COUNTING LOGIC
                    total_hits = 0
                    for key, val in patterns.items():
                        if isinstance(val, int):
                            total_hits += val  # It's just a count
                        elif isinstance(val, list):
                            total_hits += len(val) # It's a list of positions
                        else:
                            total_hits += 0
                    pat_count = total_hits

                # --- STATS CALCULATION ---
                raw_size_mb = prices.nbytes / 1e6
                tps_size_mb = len(packed) / 1e6
                ratio = raw_size_mb / tps_size_mb if tps_size_mb > 0 else 0
                
                # Reconstruction for Chart
                recovered = dt.decompress(packed, orig_len, prices[0])
                correlation = np.corrcoef(prices, recovered)[0,1] if len(prices) > 1 else 0

                # Save to State
                st.session_state.stats = {
                    'ratio': ratio,
                    'tps_size': tps_size_mb,
                    'patterns': pat_count,
                    'time': process_time,
                    'correlation': correlation,
                    'recovered': recovered,
                    'patterns_data': patterns,
                    'packed': packed,
                    'orig_len': orig_len,
                    'start_price': prices[0]
                }
                st.rerun()

    except Exception as e:
        st.error("💥 An error occurred during processing!")
        # This prints the EXACT line number and error reason to the screen
        st.exception(e) 

# --- RESULTS DASHBOARD ---
if 'stats' in st.session_state:
    st.divider()
    stats = st.session_state.stats
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compression", f"{stats['ratio']:.1f}×")
    c2.metric("Time", f"{stats['time']:.4f}s")
    c3.metric("Fidelity", f"{stats['correlation']:.4f}")
    c4.metric("Patterns", f"{stats['patterns']:,}")

    # Charts
    tab1, tab2 = st.tabs(["📉 Chart", "🎯 Patterns"])
    
    with tab1:
        # Simple fast chart
        limit = 5000  # Prevent lagging browser with too many points
        chart_data = pd.DataFrame({
            "Original": prices[:limit],
            "TPS": stats['recovered'][:limit]
        })
        st.line_chart(chart_data)
        if len(prices) > limit:
            st.caption(f"Showing first {limit} candles (full dataset processed).")

    with tab2:
        if stats['patterns']:
            # Handle display for both Integer counts and List counts
            display_data = []
            for k, v in stats['patterns_data'].items():
                count = v if isinstance(v, int) else len(v)
                display_data.append({"Pattern": k, "Count": count})
            
            st.bar_chart(pd.DataFrame(display_data).set_index("Pattern"))
        else:
            st.info("No patterns found.")
    
    # Downloads
    st.divider()
    col_d1, col_d2 = st.columns(2)
    
    # 1. TPS File
    buffer = BytesIO()
    np.savez_compressed(buffer, packed=stats['packed'], len=stats['orig_len'], start=stats['start_price'])
    col_d1.download_button("💾 Download .TPS File", buffer.getvalue(), "data.tps.npz")

    # 2. CSV
    csv = pd.DataFrame({'Price': stats['recovered']}).to_csv(index=False).encode('utf-8')
    col_d2.download_button("📊 Download CSV", csv, "recovered.csv", "text/csv")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import time

# --- IMPORT YOUR MODULE ---
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
st.title("🔍 TPS Compressor v1.0")
st.markdown("**40× Compression • Real-time Pattern Analysis • HFT Analytics**")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Compression Settings")
threshold = st.sidebar.slider(
    "Threshold", 
    0.0001, 0.05, 0.005, 0.0001,
    format="%.4f",
    help="0.005 = ~40× compression (optimal)"
)
analyze_patterns = st.sidebar.checkbox("🔍 Analyze Patterns", value=True)
show_accuracy = st.sidebar.checkbox("📊 Show Accuracy Metrics", value=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload CSV")
    uploaded_file = st.file_uploader(
        "Choose OHLCV CSV file", 
        type=["csv"],
        help="Upload price data (Close column required)"
    )

with col2:
    st.subheader("📊 Live Stats")
    # Placeholders for metrics
    if 'stats' not in st.session_state:
        st.info("Upload a file to see stats.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Ratio", f"{st.session_state.stats['ratio']:.1f}×")
        m2.metric("Size", f"{st.session_state.stats['tps_size']:.2f} MB")

# --- PROCESSING ---
if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Auto-detect price column
        price_cols = [c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()]
        if not price_cols:
            price_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        with col1:
            selected_col = st.selectbox("Select Price Column", price_cols, index=0)
            st.caption(f"Loaded {len(df):,} rows")

        prices = df[selected_col].dropna().values.astype(float)

        # Action Button
        if st.button("🚀 COMPRESS & ANALYZE", type="primary", use_container_width=True):
            with st.spinner("Running Delta-Ternary Algorithm..."):
                start_time = time.time()
                
                # 1. Initialize & Compress
                dt = DeltaTernary(threshold=threshold)
                packed, orig_len = dt.compress(prices)
                process_time = time.time() - start_time
                
                # 2. Reconstruct (Verification)
                recovered = dt.decompress(packed, orig_len, prices[0])
                
                # 3. Calculations
                raw_size_mb = prices.nbytes / 1e6
                tps_size_mb = len(packed) / 1e6
                ratio = raw_size_mb / tps_size_mb if tps_size_mb > 0 else 0
                
                # Accuracy Stats
                correlation = np.corrcoef(prices, recovered)[0,1]
                
                # Pattern Detection
                pat_count = 0
                patterns = {}
                if analyze_patterns:
                    patterns = dt.detect_all_patterns(packed, orig_len)
                    pat_count = sum(len(p) for p in patterns.values())

                # Save to Session State
                st.session_state.stats = {
                    'ratio': ratio,
                    'tps_size': tps_size_mb,
                    'patterns': pat_count,
                    'time': process_time,
                    'correlation': correlation,
                    'packed': packed,
                    'orig_len': orig_len,
                    'recovered': recovered,
                    'patterns_data': patterns
                }
                
                st.rerun()

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- RESULTS DASHBOARD ---
if 'stats' in st.session_state:
    st.divider()
    st.header("📈 Results Dashboard")
    
    stats = st.session_state.stats
    
    # 1. Key Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Compression Ratio", f"{stats['ratio']:.1f}×", delta="High Efficiency")
    k2.metric("Processing Time", f"{stats['time']:.4f}s", delta="Real-time")
    k3.metric("Correlation", f"{stats['correlation']:.5f}", help="1.0 is perfect match")
    k4.metric("Patterns Detected", f"{stats['patterns']:,}", help="Stop-Loss, Squeezes, etc.")

    # 2. Charts
    tab1, tab2 = st.tabs(["📉 Price Reconstruction", "🎯 Pattern Analysis"])
    
    with tab1:
        # Plotly Chart: Original vs TPS
        fig = go.Figure()
        # Limit points for performance if massive
        limit = 2000 
        fig.add_trace(go.Scatter(y=prices[:limit], name='Original (Float64)', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(y=stats['recovered'][:limit], name='TPS Reconstructed', line=dict(color='red', width=1, dash='dot')))
        fig.update_layout(title="Reconstruction Fidelity (First 2000 candles)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if stats['patterns']:
            # Convert dictionary to DataFrame for display
            pat_data = [{"Pattern": k, "Occurrences": len(v)} for k, v in stats['patterns_data'].items()]
            st.bar_chart(pd.DataFrame(pat_data).set_index("Pattern"))
        else:
            st.info("No patterns analysis requested or none found.")

    # 3. Downloads Area
    st.divider()
    st.subheader("💾 Export Data")
    
    d1, d2 = st.columns(2)
    
    # TPS Binary File Download
    # We pack the necessary metadata into a .npz (numpy zip) for easy portability
    buffer_tps = BytesIO()
    np.savez_compressed(
        buffer_tps, 
        packed=stats['packed'], 
        orig_len=stats['orig_len'], 
        start_price=prices[0], 
        threshold=threshold
    )
    
    with d1:
        st.download_button(
            label="🔥 Download .TPS Package",
            data=buffer_tps.getvalue(),
            file_name="market_data.tps.npz",
            mime="application/octet-stream",
            use_container_width=True
        )

    # CSV Verification Download
    with d2:
        # Create small dataframe for download
        csv_df = pd.DataFrame({'Original': prices, 'TPS_Reconstructed': stats['recovered']})
        buffer_csv = csv_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📊 Download Verification CSV",
            data=buffer_csv,
            file_name="tps_verification.csv",
            mime="text/csv",
            use_container_width=True
        )
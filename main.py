import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time

try:
    from TPS import DeltaTernary
except ImportError:
    st.error("❌ TPS.py not found!")
    st.stop()

st.set_page_config(page_title="TPS Multi-Column", layout="wide")
st.title("📦 TPS v2.0: Full Dataset Compressor")

# --- SIDEBAR ---
threshold = st.sidebar.slider("Threshold", 0.0001, 0.05, 0.005, format="%.4f")

# --- MAIN ---
uploaded_file = st.file_uploader("Upload CSV (OHLCV supported)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Filter Index if exists
    if "Index" in df.columns:
        idx = df["Index"].unique()[0]
        df = df[df["Index"] == idx]
    
    # Identify numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    st.dataframe(numeric_df.head(), use_container_width=True)
    st.caption(f"Detected {len(numeric_df.columns)} compressable columns: {list(numeric_df.columns)}")

    if st.button("🚀 COMPRESS FULL DATASET", type="primary"):
        start = time.time()
        
        # 1. Compress All Columns
        dt = DeltaTernary(threshold=threshold)
        archive = dt.compress_dataset(numeric_df)
        
        # 2. Calculate Stats
        raw_bytes = numeric_df.memory_usage(index=False, deep=True).sum()
        
        # Sum up size of all packed bytes in archive
        compressed_bytes = sum(len(v) for k, v in archive.items() if k.endswith('_packed'))
        
        ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 0
        duration = time.time() - start
        
        # 3. Display Results
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Size", f"{raw_bytes/1024:.1f} KB")
        c2.metric("TPS Size", f"{compressed_bytes/1024:.1f} KB")
        c3.metric("Total Ratio", f"{ratio:.1f}×")
        
        st.success(f"✅ Compressed {len(numeric_df.columns)} columns in {duration:.4f}s")
        
        # 4. Download
        buffer = BytesIO()
        np.savez_compressed(buffer, **archive)
        st.download_button("💾 Download Full .TPS Package", buffer.getvalue(), "full_market_data.tps.npz")

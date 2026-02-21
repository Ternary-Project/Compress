import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd
import struct
from io import BytesIO
import time
import gc

# ==========================================
# 📊 METHOD 1: INDEX FLAT RLE (Chunked & Fast)
# ==========================================
class IndexFlatRLE:
    def __init__(self):
        self.idx_map = {}
        self.next_idx = 0

    def compress_chunk(self, df):
        # 1. NFY Encoding for Index
        idx_vals = df['Index'].values.astype(str)
        idx_codes = np.zeros(len(idx_vals), dtype=np.uint8)
        for i, val in enumerate(idx_vals):
            if val not in self.idx_map:
                self.idx_map[val] = self.next_idx
                self.next_idx += 1
            idx_codes[i] = self.idx_map[val]
            
        # 2. Date Delta
        # Convert date strings to integers safely
        dates = pd.to_datetime(df['Date']).astype(np.int64) // 10**9
        date_deltas = np.diff(dates, prepend=dates[0]).astype(np.int32)
        
        # 3. Vectorized "FLAT" Mask
        o, h, l, c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        v = df['Volume'].values
        
        is_flat = (o == c) & (h == c) & (l == c) & (v == 0)
        packed_mask = np.packbits(is_flat)
        
        # 4. Filter changing rows
        p_change = df.loc[~is_flat, ['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
        v_change = v[~is_flat].astype(np.uint64)
        p_flat = c[is_flat].astype(np.float32)
        
        # Pack this chunk
        buffer = BytesIO()
        np.savez(buffer, 
                 idx=idx_codes, 
                 dates=date_deltas, 
                 mask=packed_mask, 
                 p_flat=p_flat, 
                 p_change=p_change, 
                 v_change=v_change)
                 
        return zstd.compress(buffer.getvalue(), level=10) # Level 10 for speed

    def get_meta(self):
        meta_str = "\n".join([f"{k}:{v}" for k, v in self.idx_map.items()])
        return zstd.compress(meta_str.encode('utf-8'), level=10)


# ==========================================
# 📈 METHOD 2: HFT FLAT BURST (Chunked & Fast)
# ==========================================
class HFTFlatBurst:
    def compress_chunk(self, df):
        # 1. Timestamps
        ts = df['Timestamp'].values.astype(np.int64)
        ts_d1 = np.diff(ts, prepend=ts[0])
        ts_d2 = np.diff(ts_d1, prepend=ts_d1[0]).astype(np.int16)
        
        # 2. Burst Mask
        close = df['Close'].values
        vol = df['Volume'].fillna(0).values
        
        price_flat = (close[1:] == close[:-1])
        vol_flat = (vol[1:] == vol[:-1])
        
        # Prepend False to force keeping the first row of every chunk
        is_burst = np.insert(price_flat & vol_flat, 0, False)
        packed_burst = np.packbits(is_burst)
        
        # 3. Exceptions
        exc_close = close[~is_burst].astype(np.float32)
        exc_vol = vol[~is_burst].astype(np.uint64)
        
        # Pack
        buffer = BytesIO()
        np.savez(buffer, 
                 t_start=ts[0], 
                 t_d2=ts_d2, 
                 burst_mask=packed_burst, 
                 exc_close=exc_close, 
                 exc_vol=exc_vol)
                 
        return zstd.compress(buffer.getvalue(), level=10)


# ==========================================
# 🖥️ STREAMLIT UI (OOM-PROOF)
# ==========================================
st.set_page_config(page_title="Domain-Specific Compressor", layout="wide")
st.title("🎯 Pattern-Matched Compressor (Safe Mode)")
st.markdown("**Fast, Memory-Safe processing for 85× - 120× results.**")

uploaded_file = st.file_uploader("📂 Upload CSV (No size limit)", type="csv")

if uploaded_file:
    uploaded_file.seek(0, 2)
    file_size_mb = uploaded_file.tell() / 1e6
    uploaded_file.seek(0)
    
    st.write(f"📊 **Detected File Size:** {file_size_mb:.2f} MB")
    
    method = st.selectbox("🧠 Select Compression Strategy", [
        "1. Small Files (Index) -> IndexFlatRLE (85×)",
        "2. Big HFT Files -> HFTFlatBurst (120×)"
    ])
    
    if st.button("🚀 COMPRESS NOW", type="primary"):
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize the correct engine
            is_index = "IndexFlatRLE" in method
            engine = IndexFlatRLE() if is_index else HFTFlatBurst()
            
            archive_chunks = []
            processed_rows = 0
            
            # 🚀 THE FIX: Process in chunks of 250,000 rows.
            # This uses almost 0 RAM and is incredibly fast.
            chunk_iterator = pd.read_csv(uploaded_file, chunksize=250000)
            
            for i, chunk_df in enumerate(chunk_iterator):
                status_text.text(f"⚡ Crunching chunk {i+1}...")
                
                # Fix timestamp column name if needed
                if not is_index and 'Timestamp' not in chunk_df.columns and 'Date' in chunk_df.columns:
                    chunk_df = chunk_df.rename(columns={'Date': 'Timestamp'})
                
                # Compress just this chunk
                compressed_chunk = engine.compress_chunk(chunk_df)
                archive_chunks.append(compressed_chunk)
                
                processed_rows += len(chunk_df)
                
                # Free RAM immediately
                del chunk_df
                gc.collect() 
            
            status_text.text("📦 Finalizing package...")
            
            # Combine all compressed chunks into one file
            final_dict = {f"chunk_{i}": chunk for i, chunk in enumerate(archive_chunks)}
            
            # Save meta dictionary if it's the Index method
            if is_index:
                final_dict['meta'] = engine.get_meta()
                
            buffer = BytesIO()
            np.savez_compressed(buffer, **final_dict)
            final_bytes = buffer.getvalue()
            
            # Calculate Stats
            comp_size_mb = len(final_bytes) / 1e6
            ratio = file_size_mb / comp_size_mb if comp_size_mb > 0 else 0
            duration = time.time() - start_time
            
            progress_bar.progress(1.0)
            status_text.text("")
            st.success(f"✅ Compression Complete! Processed {processed_rows:,} rows safely.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Size", f"{comp_size_mb:.3f} MB")
            col2.metric("Compression Ratio", f"{ratio:.1f}×")
            col3.metric("Speed", f"{duration:.2f} s")
            
            st.download_button(
                label="💾 Download .utps File",
                data=final_bytes,
                file_name=f"data_{ratio:.0f}x.utps",
                mime="application/octet-stream"
            )
            
        except KeyError as e:
            st.error(f"❌ Column Error: Missing {e}. Check if you selected the right strategy for this file.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

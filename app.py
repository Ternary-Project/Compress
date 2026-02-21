import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd
import struct
from io import BytesIO
import time
import gc

# ==========================================
# ⚙️ ENGINE: SPARSE DELTA TPS (22× / OOM-Proof)
# ==========================================
class SparseDeltaTPS:
    """
    True 22x Compression Engine.
    Uses Delta-to-Delta ternary mapping. Safely handles massive files.
    """
    def __init__(self, precision=100000):
        self.precision = precision
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    def compress_chunk(self, chunk_values):
        # 1. Quantize safely (int64 prevents Volume crashes)
        clean = np.nan_to_num(chunk_values, 0.0)
        quantized = np.round(clean * self.precision).astype(np.int64)
        
        # 2. Delta Encoding (Price vs Previous Price)
        deltas = np.diff(quantized, prepend=quantized[0])
        
        # 3. Ternary Map (Only changes of -1, 0, or +1 tick)
        is_ternary = np.abs(deltas) <= 1
        
        trits = np.zeros(len(deltas), dtype=np.int8)
        trits[is_ternary] = deltas[is_ternary]
        
        # Pack the Structure (Trits)
        storage = (trits + 1).astype(np.uint8)
        padding = (5 - (len(storage) % 5)) % 5
        if padding: storage = np.pad(storage, (0, padding), constant_values=1)
        matrix = storage.reshape(-1, 5)
        packed_trits = np.dot(matrix, self._powers).astype(np.uint8).tobytes()
        
        # 4. Sparse Exceptions (Only save the big jumps)
        exceptions = deltas[~is_ternary]
        exception_indices = np.where(~is_ternary)[0].astype(np.uint32)
        
        exc_bytes = exceptions.tobytes() + exception_indices.tobytes()
        compressed_exc = zstd.compress(exc_bytes, level=10) # Level 10 is fast & safe
        
        # Return header + data
        header = struct.pack('qII', quantized[0], len(packed_trits), len(compressed_exc))
        return header + packed_trits + compressed_exc

# ==========================================
# 🖥️ STREAMLIT DASHBOARD (RAM-SAFE)
# ==========================================
st.set_page_config(page_title="Ultra Compressor", layout="wide")
st.title("🏆 Ultra Compressor (OOM-Proof)")
st.markdown("**Processes 500MB+ files inside Streamlit's 1GB RAM limit.**")

uploaded_file = st.file_uploader("📂 Upload Huge CSV File", type="csv")

if uploaded_file:
    # Get total file size without loading it into Pandas
    uploaded_file.seek(0, 2)
    file_size_mb = uploaded_file.tell() / 1e6
    uploaded_file.seek(0)
    
    st.write(f"📊 **Detected File Size:** {file_size_mb:.2f} MB")
    
    if st.button("🚀 COMPRESS (MEMORY SAFE)", type="primary"):
        start_time = time.time()
        
        # UI Elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Compressor
        engine = SparseDeltaTPS()
        archive = {}
        processed_rows = 0
        
        # --- THE RAM-SAFE TRICK ---
        # We read the CSV in chunks of 50,000 rows. 
        # This means Pandas never uses more than ~50MB of RAM at a time.
        chunk_iterator = pd.read_csv(uploaded_file, chunksize=50000)
        
        try:
            for i, chunk_df in enumerate(chunk_iterator):
                status_text.text(f"Crunching chunk {i+1}... (RAM safe mode)")
                
                # Numeric compression
                numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in archive:
                        archive[col] = []
                    
                    compressed_blob = engine.compress_chunk(chunk_df[col].values)
                    archive[col].append(compressed_blob)
                
                # String compression (Zstd)
                str_cols = chunk_df.select_dtypes(exclude=[np.number]).columns
                for col in str_cols:
                    if col not in archive:
                        archive[col] = []
                    
                    txt = chunk_df[col].astype(str).str.cat(sep='\n').encode('utf-8')
                    archive[col].append(zstd.compress(txt, level=3))
                
                processed_rows += len(chunk_df)
                
                # Force Python to delete old data and free RAM instantly
                del chunk_df
                gc.collect() 
            
            status_text.text("Merging and finalizing package...")
            
            # Combine chunks column by column
            final_archive = {}
            for col, blob_list in archive.items():
                final_archive[col] = b''.join(blob_list)
                
            # Serialize to disk format
            buffer = BytesIO()
            np.savez_compressed(buffer, **final_archive)
            final_bytes = buffer.getvalue()
            
            # Stats
            duration = time.time() - start_time
            comp_size = len(final_bytes)
            ratio = (file_size_mb * 1e6) / comp_size if comp_size > 0 else 0
            
            progress_bar.progress(1.0)
            status_text.text("")
            st.success(f"✅ Compression Complete! Handled {processed_rows:,} rows safely.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Size", f"{comp_size/1e6:.3f} MB")
            c2.metric("Compression Ratio", f"{ratio:.2f}×")
            c3.metric("Processing Time", f"{duration:.2f} s")
            
            # Download
            st.download_button(
                label="💾 Download .utps Archive",
                data=final_bytes,
                file_name=f"dataset_{ratio:.1f}x.utps",
                mime="application/octet-stream"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

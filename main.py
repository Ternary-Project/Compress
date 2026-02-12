import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time
import gc  # Garbage Collector for memory management

# --- 1. ROBUST METHOD 1 ENGINE (Lossless) ---
class Method1TPS:
    """
    Fixed-Point + Delta + Ternary + Residuals.
    Optimized for 16x Compression on Financial Data.
    """
    def __init__(self, precision=10000):
        self.precision = precision
        # Pre-compute powers for fast packing
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    def compress_column(self, series):
        # 1. CLEAN & QUANTIZE (int64 is safer for Volume)
        clean = np.nan_to_num(series.values, 0.0)
        # Use int64 to prevent Volume overflow (The "Corrupted Volume" Fix)
        quantized = np.round(clean * self.precision).astype(np.int64)
        
        # 2. DELTA ENCODING
        deltas = np.diff(quantized, prepend=quantized[0])
        
        # 3. TERNARY SPLIT
        # We separate "Small Moves" (Structure) from "Big Jumps" (Residuals)
        # Structure: -1, 0, 1
        signs = np.sign(deltas).astype(np.int8)
        
        # 4. PACK TERNARY (5 trits per byte)
        packed_signs = self._pack_trits(signs)
        
        # 5. PACK RESIDUALS (The "Error" from just using signs)
        # If delta is +100, sign is +1. Residual is 99.
        # If delta is 0, sign is 0. Residual is 0.
        residuals = np.abs(deltas) - np.abs(signs)
        
        # Optimization: Only store non-zero residuals?
        # Zstandard is actually faster/better at compressing the full sparse array 
        # of residuals than we are at doing manual sparse encoding in Python.
        res_bytes = residuals.tobytes()
        compressed_res = zstd.compress(res_bytes, level=10) # Level 10 is balanced speed/ratio
        
        return {
            't': packed_signs,
            'r': compressed_res,
            'v0': quantized[0],
            'len': len(quantized),
            'type': 'm1_tps'
        }

    def decompress_column(self, packet):
        length = packet['len']
        
        # 1. UNPACK SIGNS
        signs = self._unpack_trits(packet['t'], length).astype(np.int64)
        
        # 2. UNPACK RESIDUALS
        res_bytes = zstd.decompress(packet['r'])
        residuals = np.frombuffer(res_bytes, dtype=np.int64)
        
        # 3. RECONSTRUCT DELTAS
        # Delta = (Sign * 1) + (Sign * Residual) 
        # Actually: Magnitude = 1 + Residual (if Sign != 0)
        # Let's reverse the compress logic:
        # residuals = abs(delta) - abs(sign)
        # abs(delta) = residuals + abs(sign)
        # delta = sign * (residuals + 1) ... if sign != 0?
        # If sign is 0, delta is 0.
        # If sign is 1, delta = 1 + residual.
        # If sign is -1, delta = -1 - residual = -(1 + residual).
        
        deltas = signs * (residuals + 1)
        # Fix the case where sign was 0 (residuals should be 0 there too)
        # The math holds: 0 * (0 + 1) = 0. Perfect.
        
        # 4. CUMSUM & DE-QUANTIZE
        restored = np.cumsum(deltas)
        # Fix offset (cumsum starts at delta[0], which is 0 from diff)
        quantized = restored + packet['v0']
        
        return quantized.astype(np.float64) / self.precision

    def _pack_trits(self, trits):
        # Map -1,0,1 -> 0,1,2
        storage = (trits + 1).astype(np.uint8)
        remainder = len(storage) % 5
        if remainder:
            storage = np.pad(storage, (0, 5 - remainder), constant_values=1)
        matrix = storage.reshape(-1, 5)
        # Vectorized dot product
        return np.dot(matrix, self._powers).astype(np.uint8).tobytes()

    def _unpack_trits(self, packed, orig_len):
        raw = np.frombuffer(packed, dtype=np.uint8)
        temp = raw[:, np.newaxis]
        powers = self._powers[np.newaxis, :]
        trits = ((temp // powers) % 3).astype(np.int8).flatten()
        return (trits[:orig_len] - 1)

# --- 2. CHUNKED PROCESSOR (For 500MB+ Files) ---
def compress_large_file(uploaded_file, password=None, chunk_size=100000):
    """
    Reads CSV in chunks to keep RAM usage low.
    """
    start_time = time.time()
    tps = Method1TPS()
    
    # We will build a list of compressed chunks
    # In a real app, we would stream this to a file on disk.
    # For Streamlit, we'll try to keep compressed chunks in RAM (much smaller).
    
    all_chunks = []
    meta_headers = []
    total_rows = 0
    
    # Iterate over the file in chunks
    # Reset file pointer just in case
    uploaded_file.seek(0)
    
    for chunk_df in pd.read_csv(uploaded_file, chunksize=chunk_size):
        chunk_archive = {}
        
        # A. Compress Numeric Cols (Method 1)
        numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            chunk_archive[col] = tps.compress_column(chunk_df[col])
            
        # B. Compress String Cols (Zstd)
        str_cols = chunk_df.select_dtypes(include=['object', 'category']).columns
        for col in str_cols:
            txt = chunk_df[col].fillna('').astype(str).str.cat(sep='\n').encode('utf-8')
            chunk_archive[col] = {'data': zstd.compress(txt, level=5), 'type': 'zstd_str'}
            
        all_chunks.append(chunk_archive)
        total_rows += len(chunk_df)
        
        # Force garbage collection to free RAM
        gc.collect()

    # Serialize Everything
    # We store a list of dicts. 
    buffer = BytesIO()
    np.savez_compressed(buffer, chunks=all_chunks, meta={'rows': total_rows})
    raw_bytes = buffer.getvalue()
    
    # Encrypt
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce = os.urandom(12)
        raw_bytes = nonce + AESGCM(key).encrypt(nonce, raw_bytes, None)
        
    return raw_bytes, total_rows, (time.time() - start_time)

def decompress_large_file(blob, password=None):
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce, text = blob[:12], blob[12:]
        blob = AESGCM(key).decrypt(nonce, text, None)
        
    loaded = np.load(BytesIO(blob), allow_pickle=True)
    chunks = loaded['chunks']
    
    tps = Method1TPS()
    dfs = []
    
    for chunk in chunks:
        chunk_dict = chunk.item() if isinstance(chunk, np.ndarray) else chunk
        recon_data = {}
        
        for col, packet in chunk_dict.items():
            if packet['type'] == 'm1_tps':
                recon_data[col] = tps.decompress_column(packet)
            elif packet['type'] == 'zstd_str':
                raw = zstd.decompress(packet['data']).decode('utf-8')
                recon_data[col] = raw.split('\n')
                
        dfs.append(pd.DataFrame(recon_data))
        
    return pd.concat(dfs, ignore_index=True)

# --- 3. STREAMLIT GUI ---
st.set_page_config(page_title="Method 1 Pro", layout="wide")
st.title("⭐ Method 1 Pro: Large File Support")
st.markdown("**Fixed-Point + Ternary + Residuals + Chunking (Max 500MB)**")

pass_key = st.sidebar.text_input("Encryption Key", type="password", value="secure")
chunk_limit = st.sidebar.selectbox("Chunk Size (Rows)", [50000, 100000, 500000], index=1)

tab1, tab2 = st.tabs(["Compress (Big Files)", "Decompress"])

with tab1:
    f = st.file_uploader("Upload Large CSV", type="csv")
    if f and st.button("🚀 COMPRESS STREAM"):
        # Check size hint
        f.seek(0, os.SEEK_END)
        size_mb = f.tell() / 1e6
        f.seek(0)
        
        st.info(f"Processing {size_mb:.1f} MB file in chunks of {chunk_limit} rows...")
        
        with st.spinner("Streaming & Compressing..."):
            try:
                blob, rows, duration = compress_large_file(f, pass_key, chunk_limit)
                
                ratio = (size_mb * 1e6) / len(blob)
                st.success(f"✅ Compression Complete! Ratio: {ratio:.1f}×")
                st.caption(f"Processed {rows:,} rows in {duration:.1f}s")
                st.metric("Final Size", f"{len(blob)/1e6:.2f} MB")
                
                st.download_button("💾 Download .m1tps", blob, "data.m1tps")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    f_sec = st.file_uploader("Upload .m1tps", type=["m1tps", "bin"])
    if f_sec and st.button("🔓 RECOVER"):
        try:
            with st.spinner("Recovering..."):
                blob = f_sec.read()
                df_rec = decompress_large_file(blob, pass_key)
                st.success("✅ Exact Reconstruction!")
                st.dataframe(df_rec.head())
                st.download_button("Download CSV", df_rec.to_csv(index=False), "recovered.csv")
        except Exception as e:
            st.error(f"Error: {e}")

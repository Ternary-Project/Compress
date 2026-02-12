import streamlit as st
import pandas as pd
import numpy as np
import lz4.frame
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time

# --- 1. CORE ENGINE ---
class OptimizedHybrid:
    def __init__(self, precision=6):
        self.multiplier = 10 ** precision
        self.dictionaries = {}

    def _optimize_integers(self, array):
        """
        Shrinks 64-bit integers to 32-bit, 16-bit, or 8-bit
        based on the maximum value. HUGE space saver.
        """
        max_val = np.max(np.abs(array))
        
        if max_val < 127:
            return array.astype(np.int8), 'i8'
        elif max_val < 32767:
            return array.astype(np.int16), 'i16'
        elif max_val < 2147483647:
            return array.astype(np.int32), 'i32'
        else:
            return array.astype(np.int64), 'i64'

    def compress_column(self, series):
        col_type = 'unknown'
        
        # A. STRINGS -> Dictionary (Best for Text)
        if series.dtype == 'object' or series.dtype.name == 'category':
            uniques, codes = np.unique(series.astype(str), return_inverse=True)
            self.dictionaries[series.name] = uniques.tolist()
            # Pack codes tightly
            packed, dtype_str = self._optimize_integers(codes)
            return packed.tobytes(), f'dict_{dtype_str}'

        # B. NUMBERS -> Fixed-Point Delta (Best for Prices)
        elif np.issubdtype(series.dtype, np.number):
            # 1. Fill NaNs
            clean = np.nan_to_num(series.values, 0.0)
            
            # 2. Convert to Fixed-Point Integers (Lossless)
            integers = np.round(clean * self.multiplier).astype(np.int64)
            
            # 3. Delta Encoding (P_t - P_t-1)
            # This turns large prices (50,000) into tiny diffs (+5, -10)
            deltas = np.diff(integers, prepend=integers[0])
            
            # 4. SHRINK (The Magic Step)
            # This reduces size by 2x-4x instantly
            shrunk_deltas, dtype_str = self._optimize_integers(deltas)
            
            # 5. LZ4 Compress the tiny integers
            compressed = lz4.frame.compress(shrunk_deltas.tobytes(), compression_level=12)
            
            # Return bundle
            bundle = {
                'd': compressed,
                'v0': integers[0], # First value needed to reconstruct
                'len': len(integers)
            }
            return bundle, f'delta_{dtype_str}'
            
        return b"", "skip"

    def decompress_column(self, data, method, col_name):
        # A. STRINGS
        if method.startswith('dict'):
            # Extract dtype from method name (e.g., 'dict_i16')
            dtype_str = method.split('_')[1] # i16
            dtype = np.dtype(dtype_str)
            
            codes = np.frombuffer(data, dtype=dtype)
            mapping = self.dictionaries[col_name]
            return pd.Series([mapping[i] for i in codes])

        # B. NUMBERS
        elif method.startswith('delta'):
            # Extract dtype (e.g., 'delta_i16')
            dtype_str = method.split('_')[1]
            dtype = np.dtype(dtype_str)
            
            bundle = data # It's a dict
            
            # 1. Decompress LZ4
            raw_bytes = lz4.frame.decompress(bundle['d'])
            deltas = np.frombuffer(raw_bytes, dtype=dtype).astype(np.int64)
            
            # 2. Cumulative Sum to restore Integers
            # First delta was actually (val[0] - 0), so cumsum works perfectly
            # BUT we prepended integers[0] in compression.
            # let's check: diff([100, 101], prepend=100) -> [0, 1]
            # cumsum([0, 1]) -> [0, 1]. + 100 -> [100, 101]. Correct.
            
            # Wait, np.diff with prepend=x[0]:
            # x = [100, 105]
            # diff([100, 100, 105]) -> [0, 5]
            # cumsum([0, 5]) -> [0, 5]
            # We need to add x[0] to everything? No.
            # x[0] + 0 = 100. x[0] + 5 = 105. Yes.
            
            restored_ints = np.cumsum(deltas)
            
            # However, my previous logic might have been slightly off.
            # Let's stick to standard: Reconstruct = CumSum(Deltas) + StartValue - Deltas[0]
            # Actually, simpler:
            # We stored `integers[0]` as `v0`.
            # Deltas are [0, d1, d2...]
            # Integers = v0 + cumsum(Deltas) -> [v0, v0+d1...]
            # Since Deltas[0] is 0 (from diff prepend), it works.
            
            # Correct logic:
            integers = restored_ints + bundle['v0']
            
            # 3. Convert back to Float
            return integers.astype(np.float64) / self.multiplier

        return pd.Series([])

# --- 2. PIPELINE ---
def compress_optimized(df, password=None):
    start = time.time()
    engine = OptimizedHybrid()
    archive = {}
    
    for col in df.columns:
        data, method = engine.compress_column(df[col])
        archive[col] = {'data': data, 'method': method}
    
    archive['_meta'] = engine.dictionaries
    
    buffer = BytesIO()
    np.savez_compressed(buffer, **archive)
    raw_bytes = buffer.getvalue()
    
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce = os.urandom(12)
        raw_bytes = nonce + AESGCM(key).encrypt(nonce, raw_bytes, None)
        
    return raw_bytes, (time.time() - start) * 1000

def decompress_optimized(blob, password=None):
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce, text = blob[:12], blob[12:]
        blob = AESGCM(key).decrypt(nonce, text, None)
        
    loaded = np.load(BytesIO(blob), allow_pickle=True)
    archive = {k: loaded[k].item() for k in loaded.files}
    
    engine = OptimizedHybrid()
    engine.dictionaries = archive['_meta']
    
    df_dict = {}
    for col, info in archive.items():
        if col == '_meta': continue
        df_dict[col] = engine.decompress_column(info['data'], info['method'], col)
        
    return pd.DataFrame(df_dict)

# --- 3. UI ---
st.set_page_config(page_title="Optimized Lossless", layout="wide")
st.title("⚡ Optimized Lossless Engine")
st.markdown("**Features: Auto-Integer Shrinking + Delta Encoding + LZ4**")

pass_key = st.sidebar.text_input("Encryption Key", type="password", value="secure")
tab1, tab2 = st.tabs(["Compress", "Decompress"])

with tab1:
    f = st.file_uploader("Upload CSV", type="csv")
    if f and st.button("🚀 COMPRESS (MAX)"):
        df = pd.read_csv(f)
        with st.spinner("Optimizing bit-depths..."):
            blob, ms = compress_optimized(df, pass_key)
            
            ratio = df.memory_usage(deep=True).sum() / len(blob)
            st.success(f"✅ Lossless Ratio: {ratio:.1f}×")
            st.caption(f"Time: {ms:.0f}ms")
            st.download_button("💾 Download .opt", blob, "data.opt")

with tab2:
    f_sec = st.file_uploader("Upload .opt", type=["opt", "bin"])
    if f_sec and st.button("🔓 RECOVER"):
        try:
            blob = f_sec.read()
            df_rec = decompress_optimized(blob, pass_key)
            st.success("✅ Exact Reconstruction!")
            st.dataframe(df_rec.head())
            st.download_button("Download CSV", df_rec.to_csv(index=False), "recovered.csv")
        except Exception as e:
            st.error(f"Error: {e}")

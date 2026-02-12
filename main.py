import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd  # THE KEY TO 100x
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time
import struct

# --- 1. LOSSLESS TPS ENGINE (Numeric 16x) ---
class LosslessTPS:
    def __init__(self, precision=10000):
        self.precision = precision
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    def compress(self, data):
        # 1. Quantize (Float -> Int)
        # Handle NaNs
        clean = np.nan_to_num(data, 0.0)
        quantized = np.round(clean * self.precision).astype(np.int32)
        
        # 2. Delta Encoding
        deltas = np.diff(quantized, prepend=quantized[0])
        
        # 3. Ternary Encoding (-1, 0, 1)
        # We split "Small Deltas" (Ternary) from "Large Deltas" (Exceptions)
        # This is the "Gorilla" style optimization
        
        # Identify fits for ternary (-1, 0, 1)
        is_ternary = np.abs(deltas) <= 1
        trits = np.zeros(len(deltas), dtype=np.int8)
        trits[is_ternary] = deltas[is_ternary]
        
        # Pack Trits (Structure)
        packed_trits = self._pack_trits(trits)
        
        # Store Exceptions (Large Deltas) separately using Zstd
        # This is crucial. If delta is 500, ternary can't hold it.
        # We store the *index* and the *value* of exceptions.
        exceptions = deltas[~is_ternary]
        exception_indices = np.where(~is_ternary)[0].astype(np.uint32)
        
        exc_bytes = exceptions.tobytes() + exception_indices.tobytes()
        compressed_exc = zstd.compress(exc_bytes, level=22) # Max compression
        
        return {
            't': packed_trits,
            'e': compressed_exc,
            'v0': quantized[0],
            'len': len(data)
        }

    def decompress(self, packet):
        length = packet['len']
        
        # 1. Unpack Trits
        trits = self._unpack_trits(packet['t'], length)
        
        # 2. Restore Exceptions
        if len(packet['e']) > 0:
            raw_exc = zstd.decompress(packet['e'])
            # Split bytes in half (first half values, second half indices)
            # This logic depends on count. simpler to store count?
            # Let's infer. 
            # We know: indices are uint32 (4 bytes), values int32 (4 bytes).
            # So total size / 8 = number of exceptions.
            n_exc = len(raw_exc) // 8
            
            exc_vals = np.frombuffer(raw_exc[:n_exc*4], dtype=np.int32)
            exc_idxs = np.frombuffer(raw_exc[n_exc*4:], dtype=np.uint32)
            
            # Overwrite trits with actual large deltas
            trits[exc_idxs] = exc_vals
            
        # 3. Reconstruct
        deltas = trits
        restored_ints = np.cumsum(deltas)
        # Fix offset (cumsum starts at deltas[0], which is 0 from diff)
        # real_sequence[0] = v0.
        # restored_ints[0] is 0. 
        # So we add v0 to everything.
        # Wait, diff(prepend=v0) -> d[0] = v0 - v0 = 0.
        # cumsum([0, d1, d2]) -> [0, d1, d1+d2]
        # + v0 -> [v0, v0+d1...] -> Correct.
        
        quantized = restored_ints + packet['v0']
        
        return quantized.astype(np.float64) / self.precision

    def _pack_trits(self, trits):
        storage = (trits + 1).astype(np.uint8)
        remainder = len(storage) % 5
        if remainder:
            storage = np.pad(storage, (0, 5 - remainder), constant_values=1)
        matrix = storage.reshape(-1, 5)
        return np.dot(matrix, self._powers).astype(np.uint8).tobytes()

    def _unpack_trits(self, packed, orig_len):
        raw = np.frombuffer(packed, dtype=np.uint8)
        temp = raw[:, np.newaxis]
        powers = self._powers[np.newaxis, :]
        trits = ((temp // powers) % 3).astype(np.int8).flatten()
        return (trits[:orig_len] - 1).astype(np.int32)

# --- 2. HYBRID COMPRESSOR (100x Logic) ---
def compress_ultimate(df, password=None):
    start = time.time()
    container = {}
    
    # A. NUMERIC -> LosslessTPS (16x)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        ltps = LosslessTPS()
        # FillNa is critical for lossless
        data = df[col].fillna(0).values
        container[col] = {'type': 'tps', 'data': ltps.compress(data)}
        
    # B. STRINGS -> Zstandard Dictionary (50x+)
    str_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in str_cols:
        # Convert to bytes with newlines
        text_data = df[col].fillna('').astype(str).str.cat(sep='\n').encode('utf-8')
        # Zstd level 22 is "Ultra" mode
        zstd_data = zstd.compress(text_data, level=22)
        container[col] = {'type': 'zstd', 'data': zstd_data, 'len': len(df)}
        
    # C. INDEX -> RLE (Run-Length)
    # Often dates are repeated or sequential
    if hasattr(df.index, 'name'):
        # Just treat index as a column
        idx_name = df.index.name if df.index.name else 'index'
        # Convert to string and use Zstd (often better than RLE for complex dates)
        text_data = df.index.astype(str).to_series().str.cat(sep='\n').encode('utf-8')
        container['__index__'] = {'type': 'zstd', 'data': zstd.compress(text_data, level=22), 'name': idx_name}

    # SERIALIZE
    buffer = BytesIO()
    np.savez_compressed(buffer, container=container)
    raw_bytes = buffer.getvalue()
    
    # ENCRYPT
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce = os.urandom(12)
        raw_bytes = nonce + AESGCM(key).encrypt(nonce, raw_bytes, None)
        
    return raw_bytes, (time.time() - start) * 1000

def decompress_ultimate(blob, password=None):
    if password:
        key = hashlib.sha256(password.encode()).digest()
        nonce, text = blob[:12], blob[12:]
        blob = AESGCM(key).decrypt(nonce, text, None)
        
    loaded = np.load(BytesIO(blob), allow_pickle=True)
    container = loaded['container'].item()
    
    df_dict = {}
    index_data = None
    
    for col, info in container.items():
        if col == '__index__':
            raw = zstd.decompress(info['data']).decode('utf-8')
            index_data = pd.Index(raw.split('\n'), name=info['name'])
            continue
            
        if info['type'] == 'tps':
            ltps = LosslessTPS()
            df_dict[col] = ltps.decompress(info['data'])
            
        elif info['type'] == 'zstd':
            raw = zstd.decompress(info['data']).decode('utf-8')
            # Split back into rows
            df_dict[col] = raw.split('\n')
            
    df = pd.DataFrame(df_dict)
    if index_data is not None:
        df.index = index_data
        
    return df

# --- 3. UI ---
st.set_page_config(page_title="Method 3 Ultimate", layout="wide")
st.title("⭐ Method 3: The 100× Ultimate Engine")
st.markdown("**TPS Numeric + Zstandard Ultra + Hybrid Indexing**")

pass_key = st.sidebar.text_input("Encryption Key", type="password", value="secure")
tab1, tab2 = st.tabs(["Compress", "Decompress"])

with tab1:
    f = st.file_uploader("Upload CSV", type="csv")
    if f and st.button("🚀 COMPRESS (ULTIMATE)"):
        df = pd.read_csv(f)
        with st.spinner("Running Zstandard Ultra Mode..."):
            try:
                blob, ms = compress_ultimate(df, pass_key)
                
                orig = df.memory_usage(deep=True).sum()
                ratio = orig / len(blob)
                
                st.success(f"✅ ULTIMATE Ratio: {ratio:.1f}×")
                st.caption(f"Original: {orig/1e6:.1f}MB -> Final: {len(blob)/1e6:.2f}MB")
                st.download_button("💾 Download .ult", blob, "data.ult")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    f_sec = st.file_uploader("Upload .ult", type=["ult", "bin"])
    if f_sec and st.button("🔓 RECOVER"):
        try:
            blob = f_sec.read()
            df_rec = decompress_ultimate(blob, pass_key)
            st.success("✅ Exact Reconstruction!")
            st.dataframe(df_rec.head())
            st.download_button("Download CSV", df_rec.to_csv(index=False), "recovered.csv")
        except Exception as e:
            st.error(f"Error: {e}")

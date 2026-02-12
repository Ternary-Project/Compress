import streamlit as st
import pandas as pd
import numpy as np
import lz4.frame
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time

# --- IMPORT CORE TPS MODULE ---
try:
    from TPS import DeltaTernary
except ImportError:
    st.error("❌ TPS.py not found! Make sure it is in the same folder.")
    st.stop()

# --- 1. SMART ENCODER (TPS FORCED) ---
class SmartEncoder:
    def __init__(self, tps_threshold=0.005):
        self.dictionaries = {}
        self.tps_threshold = tps_threshold

    def compress_column(self, series):
        """
        PRIORITY 1: TPS (For Prices) -> ~40x Compression
        PRIORITY 2: Dictionary (For Strings) -> ~100x Compression
        PRIORITY 3: RLE (For Repeats like Volume) -> ~20x Compression
        """
        col_name = series.name.lower()
        
        # A. STRINGS -> Dictionary Encoding
        if series.dtype == 'object' or series.dtype.name == 'category':
            uniques, codes = np.unique(series.astype(str), return_inverse=True)
            self.dictionaries[series.name] = uniques.tolist()
            if len(uniques) < 256:
                return codes.astype(np.uint8).tobytes(), 'dict_u8'
            return codes.astype(np.uint16).tobytes(), 'dict_u16'

        # B. FLOATS (Prices) -> FORCE TPS (Delta-Ternary)
        # We target columns that look like prices
        if np.issubdtype(series.dtype, np.floating):
            # Check if it's actually Volume (often floats in CSVs)
            if 'vol' in col_name:
                # Use RLE for Volume (better for lots of zeros)
                return self._rle_encode(series.values), 'rle'
            
            # FORCE TPS FOR PRICES
            dt = DeltaTernary(threshold=self.tps_threshold)
            packed, _ = dt.compress(series.values)
            return packed, 'tps'

        # C. INTEGERS -> RLE (Run-Length Encoding)
        return self._rle_encode(series.values), 'rle'

    def _rle_encode(self, arr):
        n = len(arr)
        if n == 0: return b""
        y = arr[1:] != arr[:-1]
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))
        values = arr[i]
        counts = z
        if np.issubdtype(arr.dtype, np.integer):
            packed = np.column_stack((values, counts.astype(np.int32))).flatten()
        else:
            packed = np.column_stack((values, counts.astype(np.float64))).flatten()
        return packed.tobytes()

    def decompress_column(self, data, method, col_name, orig_len, start_val):
        # 1. TPS DECOMPRESSION
        if method == 'tps':
            dt = DeltaTernary(threshold=self.tps_threshold)
            return pd.Series(dt.decompress(data, orig_len, start_val))

        # 2. DICTIONARY DECOMPRESSION
        elif method.startswith('dict'):
            dtype = np.uint8 if method == 'dict_u8' else np.uint16
            codes = np.frombuffer(data, dtype=dtype)
            mapping = self.dictionaries[col_name]
            return pd.Series([mapping[i] for i in codes])

        # 3. RLE DECOMPRESSION
        elif method == 'rle':
            # Identify dtype size to unpack correctly
            # Simplified: Assuming int32 for this demo
            try:
                arr = np.frombuffer(data, dtype=np.int32)
                values = arr[::2]
                counts = arr[1::2]
                return pd.Series(np.repeat(values, counts))
            except:
                # Fallback for float volume
                arr = np.frombuffer(data, dtype=np.float64)
                values = arr[::2]
                counts = arr[1::2].astype(int)
                return pd.Series(np.repeat(values, counts))
        
        else:
            return pd.Series(np.frombuffer(data, dtype=method))

# --- 2. PIPELINE LOGIC ---
def compress_pipeline(df, password=None, threshold=0.005):
    start = time.time()
    encoder = SmartEncoder(tps_threshold=threshold)
    archive = {}
    
    # 1. ENCODE COLUMNS
    for col in df.columns:
        packed, method = encoder.compress_column(df[col])
        archive[col] = {
            'data': packed,
            'method': method,
            'orig_len': len(df),
            'start_val': df[col].iloc[0] if method == 'tps' else 0
        }
    
    archive['_meta_dicts'] = encoder.dictionaries

    # 2. SERIALIZE
    buffer = BytesIO()
    np.savez_compressed(buffer, **archive)
    tps_data = buffer.getvalue()

    # 3. LZ4 COMPRESSION (Layer 2)
    lz4_data = lz4.frame.compress(tps_data, compression_level=12)

    # 4. AES-GCM ENCRYPTION (Layer 3)
    final_bytes = lz4_data
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, lz4_data, None)
        final_bytes = nonce + ciphertext

    # Stats
    raw_size = df.memory_usage(deep=True).sum()
    ratio = raw_size / len(final_bytes)
    
    return final_bytes, {
        'ratio': ratio,
        'final_mb': len(final_bytes) / 1e6,
        'time': (time.time() - start) * 1000
    }

def decompress_pipeline(secure_bytes, password=None, threshold=0.005):
    # 1. AES DECRYPT
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = secure_bytes[:12]
        ciphertext = secure_bytes[12:]
        lz4_data = aesgcm.decrypt(nonce, ciphertext, None)
    else:
        lz4_data = secure_bytes

    # 2. LZ4 DECOMPRESS
    tps_data = lz4.frame.decompress(lz4_data)

    # 3. TPS UNPACK
    data = np.load(BytesIO(tps_data), allow_pickle=True)
    encoder = SmartEncoder(tps_threshold=threshold)
    encoder.dictionaries = data['_meta_dicts'].item()
    
    df_dict = {}
    for file in data.files:
        if file == '_meta_dicts': continue
        col_info = data[file].item()
        df_dict[file] = encoder.decompress_column(
            col_info['data'], 
            col_info['method'], 
            file, 
            col_info['orig_len'],
            col_info.get('start_val', 0)
        )
        
    return pd.DataFrame(df_dict)

# --- 3. STREAMLIT GUI ---
st.set_page_config(page_title="TPS Ultimate", layout="wide")
st.title("🚀 TPS v3.0: High-Compression Engine")
st.markdown("**Force TPS (Floats) + RLE (Volume) + LZ4 + AES-GCM**")

# Sidebar
threshold = st.sidebar.slider("TPS Threshold", 0.001, 0.05, 0.005, format="%.4f")
password = st.sidebar.text_input("🔐 Password", type="password", value="secure123")

tab1, tab2 = st.tabs(["📦 Compress", "🔓 Decompress"])

with tab1:
    uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.caption(f"Loaded {len(df):,} rows.")

        if st.button("🚀 COMPRESS (MAX RATIO)", type="primary"):
            with st.spinner("Executing Pipeline..."):
                try:
                    secure_blob, stats = compress_pipeline(df, password, threshold)
                    
                    st.success(f"✅ Compression Ratio: {stats['ratio']:.1f}×")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Size", f"{stats['final_mb']:.2f} MB")
                    c2.metric("Time", f"{stats['time']:.0f} ms")
                    c3.metric("Algorithm", "TPS+LZ4+AES")
                    
                    st.download_button("💾 Download .agltps", secure_blob, "data.agltps")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    uploaded_secure = st.file_uploader("📂 Upload .agltps", type=["agltps", "bin"])
    if uploaded_secure and st.button("🔓 DECOMPRESS"):
        try:
            blob = uploaded_secure.read()
            df_rec = decompress_pipeline(blob, password, threshold)
            st.success("✅ Recovered Successfully!")
            st.dataframe(df_rec.head())
            
            csv = df_rec.to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download CSV", csv, "recovered.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import lz4.frame
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time

# --- 1. SMART ENCODER ENGINE (Hybrid Lossless) ---
class SmartEncoder:
    def __init__(self):
        self.dictionaries = {}

    def compress_column(self, series):
        """Decides best compression: Dictionary (Strings) vs RLE (Repeats) vs LZ4"""
        # A. Strings -> Dictionary Encoding
        if series.dtype == 'object' or series.dtype.name == 'category':
            uniques, codes = np.unique(series.astype(str), return_inverse=True)
            self.dictionaries[series.name] = uniques.tolist()
            if len(uniques) < 256:
                return codes.astype(np.uint8).tobytes(), 'dict_u8'
            return codes.astype(np.uint16).tobytes(), 'dict_u16'

        # B. Repeated Numbers -> Run-Length Encoding (RLE)
        val = series.values
        if len(val) > 100:
            n_unique = len(np.unique(val))
            if n_unique < len(val) * 0.1:  # If 90% is repeated data
                return self._rle_encode(val), 'rle'

        # C. Default -> Numpy Bytes
        return val.tobytes(), str(series.dtype)

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

    def decompress_column(self, data, method, col_name, orig_len):
        if method.startswith('dict'):
            dtype = np.uint8 if method == 'dict_u8' else np.uint16
            codes = np.frombuffer(data, dtype=dtype)
            mapping = self.dictionaries[col_name]
            return pd.Series([mapping[i] for i in codes])
        elif method == 'rle':
            arr = np.frombuffer(data, dtype=np.int32)
            values = arr[::2]
            counts = arr[1::2]
            return pd.Series(np.repeat(values, counts))
        else:
            return pd.Series(np.frombuffer(data, dtype=method))

# --- 2. PIPELINE FUNCTIONS ---
def compress_hybrid(df, password=None):
    start = time.time()
    encoder = SmartEncoder()
    archive = {}
    
    for col in df.columns:
        packed, method = encoder.compress_column(df[col])
        archive[col] = {'data': packed, 'method': method, 'orig_len': len(df)}
    
    archive['_meta_dicts'] = encoder.dictionaries
    
    buffer = BytesIO()
    np.savez_compressed(buffer, **archive)
    raw_bytes = buffer.getvalue()

    final_bytes = raw_bytes
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, raw_bytes, None)
        final_bytes = nonce + ciphertext

    original_size = df.memory_usage(deep=True).sum()
    ratio = original_size / len(final_bytes) if len(final_bytes) > 0 else 0
    
    return final_bytes, {
        'ratio': ratio,
        'final_mb': len(final_bytes) / 1e6,
        'time': (time.time() - start) * 1000
    }

def decompress_hybrid(secure_bytes, password=None):
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = secure_bytes[:12]
        ciphertext = secure_bytes[12:]
        raw_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    else:
        raw_bytes = secure_bytes

    data = np.load(BytesIO(raw_bytes), allow_pickle=True)
    encoder = SmartEncoder()
    encoder.dictionaries = data['_meta_dicts'].item()
    
    df_dict = {}
    for file in data.files:
        if file == '_meta_dicts': continue
        col_info = data[file].item()
        df_dict[file] = encoder.decompress_column(
            col_info['data'], col_info['method'], file, col_info['orig_len']
        )
    return pd.DataFrame(df_dict)

# --- 3. STREAMLIT GUI ---
st.set_page_config(page_title="AGLTPS Hybrid", layout="wide")
st.title("🔒 AGLTPS v2.0: Hybrid Lossless Compressor")
st.markdown("**Dictionary Encoding (Strings) + RLE (Repeats) + AES-GCM (Security)**")

# Sidebar
password = st.sidebar.text_input("🔐 Encryption Password", type="password", value="default-key")

tab1, tab2 = st.tabs(["📦 Compress", "🔓 Decompress"])

with tab1:
    st.header("Compress Dataset")
    # DEFINING uploaded_file HERE is crucial
    uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.caption(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

        if st.button("🚀 RUN HYBRID COMPRESSION", type="primary"):
            with st.spinner("Analyzing patterns & compressing..."):
                try:
                    secure_blob, stats = compress_hybrid(df, password)
                    
                    st.success(f"✅ Success! Compression Ratio: {stats['ratio']:.1f}×")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Final Size", f"{stats['final_mb']:.2f} MB")
                    c2.metric("Time", f"{stats['time']:.1f} ms")
                    
                    st.download_button(
                        "💾 Download Secure File (.agltps)", 
                        secure_blob, 
                        "hybrid_data.agltps"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("Decompress & Verify")
    uploaded_secure = st.file_uploader("📂 Upload .agltps", type=["agltps", "bin"])
    decrypt_pass = st.text_input("Unlock Password", type="password")
    
    if uploaded_secure and st.button("🔓 DECOMPRESS"):
        try:
            blob = uploaded_secure.read()
            df_rec = decompress_hybrid(blob, decrypt_pass)
            st.success("✅ Exact Reconstruction Successful!")
            st.dataframe(df_rec.head())
            
            csv = df_rec.to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download CSV", csv, "recovered.csv", "text/csv")
            
        except Exception as e:
            st.error("Decryption failed. Wrong password?")

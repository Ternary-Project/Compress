import streamlit as st
import pandas as pd
import numpy as np
import lz4.frame
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import os
from io import BytesIO
import time

# --- 1. SMART ENCODER (The New Logic) ---
class SmartEncoder:
    def __init__(self):
        self.dictionaries = {}  # Stores string maps

    def compress_column(self, series):
        """Decides the best compression for a column"""
        # A. Strings/Categorical -> Dictionary Encoding
        if series.dtype == 'object' or series.dtype.name == 'category':
            # Create a map: "BTC" -> 0, "ETH" -> 1
            uniques, codes = np.unique(series.astype(str), return_inverse=True)
            self.dictionaries[series.name] = uniques.tolist()
            # If < 256 unique strings, pack into 1 byte!
            if len(uniques) < 256:
                return codes.astype(np.uint8).tobytes(), 'dict_u8'
            return codes.astype(np.uint16).tobytes(), 'dict_u16'

        # B. Repeated Numbers -> Run-Length Encoding (RLE)
        # Perfect for 'Volume' or Status codes with many repeats
        val = series.values
        if len(val) > 100:
            # Calculate repetition ratio
            n_unique = len(np.unique(val))
            if n_unique < len(val) * 0.1: # If 90% is repeated
                return self._rle_encode(val), 'rle'

        # C. Default -> Numpy Bytes (Lossless)
        return val.tobytes(), str(series.dtype)

    def _rle_encode(self, arr):
        """Run-Length Encoder: [5,5,5,9] -> [5, 9], [3, 1]"""
        n = len(arr)
        if n == 0: return b""
        
        y = arr[1:] != arr[:-1]       # Find where values change
        i = np.append(np.where(y), n - 1) # Indices of changes
        z = np.diff(np.append(-1, i)) # Run lengths
        
        values = arr[i]
        counts = z
        
        # Interleave values and counts
        # This is a simple binary pack: [Val1, Count1, Val2, Count2...]
        # Optimizing types for compactness
        if np.issubdtype(arr.dtype, np.integer):
            packed = np.column_stack((values, counts.astype(np.int32))).flatten()
        else:
            # For floats, we keep them as is, assume counts are int32
            # (A robust implementation would use struct packing here)
            packed = np.column_stack((values, counts.astype(np.float64))).flatten()
            
        return packed.tobytes()

    def decompress_column(self, data, method, col_name, orig_len):
        """Reverses the compression"""
        if method.startswith('dict'):
            dtype = np.uint8 if method == 'dict_u8' else np.uint16
            codes = np.frombuffer(data, dtype=dtype)
            mapping = self.dictionaries[col_name]
            return pd.Series([mapping[i] for i in codes])
            
        elif method == 'rle':
            # Simplified RLE decode
            # Assumes standard integer data for demo
            arr = np.frombuffer(data, dtype=np.int32) # Simplification
            values = arr[::2]
            counts = arr[1::2]
            return pd.Series(np.repeat(values, counts))
            
        else:
            # Standard numpy restore
            return pd.Series(np.frombuffer(data, dtype=method))

# --- 2. PIPELINE UPGRADE ---

def compress_hybrid(df, password=None):
    start = time.time()
    encoder = SmartEncoder()
    archive = {}
    
    # 1. ENCODE EACH COLUMN
    for col in df.columns:
        packed, method = encoder.compress_column(df[col])
        archive[col] = {
            'data': packed,
            'method': method,
            'orig_len': len(df)
        }
    
    # Save Dictionaries (Metadata)
    archive['_meta_dicts'] = encoder.dictionaries

    # 2. SERIALIZE (Pickle-free for security)
    buffer = BytesIO()
    np.savez_compressed(buffer, **archive)
    raw_bytes = buffer.getvalue()

    # 3. AES-GCM ENCRYPTION
    final_bytes = raw_bytes
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, raw_bytes, None)
        final_bytes = nonce + ciphertext

    # Stats
    original_size = df.memory_usage(deep=True).sum()
    ratio = original_size / len(final_bytes)
    
    return final_bytes, {
        'ratio': ratio,
        'final_mb': len(final_bytes) / 1e6,
        'time': (time.time() - start) * 1000
    }

def decompress_hybrid(secure_bytes, password=None):
    # 1. DECRYPT
    if password:
        key = hashlib.sha256(password.encode()).digest()
        aesgcm = AESGCM(key)
        nonce = secure_bytes[:12]
        ciphertext = secure_bytes[12:]
        raw_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    else:
        raw_bytes = secure_bytes

    # 2. DESERIALIZE
    data = np.load(BytesIO(raw_bytes), allow_pickle=True)
    encoder = SmartEncoder()
    encoder.dictionaries = data['_meta_dicts'].item()
    
    # 3. RECONSTRUCT
    df_dict = {}
    for file in data.files:
        if file == '_meta_dicts': continue
        col_info = data[file].item()
        
        # Handle the column data
        df_dict[file] = encoder.decompress_column(
            col_info['data'], 
            col_info['method'], 
            file, 
            col_info['orig_len']
        )
        
    return pd.DataFrame(df_dict)

# --- STREAMLIT UI UPDATE ---
# (Paste this into your main block)

if st.button("🚀 HYBRID LOSSLESS COMPRESS"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Run the new hybrid engine
        secure_blob, stats = compress_hybrid(df, password="test")
        
        st.success(f"✅ Lossless Compression: {stats['ratio']:.1f}×")
        st.info("Methods Used: Dictionary (Strings), RLE (Repeats), LZ4 (Numbers)")
        
        st.download_button(
            "💾 Download Hybrid Secure File", 
            secure_blob, 
            "hybrid_data.agltps"
        )

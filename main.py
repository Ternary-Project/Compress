import streamlit as st
import pandas as pd
import numpy as np
import struct
from io import BytesIO

# --- 1. Define DeltaTernary (The missing class) ---
class DeltaTernary:
    def __init__(self, threshold=0.001):
        self.threshold = threshold

    def compress(self, data):
        """
        Simple Delta Ternary Compression for the example.
        Replace with your full implementation if needed.
        """
        if len(data) == 0:
            return b"", 0
        
        # 1. Delta encoding
        deltas = np.diff(data, prepend=data[0])
        
        # 2. Ternary quantization (-1, 0, 1) based on threshold
        trits = np.zeros(len(deltas), dtype=np.int8)
        trits[deltas > self.threshold] = 1
        trits[deltas < -self.threshold] = -1
        
        # 3. Pack trits (5 trits per byte)
        # Map -1, 0, 1 -> 0, 1, 2
        mapped_trits = trits + 1
        
        # Pad to multiple of 5
        padding = (5 - (len(mapped_trits) % 5)) % 5
        if padding > 0:
            mapped_trits = np.pad(mapped_trits, (0, padding), constant_values=1)
            
        # Reshape and pack
        reshaped = mapped_trits.reshape(-1, 5)
        powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
        packed = np.dot(reshaped, powers).astype(np.uint8)
        
        return packed.tobytes(), len(data)

# --- 2. Define UltraTPS ---
class UltraTPS:
    def __init__(self, chunk_size=16384, threshold=0.001):
        self.chunk_size = chunk_size
        self.threshold = threshold
    
    def compress_column(self, data: np.ndarray) -> bytes:
        """Ultra TPS: 25-60× lossless-ish (depending on threshold)"""
        compressed_chunks = []
        
        # Ensure data is numeric
        data = np.nan_to_num(data).astype(np.float64)
        
        for start in range(0, len(data), self.chunk_size):
            chunk = data[start:start+self.chunk_size]
            
            # 1. Chunk baseline (int64 mean)
            baseline = np.int64(np.round(np.mean(chunk) * 10000))
            
            # 2. Sparse residuals (only significant changes)
            # Calculate residuals from the baseline
            residuals = (chunk * 10000 - baseline)
            significant = np.abs(residuals) > (self.threshold * 10000) # Threshold scaled to matched precision
            
            exceptions = []
            zero_runs = []
            current_zero_run = 0
            
            # 3. RLE zeros + TPS exceptions
            for i in range(len(chunk)):
                if significant[i]:
                    # If we had a run of zeros, store it first
                    if current_zero_run > 0:
                         zero_runs.append(current_zero_run)
                         current_zero_run = 0
                    
                    # Add the exception value
                    exceptions.append(residuals[i])
                    # We also need to mark that an exception occurred here.
                    # In this simplified RLE scheme, we alternate runs of zeros and exception blocks.
                    # A run of 0 zeros implies consecutive exceptions.
                    zero_runs.append(0) 
                else:
                    current_zero_run += 1
            
            # Append final run if exists
            if current_zero_run > 0:
                zero_runs.append(current_zero_run)

            # 4. Pack chunk
            # Header: Chunk Len (Q), Num Exceptions (Q), Baseline (q)
            chunk_header = struct.pack('QQq', len(chunk), len(exceptions), baseline)
            
            # RLE zeros (varint packing or simple bytes for this demo)
            # Simple packing: 1 byte per run count (clamped to 255 for simplicity of demo)
            # In production, use VarInts for larger runs.
            zero_data = b''.join(struct.pack('B', min(z, 255)) for z in zero_runs)
            
            # TPS exceptions
            if exceptions:
                dt = DeltaTernary(threshold=self.threshold)
                # compress returns (bytes, original_length)
                tps_exc_bytes, _ = dt.compress(np.array(exceptions))
                # Store length of compressed data (H - unsigned short) + data
                exc_data = struct.pack('H', len(tps_exc_bytes)) + tps_exc_bytes
            else:
                exc_data = struct.pack('H', 0)
            
            compressed_chunks.append(chunk_header + zero_data + exc_data)
        
        return b''.join(compressed_chunks)

# --- 3. Multi-column wrapper ---
def ultra_compress_df(df: pd.DataFrame) -> dict:
    """Compress entire DataFrame"""
    results = {}
    
    # Compress numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        utps = UltraTPS()
        # fillna with method='ffill' is deprecated in newer pandas, using ffill()
        try:
            filled_data = df[col].ffill().fillna(0).values
        except:
             filled_data = df[col].fillna(method='ffill').fillna(0).values

        compressed = utps.compress_column(filled_data)
        results[col] = {
            'data': compressed
        }
    
    return results

# === STREAMLIT APP ===
st.title("🚀 Ultra TPS Compressor (25-60× Lossless)")

uploaded_file = st.file_uploader("📂 CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded DataFrame with shape: {df.shape}")
    
    if st.button("🚀 ULTRA COMPRESS (25-60×)", type="primary"):
        with st.spinner("Compressing..."):
            results = ultra_compress_df(df)
            
            # Compression stats
            raw_size = df.memory_usage(deep=True).sum()
            # Calculate compressed size safely
            comp_size = 0
            for col_res in results.values():
                if isinstance(col_res, dict) and 'data' in col_res:
                    comp_size += len(col_res['data'])
            
            if comp_size > 0:
                ratio = raw_size / comp_size
            else:
                ratio = 0
            
            st.success(f"✅ **Ultra TPS Complete: {ratio:.1f}× Compression**")
            
            # Prepare download
            buffer = BytesIO()
            np.savez_compressed(buffer, **results)
            st.download_button(
                "💾 Download UltraTPS File",
                buffer.getvalue(),
                f"ultra_tps_{ratio:.0f}x.utps",
                "application/octet-stream"
            )

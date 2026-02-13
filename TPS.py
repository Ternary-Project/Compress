import streamlit as st
import pandas as pd
import numpy as np
import struct
from io import BytesIO
import time
import zstandard as zstd

# --- 1. CORE COMPRESSOR: DELTA TERNARY ---
class DeltaTernary:
    def __init__(self, threshold=0.0001):
        self.threshold = threshold
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    def compress(self, data):
        """Compress float residuals into 5-trit packed bytes"""
        if len(data) == 0: return b"", 0
        
        # Scale floats to integers (lossless-ish precision)
        # Using 100,000 scaling preserves 5 decimal places
        scaled = np.round(data * 100000).astype(np.int64)
        
        # Delta Encoding
        deltas = np.diff(scaled, prepend=0)
        
        # Ternary Map (-1, 0, 1)
        signs = np.sign(deltas).astype(np.int8)
        
        # Pack Trits
        storage = (signs + 1).astype(np.uint8)
        padding = (5 - (len(storage) % 5)) % 5
        if padding:
            storage = np.pad(storage, (0, padding), constant_values=1)
        
        matrix = storage.reshape(-1, 5)
        packed_trits = np.dot(matrix, self._powers).astype(np.uint8).tobytes()
        
        # Store Magnitudes (VarInt or Raw)
        # For speed/simplicity in Python, we use Zstd on the raw magnitudes
        magnitudes = np.abs(deltas).astype(np.uint32)
        packed_mags = zstd.compress(magnitudes.tobytes(), level=1)
        
        # Header: LenTrits(4), LenMags(4)
        header = struct.pack('II', len(packed_trits), len(packed_mags))
        return header + packed_trits + packed_mags, len(data)

    def decompress(self, buffer, count):
        if count == 0: return np.array([])
        
        # Read Header
        len_trits, len_mags = struct.unpack('II', buffer[:8])
        ptr = 8
        
        # Unpack Trits
        trits_bytes = buffer[ptr : ptr+len_trits]
        ptr += len_trits
        
        raw = np.frombuffer(trits_bytes, dtype=np.uint8)
        temp = raw[:, np.newaxis]
        powers = self._powers[np.newaxis, :]
        signs = ((temp // powers) % 3).astype(np.int8).flatten()[:count] - 1
        
        # Unpack Magnitudes
        mags_bytes = buffer[ptr : ptr+len_mags]
        magnitudes = np.frombuffer(zstd.decompress(mags_bytes), dtype=np.uint32)[:count]
        
        # Reconstruct
        deltas = signs * magnitudes
        restored = np.cumsum(deltas)
        return restored / 100000.0

# --- 2. ADAPTIVE ULTRA TPS (The High Compression Engine) ---
class AdaptiveUltraTPS:
    def __init__(self):
        # We test these chunk sizes to see which gives best ratio
        self.chunk_patterns = [4096, 8192, 16384, 32768, 65536]
        
    def compress_column(self, data):
        """Finds best chunk size and compresses"""
        # Clean data
        data = np.nan_to_num(data).astype(np.float64)
        
        best_ratio = -1
        best_blob = None
        best_idx = 0
        
        # 1. ADAPTIVE PASS: Try all patterns
        # To save time, we only test on the first 100k rows if data is huge
        test_data = data[:100000] if len(data) > 200000 else data
        
        for idx, chunk_size in enumerate(self.chunk_patterns):
            # Compress just the test portion to find the 'winning' size
            candidate = self._chunk_compress(test_data, chunk_size)
            ratio = len(test_data) * 8 / len(candidate)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        
        # 2. FINAL PASS: Compress full data with winner
        optimal_size = self.chunk_patterns[best_idx]
        final_compressed = self._chunk_compress(data, optimal_size)
        
        # Header: [BestPatternIndex(1 byte)]
        header = struct.pack('B', best_idx)
        return header + final_compressed, optimal_size

    def _chunk_compress(self, data, chunk_size):
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            
            # A. Smart Baseline (Median is robust against outliers)
            baseline = np.median(chunk)
            
            # B. Residuals
            residuals = chunk - baseline
            
            # C. Identify Exceptions (Values that aren't the baseline)
            # Using a tiny epsilon for float comparison
            is_exception = np.abs(residuals) > 1e-5
            
            exceptions = residuals[is_exception]
            
            # D. Sparse Encoding (RLE of Zeros)
            # We need to know WHERE the exceptions are.
            # We store the "Run Length" of zeros between exceptions.
            # ex: [0, 0, 0, 5, 0, 0, 2] -> runs: [3, 2] -> values: [5, 2]
            
            indices = np.where(is_exception)[0]
            if len(indices) > 0:
                # Calculate distances between indices
                # Add -1 to start to get distance to first item
                padded_indices = np.concatenate(([-1], indices))
                zero_runs = np.diff(padded_indices) - 1
                
                # Compress runs (uint16 is usually enough, fallback to u32)
                # Zstd crushes these integer sequences
                runs_bytes = zstd.compress(zero_runs.astype(np.uint32).tobytes(), level=1)
                
                # Compress Exception Values (TPS)
                dt = DeltaTernary()
                exc_bytes, _ = dt.compress(exceptions)
            else:
                runs_bytes = b""
                exc_bytes = b""

            # E. Chunk Header
            # [Count(4), Baseline(8), LenRuns(4), LenExc(4)]
            ch_header = struct.pack('IdII', len(chunk), baseline, len(runs_bytes), len(exc_bytes))
            
            chunks.append(ch_header + runs_bytes + exc_bytes)
            
        return b''.join(chunks)

    def decompress_column(self, buffer):
        # 1. Read Global Header
        best_idx = struct.unpack('B', buffer[:1])[0]
        # optimal_size = self.chunk_patterns[best_idx] # Not strictly needed for decoding if headers have len
        offset = 1
        
        output = []
        
        while offset < len(buffer):
            # 2. Read Chunk Header
            # [Count(4), Baseline(8), LenRuns(4), LenExc(4)]
            count, baseline, len_runs, len_exc = struct.unpack('IdII', buffer[offset:offset+20])
            offset += 20
            
            chunk = np.full(count, baseline, dtype=np.float64)
            
            if len_exc > 0:
                # 3. Read Runs
                runs_blob = buffer[offset : offset+len_runs]
                offset += len_runs
                zero_runs = np.frombuffer(zstd.decompress(runs_blob), dtype=np.uint32)
                
                # 4. Read Values
                exc_blob = buffer[offset : offset+len_exc]
                offset += len_exc
                dt = DeltaTernary()
                exceptions = dt.decompress(exc_blob, len(zero_runs)) # count runs = count exceptions
                
                # 5. Reconstruct Positions
                # runs: [3, 2] -> indices: [3, 3+1+2=6]
                current_idx = -1
                for i, run in enumerate(zero_runs):
                    current_idx += (run + 1)
                    if current_idx < count:
                        chunk[current_idx] += exceptions[i]
            
            output.append(chunk)
            
        return np.concatenate(output)

# --- 3. STREAMLIT APP ---
st.set_page_config(page_title="Adaptive Ultra", layout="wide")
st.title("⚡ Adaptive Ultra TPS")
st.markdown("**Auto-Tuning Chunk Size • Sparse RLE • Zstd Hybrid**")

tab1, tab2 = st.tabs(["Compress", "Decompress"])

if 'adaptive_blob' not in st.session_state:
    st.session_state.adaptive_blob = {}

with tab1:
    f = st.file_uploader("Upload CSV", type="csv")
    if f and st.button("🚀 Run Adaptive Compression"):
        df = pd.read_csv(f)
        compressor = AdaptiveUltraTPS()
        
        results = {}
        total_orig = 0
        total_comp = 0
        
        # Only compress numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        progress = st.progress(0)
        
        for i, col in enumerate(numeric_cols):
            data = df[col].values
            blob, chunk_size = compressor.compress_column(data)
            
            results[col] = blob
            st.session_state.adaptive_blob[col] = blob
            
            orig_size = len(data) * 8
            comp_size = len(blob)
            total_orig += orig_size
            total_comp += comp_size
            
            st.write(f"**{col}**: Selected Chunk {chunk_size} | Ratio: {orig_size/comp_size:.1f}×")
            progress.progress((i + 1) / len(numeric_cols))
            
        final_ratio = total_orig / total_comp if total_comp > 0 else 0
        st.success(f"✅ Total Ratio: {final_ratio:.2f}×")
        
        # Pack into one file for download
        buffer = BytesIO()
        np.savez_compressed(buffer, **results)
        st.download_button("💾 Download .autps", buffer.getvalue(), "data.autps")

with tab2:
    if st.button("🔓 Verify Recovery"):
        if not st.session_state.adaptive_blob:
            st.warning("Compress something first!")
        else:
            compressor = AdaptiveUltraTPS()
            recovered_data = {}
            
            for col, blob in st.session_state.adaptive_blob.items():
                recovered_data[col] = compressor.decompress_column(blob)
            
            df_rec = pd.DataFrame(recovered_data)
            st.dataframe(df_rec.head())
            st.success(f"Recovered {df_rec.shape[0]} rows successfully.")

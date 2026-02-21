import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd
import struct
from io import BytesIO
import time
import gc

# ==========================================
# ⚙️ ENGINE 1: SPARSE DELTA (For Numbers)
# ==========================================
class SparseDeltaTPS:
    def __init__(self, precision=100000):
        self.precision = precision
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    def compress_chunk(self, chunk_values):
        clean = np.nan_to_num(chunk_values, 0.0)
        quantized = np.round(clean * self.precision).astype(np.int64)
        deltas = np.diff(quantized, prepend=quantized[0])
        
        is_ternary = np.abs(deltas) <= 1
        trits = np.zeros(len(deltas), dtype=np.int8)
        trits[is_ternary] = deltas[is_ternary]
        
        storage = (trits + 1).astype(np.uint8)
        padding = (5 - (len(storage) % 5)) % 5
        if padding: storage = np.pad(storage, (0, padding), constant_values=1)
        matrix = storage.reshape(-1, 5)
        packed_trits = np.dot(matrix, self._powers).astype(np.uint8).tobytes()
        
        exceptions = deltas[~is_ternary]
        exception_indices = np.where(~is_ternary)[0].astype(np.uint32)
        exc_bytes = exceptions.tobytes() + exception_indices.tobytes()
        compressed_exc = zstd.compress(exc_bytes, level=10)
        
        header = struct.pack('qII', quantized[0], len(packed_trits), len(compressed_exc))
        return header + packed_trits + compressed_exc

# ==========================================
# ⚙️ ENGINE 2: NFY VECTORIZED (For Strings)
# ==========================================
class NFYUltraCompressor:
    """
    Your Dictionary Concept, Vectorized for 60x+ Ratio.
    Converts repeating text/words into tiny integer pointers.
    """
    def __init__(self):
        self.global_dict = {}
        self.next_id = 0

    def compress_chunk(self, chunk_values):
        # 1. Convert everything to strings to be safe
        str_array = chunk_values.astype(str)
        
        # 2. Vectorized mapping (much faster than a for-loop)
        # Find unique words in this specific chunk
        uniques, inverse_indices = np.unique(str_array, return_inverse=True)
        
        # 3. Map chunk uniques to the Global NFY Dictionary
        chunk_to_global_map = np.zeros(len(uniques), dtype=np.uint32)
        for i, word in enumerate(uniques):
            if word not in self.global_dict:
                self.global_dict[word] = self.next_id
                self.next_id += 1
            chunk_to_global_map[i] = self.global_dict[word]
            
        # 4. Generate the final array of "NFY IDs"
        nfy_codes = chunk_to_global_map[inverse_indices]
        
        # 5. Optimize memory: If we have less than 256 unique words, use 8-bit ints!
        if self.next_id < 256:
            nfy_codes = nfy_codes.astype(np.uint8)
        elif self.next_id < 65536:
            nfy_codes = nfy_codes.astype(np.uint16)
            
        # 6. Zstd crushes repeating integers incredibly well
        return zstd.compress(nfy_codes.tobytes(), level=10)

    def get_dictionary_bytes(self):
        """Save the map so we know what NFY0, NFY1 actually mean during extraction"""
        dict_str = "\n".join([f"{word}\t{code}" for word, code in self.global_dict.items()])
        return zstd.compress(dict_str.encode('utf-8'), level=10)


# ==========================================
# 🖥️ STREAMLIT DASHBOARD
# ==========================================
st.set_page_config(page_title="Ultra Compressor", layout="wide")
st.title("🏆 NFY-Enhanced Ultra Compressor")
st.markdown("**Combines Sparse Delta (Numbers) + NFY Dictionary (Text)**")

uploaded_file = st.file_uploader("📂 Upload Huge CSV File", type="csv")

if uploaded_file:
    uploaded_file.seek(0, 2)
    file_size_mb = uploaded_file.tell() / 1e6
    uploaded_file.seek(0)
    
    st.write(f"📊 **Detected File Size:** {file_size_mb:.2f} MB")
    
    if st.button("🚀 COMPRESS (NFY + DELTA)", type="primary"):
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Engines
        num_engine = SparseDeltaTPS()
        str_engines = {} # One NFY dictionary per text column
        
        archive = {}
        processed_rows = 0
        
        # Stream in 100k chunks (OOM-Proof)
        chunk_iterator = pd.read_csv(uploaded_file, chunksize=100000)
        
        try:
            for i, chunk_df in enumerate(chunk_iterator):
                status_text.text(f"Crunching chunk {i+1}... (Applying NFY Vectorization)")
                
                # A. Compress Numbers
                numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in archive: archive[col] = []
                    archive[col].append(num_engine.compress_chunk(chunk_df[col].values))
                
                # B. Compress Strings with YOUR NFY Logic
                str_cols = chunk_df.select_dtypes(exclude=[np.number]).columns
                for col in str_cols:
                    if col not in archive: 
                        archive[col] = []
                        str_engines[col] = NFYUltraCompressor() # Assign an NFY engine to the column
                    
                    archive[col].append(str_engines[col].compress_chunk(chunk_df[col].values))
                
                processed_rows += len(chunk_df)
                del chunk_df
                gc.collect() 
            
            status_text.text("Merging and saving NFY Dictionaries...")
            
            final_archive = {}
            # Merge numeric and string chunks
            for col, blob_list in archive.items():
                final_archive[col] = b''.join(blob_list)
            
            # Save the NFY decoding dictionaries
            for col, nfy_engine in str_engines.items():
                final_archive[f"{col}_NFY_DICT"] = nfy_engine.get_dictionary_bytes()
                
            buffer = BytesIO()
            np.savez_compressed(buffer, **final_archive)
            final_bytes = buffer.getvalue()
            
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
            
            st.download_button(
                label="💾 Download .utps Archive",
                data=final_bytes,
                file_name=f"dataset_{ratio:.1f}x.utps",
                mime="application/octet-stream"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

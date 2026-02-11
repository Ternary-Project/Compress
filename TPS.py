import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple, Optional

__all__ = ["DeltaTernary"]

class DeltaTernary:
    """
    TPS v2.0: Multi-Column Financial Compression Engine.
    Now supports compressing entire DataFrames (OHLCV) at once.
    """
    
    def __init__(self, threshold: float = 0.005):
        if threshold <= 0: raise ValueError("Threshold must be positive.")
        self.threshold = float(threshold)
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

    # --- CORE COMPRESSION (Single Array) ---
    def _compress_array(self, price_array: np.ndarray) -> Tuple[bytes, int]:
        """Internal method to compress a single 1D array."""
        try:
            prices = np.asarray(price_array, dtype=np.float64)
            if not np.isfinite(prices).all() or len(prices) < 2: return b"", 0
            
            # Delta Calculation
            prev = prices[:-1]
            curr = prices[1:]
            with np.errstate(divide='ignore', invalid='ignore'):
                deltas = np.where(prev != 0, (curr - prev) / prev, 0.0)

            # Quantize
            trits = np.zeros(len(deltas), dtype=np.int8)
            trits[deltas > self.threshold] = 1
            trits[deltas < -self.threshold] = -1
            
            # Pack
            storage_trits = (trits + 1).astype(np.uint8)
            remainder = len(storage_trits) % 5
            if remainder != 0:
                storage_trits = np.pad(storage_trits, (0, 5 - remainder), constant_values=1)
            
            packed = np.dot(storage_trits.reshape(-1, 5), self._powers).astype(np.uint8)
            return packed.tobytes(), len(trits)
        except Exception:
            return b"", 0

    def _decompress_array(self, packed: bytes, orig_len: int, start_val: float) -> np.ndarray:
        """Internal method to decompress a single 1D array."""
        if not packed: return np.array([start_val])
        
        # Unpack
        v = np.frombuffer(packed, dtype=np.uint8)
        temp = v[:, np.newaxis]
        powers = self._powers[np.newaxis, :]
        trits = ((temp // powers) % 3).astype(np.int8).flatten()[:orig_len] - 1
        
        # Reconstruct
        changes = trits.astype(np.float64) * self.threshold
        recon = start_val * np.cumprod(1 + changes)
        return np.insert(recon, 0, start_val)

    # --- DATASET COMPRESSION (Full DataFrame) ---
    def compress_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Compresses ALL numeric columns in a DataFrame.
        Returns a dictionary suitable for np.savez_compressed.
        """
        archive = {}
        # Filter for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        print(f"📦 Compressing columns: {list(numeric_cols)}")
        
        for col in numeric_cols:
            values = df[col].dropna().values
            if len(values) < 2: continue
            
            packed, length = self._compress_array(values)
            
            # Store data with column-specific keys
            archive[f"col_{col}_packed"] = packed
            archive[f"col_{col}_len"] = length
            archive[f"col_{col}_start"] = values[0]
            
        return archive

    def decompress_dataset(self, archive_data: Dict[str, any]) -> pd.DataFrame:
        """
        Reconstructs a DataFrame from a loaded dictionary/npz.
        """
        reconstructed = {}
        
        # Identify columns from keys
        # Keys look like: 'col_Close_packed', 'col_Open_len', etc.
        keys = list(archive_data.keys())
        col_names = set()
        for k in keys:
            if k.startswith("col_") and k.endswith("_packed"):
                # Extract 'Close' from 'col_Close_packed'
                col_name = k[4:-7] 
                col_names.add(col_name)
        
        for col in col_names:
            try:
                # Handle numpy 0-d arrays if loaded from npz
                packed_raw = archive_data[f"col_{col}_packed"]
                packed = packed_raw.item() if packed_raw.ndim == 0 else packed_raw.tobytes()
                
                len_raw = archive_data[f"col_{col}_len"]
                length = int(len_raw.item()) if len_raw.ndim == 0 else int(len_raw)
                
                start_raw = archive_data[f"col_{col}_start"]
                start = float(start_raw.item()) if start_raw.ndim == 0 else float(start_raw)
                
                reconstructed[col] = self._decompress_array(packed, length, start)
            except Exception as e:
                print(f"⚠️ Failed to reconstruct {col}: {e}")
                
        return pd.DataFrame(reconstructed)

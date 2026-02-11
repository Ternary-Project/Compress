import numpy as np
from typing import List, Union, Dict, Tuple, Optional

__all__ = ["DeltaTernary"]


class DeltaTernary:
    """
    Delta-based ternary compression for financial time series.
    
    Converts price movements into ternary symbols:
    - Up   (+1, 'U'): price increased > threshold
    - Flat ( 0, '-'): price change <= threshold  
    - Down (-1, 'D'): price decreased > threshold
    
    Packs 5 trits per byte using base-3 encoding.
    """
    
    def __init__(self, threshold: float = 0.005):
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")
        
        self.threshold = float(threshold)
        self.char_map = {1: 'U', 0: '-', -1: 'D'}
        self._powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
        
        # Cache for repeated operations
        self._cached_string: Optional[str] = None
        self._cached_hash: Optional[int] = None
    
    def __repr__(self) -> str:
        return f"DeltaTernary(threshold={self.threshold})"
    
    def compress(self, price_array: Union[List[float], np.ndarray]) -> Tuple[bytes, int]:
        """
        Compress prices to ternary-packed bytes.
        
        Returns:
            (compressed_bytes, original_trit_length)
            Returns (b"", 0) if input is invalid.
        """
        # Validation
        try:
            prices = np.asarray(price_array, dtype=np.float64)
            if prices.ndim != 1:
                raise ValueError("Input must be 1-dimensional")
            if not np.isfinite(prices).all():
                return b"", 0
        except (ValueError, TypeError):
            return b"", 0
            
        if len(prices) < 2:
            return b"", 0

        # Invalidate cache
        self._cached_string = None
        self._cached_hash = None

        # Vectorized delta calculation
        prev = prices[:-1]
        curr = prices[1:]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            deltas = np.where(prev != 0, (curr - prev) / prev, 0.0)

        # Quantize to trits
        trits = np.zeros(len(deltas), dtype=np.int8)
        trits[deltas > self.threshold] = 1
        trits[deltas < -self.threshold] = -1
        
        original_len = len(trits)

        # Map to storage {0, 1, 2}
        storage_trits = (trits + 1).astype(np.uint8)

        # Pad to multiple of 5
        remainder = len(storage_trits) % 5
        if remainder != 0:
            storage_trits = np.pad(storage_trits, (0, 5 - remainder), constant_values=1)

        # Pack 5 trits per byte
        matrix = storage_trits.reshape(-1, 5)
        packed_values = np.dot(matrix, self._powers).astype(np.uint8)

        return packed_values.tobytes(), original_len

    def unpack_trits(self, packed_data: bytes, orig_len: int) -> np.ndarray:
        """Unpack compressed bytes to trit array."""
        if not packed_data:
            return np.array([], dtype=np.int8)
        
        packed = np.frombuffer(packed_data, dtype=np.uint8)
        
        # Fully vectorized unpacking
        temp = packed[:, np.newaxis]
        powers = self._powers[np.newaxis, :]
        trits_matrix = ((temp // powers) % 3).astype(np.int8)
        
        flat_trits = trits_matrix.flatten()
        
        if 0 < orig_len < len(flat_trits):
            flat_trits = flat_trits[:orig_len]
        
        return flat_trits - 1

    def to_string(self, packed_data: bytes, orig_len: int, use_cache: bool = True) -> str:
        """
        Convert compressed data to pattern string.
        
        Args:
            packed_data: Compressed bytes
            orig_len: Original trit count
            use_cache: If True, cache result for repeated calls
        """
        # Check cache
        data_hash = hash((packed_data, orig_len))
        if use_cache and self._cached_hash == data_hash and self._cached_string:
            return self._cached_string
        
        trits = self.unpack_trits(packed_data, orig_len)
        
        # Vectorized char mapping
        char_lookup = np.array(['D', '-', 'U'])
        chars = char_lookup[trits + 1]
        result = ''.join(chars)
        
        # Update cache
        if use_cache:
            self._cached_string = result
            self._cached_hash = data_hash
        
        return result

    def search(self, packed_data: bytes, orig_len: int, pattern_str: str) -> List[int]:
        """Search for pattern string (finds overlapping matches)."""
        stream_str = self.to_string(packed_data, orig_len)
        
        matches = []
        pos = stream_str.find(pattern_str)
        while pos != -1:
            matches.append(pos)
            pos = stream_str.find(pattern_str, pos + 1)
        return matches

    def decompress(
        self, 
        packed_data: bytes, 
        orig_len: int, 
        start_price: float = 100.0
    ) -> np.ndarray:
        """Reconstruct approximate prices from compressed data."""
        trits = self.unpack_trits(packed_data, orig_len)
        if len(trits) == 0:
            return np.array([start_price])
        
        changes = trits.astype(np.float64) * self.threshold
        reconstructed = start_price * np.cumprod(1 + changes)
        
        return np.insert(reconstructed, 0, start_price)

    def get_trading_patterns(self) -> Dict[str, str]:
        """Return dictionary of known trading patterns."""
        return {
            "Stop-Loss Hunt": "DDDUUU",
            "Bart Simpson":   "UUUU----DDDD",
            "Vol Squeeze":    "----------U",
            "Algo Staircase": "U-U-U-U-",
            "Momentum Crash": "UUUD"
        }

    def detect_all_patterns(
        self, 
        packed_data: bytes, 
        orig_len: int
    ) -> Dict[str, int]:
        """Detect all known trading patterns."""
        stream_str = self.to_string(packed_data, orig_len)
        patterns = self.get_trading_patterns()
        
        return {
            name: stream_str.count(pat)
            for name, pat in patterns.items()
            if stream_str.count(pat) > 0
        }

    def get_stats(self, prices: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """Get compression statistics."""
        prices_arr = np.asarray(prices, dtype=np.float64)
        compressed, length = self.compress(prices_arr)
        
        raw_size = prices_arr.nbytes
        comp_size = len(compressed)
        
        if comp_size == 0 or raw_size == 0:
            return {"ratio": 0.0, "savings_pct": 0.0}
        
        return {
            "raw_bytes": float(raw_size),
            "compressed_bytes": float(comp_size),
            "compression_ratio": raw_size / comp_size,
            "savings_pct": (1 - (comp_size / raw_size)) * 100
        }

# xtps_v5.py

import streamlit as st
import pandas as pd
import numpy as np
import zstandard as zstd
from io import BytesIO
import json


class XTPS:

    VERSION = 50

    def __init__(self, precision='full'):
        self.precision = precision
        if precision == 'full':
            self.decimals = None
        else:
            self.decimals = int(precision.replace('dp', ''))

    def compress(self, df, chunk_report=None):
        df = df.reset_index(drop=True)
        n_rows = len(df)

        group_col = self._find_group_column(df.columns.tolist())

        numeric_cols = []
        string_cols = []
        int_cols = []

        for col in df.columns:
            if col == group_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    int_cols.append(col)
                numeric_cols.append(col)
            else:
                string_cols.append(col)

        if group_col is not None:
            group_values = df[group_col].astype(str).values
            groups, group_starts, group_lengths = self._compute_groups(group_values)
        else:
            groups = ['__ALL__']
            group_starts = [0]
            group_lengths = [n_rows]

        n_groups = len(groups)
        scale = 10.0 ** self.decimals if self.decimals is not None else 1.0

        save_dict = {}
        col_nan_info = {}

        for col in numeric_cols:
            raw = df[col].values.astype(np.float64)
            nan_mask = np.isnan(raw)
            has_nans = bool(nan_mask.any())

            filled = raw.copy()
            for g_idx in range(n_groups):
                s = group_starts[g_idx]
                e = s + group_lengths[g_idx]
                chunk = filled[s:e]
                last = np.nan
                for i in range(len(chunk)):
                    if np.isnan(chunk[i]):
                        if not np.isnan(last):
                            chunk[i] = last
                    else:
                        last = chunk[i]
                first = np.nan
                for i in range(len(chunk)):
                    if not np.isnan(chunk[i]):
                        first = chunk[i]
                        break
                if not np.isnan(first):
                    for i in range(len(chunk)):
                        if np.isnan(chunk[i]):
                            chunk[i] = first
                        else:
                            break
                filled[s:e] = chunk

            col_starts = np.empty(n_groups, dtype=np.float64)
            delta_parts = []
            for g_idx in range(n_groups):
                s = group_starts[g_idx]
                g_len = group_lengths[g_idx]
                e = s + g_len
                g_filled = filled[s:e]
                col_starts[g_idx] = g_filled[0] if g_len > 0 else 0.0
                if g_len > 1:
                    delta_parts.append(np.diff(g_filled))

            if delta_parts:
                all_deltas = np.concatenate(delta_parts)
            else:
                all_deltas = np.array([], dtype=np.float64)

            if self.decimals is not None:
                col_starts = np.round(col_starts * scale).astype(np.int64)
                all_deltas = np.round(all_deltas * scale).astype(np.int64)
                if all_deltas.size > 0 and np.abs(all_deltas).max() < 2**30:
                    all_deltas = all_deltas.astype(np.int32)

            safe = col.replace(' ', '_').replace('.', '_')
            save_dict[f's_{safe}'] = col_starts
            save_dict[f'd_{safe}'] = all_deltas
            if has_nans:
                save_dict[f'n_{safe}'] = np.packbits(nan_mask)

            col_nan_info[col] = {
                'has_nans': has_nans,
                'nan_count': int(nan_mask.sum()),
                'is_int': col in int_cols,
            }

            del raw, filled, nan_mask, col_starts, all_deltas, delta_parts

        string_data = {}
        for col in string_cols:
            string_data[col] = df[col].astype(str).tolist()

        meta = {
            'v': self.VERSION,
            'p': self.precision,
            'nr': n_rows,
            'ng': n_groups,
            'g': groups,
            'gl': [int(x) for x in group_lengths],
            'gc': group_col if group_col else '',
            'nc': numeric_cols,
            'sc': string_cols,
            'ic': int_cols,
            'co': df.columns.tolist(),
            'ni': col_nan_info,
        }

        save_dict['meta'] = np.void(json.dumps(meta).encode('utf-8'))
        save_dict['scale'] = np.array([scale], dtype=np.float64)
        save_dict['strs'] = np.void(json.dumps(string_data).encode('utf-8'))

        if group_col:
            save_dict['gv'] = np.array(df[group_col].astype(str).values, dtype=object)

        buf = BytesIO()
        np.savez_compressed(buf, **save_dict)
        compressed = buf.getvalue()
        del save_dict, buf

        return zstd.compress(compressed, level=19)

    @staticmethod
    def decompress(compressed):
        raw = zstd.decompress(compressed)
        data = np.load(BytesIO(raw), allow_pickle=True)
        del raw

        if 'meta' in data and 'scale' in data and 'strs' in data:
            return XTPS._decompress_v50(data)

        raise ValueError("Unknown XTPS format")

    @staticmethod
    def _decompress_v50(data):
        meta = json.loads(bytes(data['meta']).decode('utf-8'))

        n_rows = meta['nr']
        n_groups = meta['ng']
        group_lengths = meta['gl']
        group_col = meta['gc'] if meta['gc'] else None
        numeric_cols = meta['nc']
        string_cols = meta['sc']
        int_cols = meta['ic']
        column_order = meta['co']
        col_nan_info = meta['ni']
        scale = float(data['scale'][0])

        result = {}

        for col in numeric_cols:
            safe = col.replace(' ', '_').replace('.', '_')
            info = col_nan_info[col]

            starts = data[f's_{safe}'].astype(np.float64)
            deltas = data[f'd_{safe}'].astype(np.float64)

            if scale != 1.0:
                starts = starts / scale
                deltas = deltas / scale

            prices = np.empty(n_rows, dtype=np.float64)
            d_off = 0
            p_off = 0

            for g_idx in range(n_groups):
                g_len = group_lengths[g_idx]
                nd = g_len - 1
                prices[p_off] = starts[g_idx]
                if nd > 0:
                    gd = deltas[d_off:d_off + nd]
                    prices[p_off + 1:p_off + g_len] = starts[g_idx] + np.cumsum(gd)
                    d_off += nd
                p_off += g_len

            if info['has_nans']:
                packed = data[f'n_{safe}']
                mask = np.unpackbits(packed)[:n_rows].astype(bool)
                prices[mask] = np.nan

            if info['is_int']:
                if info['has_nans']:
                    result[col] = pd.array(
                        [int(x) if not np.isnan(x) else pd.NA for x in prices],
                        dtype='Int64'
                    )
                else:
                    result[col] = prices.astype(np.int64)
            else:
                result[col] = prices

            del starts, deltas, prices

        string_data = json.loads(bytes(data['strs']).decode('utf-8'))
        for col in string_cols:
            vals = string_data[col]
            result[col] = [np.nan if v in ('nan', 'None', 'NaN', '') else v for v in vals]

        if group_col and 'gv' in data:
            result[group_col] = [str(x) for x in data['gv']]

        df = pd.DataFrame(result)
        del result

        final = [c for c in column_order if c in df.columns]
        for c in df.columns:
            if c not in final:
                final.append(c)

        return df[final]

    @staticmethod
    def _find_group_column(columns):
        for c in columns:
            if str(c).lower().strip() in ['index', 'symbol', 'ticker', 'stock', 'name']:
                return c
        return None

    @staticmethod
    def _compute_groups(group_values):
        groups = []
        starts = []
        lengths = []
        if len(group_values) == 0:
            return groups, starts, lengths
        cur = group_values[0]
        s = 0
        for i in range(1, len(group_values)):
            if group_values[i] != cur:
                groups.append(str(cur))
                starts.append(s)
                lengths.append(i - s)
                cur = group_values[i]
                s = i
        groups.append(str(cur))
        starts.append(s)
        lengths.append(len(group_values) - s)
        return groups, starts, lengths

    @staticmethod
    def to_csv_bytes(df):
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                nn = out[col].dropna()
                if len(nn) > 0 and (nn == nn.round(0)).all() and (nn.abs() < 2**53).all():
                    out[col] = out[col].astype('Int64')
        return out.to_csv(index=False).encode('utf-8')


def main():
    st.set_page_config(page_title="XTPS v5.0", page_icon="⚡", layout="wide")
    st.title("⚡ XTPS v5.0 — Lossless Financial Compressor")

    tab1, tab2 = st.tabs(["🚀 Compress", "📥 Decompress"])

    with tab1:
        uploaded = st.file_uploader("Upload CSV", type="csv", key="comp")
        if uploaded:
            sample = pd.read_csv(uploaded, nrows=5)
            uploaded.seek(0)
            st.dataframe(sample, use_container_width=True)

            precision = st.selectbox(
                "Precision",
                ['full', '6dp', '4dp', '2dp'],
                format_func=lambda x: {
                    'full': 'Full float64 — 100% lossless',
                    '6dp': '6 decimals',
                    '4dp': '4 decimals',
                    '2dp': '2 decimals',
                }[x]
            )

            if st.button("Compress", type="primary", use_container_width=True):
                with st.spinner("Reading CSV..."):
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded)
                    orig_size = uploaded.size

                st.info(f"{len(df):,} rows × {len(df.columns)} columns | {df.isnull().sum().sum():,} NaN values")

                with st.spinner("Compressing..."):
                    comp = XTPS(precision)
                    compressed = comp.compress(df)
                    del df

                ratio = orig_size / len(compressed)
                c1, c2, c3 = st.columns(3)
                c1.metric("Original", f"{orig_size:,} B")
                c2.metric("Compressed", f"{len(compressed):,} B")
                c3.metric("Ratio", f"{ratio:.1f}×")

                st.download_button(
                    "Download .xtps",
                    compressed,
                    f"data_{ratio:.0f}x.xtps",
                    "application/octet-stream",
                    use_container_width=True,
                )

    with tab2:
        st.caption("All settings auto-detected from file. No configuration needed.")
        xtps = st.file_uploader("Upload .xtps", type="xtps", key="dec")
        if xtps:
            st.info(f"{xtps.name} — {xtps.size:,} bytes")
            if st.button("Decompress", type="primary", use_container_width=True):
                with st.spinner("Decompressing..."):
                    raw = xtps.read()
                    df = XTPS.decompress(raw)
                    del raw

                st.success(f"{len(df):,} rows × {len(df.columns)} columns recovered")

                st.markdown("**First 10 rows**")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("**Last 10 rows**")
                st.dataframe(df.tail(10), use_container_width=True)

                nulls = df.isnull().sum()
                if nulls.sum() > 0:
                    st.markdown("**Preserved NaN counts**")
                    st.dataframe(nulls[nulls > 0].to_frame("NaN"), use_container_width=True)

                csv = XTPS.to_csv_bytes(df)
                st.download_button(
                    "Download CSV",
                    csv,
                    "recovered_data.csv",
                    "text/csv",
                    use_container_width=True,
                )
                del df


if __name__ == "__main__":
    main()

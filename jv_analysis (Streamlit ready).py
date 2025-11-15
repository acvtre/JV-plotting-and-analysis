import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime


# ---------- PV PARAMETER CALCULATION ----------

def calculate_pv_parameters(voltage, current):
    """
    Calculate PV parameters from JV curve data.

    Returns dict with keys: 'Voc', 'Isc', 'FF', 'Pmax', 'Vmp', 'Imp'
    """
    voltage = np.array(voltage)
    current = np.array(current)

    if len(voltage) < 2 or len(current) < 2:
        return {'Voc': np.nan, 'Isc': np.nan, 'FF': np.nan,
                'Pmax': np.nan, 'Vmp': np.nan, 'Imp': np.nan}

    # Voc: interpolate to find voltage when current = 0
    if np.min(current) <= 0 <= np.max(current):
        Voc = np.interp(0, current, voltage)
    else:
        Voc = voltage[np.argmin(np.abs(current))]

    # Isc: interpolate to find current when voltage = 0
    if np.min(voltage) <= 0 <= np.max(voltage):
        Isc = np.interp(0, voltage, current)
    else:
        Isc = current[np.argmin(np.abs(voltage))]

    power = voltage * current

    # Handle sign convention: negative current => min(power) is max magnitude
    if np.mean(current) < 0:
        max_power_idx = np.argmin(power)
    else:
        max_power_idx = np.argmax(power)

    Pmax = abs(power[max_power_idx])
    Vmp = voltage[max_power_idx]
    Imp = current[max_power_idx]

    if Voc != 0 and Isc != 0:
        FF = (Pmax / abs(Voc * Isc)) * 100
    else:
        FF = np.nan

    return {
        'Voc': Voc,
        'Isc': Isc,
        'FF': FF,
        'Pmax': Pmax,
        'Vmp': Vmp,
        'Imp': Imp
    }


# ---------- COLUMN DETECTION HELPERS ----------

def is_voltage_column(col_name: str) -> bool:
    col_lower = col_name.lower()
    return any([
        'voltage' in col_lower,
        col_lower.startswith('v '),
        col_lower.startswith('v('),
        col_lower.startswith('v_'),
        col_lower == 'v',
        'volt' in col_lower,
        col_lower.endswith(' v'),
        col_lower.endswith('(v)'),
        col_lower.endswith('[v]')
    ])


def is_current_column(col_name: str) -> bool:
    col_lower = col_name.lower()
    return any([
        'current' in col_lower,
        col_lower.startswith('i '),
        col_lower.startswith('i('),
        col_lower.startswith('i_'),
        col_lower == 'i',
        'amp' in col_lower,
        col_lower.endswith(' a'),
        col_lower.endswith('(a)'),
        col_lower.endswith('[a]'),
        col_lower.endswith(' ma'),
        col_lower.endswith('(ma)'),
        col_lower.endswith('[ma]'),
        'ma' in col_lower and 'max' not in col_lower,
    ])


def extract_identifier(v_col: str, i_col: str) -> str | None:
    v_lower = v_col.lower()
    i_lower = i_col.lower()

    for keyword in ['2w', '4w', 'ref', 'sample', 'cell', 'device']:
        if keyword in v_lower and keyword in i_lower:
            parts = v_col.split('_')
            for part in parts:
                if keyword in part.lower():
                    return part

    v_match = re.search(r'[\(\[]([^\)\]]+)[\)\]]', v_col)
    if v_match:
        return v_match.group(1)

    if '_' in v_col:
        parts = v_col.split('_')
        return '_'.join(parts[1:])

    return None


# ---------- CORE PROCESSING (NO DISK I/O) ----------

def process_uploaded_files(uploaded_files):
    """
    Process a list of Streamlit UploadedFile objects.

    Returns:
        - pv_parameters_df: DataFrame of parameters per curve
        - stats_df: DataFrame of statistics
        - fig: matplotlib Figure with all JV curves
    """
    pv_parameters = []

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.replace('.csv', '')

        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except Exception:
            uploaded_file.seek(0)  # reset pointer
            df = pd.read_csv(uploaded_file)

        columns = df.columns.tolist()
        pairs = []

        # Strategy 1: sequential pairing (V, I, V, I, ...)
        for i in range(0, len(columns) - 1, 2):
            v_col = columns[i]
            i_col = columns[i + 1]
            if is_voltage_column(v_col) and is_current_column(i_col):
                identifier = extract_identifier(v_col, i_col)
                label = f"{file_name} - {identifier}" if identifier else f"{file_name} - Curve {len(pairs) + 1}"
                pairs.append((v_col, i_col, label))

        # Strategy 2: match by proximity if none found
        if len(pairs) == 0:
            voltage_cols = [(i, col) for i, col in enumerate(columns) if is_voltage_column(col)]
            current_cols = [(i, col) for i, col in enumerate(columns) if is_current_column(col)]

            if len(voltage_cols) > 0 and len(current_cols) > 0:
                for v_idx, v_col in voltage_cols:
                    candidates_after = [(i_idx, i_col) for i_idx, i_col in current_cols if i_idx > v_idx]
                    if candidates_after:
                        i_idx, i_col = min(candidates_after, key=lambda x: x[0])
                    else:
                        candidates_before = [(i_idx, i_col) for i_idx, i_col in current_cols if i_idx < v_idx]
                        if not candidates_before:
                            continue
                        i_idx, i_col = max(candidates_before, key=lambda x: x[0])

                    identifier = extract_identifier(v_col, i_col)
                    label = f"{file_name} - {identifier}" if identifier else f"{file_name} - Curve {len(pairs) + 1}"
                    pairs.append((v_col, i_col, label))

        if len(pairs) == 0:
            st.warning(
                f"Could not identify V–I pairs in **{uploaded_file.name}**.\n"
                f"Columns: {columns}"
            )
            continue

        # Plot and compute parameters per pair
        for v_col, i_col, label in pairs:
            voltage = df[v_col].dropna()
            current = df[i_col].dropna()

            min_len = min(len(voltage), len(current))
            voltage = voltage.iloc[:min_len]
            current = current.iloc[:min_len]

            # convert A → mA if needed
            if current.abs().mean() < 1:
                current = current * 1000

            params = calculate_pv_parameters(voltage, current)
            pv_parameters.append({
                'Curve': label,
                'File': file_name,
                'Voc (V)': params['Voc'],
                'Isc (mA)': params['Isc'],
                'FF (%)': params['FF'],
                'Pmax (mW)': params['Pmax'],
                'Vmp (V)': params['Vmp'],
                'Imp (mA)': params['Imp']
            })

            ax.plot(
                voltage,
                current,
                marker='o',
                markersize=3,
                linewidth=1.5,
                label=label,
                color=colors[color_idx % len(colors)],
                alpha=0.7,
            )
            color_idx += 1

    if len(pv_parameters) == 0:
        return None, None, None

    ax.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (mA)', fontsize=12, fontweight='bold')
    ax.set_title('JV Curves Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    pv_df = pd.DataFrame(pv_parameters)
    stats_df = compute_statistics(pv_df)

    return pv_df, stats_df, fig


# ---------- STATISTICS (IN-MEMORY VERSION) ----------

def compute_statistics(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics similar to your original save_parameters_csv logic.
    Returns a DataFrame 'df_stats'.
    """
    stats_data = []

    unique_files = df_params['File'].unique()

    if len(unique_files) > 1:
        file_stats_list = []

        for file in unique_files:
            file_data = df_params[df_params['File'] == file]
            file_stats = {
                'Group': file,
                'N_curves': len(file_data),
                'Voc_median (V)': file_data['Voc (V)'].median(),
                'Voc_std (V)': file_data['Voc (V)'].std(),
                'Isc_median (mA)': file_data['Isc (mA)'].median(),
                'Isc_std (mA)': file_data['Isc (mA)'].std(),
                'FF_median (%)': file_data['FF (%)'].median(),
                'FF_std (%)': file_data['FF (%)'].std(),
                'Pmax_median (mW)': file_data['Pmax (mW)'].median(),
                'Pmax_std (mW)': file_data['Pmax (mW)'].std()
            }
            stats_data.append(file_stats)
            file_stats_list.append(file_stats)

        # If exactly two files, add percentage difference row
        if len(unique_files) == 2:
            f1, f2 = file_stats_list

            def calc_percent_diff(val1, val2):
                if val1 == 0 or pd.isna(val1) or pd.isna(val2):
                    return np.nan
                return ((val2 - val1) / abs(val1)) * 100

            comparison = {
                'Group': f'Δ% ({unique_files[1]} vs {unique_files[0]})',
                'N_curves': np.nan,
                'Voc_median (V)': calc_percent_diff(f1['Voc_median (V)'], f2['Voc_median (V)']),
                'Voc_std (V)': calc_percent_diff(f1['Voc_std (V)'], f2['Voc_std (V)']),
                'Isc_median (mA)': calc_percent_diff(f1['Isc_median (mA)'], f2['Isc_median (mA)']),
                'Isc_std (mA)': calc_percent_diff(f1['Isc_std (mA)'], f2['Isc_std (mA)']),
                'FF_median (%)': calc_percent_diff(f1['FF_median (%)'], f2['FF_median (%)']),
                'FF_std (%)': calc_percent_diff(f1['FF_std (%)'], f2['FF_std (%)']),
                'Pmax_median (mW)': calc_percent_diff(f1['Pmax_median (mW)'], f2['Pmax_median (mW)']),
                'Pmax_std (mW)': calc_percent_diff(f1['Pmax_std (mW)'], f2['Pmax_std (mW)']),
            }
            stats_data.append(comparison)

        # Overall stats across all curves
        stats_data.append({
            'Group': 'ALL',
            'N_curves': len(df_params),
            'Voc_median (V)': df_params['Voc (V)'].median(),
            'Voc_std (V)': df_params['Voc (V)'].std(),
            'Isc_median (mA)': df_params['Isc (mA)'].median(),
            'Isc_std (mA)': df_params['Isc (mA)'].std(),
            'FF_median (%)': df_params['FF (%)'].median(),
            'FF_std (%)': df_params['FF (%)'].std(),
            'Pmax_median (mW)': df_params['Pmax (mW)'].median(),
            'Pmax_std (mW)': df_params['Pmax (mW)'].std(),
        })

    else:
        group_name = unique_files[0] if len(unique_files) > 0 else 'ALL'
        stats_data.append({
            'Group': group_name,
            'N_curves': len(df_params),
            'Voc_median (V)': df_params['Voc (V)'].median(),
            'Voc_std (V)': df_params['Voc (V)'].std(),
            'Isc_median (mA)': df_params['Isc (mA)'].median(),
            'Isc_std (mA)': df_params['Isc (mA)'].std(),
            'FF_median (%)': df_params['FF (%)'].median(),
            'FF_std (%)': df_params['FF (%)'].std(),
            'Pmax_median (mW)': df_params['Pmax (mW)'].median(),
            'Pmax_std (mW)': df_params['Pmax (mW)'].std(),
        })

    return pd.DataFrame(stats_data)


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="JV Curves Plotter", layout="wide")

    st.title("JV Curves Plotting & PV Parameter Analysis")

    st.markdown(
        "Upload one or more CSV files containing JV data. "
        "The app will try to detect voltage/current columns, plot all curves, "
        "and compute PV parameters (Voc, Isc, FF, Pmax, Vmp, Imp)."
    )

    uploaded_files = st.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 Upload at least one CSV file to begin.")
        return

    if st.button("Process JV Curves"):
        with st.spinner("Processing files..."):
            pv_df, stats_df, fig = process_uploaded_files(uploaded_files)

        if pv_df is None:
            st.error("No valid JV curves found in the uploaded files.")
            return

        # Plot
        st.subheader("JV Curves")
        st.pyplot(fig)

        # Parameters table
        st.subheader("PV Parameters per Curve")
        st.dataframe(pv_df, use_container_width=True)

        # Statistics table
        st.subheader("Statistics (Median & Std Dev)")
        st.dataframe(stats_df, use_container_width=True)

        # Download buttons
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        params_csv = pv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download PV Parameters CSV",
            data=params_csv,
            file_name=f"pv_parameters_{ts}.csv",
            mime="text/csv",
        )

        stats_csv = stats_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Statistics CSV",
            data=stats_csv,
            file_name=f"pv_statistics_{ts}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

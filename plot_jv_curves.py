import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
from datetime import datetime

def calculate_pv_parameters(voltage, current):
    """
    Calculate PV parameters from JV curve data.

    Parameters:
    -----------
    voltage : array-like
        Voltage values in V
    current : array-like
        Current values in mA

    Returns:
    --------
    dict with keys: 'Voc', 'Isc', 'FF', 'Pmax', 'Vmp', 'Imp'
    """
    voltage = np.array(voltage)
    current = np.array(current)

    # Calculate Voc (voltage at zero current)
    # Find zero crossing by interpolation
    if len(voltage) < 2 or len(current) < 2:
        return {'Voc': np.nan, 'Isc': np.nan, 'FF': np.nan, 'Pmax': np.nan, 'Vmp': np.nan, 'Imp': np.nan}

    # Voc: interpolate to find voltage when current = 0
    if np.min(current) <= 0 <= np.max(current):
        Voc = np.interp(0, current, voltage)
    else:
        # If no zero crossing, take the voltage at minimum absolute current
        Voc = voltage[np.argmin(np.abs(current))]

    # Isc: interpolate to find current when voltage = 0
    if np.min(voltage) <= 0 <= np.max(voltage):
        Isc = np.interp(0, voltage, current)
    else:
        # If no zero crossing, take the current at minimum absolute voltage
        Isc = current[np.argmin(np.abs(voltage))]

    # Calculate power (P = V * I, in mW since I is in mA)
    power = voltage * current

    # Find maximum power point
    # For negative current convention (solar cells), power is negative, so find minimum (most negative)
    # For positive current convention, find maximum
    if np.mean(current) < 0:
        # Negative current convention - find minimum power (most negative = highest magnitude)
        max_power_idx = np.argmin(power)
    else:
        # Positive current convention - find maximum power
        max_power_idx = np.argmax(power)

    Pmax = abs(power[max_power_idx])  # Take absolute value for reporting
    Vmp = voltage[max_power_idx]
    Imp = current[max_power_idx]

    # Calculate Fill Factor: FF = Pmax / (|Voc * Isc|) * 100%
    # Use absolute values to handle both current conventions
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


def plot_jv_curves(file_paths, output_file='jv_curves_plot.png'):
    """
    Plot JV curves from multiple CSV files with different data organizations.

    Parameters:
    -----------
    file_paths : list
        List of paths to CSV files containing JV curve data
    output_file : str
        Name of the output plot file
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    # Store PV parameters for each curve
    pv_parameters = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue

        print(f"\nProcessing file: {file_path}")

        # Read the CSV file
        try:
            # Try reading with UTF-8 BOM encoding
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except:
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # Get file basename for legend
        file_name = os.path.basename(file_path).replace('.csv', '')

        # Identify voltage and current column pairs
        columns = df.columns.tolist()
        pairs = []

        def is_voltage_column(col_name):
            """Check if column name indicates voltage data"""
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

        def is_current_column(col_name):
            """Check if column name indicates current data"""
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
                'ma' in col_lower and 'max' not in col_lower
            ])

        def extract_identifier(v_col, i_col):
            """Extract a unique identifier from column names"""
            # Try to find common suffix/identifier in both column names
            v_lower = v_col.lower()
            i_lower = i_col.lower()

            # Look for common patterns like "2W", "4W", "ref", "sample 1", etc.
            for keyword in ['2w', '4w', 'ref', 'sample', 'cell', 'device']:
                if keyword in v_lower and keyword in i_lower:
                    # Extract the full identifier around this keyword
                    parts = v_col.split('_')
                    for part in parts:
                        if keyword in part.lower():
                            return part

            # Try to extract from parentheses or brackets
            v_match = re.search(r'[\(\[]([^\)\]]+)[\)\]]', v_col)
            if v_match:
                return v_match.group(1)

            # Try to extract suffix after underscore
            if '_' in v_col:
                parts = v_col.split('_')
                return '_'.join(parts[1:])

            return None

        # Strategy 1: Sequential pairing (V, I, V, I, ...)
        for i in range(0, len(columns) - 1, 2):
            v_col = columns[i]
            i_col = columns[i + 1]

            if is_voltage_column(v_col) and is_current_column(i_col):
                identifier = extract_identifier(v_col, i_col)
                if identifier:
                    label = f"{file_name} - {identifier}"
                else:
                    label = f"{file_name} - Curve {len(pairs) + 1}"

                pairs.append((v_col, i_col, label))

        # Strategy 2: If no pairs found, try matching by index/suffix
        if len(pairs) == 0:
            voltage_cols = [(i, col) for i, col in enumerate(columns) if is_voltage_column(col)]
            current_cols = [(i, col) for i, col in enumerate(columns) if is_current_column(col)]

            print(f"  Found {len(voltage_cols)} voltage columns and {len(current_cols)} current columns")

            if len(voltage_cols) > 0 and len(current_cols) > 0:
                # Try to match by proximity (closest current column to each voltage column)
                for v_idx, v_col in voltage_cols:
                    # Find the nearest current column after this voltage column
                    candidates = [(i_idx, i_col) for i_idx, i_col in current_cols if i_idx > v_idx]
                    if candidates:
                        i_idx, i_col = min(candidates, key=lambda x: x[0])

                        identifier = extract_identifier(v_col, i_col)
                        if identifier:
                            label = f"{file_name} - {identifier}"
                        else:
                            label = f"{file_name} - Curve {len(pairs) + 1}"

                        pairs.append((v_col, i_col, label))
                    else:
                        # Try the closest current column before this voltage column
                        candidates = [(i_idx, i_col) for i_idx, i_col in current_cols if i_idx < v_idx]
                        if candidates:
                            i_idx, i_col = max(candidates, key=lambda x: x[0])

                            identifier = extract_identifier(v_col, i_col)
                            if identifier:
                                label = f"{file_name} - {identifier}"
                            else:
                                label = f"{file_name} - Curve {len(pairs) + 1}"

                            pairs.append((v_col, i_col, label))

        # If still no pairs found, report error
        if len(pairs) == 0:
            print(f"  Warning: Could not identify V-I pairs in {file_path}")
            print(f"  Columns: {columns}")
            print(f"  Please check that columns are named with 'voltage'/'current' or 'V'/'I'")
            continue

        print(f"Found {len(pairs)} voltage-current pairs")

        # Plot each pair
        for v_col, i_col, label in pairs:
            # Get data and remove NaN values
            voltage = df[v_col].dropna()
            current = df[i_col].dropna()

            # Ensure same length
            min_len = min(len(voltage), len(current))
            voltage = voltage.iloc[:min_len]
            current = current.iloc[:min_len]

            # Convert current to milliampere if needed
            # Check the magnitude to determine if conversion is needed
            if current.abs().mean() < 1:  # Likely in Amperes
                current = current * 1000  # Convert to mA

            # Calculate PV parameters
            params = calculate_pv_parameters(voltage, current)
            pv_parameters.append({
                'Curve': label,
                'Voc (V)': params['Voc'],
                'Isc (mA)': params['Isc'],
                'FF (%)': params['FF'],
                'Pmax (mW)': params['Pmax'],
                'Vmp (V)': params['Vmp'],
                'Imp (mA)': params['Imp']
            })

            # Plot
            ax.plot(voltage, current, marker='o', markersize=3,
                   linewidth=1.5, label=label, color=colors[color_idx % len(colors)],
                   alpha=0.7)

            color_idx += 1

    # Formatting
    ax.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (mA)', fontsize=12, fontweight='bold')
    ax.set_title('JV Curves Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    plt.show()

    # Save PV parameters to CSV with statistics
    if len(pv_parameters) > 0:
        save_parameters_csv(pv_parameters, output_file)

    return pv_parameters


def save_parameters_csv(pv_parameters, jv_plot_file):
    """
    Save PV parameters to CSV with median and standard deviation.

    Parameters:
    -----------
    pv_parameters : list of dict
        List of dictionaries containing PV parameters for each curve
    jv_plot_file : str
        Path to the JV plot file (used to generate CSV filename)
    """
    # Create DataFrame from parameters
    df_params = pd.DataFrame(pv_parameters)

    # Extract file names from curve labels to group by file
    df_params['File'] = df_params['Curve'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else 'Unknown')

    # Save individual curve parameters
    csv_file = jv_plot_file.replace('.png', '_parameters.csv')
    df_params.to_csv(csv_file, index=False)
    print(f"\nParameters data saved as: {csv_file}")

    # Calculate statistics
    stats_data = []

    # Get unique files
    unique_files = df_params['File'].unique()

    # If multiple files, calculate statistics per file AND across all data
    if len(unique_files) > 1:
        print("\n--- Statistics by File ---")
        file_stats_list = []  # Store for comparison calculation

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

            print(f"\n{file}:")
            print(f"  Voc: {file_data['Voc (V)'].median():.4f} ± {file_data['Voc (V)'].std():.4f} V")
            print(f"  Isc: {file_data['Isc (mA)'].median():.4f} ± {file_data['Isc (mA)'].std():.4f} mA")
            print(f"  FF:  {file_data['FF (%)'].median():.2f} ± {file_data['FF (%)'].std():.2f} %")
            print(f"  Pmax: {file_data['Pmax (mW)'].median():.4f} ± {file_data['Pmax (mW)'].std():.4f} mW")

        # Add comparison row if exactly 2 files
        if len(unique_files) == 2:
            file1_stats = file_stats_list[0]
            file2_stats = file_stats_list[1]

            def calc_percent_diff(val1, val2):
                """Calculate percentage difference between two values"""
                if val1 == 0 or pd.isna(val1) or pd.isna(val2):
                    return np.nan
                return ((val2 - val1) / abs(val1)) * 100

            comparison = {
                'Group': f'Δ% ({unique_files[1]} vs {unique_files[0]})',
                'N_curves': np.nan,
                'Voc_median (V)': calc_percent_diff(file1_stats['Voc_median (V)'], file2_stats['Voc_median (V)']),
                'Voc_std (V)': calc_percent_diff(file1_stats['Voc_std (V)'], file2_stats['Voc_std (V)']),
                'Isc_median (mA)': calc_percent_diff(file1_stats['Isc_median (mA)'], file2_stats['Isc_median (mA)']),
                'Isc_std (mA)': calc_percent_diff(file1_stats['Isc_std (mA)'], file2_stats['Isc_std (mA)']),
                'FF_median (%)': calc_percent_diff(file1_stats['FF_median (%)'], file2_stats['FF_median (%)']),
                'FF_std (%)': calc_percent_diff(file1_stats['FF_std (%)'], file2_stats['FF_std (%)']),
                'Pmax_median (mW)': calc_percent_diff(file1_stats['Pmax_median (mW)'], file2_stats['Pmax_median (mW)']),
                'Pmax_std (mW)': calc_percent_diff(file1_stats['Pmax_std (mW)'], file2_stats['Pmax_std (mW)'])
            }
            stats_data.append(comparison)

            print(f"\n--- Comparison: {unique_files[1]} vs {unique_files[0]} ---")
            print(f"  Voc median: {comparison['Voc_median (V)']:+.2f}%")
            print(f"  Voc std: {comparison['Voc_std (V)']:+.2f}%")
            print(f"  Isc median: {comparison['Isc_median (mA)']:+.2f}%")
            print(f"  Isc std: {comparison['Isc_std (mA)']:+.2f}%")
            print(f"  FF median: {comparison['FF_median (%)']:+.2f}%")
            print(f"  FF std: {comparison['FF_std (%)']:+.2f}%")
            print(f"  Pmax median: {comparison['Pmax_median (mW)']:+.2f}%")
            print(f"  Pmax std: {comparison['Pmax_std (mW)']:+.2f}%")

        # Overall statistics across all files
        print("\n--- Overall Statistics (All Curves) ---")
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
            'Pmax_std (mW)': df_params['Pmax (mW)'].std()
        })

        print(f"  Voc: {df_params['Voc (V)'].median():.4f} ± {df_params['Voc (V)'].std():.4f} V")
        print(f"  Isc: {df_params['Isc (mA)'].median():.4f} ± {df_params['Isc (mA)'].std():.4f} mA")
        print(f"  FF:  {df_params['FF (%)'].median():.2f} ± {df_params['FF (%)'].std():.2f} %")
        print(f"  Pmax: {df_params['Pmax (mW)'].median():.4f} ± {df_params['Pmax (mW)'].std():.4f} mW")

    else:
        # Single file - just calculate overall statistics
        print("\n--- Statistics (All Curves) ---")
        stats_data.append({
            'Group': unique_files[0] if len(unique_files) > 0 else 'ALL',
            'N_curves': len(df_params),
            'Voc_median (V)': df_params['Voc (V)'].median(),
            'Voc_std (V)': df_params['Voc (V)'].std(),
            'Isc_median (mA)': df_params['Isc (mA)'].median(),
            'Isc_std (mA)': df_params['Isc (mA)'].std(),
            'FF_median (%)': df_params['FF (%)'].median(),
            'FF_std (%)': df_params['FF (%)'].std(),
            'Pmax_median (mW)': df_params['Pmax (mW)'].median(),
            'Pmax_std (mW)': df_params['Pmax (mW)'].std()
        })

        print(f"  Voc: {df_params['Voc (V)'].median():.4f} ± {df_params['Voc (V)'].std():.4f} V")
        print(f"  Isc: {df_params['Isc (mA)'].median():.4f} ± {df_params['Isc (mA)'].std():.4f} mA")
        print(f"  FF:  {df_params['FF (%)'].median():.2f} ± {df_params['FF (%)'].std():.2f} %")
        print(f"  Pmax: {df_params['Pmax (mW)'].median():.4f} ± {df_params['Pmax (mW)'].std():.4f} mW")

    # Save statistics to CSV
    df_stats = pd.DataFrame(stats_data)
    stats_csv_file = jv_plot_file.replace('.png', '_statistics.csv')
    df_stats.to_csv(stats_csv_file, index=False)
    print(f"\nStatistics saved as: {stats_csv_file}")

    # Create statistics table image
    create_statistics_table_image(df_stats, jv_plot_file)


def create_statistics_table_image(df_stats, jv_plot_file):
    """
    Create and save an image of the statistics table.

    Parameters:
    -----------
    df_stats : DataFrame
        DataFrame containing statistics
    jv_plot_file : str
        Path to the JV plot file (used to generate table filename)
    """
    # Format the dataframe for display
    df_display = df_stats.copy()

    # Format numeric columns to show fewer decimal places
    for col in df_display.columns:
        if 'median' in col or 'std' in col:
            # Check if this is a comparison row (contains percentage differences)
            def format_value(x, row_group):
                if pd.isna(x):
                    return 'N/A'
                # If this is a comparison row, format as percentage with sign
                if pd.notna(row_group) and 'Δ%' in str(row_group):
                    return f'{x:+.2f}%'
                # Otherwise format as normal value
                elif 'FF' in col:
                    return f'{x:.2f}'
                else:
                    return f'{x:.4f}'

            df_display[col] = df_display.apply(lambda row: format_value(row[col], row['Group']), axis=1)
        elif col == 'N_curves':
            df_display[col] = df_display[col].apply(lambda x: f'{int(x)}' if pd.notna(x) else '-')

    # Calculate figure size based on number of rows
    n_rows = len(df_display)
    n_cols = len(df_display.columns)
    fig_height = max(3, 1 + n_rows * 0.5)
    fig_width = max(10, n_cols * 1.2)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table_data = [df_display.columns.tolist()] + df_display.values.tolist()

    # Calculate column widths dynamically
    col_widths = []
    for i, col in enumerate(df_display.columns):
        if col == 'Group':
            col_widths.append(0.25)
        elif col == 'N_curves':
            col_widths.append(0.08)
        else:
            col_widths.append(0.10)

    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Header styling
    for i in range(len(df_display.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(df_display.columns)):
            cell = table[(i, j)]
            row_group = df_display.iloc[i-1]['Group']

            # Highlight comparison row (Δ%)
            if pd.notna(row_group) and 'Δ%' in str(row_group):
                cell.set_facecolor('#E6F2FF')  # Light blue for comparison
                if j == 0:
                    cell.set_text_props(weight='bold', style='italic')
                else:
                    cell.set_text_props(weight='bold')
            # Highlight the "ALL" row if present
            elif row_group == 'ALL':
                cell.set_facecolor('#FFE699')
                if j == 0:
                    cell.set_text_props(weight='bold')
            # Regular alternating colors
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#F2F2F2')
                else:
                    cell.set_facecolor('white')

    # Add title
    plt.title('PV Parameters Statistics (Median ± Std Dev)', fontsize=14, fontweight='bold', pad=20)

    # Save table image
    table_file = jv_plot_file.replace('.png', '_statistics.png')
    plt.savefig(table_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Statistics table image saved as: {table_file}")
    plt.show()


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if files were dragged onto the script
    if len(sys.argv) > 1:
        # Files were dragged onto the script
        file_paths = sys.argv[1:]
        print(f"Received {len(file_paths)} file(s) via drag and drop:")
        for fp in file_paths:
            print(f"  - {fp}")
    else:
        # No files dragged - use default files or prompt user
        print("No files provided!")
        print("\nUsage:")
        print("  1. Drag and drop CSV files onto this script")
        print("  2. Or run from command line: python plot_jv_curves.py file1.csv file2.csv ...")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

    # Filter to only include CSV files
    csv_files = [f for f in file_paths if f.lower().endswith('.csv')]

    if len(csv_files) == 0:
        print("\nError: No CSV files found in the provided files!")
        print("Press Enter to exit...")
        input()
        sys.exit(1)

    print(f"\nProcessing {len(csv_files)} CSV file(s)...")

    # Generate output filename based on input CSV files
    if len(csv_files) == 1:
        # Single file - use its name
        base_name = os.path.splitext(os.path.basename(csv_files[0]))[0]
        output_file = os.path.join(script_dir, f'{base_name}_jv_curves.png')
    else:
        # Multiple files - combine names or use timestamp
        if len(csv_files) <= 3:
            # For 2-3 files, combine names
            names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
            combined_name = '_vs_'.join(names)
            # Limit filename length
            if len(combined_name) > 100:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(script_dir, f'jv_curves_{timestamp}.png')
            else:
                output_file = os.path.join(script_dir, f'{combined_name}_jv_curves.png')
        else:
            # For many files, use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(script_dir, f'jv_curves_{timestamp}.png')

    # Create the plot
    try:
        plot_jv_curves(csv_files, output_file=output_file)
        print("\nSuccess! Press Enter to exit...")
        input()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

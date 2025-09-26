# Script to copy and modify `eloutRelevantData.csv` to plot the pressure vs. radial distance plots in one figure
# Assumptions: User input n is an integer
# Haena Lee, September 2025

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

#return the actual column name in df that case-insensitively matches target_lower
def find_column(df, target_lower: str) -> str:
    for c in df.columns:
        if c.strip().lower() == target_lower:
            return c
    raise KeyError(f"Required column '{target_lower}' not found in CSV header: {list(df.columns)}.")

def main():
    #find CSV in same directory as this script
    base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    csv_in = base_dir / "eloutRelevantData.csv"
    if not csv_in.exists():
        print(f"ERROR: Could not find {csv_in}. Make sure 'eloutRelevantData.csv' is in the same folder as this script.")
        sys.exit(1)

    #read CSV
    try:
        df = pd.read_csv(csv_in)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}.")
        sys.exit(1)

    #identify required columns
    try:
        time_col = find_column(df, "time")
        elem_col = find_column(df, "element")
        sxx_col = find_column(df, "sig-xx")
        syy_col = find_column(df, "sig-yy")
        szz_col = find_column(df, "sig-zz")
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    #ensure numeric types
    for c in [time_col, elem_col, sxx_col, syy_col, szz_col]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    element_size = float(input("Enter element size (e.g., 1.0): ").strip())

    #add "radial element" column after "element" column
    radial_element_series = df.groupby(time_col).cumcount() + 1
    insert_pos_after_elem = df.columns.get_loc(elem_col) + 1
    df.insert(insert_pos_after_elem, "radial element", radial_element_series)

    #add "radial distance" column after "radial element" column
    radial_distance = (df["radial element"] - 1) * element_size + 0.5 * element_size
    insert_pos_after_radial_elem = df.columns.get_loc("radial element") + 1
    df.insert(insert_pos_after_radial_elem, "radial distance", radial_distance)

    #add "pressure" column after "sig-zz" column; pressure = -(sig-xx + sig-yy + sig-zz)/3
    pressure = -(df[sxx_col] + df[syy_col] + df[szz_col]) / 3.0
    insert_pos_after_szz = df.columns.get_loc(szz_col) + 1
    df.insert(insert_pos_after_szz, "pressure", pressure)

    #find the unique time steps
    unique_times = df[time_col].dropna().drop_duplicates().tolist()
    total_steps = len(unique_times)

    n = int(input(f"There are [{total_steps}] time steps. Plot every nth time step: ").strip())

    #plot every nth time step on the same figure
    plt.figure()
    times_to_plot = unique_times[::n]
    for t in times_to_plot:
        sub = df[df[time_col] == t]
        plt.plot(sub["radial distance"], sub["pressure"], label=f"t={t}")

    plt.xlabel("Radial Distance (cm)")
    plt.ylabel("Pressure (Mbar)")
    plt.title(f"Pressure vs. Radial Distance (every {n}ᵗʰ time step)")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    #save modified CSV
    csv_out = base_dir / "eloutRelevantData_Processed.csv"
    df.to_csv(csv_out, index=False)
    print("Saved modified file to eloutRelevantData_Processed.csv.")

    #save and show figure
    fig_out = base_dir / "eloutPressures.png"
    plt.savefig(fig_out, dpi=300)
    print("Saved figure to eloutPressures.png.")
    plt.show()

if __name__ == "__main__":
    main()
# Script to copy and modify `eloutRelevantData.csv` to plot the pressure vs. radial distance plots in one figure
# Assumptions: User input n is an integer
# Haena Lee, September 2025

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#return the actual column name in df that case-insensitively matches target_lower
def find_column(df, target_lower: str) -> str:
    for c in df.columns:
        if c.strip().lower() == target_lower:
            return c
    raise KeyError(f"Required column '{target_lower}' not found in CSV header: {list(df.columns)}.")

#parse Y/N input from user
def get_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    while ans not in {"y", "n", "yes", "no"}:
        ans = input("Please enter Y or N: ").strip().lower()
    return ans.startswith("y")

#parse comma-separated floats safely
def parse_times_list(prompt: str) -> list[float]:
    raw = input(prompt).strip()
    if not raw:
        return []
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(float(piece))
        except ValueError:
            print(f"WARNING: '{piece}' is not a valid number and will be ignored.")
    return out

#plot selected time steps
def plot_selected_times(df, time_col, times_requested, label_prefix="t="):
    unique_times = np.array(sorted(df[time_col].dropna().unique()))
    plotted = []
    plotted_df_list = []  #to export the plotted data to CSV
    for t_req in times_requested:
        idx = int(np.argmin(np.abs(unique_times - t_req)))
        t_actual = float(unique_times[idx])
        if not np.isclose(t_actual, t_req, rtol=1e-10, atol=1e-12):
            print(f"Note: requested {t_req} mapped to nearest available time {t_actual}.")
        sub = df[np.isclose(df[time_col], t_actual)]
        plt.plot(sub["radial distance"], sub["pressure"], label=f"{label_prefix}{t_actual}")
        plotted.append(t_actual)
        plotted_df_list.append(sub)
    return plotted, pd.concat(plotted_df_list, ignore_index=True) if plotted_df_list else pd.DataFrame()

def main():
    #find CSV in same dCirectory as this script
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

    print(f"There are [{total_steps}] time steps.")
    do_every_n = get_yes_no("Plot every nth time step? (Y/N): ")
    do_specific = get_yes_no("Plot specific time steps? (Y/N): ")  

    plt.figure()
    plotted_df = pd.DataFrame()  #for the plotted data

    if do_specific and not do_every_n:
        requested = parse_times_list("Time steps to plot: ")
        if not requested:
            print("No valid time steps provided.")
            sys.exit(0)
        plotted, plotted_df = plot_selected_times(df, time_col, requested)
        plt.title("Pressure vs. Radial Distance (selected time steps)")
    else:
        if not do_every_n and not do_specific:
            print("Neither option selected; defaulting to plotting every time step (n=1).")
            n = 1
        else:
            n = int(input("Plot every nth time step: ").strip())
        times_to_plot = unique_times[::n]
        plotted_df_list = []
        for t in times_to_plot:
            sub = df[np.isclose(df[time_col], t)]
            plt.plot(sub["radial distance"], sub["pressure"], label=f"t={t}")
            plotted_df_list.append(sub)
        plotted_df = pd.concat(plotted_df_list, ignore_index=True)
        plt.title(f"Pressure vs. Radial Distance (every {n}ᵗʰ time step)")

    plt.xlabel("Radial Distance (cm)")
    plt.ylabel("Pressure (Mbar)")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    #save modified CSV
    csv_out = base_dir / "eloutRelevantData_Processed.csv"
    df.to_csv(csv_out, index=False)
    print("Saved modified file to eloutRelevantData_Processed.csv.")

    #save plotted data to a CSV
    if not plotted_df.empty:
        plotted_out = base_dir / "eloutRelevantData_Processed_Plotted.csv"
        plotted_df.to_csv(plotted_out, index=False)
        print("Saved plotted data to eloutRelevantData_Processed_Plotted.csv.")

    #save and show figure
    fig_out = base_dir / "eloutPressures.png"
    plt.savefig(fig_out, dpi=300)
    print("Saved figure to eloutPressures.png.")
    plt.show()

if __name__ == "__main__":
    main()
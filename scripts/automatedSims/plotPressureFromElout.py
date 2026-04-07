# Script to plot pressure vs. radial distance from eloutExtractedData.csv
# This version uses pre-calculated pressure and radial distance columns
# Haena Lee

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# WARDLAW PARAMETERS
# ==============================
WARDLAW_REFERENCE_RADIUS = 16.0  #radius (cm) at which t=0 is defined in Wardlaw's paper

#Return the time (microsec) when shock front reaches target radius
def find_shock_time_at_radius(shock_csv_path, target_radius, radius_col="Shock front radial distance (cm)", time_col="Time (µs)"):
    try:
        shock_df = pd.read_csv(shock_csv_path)

        #Find the time when shock is closest to target radius
        idx = int(np.argmin(np.abs(shock_df[radius_col] - target_radius)))
        t_at_radius = shock_df[time_col].iloc[idx]
        r_actual = shock_df[radius_col].iloc[idx]
        
        #print(f"Shock front reaches r={target_radius} cm at t={t_at_radius:.4f} µs (actual r={r_actual:.2f} cm)")
        return t_at_radius
    
    except FileNotFoundError:
        print(f"WARNING: {shock_csv_path} not found. Using t=0 as reference.")
        return 0.0
    
    except Exception as e:
        print(f"WARNING: Could not read shock front data: {e}. Using t=0 as reference.")
        return 0.0

#Return the actual column name in df that case-insensitively matches target_lower
def find_column(df, target_lower: str) -> str:
    for c in df.columns:
        if c.strip().lower() == target_lower:
            return c
    raise KeyError(f"Required column '{target_lower}' not found in CSV header: {list(df.columns)}.")

#Parse Y/N input from user
def get_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    while ans not in {"y", "n", "yes", "no"}:
        ans = input("Please enter Y or N: ").strip().lower()
    
    return ans.startswith("y")

#Parse comma-separated floats safely
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

#Plot selected time steps
def plot_selected_times(df, time_col, radial_col, pressure_col, times_requested, label_prefix="t="):
    unique_times = np.array(sorted(df[time_col].dropna().unique()))
    plotted = []
    plotted_df_list = []  #to export the plotted data to CSV

    for t_req in times_requested:
        idx = int(np.argmin(np.abs(unique_times - t_req)))
        t_actual = float(unique_times[idx])
        if not np.isclose(t_actual, t_req, rtol=1e-10, atol=1e-12):
            print(f"Note: requested {t_req} mapped to nearest available time {t_actual}.")
        
        sub = df[np.isclose(df[time_col], t_actual)].sort_values(radial_col)
        pressure_dyne = sub[pressure_col].values * 1e12     #convert pressure from Mbar to dyne/cm^2
        plt.plot(sub[radial_col].values, pressure_dyne, 'o-', markersize=5, linewidth=2, label=f"{label_prefix}{t_actual}")
        plotted.append(t_actual)
        plotted_df_list.append(sub)

    return plotted, pd.concat(plotted_df_list, ignore_index=True) if plotted_df_list else pd.DataFrame()

def main():
    #find CSV in same directory as this script
    base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    csv_in = base_dir / "eloutExtractedData.csv"
    if not csv_in.exists():
        print(f"ERROR: Could not find {csv_in}. Make sure 'eloutExtractedData.csv' is in the same folder as this script.")
        sys.exit(1)

    #read CSV
    try:
        df = pd.read_csv(csv_in)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}.")
        sys.exit(1)

    #identify required columns (look for flexible matches)
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)
    if not time_col:
        print(f"ERROR: Could not find 'time' column in CSV")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    radial_col = next((c for c in df.columns if 'radial distance' in c.lower()), None)
    if not radial_col:
        print(f"ERROR: Could not find 'radial distance' column in CSV")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    pressure_col = next((c for c in df.columns if 'pressure' in c.lower()), None)
    if not pressure_col:
        print(f"ERROR: Could not find 'pressure' column in CSV")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Found columns: time='{time_col}', radial='{radial_col}', pressure='{pressure_col}'")

    #ensure numeric types
    for c in [time_col, radial_col, pressure_col]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    #find the unique time steps
    unique_times = sorted(df[time_col].dropna().unique())
    total_steps = len(unique_times)

    print(f"Total time steps available: {total_steps}")
    print(f"Time range: {unique_times[0]:.2e} to {unique_times[-1]:.2e}")
    
    plot_wardlaw = get_yes_no("\nPlot against Wardlaw time steps? (Y/N): ")
    
    fig = plt.figure(figsize=(11, 8))
    plotted_df = pd.DataFrame()  #for the plotted data
    
    if plot_wardlaw:
        #calculate time offset based on shock front reaching reference radius
        base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
        shock_csv = base_dir / "shockFront.csv"
    
        print(f"\nCalculating Wardlaw time offset...")
        t_offset = find_shock_time_at_radius(str(shock_csv), WARDLAW_REFERENCE_RADIUS)
        
        #Wardlaw times in seconds, convert to microseconds to match CSV
        wardlaw_times_sec = np.array([0, 2.1e-6, 4.3e-6, 8.8e-6, 1.9e-5, 4.4e-5])
        wardlaw_times_us = wardlaw_times_sec * 1e6
        
        #add time offset to Wardlaw times (Wardlaw t=0 -> our simulation time t_offset)
        wardlaw_times_adjusted = wardlaw_times_us + t_offset
        
        print(f"Wardlaw reference radius: {WARDLAW_REFERENCE_RADIUS} cm")
        print(f"Time offset from shock front: {t_offset:.4f} µs")
        print(f"Adjusted Wardlaw time steps (µs): {wardlaw_times_adjusted}")
        
        plotted_df_list = []
        
        for t_req, t_ward in zip(wardlaw_times_adjusted, wardlaw_times_us):
            idx = int(np.argmin(np.abs(np.array(unique_times) - t_req)))
            t_actual = unique_times[idx]
            error = abs(t_actual - t_req)
            print(f"  Wardlaw t={t_ward:.2e}µs (→ sim t={t_req:.2e}µs) → Found t={t_actual:.2e}µs (Δt={error:.2e}µs)")
            
            sub = df[np.isclose(df[time_col], t_actual)].sort_values(radial_col)
            pressure_dyne = sub[pressure_col].values * 1e12     #convert pressure from Mbar to dyne/cm^2
            #label shows Wardlaw time (not adjusted)
            plt.plot(sub[radial_col].values, pressure_dyne, 'o-', markersize=5, linewidth=2, label=f"t={t_ward:.2g}µs")
            plotted_df_list.append(sub)
        
        plotted_df = pd.concat(plotted_df_list, ignore_index=True)
        plt.title(f"Pressure vs. Radial Distance - Wardlaw Comparison")
    
    else:
        do_every_n = get_yes_no("\nPlot every nth time step? (Y/N): ")
        do_specific = get_yes_no("Plot specific time steps? (Y/N): ")
        
        if do_specific and not do_every_n:
            requested = parse_times_list("\nEnter time steps to plot (comma-separated, e.g., 0, 1.8, 4.2, 8.7, 19, 44): ")

            if not requested:
                print("No valid time steps provided.")
                sys.exit(0)
            
            plotted, plotted_df = plot_selected_times(df, time_col, radial_col, pressure_col, requested)
            plt.title("Pressure vs. Radial Distance (selected time steps)")
        else:
            if not do_every_n and not do_specific:
                print("Neither option selected; defaulting to plotting every time step (n=1).")
                n = 1
            else:
                n = int(input("\nPlot every nth time step (n=): ").strip())
            times_to_plot = unique_times[::n]
            plotted_df_list = []

            for t in times_to_plot:
                sub = df[np.isclose(df[time_col], t)].sort_values(radial_col)
                pressure_dyne = sub[pressure_col].values * 1e12
                plt.plot(sub[radial_col].values, pressure_dyne, 'o-', markersize=5, linewidth=2, label=f"t={t}")
                plotted_df_list.append(sub)
            
            plotted_df = pd.concat(plotted_df_list, ignore_index=True)
            plt.title(f"Pressure vs. Radial Distance (every {n}ᵗʰ time step)")

    #set up axes with log scale
    ax = plt.gca()
    ax.set_yscale('log')
    
    #set y-axis limits and ticks to match Wardlaw plot
    ax.set_ylim(1e7, 1e12)
    ax.set_yticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    ax.set_yticklabels(['1e7', '1e8', '1e9', '1e10', '1e11', '1e12'])
    
    #set x-axis limits
    ax.set_xlim(0, 35)
    
    plt.xlabel("Radial Distance (cm)", fontsize=12, fontweight='bold')
    plt.ylabel("Pressure (d/cm²)", fontsize=12, fontweight='bold')
    plt.grid(False)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    #save figure
    fig_out = base_dir / "eloutPressures.png"
    try:
        plt.savefig(fig_out, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure to {fig_out}")
    except OSError:
        # If directory is read-only, save to home
        home_dir = Path.home()
        fig_out = home_dir / "eloutPressures.png"
        plt.savefig(fig_out, dpi=300, bbox_inches='tight')
        print(f"\n⚠ Directory is read-only. Saved figure to {fig_out}")

    #save plotted data to a CSV
    if not plotted_df.empty:
        try:
            plotted_out = base_dir / "eloutExtractedData_Plotted.csv"
            plotted_df.to_csv(plotted_out, index=False)
            print(f"✓ Saved plotted data to {plotted_out}")
        except OSError:
            home_dir = Path.home()
            plotted_out = home_dir / "eloutExtractedData_Plotted.csv"
            plotted_df.to_csv(plotted_out, index=False)
            print(f"⚠ Saved plotted data to {plotted_out}")

    plt.show()

if __name__ == "__main__":
    main()

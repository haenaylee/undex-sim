# Script to generate a Latin Hypercube sample of parameter sets, create corresponding simulation directories with patched input files, 
# submit Slurm jobs, monitor their completion, and post-process elout/nodout files to extract pressure and velocity data along with bubble and shock fronts.
#!/usr/bin/env python3
# Haena Lee, February 2026
import csv
import shutil
import subprocess
import sys
import re
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------
# Constants & parameters - CHANGE AS NEEDED
# -------------------------------
ELEMENT_SIZE_CM = 0.2
EXPL_RADIUS_CM = 16.0
REGION_X_DIM = 36
REGION_Y_DIM = 36
REGION_Z_DIM = 36
SIM_ENDTIME = 70.0
WRITE_D3PLOT = 1
WRITE_GLSTAT = 0.5
WRITE_ELOUT = 0.5
WRITE_NODOUT = 0.5
PARAMETERS = {
    "A": {"bounds": (3.712, 3.712), "scale": "linear"},
    "B": {"bounds": (0.0323, 0.0323), "scale": "linear"},
    "R1": {"bounds": (4.15, 4.15), "scale": "linear"},
    "R2": {"bounds": (0.95, 0.95), "scale": "linear"},
    "omega": {"bounds": (0.3, 0.3), "scale": "linear"},
    "rho0": {"bounds": (1.63, 1.63), "scale": "linear"},
    "e0": {"bounds": (4.29e-2, 4.29e-2), "scale": "linear"},
    "Pcj": {"bounds": (0.21, 0.21), "scale": "linear"},
    "D": {"bounds": (0.693, 0.693), "scale": "linear"},
}
TEMPLATE_MODEL_PARAMS = "TEMPLATE_modelParams.csv"
TEMPLATE_MESH_SCRIPT = "TEMPLATE_generateRectMeshFile_sensorAndTracerElmts_sphericalCharge_3DField.py"
TEMPLATE_INPUT_K = "TEMPLATE_input.k"
SCRIPT_SH = "script.sh"
SEED = 123
POLL_INTERVAL_SECONDS = 10
CENTERLINE_TOL = 1.0e-9


# -------------------------------
# Progress/logging/post-processing helpers
# -------------------------------
SCRIPT_T0 = time.time()
SUPPRESS_POSTPROCESS_LOGGING = True  #set to True to suppress log_status output
POSTPROCESS_ONLY = False
POSTPROCESS_EXPERIMENT_DIR = None

def log_status(message: str):
    if SUPPRESS_POSTPROCESS_LOGGING:
        return
    elapsed = time.time() - SCRIPT_T0
    print(f"[{elapsed:8.1f}s] {message}", flush=True)

def timed_step(label: str, func, *args, **kwargs):
    log_status(f"START {label}")
    t0 = time.time()
    result = func(*args, **kwargs)
    log_status(f"DONE  {label} ({time.time() - t0:.1f}s)")
    return result

#Post-processing regex/helpers
float_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-\+]?\d+)?|\.\d+(?:[EeDd][-\+]?\d+)?")
time_re = re.compile(r"\(\s*at\s*time\s*([^\)]+)\)", re.IGNORECASE)


# -------------------------------
# General helpers
# -------------------------------
def next_experiment_dir(base: Path) -> Path:
    i = 1
    while True:
        p = base / f"experiment{i}"
        if not p.exists():
            return p
        i += 1

def prompt_positive_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            n = int(s)
        except ValueError:
            print("Please enter a positive integer.")
            continue
        if n <= 0:
            print("Please enter a positive integer.")
            continue
        return n

#LHS in [0,1] for each parameter then map to bounds
def latin_hypercube(n: int, param_names, seed: int):
    rng = np.random.RandomState(seed)
    samples01 = {}
    for name in param_names:
        u = rng.rand(n)     #stratified bins
        perm = rng.permutation(n)
        x01 = (perm + u) / float(n)
        samples01[name] = x01
    out = {}
    for name in param_names:
        lo, hi = PARAMETERS[name]["bounds"]
        scale = PARAMETERS[name]["scale"]
        x01 = samples01[name]
        if scale == "linear":
            x = lo + (hi - lo) * x01
        elif scale == "log":
            x = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * x01)
        else:
            raise ValueError(f"Unknown scale '{scale}' for parameter {name}")
        out[name] = x
    return out

def fmt_sci_3(x: float) -> str:
    return f"{x:.3e}".replace("E", "e")

def fmt_fix_3(x: float) -> str:
    return f"{x:.3f}"

#Convert Fortran-style D exponent to float-friendly E
def f2(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "E"))

#Return sorted times with near-duplicates merged into one canonical value to prevent duplicate rows (e.g., 0.316572 and 0.31657237)
#when different files have the same physical timestep but with slightly different floats
def canonicalize_time_list(times, rel_tol: float = 1.0e-6, abs_tol: float = 1.0e-9):
    vals = sorted(float(x) for x in times if pd.notna(x))
    if not vals:
        return []
    merged = [vals[0]]
    for t in vals[1:]:
        last = merged[-1]
        if np.isclose(t, last, rtol=rel_tol, atol=abs_tol):
            continue
        merged.append(t)
    return merged

#Return actual column name in df that case-insensitively matches target_lower
def find_column(df, target_lower: str) -> str:
    for c in df.columns:
        if c.strip().lower() == target_lower:
            return c
    raise KeyError(f"Required column '{target_lower}' not found in CSV header: {list(df.columns)}.")


# -------------------------------
# LS-DYNA formatting (10 chars per field)
# -------------------------------
def fit10(s: str) -> str:
    s = str(s)
    if len(s) > 10:
        raise ValueError(f"'{s}' exceeds 10 characters; cannot fit LS-DYNA 10-char field.")
    return s.rjust(10)

def replace_field10(line: str, field_1based: int, s: str) -> str:
    start = (field_1based - 1) * 10
    end = start + 10
    if len(line) < end:
        line = line.rstrip("\n").ljust(end) + "\n"
    return line[:start] + fit10(s) + line[end:]

def patch_input_k(path: Path, model_row_vals: dict, sim_params: dict):
    lines = path.read_text().splitlines(True)
    def patch_after(keyword: str, patch_fn):
        for i, ln in enumerate(lines):
            if keyword in ln:
                if i + 1 >= len(lines):
                    raise RuntimeError(f"Line after {keyword} not found in {path}")
                lines[i + 1] = patch_fn(lines[i + 1])
                return
        raise RuntimeError(f"{keyword} not found in {path}")
    
    #Control & database lines
    patch_after(
        "*CONTROL_TERMINATION",
        lambda ln: replace_field10(ln, 1, fmt_fix_3(model_row_vals["SIM_ENDTIME"]))
    )
    patch_after(
        "*DATABASE_BINARY_D3PLOT",
        lambda ln: replace_field10(ln, 1, fmt_fix_3(model_row_vals["WRITE_D3PLOT"]))
    )
    patch_after(
        "*DATABASE_GLSTAT",
        lambda ln: replace_field10(ln, 1, fmt_fix_3(model_row_vals["WRITE_GLSTAT"]))
    )
    patch_after(
        "*DATABASE_ELOUT",
        lambda ln: replace_field10(ln, 1, fmt_fix_3(model_row_vals["WRITE_ELOUT"]))
    )
    patch_after(
        "*DATABASE_NODOUT",
        lambda ln: replace_field10(ln, 1, fmt_fix_3(model_row_vals["WRITE_NODOUT"]))
    )
    
    #MAT card line
    def patch_mat(ln: str) -> str:
        ln2 = replace_field10(ln, 2, fmt_fix_3(sim_params["rho0"]))
        ln2 = replace_field10(ln2, 3, fmt_fix_3(sim_params["D"]))
        ln2 = replace_field10(ln2, 4, fmt_sci_3(sim_params["Pcj"]))
        return ln2
    
    #EOS_JWL line
    def patch_jwl(ln: str) -> str:
        ln2 = replace_field10(ln, 2, fmt_sci_3(sim_params["A"]))
        ln2 = replace_field10(ln2, 3, fmt_sci_3(sim_params["B"]))
        ln2 = replace_field10(ln2, 4, fmt_fix_3(sim_params["R1"]))
        ln2 = replace_field10(ln2, 5, fmt_fix_3(sim_params["R2"]))
        ln2 = replace_field10(ln2, 6, fmt_fix_3(sim_params["omega"]))
        ln2 = replace_field10(ln2, 7, fmt_sci_3(sim_params["e0"]))
        return ln2
    
    patch_after("*MAT_HIGH_EXPLOSIVE_BURN", patch_mat)
    patch_after("*EOS_JWL", patch_jwl)
    path.write_text("".join(lines))


# -------------------------------
# Write to mesh script
# -------------------------------
def patch_mesh_script_constants(mesh_py: Path, model_row_vals: dict):
    txt = mesh_py.read_text()

    def repl(name, value):
        pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)([^#\n]+)(.*)$", re.MULTILINE)
        def _r(m):
            return f"{m.group(1)}{value}{m.group(3)}"
        new_txt, n = pattern.subn(_r, txt, count=1)
        return new_txt, n
    
    mapping = {
        "ELEMENT_SIZE_CM": model_row_vals["ELEMENT_SIZE_CM"],
        "EXPL_RADIUS_CM": model_row_vals["EXPL_RADIUS_CM"],
        "REGION_X_DIM": model_row_vals["REGION_X_DIM"],
        "REGION_Y_DIM": model_row_vals["REGION_Y_DIM"],
        "REGION_Z_DIM": model_row_vals["REGION_Z_DIM"],
    }
    updated = txt

    for k, v in mapping.items():
        updated2, n = repl(k, v)
        if n != 1:
            raise RuntimeError(f"Could not uniquely patch '{k}' in {mesh_py}")
        updated = updated2
        txt = updated2
    
    mesh_py.write_text(updated)


# -------------------------------
# CSV helpers
# -------------------------------
def ensure_min_cols(rows, min_cols):
    for row in rows:
        while len(row) < min_cols:
            row.append("")

def write_job_ids_to_csv(csv_path: Path, sim_job_map: dict):
    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    ensure_min_cols(rows, 11)

    for i in range(3, len(rows)):
        sim_name = rows[i][0].strip()
        if sim_name in sim_job_map:
            rows[i][10] = sim_job_map[sim_name]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


# -------------------------------
# Slurm helpers
# -------------------------------
def submit_slurm_job(sim_dir: Path) -> str:
    result = subprocess.run(
        ["sbatch", "script.sh"],
        cwd=sim_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    combined = "\n".join(x for x in [stdout, stderr] if x)
    match = re.search(r"Submitted batch job (\d+)", combined)
    if not match:
        raise RuntimeError(
            f"Could not parse Slurm Job ID from sbatch output in {sim_dir}.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return match.group(1)

#True if the job's still visible in Slurm queue, False if not
def is_job_still_running(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"squeue failed for job {job_id}.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return bool(result.stdout.strip())

def monitor_jobs(job_ids):
    remaining = set(str(j) for j in job_ids)
    print("\nMonitoring submitted jobs...")
    while remaining:
        completed_this_pass = []
        for job_id in sorted(remaining, key=int):
            if not is_job_still_running(job_id):
                print(f"Job {job_id} is complete.")
                completed_this_pass.append(job_id)
        for job_id in completed_this_pass:
            remaining.remove(job_id)
        if remaining:
            time.sleep(POLL_INTERVAL_SECONDS)


# -----------------------------
# Geometry-aware element radial distance helper
# -----------------------------
def _find_latest_experiment_dir(base: Path):
    exps = [p for p in base.iterdir() if p.is_dir() and re.match(r"^experiment\d+$", p.name)]
    if not exps:
        return None
    exps.sort(key=lambda p: int(re.search(r"\d+", p.name).group()))
    return exps[-1]

def resolve_postprocess_experiment_dir(base: Path, explicit_dir=None) -> Path:
    if explicit_dir:
        p = Path(explicit_dir).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        return p
    sim_dirs_here = [p for p in base.iterdir() if p.is_dir() and re.match(r"^sim\d+$", p.name)]

    if sim_dirs_here:
        return base
    latest = _find_latest_experiment_dir(base)

    if latest is None:
        raise FileNotFoundError(
            "POSTPROCESS_ONLY is enabled, but no sim* folders were found in the current directory and no experiment* folder exists."
        )
    return latest

def _history_channel_stats(history):
    stats = []
    if history is None:
        return stats
    arr = np.asarray(history)

    if arr.ndim != 4 or arr.shape[-1] <= 0:
        return stats
    
    for k in range(arr.shape[-1]):
        vals = arr[..., k].astype(float).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            stats.append({"index": k, "finite_count": 0})
            continue

        stats.append({
            "index": k,
            "finite_count": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "fraction_between_0_and_1": float(np.mean((vals >= -1.0e-9) & (vals <= 1.0 + 1.0e-9))),
            "fraction_positive": float(np.mean(vals > 0.0)),
        })
    return stats

def _pick_density_ie_from_history(history, elem_ids, radial_distance, times):
    arr = np.asarray(history) if history is not None else None
    if arr is None or arr.ndim != 4 or arr.shape[0] != len(times) or arr.shape[1] != len(elem_ids):
        return None, None, {"source": "history_variables", "used": False}
    nhv = arr.shape[-1]
    diag = {"source": "history_variables", "used": False, "nhv": int(nhv)}
    diag["channels"] = _history_channel_stats(arr)
    hv = arr[:, :, 0, :].astype(float)

    density_idx = 0 if nhv >= 1 else None
    vf1_idx = 1 if nhv >= 2 else None
    vf2_idx = 2 if nhv >= 3 else None
    dominant_material_idx = 3 if nhv >= 4 else None
    ie1_idx = 4 if nhv >= 5 else None
    ie2_idx = 5 if nhv >= 6 else None

    density_data = hv[:, :, density_idx] if density_idx is not None else None
    ie_data = None
    ie_strategy = None

    if ie1_idx is not None and ie2_idx is not None:
        ie1 = hv[:, :, ie1_idx]
        ie2 = hv[:, :, ie2_idx]
        if dominant_material_idx is not None:
            dominant_material = hv[:, :, dominant_material_idx]
            ie_data = np.where(dominant_material >= 1.5, ie2, ie1)
            ie_strategy = "dominant_material_id"
        elif vf1_idx is not None and vf2_idx is not None:
            vf1 = hv[:, :, vf1_idx]
            vf2 = hv[:, :, vf2_idx]
            ie_data = np.where(vf2 > vf1, ie2, ie1)
            ie_strategy = "dominant_volume_fraction"
        else:
            ie_data = ie1
            ie_strategy = "material_1_only"
    elif ie1_idx is not None:
        ie_data = hv[:, :, ie1_idx]
        ie_strategy = "single_energy_channel"

    if density_data is not None:
        dens_vals = density_data[np.isfinite(density_data)]
        if dens_vals.size:
            log_status(
                f"[d3plot] History-variable mapping: density=HV[{density_idx}] (min={np.min(dens_vals):.6g}, max={np.max(dens_vals):.6g})"
            )
    if ie_data is not None:
        ie_vals = ie_data[np.isfinite(ie_data)]
        if ie_vals.size:
            log_status(
                f"[d3plot] History-variable mapping: internal-energy-like quantity via {ie_strategy}; "
                f"HV[{ie1_idx}] / HV[{ie2_idx}] (min={np.min(ie_vals):.6g}, max={np.max(ie_vals):.6g})"
            )

    diag.update({
        "used": density_data is not None or ie_data is not None,
        "density_index": None if density_idx is None else int(density_idx),
        "vf1_index": None if vf1_idx is None else int(vf1_idx),
        "vf2_index": None if vf2_idx is None else int(vf2_idx),
        "dominant_material_index": None if dominant_material_idx is None else int(dominant_material_idx),
        "internal_energy_material_1_index": None if ie1_idx is None else int(ie1_idx),
        "internal_energy_material_2_index": None if ie2_idx is None else int(ie2_idx),
        "internal_energy_strategy": ie_strategy,
        "internal_energy_index": None,
    })
    return density_data, ie_data, diag

#Build a mapping {element_id: radial_distance_cm} from actual mesh geometry from d3plot element centroids (to avoid dependence on element numbering/order)
def try_build_element_radial_distance_map(sim_dir: Path):
    d3plot_path = sim_dir / "d3plot"
    if not d3plot_path.exists():
        return None
    try:
        from lasso.dyna import D3plot
    except Exception as exc:
        print(f"[geometry] lasso-python unavailable; cannot build element radial-distance map from d3plot. ({exc})")
        return None
    
    try:
        d3 = D3plot(str(d3plot_path))
        node_coords = np.asarray(d3.arrays["node_coordinates"])
        elem_conn = np.asarray(d3.arrays["element_solid_node_indexes"])
        elem_ids = np.asarray(
            d3.arrays.get("element_solid_ids", np.arange(1, elem_conn.shape[0] + 1))
        )
        if node_coords.ndim == 3:
            node_coords = node_coords[0]
        if elem_conn.min() == 1:
            elem_conn = elem_conn - 1
        if node_coords.ndim != 2 or node_coords.shape[1] != 3:
            print(f"[geometry] Unexpected node_coordinates shape {node_coords.shape}; cannot build element radial-distance map.")
            return None
        centroids = node_coords[elem_conn].mean(axis=1)
        radial_distance = centroids[:, 0].astype(float)
        if np.nanmax(radial_distance) <= 0:
            radial_distance = np.abs(radial_distance)
        radial_map = {int(eid): float(r) for eid, r in zip(elem_ids, radial_distance)}
        return radial_map
    except Exception as exc:
        print(f"[geometry] Failed to build element radial-distance map from d3plot. ({exc})")
        return None

def _pick_first_available_array(arrays: dict, candidate_names):
    for name in candidate_names:
        if name in arrays:
            return np.asarray(arrays[name]), name
    return None, None

def _pick_best_matching_array(arrays: dict, preferred_names, keyword_groups, expected_nt=None, expected_ne=None):
    arr, name = _pick_first_available_array(arrays, preferred_names)
    if arr is not None:
        return np.asarray(arr), name
    candidates = []

    for key, value in arrays.items():
        key_l = str(key).strip().lower()
        if not all(any(token in key_l for token in group) for group in keyword_groups):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        score = 0
        if expected_nt is not None and arr.ndim >= 1 and arr.shape[0] == expected_nt:
            score += 10
        if expected_ne is not None and arr.ndim >= 2 and arr.shape[1] == expected_ne:
            score += 10
        if arr.ndim == 2:
            score += 6
        elif arr.ndim == 4:
            score += 4
        candidates.append((score, key_l, np.asarray(value), key))
    
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, _, best_arr, best_name = candidates[0]
    return np.asarray(best_arr), best_name

def _coerce_state_array(arr, nt, ne, fallback_hv_index=None):
    if arr is None:
        return None
    arr = np.asarray(arr)

    if arr.ndim == 2 and arr.shape[0] == nt and arr.shape[1] == ne:
        return arr.astype(float)
    
    if arr.ndim == 4 and arr.shape[0] == nt and arr.shape[1] == ne:
        if fallback_hv_index is None:
            fallback_hv_index = 0
        idx_candidates = []
        if arr.shape[-1] > fallback_hv_index:
            idx_candidates.append(fallback_hv_index)
        idx_candidates.extend([i for i in range(arr.shape[-1]) if i != fallback_hv_index])

        for idx in idx_candidates:
            try:
                candidate = arr[:, :, 0, idx].astype(float)
            except Exception:
                continue
            if candidate.shape == (nt, ne):
                return candidate
    return None

def try_build_d3plot_state_dataframe(sim_dir: Path):
    d3plot_path = sim_dir / "d3plot"
    if not d3plot_path.exists():
        return pd.DataFrame()
    diagnostics = {"sim_dir": str(sim_dir), "d3plot_path": str(d3plot_path), "arrays": []}

    try:
        from lasso.dyna import D3plot
    except Exception as exc:
        diagnostics["error"] = f"lasso import failed: {exc}"
        return pd.DataFrame()
    
    try:
        d3 = D3plot(str(d3plot_path))
        arrays = d3.arrays
        for key, value in arrays.items():
            try:
                arr = np.asarray(value)
                diagnostics["arrays"].append({"name": str(key), "shape": list(arr.shape), "dtype": str(arr.dtype)})
            except Exception as exc:
                diagnostics["arrays"].append({"name": str(key), "shape": None, "dtype": f"unavailable: {exc}"})
        
        times = np.asarray(arrays.get("timesteps", []), dtype=float)
        node_coords = np.asarray(arrays["node_coordinates"])
        elem_conn = np.asarray(arrays["element_solid_node_indexes"])
        elem_ids = np.asarray(arrays.get("element_solid_ids", np.arange(1, elem_conn.shape[0] + 1)))

        if node_coords.ndim == 3:
            node_coords = node_coords[0]
        if elem_conn.min() == 1:
            elem_conn = elem_conn - 1
        
        centroids = node_coords[elem_conn].mean(axis=1)
        x_abs = np.abs(centroids[:, 0].astype(float))
        y_abs = np.abs(centroids[:, 1].astype(float))
        z_abs = np.abs(centroids[:, 2].astype(float))
        radial_distance = x_abs.copy()
        nt = len(times)
        ne = len(elem_ids)

        #Restrict d3plot state exports to centerline elements only
        min_abs_y = float(np.nanmin(y_abs)) if y_abs.size else 0.0
        min_abs_z = float(np.nanmin(z_abs)) if z_abs.size else 0.0
        centerline_mask = (np.abs(y_abs - min_abs_y) <= CENTERLINE_TOL) & (np.abs(z_abs - min_abs_z) <= CENTERLINE_TOL)

        if not np.any(centerline_mask):
            centerline_mask = (y_abs <= CENTERLINE_TOL) & (z_abs <= CENTERLINE_TOL)
        centerline_idx = np.nonzero(centerline_mask)[0]

        if centerline_idx.size == 0:
            diagnostics["centerline_selection"] = {
                "status": "failed",
                "centerline_tol": float(CENTERLINE_TOL),
                "min_abs_y": min_abs_y,
                "min_abs_z": min_abs_z,
            }
            log_status(f"[{sim_dir.name}] WARNING: no centerline elements found for d3plot state export.")
            return pd.DataFrame(columns=["time", "element", "radial distance", "density", "internal energy-like quantity"])
        
        centerline_order = np.argsort(radial_distance[centerline_idx])
        centerline_idx = centerline_idx[centerline_order]

        diagnostics["centerline_selection"] = {
            "status": "ok",
            "centerline_tol": float(CENTERLINE_TOL),
            "min_abs_y": min_abs_y,
            "min_abs_z": min_abs_z,
            "num_centerline_elements": int(centerline_idx.size),
            "first_five_element_ids": [int(x) for x in elem_ids[centerline_idx][:5]],
            "first_five_radial_distances": [float(x) for x in radial_distance[centerline_idx][:5]],
        }

        density_arr, density_name = _pick_best_matching_array(
            arrays,
            preferred_names=[
                "element_solid_density", "element_solid_densities",
                "element_solid_mass_density", "element_solid_mass_densities",
                "element_solid_rho", "element_solid_rhos",
            ],
            keyword_groups=[["solid"], ["dens", "rho"]],
            expected_nt=nt, expected_ne=ne,
        )

        ie_arr, ie_name = _pick_best_matching_array(
            arrays,
            preferred_names=[
                "element_solid_internal_energy", "element_solid_internal_energies",
                "element_solid_specific_internal_energy", "element_solid_specific_internal_energies",
                "element_solid_internal_energy_density", "element_solid_internal_energy_densities",
                "element_solid_specific_energy", "element_solid_specific_energies",
                "element_solid_energy", "element_solid_energies",
                "elem_solid_internal_energy", "elem_solid_internal_energies",
                "elem_solid_specific_internal_energy", "elem_solid_specific_internal_energies",
                "elem_solid_internal_energy_density", "elem_solid_internal_energy_densities",
                "elem_solid_specific_energy", "elem_solid_specific_energies",
                "elem_solid_energy", "elem_solid_energies",
                "element_internal_energy", "element_internal_energies",
                "element_specific_internal_energy", "element_specific_internal_energies",
                "element_internal_energy_density", "element_internal_energy_densities",
                "element_specific_energy", "element_specific_energies",
                "internal_energy", "internal_energies",
                "internal_energy_density", "internal_energy_densities",
                "specific_internal_energy", "specific_internal_energies",
                "specific_energy", "specific_energies",
            ],
            keyword_groups=[["energ"]],
            expected_nt=nt, expected_ne=ne,
        )

        if ie_arr is None:
            ie_arr, ie_name = _pick_best_matching_array(arrays, [], [["internal"], ["energ"]], expected_nt=nt, expected_ne=ne)
        if ie_arr is None:
            ie_arr, ie_name = _pick_best_matching_array(arrays, [], [["specific"], ["energ"]], expected_nt=nt, expected_ne=ne)
        
        density_data = _coerce_state_array(density_arr, nt, ne, fallback_hv_index=0)
        ie_data = None
        diagnostics["picked_named_arrays"] = {
            "density_name": density_name,
            "internal_energy_name": ie_name,
            "density_named_shape": None if density_arr is None else list(np.asarray(density_arr).shape),
            "internal_energy_named_shape": None if ie_arr is None else list(np.asarray(ie_arr).shape),
        }

        log_status(f"[{sim_dir.name}] d3plot arrays detected: {len(diagnostics['arrays'])}")
        log_status(f"[{sim_dir.name}] Named density candidate: {density_name}")
        log_status(f"[{sim_dir.name}] Named internal-energy candidate: {ie_name}")
        history = arrays.get("element_solid_history_variables")
        history_diag = {"source": "history_variables", "used": False}

        if history is not None and (density_data is None or ie_data is None):
            history_density, history_ie, history_diag = _pick_density_ie_from_history(history, elem_ids, radial_distance, times)
            if density_data is None and history_density is not None:
                density_data = history_density
            if ie_data is None and history_ie is not None:
                ie_data = history_ie
        
        diagnostics["history_fallback"] = history_diag
        diagnostics["resolved_sources"] = {
            "density_source": density_name if density_name and _coerce_state_array(density_arr, nt, ne, fallback_hv_index=0) is not None else (f"element_solid_history_variables[{history_diag.get('density_index')}]" if history_diag.get("density_index") is not None else None),
            "internal_energy_source": (
                f"element_solid_history_variables[{history_diag.get('internal_energy_material_1_index')}] / "
                f"element_solid_history_variables[{history_diag.get('internal_energy_material_2_index')}] via {history_diag.get('internal_energy_strategy')}"
                if history_diag.get("internal_energy_material_1_index") is not None
                else None
            ),
        }

        log_status(f"[{sim_dir.name}] Resolved density source: {diagnostics['resolved_sources']['density_source']}")
        log_status(f"[{sim_dir.name}] Resolved internal-energy-like source: {diagnostics['resolved_sources']['internal_energy_source']}")
        rows = []
        log_status(f"[{sim_dir.name}] Building d3plot state rows for {nt} timesteps x {centerline_idx.size} centerline elements ...")
        
        for it, t in enumerate(times):
            if it == 0 or (it + 1) % 10 == 0 or it == nt - 1:
                log_status(f"[{sim_dir.name}] d3plot row build progress: timestep {it + 1}/{nt}")
            for j in centerline_idx:
                elem_id = elem_ids[j]
                rows.append({
                    "time": float(t),
                    "element": int(elem_id),
                    "radial distance": float(radial_distance[j]),
                    "density": float(density_data[it, j]) if density_data is not None and it < density_data.shape[0] and j < density_data.shape[1] else np.nan,
                    "internal energy-like quantity": float(ie_data[it, j]) if ie_data is not None and it < ie_data.shape[0] and j < ie_data.shape[1] else np.nan,
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        diagnostics["error"] = str(exc)
        return pd.DataFrame()

def _unique_numeric_times_in_order(series):
    vals = pd.to_numeric(series, errors="coerce")
    out = []
    seen = set()

    for x in vals:
        if pd.isna(x):
            continue
        xf = float(x)
        if xf not in seen:
            seen.add(xf)
            out.append(xf)
    return out


#Return a copy of d3plot_df whose rows are duplicated/relabelled onto target_times; row labels follow elout exactly
#while stil sourcig values from the closest d3plot states
def _align_d3plot_df_to_target_times_by_nearest(d3plot_df: pd.DataFrame, target_times):
    if d3plot_df is None or d3plot_df.empty or not target_times or "time" not in d3plot_df.columns:
        return d3plot_df

    work = d3plot_df.copy()
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    work = work[work["time"].notna()].copy()
    if work.empty:
        return d3plot_df

    unique_d3_times = np.array(sorted(work["time"].dropna().unique().tolist()), dtype=float)
    if unique_d3_times.size == 0:
        return d3plot_df

    aligned_blocks = []
    for target_t in target_times:
        try:
            target_tf = float(target_t)
        except Exception:
            continue
        idx = int(np.argmin(np.abs(unique_d3_times - target_tf)))
        source_t = float(unique_d3_times[idx])
        block = work.loc[np.isclose(work["time"].to_numpy(dtype=float), source_t, rtol=0.0, atol=1.0e-12)].copy()
        if block.empty:
            continue
        block["time"] = target_tf
        aligned_blocks.append(block)

    if not aligned_blocks:
        return d3plot_df

    aligned = pd.concat(aligned_blocks, ignore_index=True)
    return aligned


# -----------------------------
# Post-processing: elout
# -----------------------------
def parse_elout(elout_path: Path, out_csv: Path, element_size: float, element_radial_distance_map=None):
    src = Path(elout_path)
    out = Path(out_csv)
    elem_line_re = re.compile(r"^\s*(\d+)\-\s*(\d+)\s*$")
    data_start_re = re.compile(r"^\s*(\d+)\s+(.*)$")
    rows = []
    current_time = None
    lines = src.read_text(errors="ignore").splitlines()

    for i, line in enumerate(lines):
        tm = time_re.search(line)
        if tm:
            tmatch = re.search(
                r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-\+]?\d+)?",
                tm.group(1).replace("D", "E").replace("d", "E"),
            )
            current_time = f2(tmatch.group(0)) if tmatch else None
            continue
        em = elem_line_re.match(line)

        if not em:
            continue
        element_id = int(em.group(1))
        material_id = int(em.group(2))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            dm = data_start_re.match(lines[j])
            if dm:
                floats = [f2(tok) for tok in float_re.findall(dm.group(2))]
                if len(floats) >= 3:
                    rows.append({
                        "time": current_time,
                        "element": element_id,
                        "material": material_id,
                        "sig-xx": floats[0],
                        "sig-yy": floats[1],
                        "sig-zz": floats[2],
                    })

    df = pd.DataFrame(rows, columns=["time", "element", "material", "sig-xx", "sig-yy", "sig-zz"])
    export_headers = [
        "Time (µs)", "Element ID", "Calculated radial distance (cm)", "Material ID",
        "Sig-xx", "Sig-yy", "Sig-zz", "Calculated pressure (Mbar)"
    ]
    if df.empty:
        pd.DataFrame(columns=export_headers).to_csv(out, index=False, encoding='utf-8-sig')
        print(f"[elout] No rows parsed from {src}. Saved empty file to {out}")
        return df
    df = df.sort_values(["time", "element"]).reset_index(drop=True)

    if element_radial_distance_map:
        df["radial distance"] = df["element"].map(element_radial_distance_map)
        missing = int(df["radial distance"].isna().sum())
        if missing > 0:
            print(f"[elout] Warning: {missing} rows had no geometric element radial distance. Falling back to order-based spacing for those rows.")
            fallback_idx = df["radial distance"].isna()
            radial_index = df.groupby("time").cumcount()
            df.loc[fallback_idx, "radial distance"] = (
                radial_index.loc[fallback_idx] * element_size + 0.5 * element_size
            )
    else:
        print("[elout] No geometric element radial-distance map available; using order-based spacing.")
        radial_index = df.groupby("time").cumcount()
        df["radial distance"] = radial_index * element_size + 0.5 * element_size
    cols = df.columns.tolist()
    cols.insert(2, cols.pop(cols.index("radial distance")))
    df = df[cols]

    #Pressure calculated as hydrostatic pressure, i.e., the negative mean normal stress
    df["pressure"] = -(df["sig-xx"] + df["sig-yy"] + df["sig-zz"]) / 3.0
    export_df = pd.DataFrame({
        "Time (µs)": df["time"],
        "Element ID": df["element"],
        "Calculated radial distance (cm)": df["radial distance"],
        "Material ID": df["material"],
        "Sig-xx": df["sig-xx"],
        "Sig-yy": df["sig-yy"],
        "Sig-zz": df["sig-zz"],
        "Calculated pressure (Mbar)": df["pressure"],
    })
    export_df.to_csv(out, index=False, encoding='utf-8-sig')
    return df


# -----------------------------
# Post-processing: nodout
# -----------------------------
#Extract only the nodes along the centerline
def parse_nodout(nodout_path: Path, out_csv: Path, element_size: float, centerline_tol: float = CENTERLINE_TOL):
    src = Path(nodout_path)
    out = Path(out_csv)
    lines = src.read_text(errors="ignore").splitlines()
    rows = []
    current_time = None
    in_nodal_block = False
    nodal_header_re = re.compile(r"nodal\s+point", re.IGNORECASE)
    node_line_re = re.compile(r"^\s*(\d+)\s+")

    for line in lines:
        line_stripped = line.strip()
        tm = time_re.search(line)
        if tm:
            tmatch = re.search(
                r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-\+]?\d+)?",
                tm.group(1).replace("D", "E").replace("d", "E"),
            )
            current_time = f2(tmatch.group(0)) if tmatch else None
            in_nodal_block = False
            continue
        if nodal_header_re.search(line):
            in_nodal_block = True
            continue
        if not in_nodal_block or not line_stripped or not node_line_re.match(line):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            node_id = int(parts[0])
        except ValueError:
            continue
        numeric_tokens = []

        for tok in parts[1:]:
            try:
                numeric_tokens.append(f2(tok))
            except Exception:
                pass
        if len(numeric_tokens) < 8:
            continue
    
        try:
            x_vel = numeric_tokens[3]
            y_vel = numeric_tokens[4]
            z_vel = numeric_tokens[5]
            x_coor = numeric_tokens[-3]
            y_coor = numeric_tokens[-2]
            z_coor = numeric_tokens[-1]
        except Exception:
            continue
        if abs(y_coor) > centerline_tol or abs(z_coor) > centerline_tol:    #only centerline nodes
            continue

        rows.append({
            "time": current_time,
            "node": node_id,
            "x-coor": x_coor,
            "y-coor": y_coor,
            "z-coor": z_coor,
            "x-vel": x_vel,
            "y-vel": y_vel,
            "z-vel": z_vel,
        })

    df = pd.DataFrame(rows, columns=["time", "node", "x-coor", "y-coor", "z-coor", "x-vel", "y-vel", "z-vel"])
    export_headers = [
        "Time (µs)", "Node ID", "X-coor", "Calculated radial distance (cm)", "X-vel"
    ]

    if df.empty:
        pd.DataFrame(columns=export_headers).to_csv(out, index=False, encoding='utf-8-sig')
        print(f"[nodout] No centerline rows parsed from {src}. Saved empty file to {out}")
        return df
    df = df.sort_values(["time", "x-coor", "node"]).reset_index(drop=True)
    df["radial distance"] = pd.to_numeric(df["x-coor"], errors="coerce")
    df["total velocity"] = np.sqrt(df["x-vel"] ** 2 + df["y-vel"] ** 2 + df["z-vel"] ** 2)
    
    # Verify that all centerline nodes have small y and z velocities
    max_y_vel = df["y-vel"].abs().max()
    max_z_vel = df["z-vel"].abs().max()
    if max_y_vel > 1.0e-6 or max_z_vel > 1.0e-6:
        print(f"[nodout] WARNING: Centerline nodes have non-trivial transverse velocities! Max |y-vel|={max_y_vel:.2e}, Max |z-vel|={max_z_vel:.2e}")
    
    export_df = pd.DataFrame({
        "Time (µs)": df["time"],
        "Node ID": df["node"],
        "X-coor": df["x-coor"],
        "Calculated radial distance (cm)": df["radial distance"],
        "X-vel": df["x-vel"],
    })

    export_df.to_csv(out, index=False, encoding='utf-8-sig')
    return df


# -----------------------------
# Matrix-format outputs
# -----------------------------
def _format_value(v):
    if pd.isna(v):
        return ""
    return v

def _write_matrix_from_df(df: pd.DataFrame, out_csv: Path, id_col: str, value_col: str, id_header: str, master_time_order=None):
    columns = [
        "Radial distance (cm)",
        id_header,
        "Time (µs)",
    ]
    if df.empty or any(col not in df.columns for col in [id_col, "radial distance", "time", value_col]):
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([columns[0]])
            writer.writerow([columns[1]])
            writer.writerow([columns[2]])
        return

    work = df[["time", id_col, "radial distance", value_col]].copy()
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    work["radial distance"] = pd.to_numeric(work["radial distance"], errors="coerce")
    work = work[work["time"].notna()].copy()

    base = (
        work[[id_col, "radial distance"]]
        .drop_duplicates()
        .sort_values(["radial distance", id_col])
        .reset_index(drop=True)
    )
    id_order = base[id_col].tolist()
    radial_order = base["radial distance"].tolist()

    def _unique_preserve_order(values):
        seen = set()
        ordered = []
        for x in values:
            if pd.isna(x):
                continue
            xf = float(x)
            if xf in seen:
                continue
            seen.add(xf)
            ordered.append(xf)
        return ordered

    #For d3plot-derived matrices, preserve the exact native d3plot timestep list and ordering
    if master_time_order is not None:
        time_order = _unique_preserve_order(master_time_order)
    else:
        time_order = canonicalize_time_list(work["time"].drop_duplicates().tolist())

    #Snap each row time to the nearest requested output time only when it is already essentially equal
    def _snap_to_time_order(t):
        if pd.isna(t) or not time_order:
            return t
        tf = float(t)
        if master_time_order is not None:
            for tc in time_order:
                if np.isclose(tf, float(tc), rtol=0.0, atol=1.0e-8):
                    return float(tc)
            return tf
        for tc in time_order:
            if np.isclose(tf, float(tc), rtol=1.0e-6, atol=1.0e-9):
                return float(tc)
        return tf

    work["time"] = work["time"].map(_snap_to_time_order)
    pivot_source = work.drop_duplicates(subset=["time", id_col], keep="first")
    pivot = pivot_source.pivot(index="time", columns=id_col, values=value_col)
    pivot = pivot.reindex(columns=id_order)

    if master_time_order is not None and len(time_order) > 0:
        pivot = pivot.reindex(index=time_order, columns=id_order)
    else:
        pivot = pivot.reindex(index=time_order, columns=id_order)

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([columns[0]] + radial_order)
        writer.writerow([columns[1]] + id_order)
        writer.writerow([columns[2]] + [""] * len(id_order))
        for t, row in pivot.iterrows():
            writer.writerow([t] + [_format_value(v) for v in row.tolist()])

def write_pressure_matrix(elout_df: pd.DataFrame, out_csv: Path):
    _write_matrix_from_df(elout_df, out_csv, id_col="element", value_col="pressure", id_header="Element ID")

def write_velocity_matrix(nodout_df: pd.DataFrame, out_csv: Path):
    columns = [
        "Radial distance (cm)",
        "Node ID",
        "Time (µs)",
    ]

    if nodout_df.empty:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([columns[0]])
            writer.writerow([columns[1]])
            writer.writerow([columns[2]])
        return
    base = (
        nodout_df[["node", "radial distance"]]
        .drop_duplicates()
        .sort_values(["radial distance", "node"])
        .reset_index(drop=True)
    )

    node_order = base["node"].tolist()
    radial_order = base["radial distance"].tolist()
    pivot = nodout_df.pivot_table(
        index="time",
        columns="node",
        values="x-vel",  # Use x-velocity (radial velocity), not total velocity
        aggfunc="first",
    )

    pivot = pivot.reindex(columns=node_order)
    pivot = pivot.sort_index()
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([columns[0]] + radial_order)
        writer.writerow([columns[1]] + node_order)
        writer.writerow([columns[2]] + [""] * len(node_order))
        for t, row in pivot.iterrows():
            writer.writerow([t] + [_format_value(v) for v in row.tolist()])

def write_d3plot_state_matrix(d3plot_df: pd.DataFrame, out_csv: Path, value_col: str, master_time_order=None):
    _write_matrix_from_df(d3plot_df, out_csv, id_col="element", value_col=value_col, id_header="Element ID", master_time_order=master_time_order)

#Bubble front & shock front outputs
#shockFront.csv: extracted from elout file (eloutExtractedData.csv)
#   -Pressures come from elout, velocities from nodout, density/internal energy from d3plot
#bubbleFront.csv: also extracted from elout file
#   -Alternately extracted from D3PLOT volume fractions if lasso-python is available
def _nearest_time_subset(df: pd.DataFrame, t: float, time_col: str = "time", rel_tol: float = 1.0e-6, abs_tol: float = 1.0e-9):
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0].copy(), np.nan
    times = pd.to_numeric(df[time_col], errors="coerce")
    valid = times.notna()

    if not valid.any():
        return df.iloc[0:0].copy(), np.nan
    
    dfv = df.loc[valid].copy()
    tv = pd.to_numeric(dfv[time_col], errors="coerce").to_numpy(dtype=float)
    close_mask = np.isclose(tv, float(t), rtol=rel_tol, atol=abs_tol)

    if np.any(close_mask):
        matched_time = tv[np.where(close_mask)[0][0]]
        return dfv.loc[close_mask].copy(), float(matched_time)

    idx = int(np.argmin(np.abs(tv - float(t))))
    matched_time = tv[idx]
    subset = dfv[np.isclose(tv, matched_time, rtol=0.0, atol=0.0)].copy()
    return subset, float(matched_time)

def _lookup_pressure_for_element(elout_df: pd.DataFrame, t: float, elem_id):
    if pd.isna(elem_id) or elout_df.empty:
        return np.nan
    sub_t, matched_time = _nearest_time_subset(elout_df, t, time_col="time")

    if sub_t.empty:
        return np.nan
    sub = sub_t[sub_t["element"] == int(elem_id)]

    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["pressure"])

def _lookup_x_velocity_behind_radius(nodout_df: pd.DataFrame, t: float, radius):
    if pd.isna(radius) or nodout_df.empty:
        return np.nan
    sub, matched_time = _nearest_time_subset(nodout_df, t, time_col="time")

    if sub.empty:
        return np.nan
    sub = sub.copy()
    sub["radial distance"] = pd.to_numeric(sub["radial distance"], errors="coerce")
    sub = sub[sub["radial distance"].notna()]

    if sub.empty:
        return np.nan
    behind = sub[sub["radial distance"] <= float(radius) + 1.0e-12].copy()

    if behind.empty:
        behind = sub.copy()
    behind["dr_behind"] = float(radius) - behind["radial distance"]
    behind.loc[behind["dr_behind"] < 0.0, "dr_behind"] = np.inf
    behind = behind.sort_values(["dr_behind", "radial distance", "node"], ascending=[True, False, True])

    if behind.empty:
        return np.nan
    return float(behind.iloc[0]["x-vel"])

def _lookup_d3plot_state_for_element(d3plot_df: pd.DataFrame, t: float, elem_id, radius=np.nan):
    if d3plot_df.empty:
        return np.nan, np.nan
    sub_t, _ = _nearest_time_subset(d3plot_df, t, time_col="time")

    if sub_t.empty:
        return np.nan, np.nan
    
    def _useful(row):
        return not (pd.isna(row.get("density", np.nan)) and pd.isna(row.get("internal energy-like quantity", np.nan)))
    
    if not pd.isna(elem_id):
        sub = sub_t[sub_t["element"] == int(elem_id)]
        if not sub.empty and _useful(sub.iloc[0]):
            row = sub.iloc[0]
            return float(row.get("density", np.nan)), float(row.get("internal energy-like quantity", np.nan))
    
    if not pd.isna(radius):
        work = sub_t.copy()
        work["radial distance"] = pd.to_numeric(work["radial distance"], errors="coerce")
        work = work[work["radial distance"].notna()]

        if not work.empty:
            behind = work[work["radial distance"] <= float(radius) + 1.0e-12].copy()

            if behind.empty:
                behind = work.copy()
            behind["dr_behind"] = float(radius) - behind["radial distance"]
            behind.loc[behind["dr_behind"] < 0.0, "dr_behind"] = np.inf
            behind = behind.sort_values(["dr_behind", "radial distance", "element"], ascending=[True, False, True])

            for _, row in behind.iterrows():
                if _useful(row):
                    return float(row.get("density", np.nan)), float(row.get("internal energy-like quantity", np.nan))
            
            if not behind.empty:
                row = behind.iloc[0]
                return float(row.get("density", np.nan)), float(row.get("internal energy-like quantity", np.nan))
    return np.nan, np.nan

def augment_bubble_front_with_pressure_velocity(bubble_df: pd.DataFrame, elout_df: pd.DataFrame, nodout_df: pd.DataFrame) -> pd.DataFrame:
    bubble_df = bubble_df.copy()
    gas_p = []
    water_p = []
    gas_v = []
    water_v = []

    for _, row in bubble_df.iterrows():
        t = row["Time (µs)"]
        gas_r = row["Radial distance gas (cm)"]
        water_r = row["Radial distance water (cm)"]
        gas_elem = row["Element ID gas"]
        water_elem = row["Element ID water"]
        gas_p.append(_lookup_pressure_for_element(elout_df, t, gas_elem))
        water_p.append(_lookup_pressure_for_element(elout_df, t, water_elem))
        gas_v.append(_lookup_x_velocity_behind_radius(nodout_df, t, gas_r))
        water_v.append(_lookup_x_velocity_behind_radius(nodout_df, t, water_r))
    
    bubble_df["Pressure (gas) (Mbar)"] = gas_p
    bubble_df["Pressure (water) (Mbar)"] = water_p
    bubble_df["Velocity at the nearest node (gas) (cm/µs)"] = gas_v
    bubble_df["Velocity at the nearest node (water) (cm/µs)"] = water_v
    return bubble_df

#Bubble front is from the centerline ordering in elout_df; looks for the first material transition from gas (2) to water (1) as radial distance increases
#Shock front is element with max pressure at each time step
#Bubble front pressures are from elout element pressures, velocities from nearest nodout node at the same time and radial distance
def write_bubble_and_shock_from_elout(elout_df: pd.DataFrame, nodout_df: pd.DataFrame, d3plot_df: pd.DataFrame, bubble_out_csv: Path, shock_out_csv: Path):
    bubble_cols = [
        "Time (µs)",
        "Radial distance gas (cm)",
        "Radial distance water (cm)",
        "Element ID gas",
        "Element ID water",
        "Pressure (gas) (Mbar)",
        "Pressure (water) (Mbar)",
        "Velocity at the nearest node (gas) (cm/µs)",
        "Velocity at the nearest node (water) (cm/µs)",
    ]
    shock_cols = [
        "Time (µs)",
        "Shock front radial distance (cm)",
        "Shock front element ID",
        "Pressure (Mbar)",
        "Velocity at NODE directly behind shock (cm/μs)",
        "Fluid density (g/cm^3)",
        "Internal energy-like quantity (from d3plot HV)",
    ]

    if elout_df.empty:
        pd.DataFrame(columns=bubble_cols).to_csv(bubble_out_csv, index=False, encoding='utf-8-sig')
        pd.DataFrame(columns=shock_cols).to_csv(shock_out_csv, index=False, encoding='utf-8-sig')
        return
    
    bubble_rows = []
    shock_rows = []
    unique_times = sorted(pd.to_numeric(elout_df["time"], errors="coerce").dropna().unique().tolist())

    for t in unique_times:
        sub, matched_time = _nearest_time_subset(elout_df, t, time_col="time")
        sub = sub.sort_values(["radial distance", "element"]).reset_index(drop=True)
        gas_r = np.nan
        water_r = np.nan
        gas_elem = np.nan
        water_elem = np.nan
        mats = sub["material"].tolist()

        for i in range(len(sub) - 1):
            if mats[i] == 2 and mats[i + 1] == 1:
                gas_r = float(sub.iloc[i]["radial distance"])
                water_r = float(sub.iloc[i + 1]["radial distance"])
                gas_elem = int(sub.iloc[i]["element"])
                water_elem = int(sub.iloc[i + 1]["element"])
                break
        
        bubble_rows.append({
            "Time (µs)": float(t),
            "Radial distance gas (cm)": gas_r,
            "Radial distance water (cm)": water_r,
            "Element ID gas": gas_elem,
            "Element ID water": water_elem,
        })

        if sub["pressure"].isna().all():
            shock_r = np.nan
            shock_elem = np.nan
            shock_pressure = np.nan
        else:
            shock_idx = sub["pressure"].astype(float).idxmax()
            shock_row = sub.loc[shock_idx]
            shock_r = float(shock_row["radial distance"])
            shock_elem = int(shock_row["element"])
            shock_pressure = float(shock_row["pressure"])
        
        shock_velocity = _lookup_x_velocity_behind_radius(nodout_df, t, shock_r)
        shock_density, shock_ie = _lookup_d3plot_state_for_element(d3plot_df, t, shock_elem, radius=shock_r)

        shock_rows.append({
            shock_cols[0]: float(t),
            shock_cols[1]: shock_r,
            shock_cols[2]: shock_elem,
            shock_cols[3]: shock_pressure,
            shock_cols[4]: shock_velocity,
            shock_cols[5]: shock_density,
            shock_cols[6]: shock_ie,
        })
    
    bubble_df = pd.DataFrame(bubble_rows)
    bubble_df = augment_bubble_front_with_pressure_velocity(bubble_df, elout_df, nodout_df)
    bubble_df = bubble_df.reindex(columns=bubble_cols)
    bubble_df.to_csv(bubble_out_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(shock_rows).reindex(columns=shock_cols).to_csv(shock_out_csv, index=False, encoding='utf-8-sig')


# -----------------------------
# Optional d3plot-based bubble front
# -----------------------------
#Optional replacement for bubble front using volume fraction mat#2 from d3plot
#Falls back by returning False if lasso-python or d3plot processing is unavailable
#Assumes that HV[2] = volume fraction mat#2 = gas
def try_write_bubble_front_from_d3plot(sim_dir: Path, out_csv: Path, elout_df: pd.DataFrame, nodout_df: pd.DataFrame, element_size: float):
    d3plot_path = sim_dir / "d3plot"
    if not d3plot_path.exists():
        return False
    
    try:
        from lasso.dyna import D3plot
    except Exception as exc:
        print(f"[bubbleFront] lasso-python unavailable; using elout-based bubble front instead. ({exc})")
        return False
    
    try:
        d3 = D3plot(str(d3plot_path))
        times = np.asarray(d3.arrays["timesteps"])
        node_coords = np.asarray(d3.arrays["node_coordinates"])
        elem_conn = np.asarray(d3.arrays["element_solid_node_indexes"])
        history = np.asarray(d3.arrays["element_solid_history_variables"])

        if history.ndim != 4 or history.shape[-1] < 3:
            print("[bubbleFront] d3plot history variable layout not as expected; using elout-based bubble front instead.")
            return False
        
        if elem_conn.min() == 1:
            elem_conn = elem_conn - 1
        
        centroids = node_coords[elem_conn].mean(axis=1)
        elem_ids = np.asarray(d3.arrays.get("element_solid_ids", np.arange(1, elem_conn.shape[0] + 1)))
        y_abs = np.abs(centroids[:, 1])
        z_abs = np.abs(centroids[:, 2])
        y0 = np.min(y_abs)
        z0 = np.min(z_abs)
        tol = max(element_size * 1.0e-3, 1.0e-9)
        centerline_mask = (np.abs(y_abs - y0) <= tol) & (np.abs(z_abs - z0) <= tol)

        if not np.any(centerline_mask):
            print("[bubbleFront] Could not identify centerline elements in d3plot; using elout-based bubble front instead.")
            return False
    
        centerline_idx = np.nonzero(centerline_mask)[0]
        centerline_r = np.linalg.norm(centroids[centerline_idx], axis=1)
        order = np.argsort(centerline_r)
        centerline_idx = centerline_idx[order]
        centerline_r = centerline_r[order]
        centerline_elem_ids = elem_ids[centerline_idx]
        vf_gas = history[:, :, 0, 2]
        rows = []
        target_times = sorted(pd.to_numeric(elout_df["time"], errors="coerce").dropna().unique().tolist()) if not elout_df.empty else [float(x) for x in times]
        
        for t in target_times:
            it = int(np.argmin(np.abs(times - float(t))))
            vg = vf_gas[it, centerline_idx]
            gas_r = np.nan
            water_r = np.nan
            gas_elem = np.nan
            water_elem = np.nan

            for i in range(len(centerline_idx) - 1):
                if vg[i] >= 0.5 and vg[i + 1] < 0.5:
                    gas_r = float(centerline_r[i])
                    water_r = float(centerline_r[i + 1])
                    gas_elem = int(centerline_elem_ids[i])
                    water_elem = int(centerline_elem_ids[i + 1])
                    break
            
            rows.append({
                "Time (µs)": float(t),
                "Radial distance gas (cm)": gas_r,
                "Radial distance water (cm)": water_r,
                "Element ID gas": gas_elem,
                "Element ID water": water_elem,
            })
        
        bubble_df = pd.DataFrame(rows)
        bubble_df = bubble_df.sort_values(["Time (µs)"]).reset_index(drop=True)
        bubble_df = augment_bubble_front_with_pressure_velocity(bubble_df, elout_df, nodout_df)
        bubble_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        return True
    
    except Exception as exc:
        print(f"[bubbleFront] d3plot-based extraction failed; using elout-based bubble front instead. ({exc})")
        return False


# -----------------------------
# Simulation folder post-processing
# -----------------------------
#Write output CSVs to Outputs folder
def process_sim_folder(sim_dir: Path, element_size: float):
    sim_t0 = time.time()
    log_status(f"[{sim_dir.name}] Starting simulation post-processing in {sim_dir}")

    elout_path = sim_dir / "elout"
    nodout_path = sim_dir / "nodout"
    d3plot_path = sim_dir / "d3plot"

    log_status(f"[{sim_dir.name}] Input presence -> elout: {elout_path.exists()}, nodout: {nodout_path.exists()}, d3plot: {d3plot_path.exists()}")

    if not elout_path.exists():
        log_status(f"[{sim_dir.name}] Skipping because no elout file was found.")
        return

    outputs_dir = sim_dir / "Outputs"
    outputs_dir.mkdir(exist_ok=True)
    log_status(f"[{sim_dir.name}] Outputs directory ready: {outputs_dir}")

    elout_csv = outputs_dir / "eloutExtractedData.csv"
    nodout_csv = outputs_dir / "nodoutExtractedData.csv"
    pressure_matrix_csv = outputs_dir / "pressure_Mbar.csv"
    velocity_matrix_csv = outputs_dir / "velocity_cmPerMicrosec.csv"
    fluid_density_matrix_csv = outputs_dir / "fluidDensity.csv"
    internal_energy_matrix_csv = outputs_dir / "internalEnergyLikeQuantity.csv"
    bubble_front_csv = outputs_dir / "bubbleFront.csv"
    shock_front_csv = outputs_dir / "shockFront.csv"

    element_radial_distance_map = timed_step(f"{sim_dir.name}: build element radial-distance map from d3plot", try_build_element_radial_distance_map, sim_dir)
    if element_radial_distance_map is None:
        log_status(f"[{sim_dir.name}] No geometric element radial-distance map was built from d3plot.")
    else:
        try:
            log_status(f"[{sim_dir.name}] Built radial-distance map for {len(element_radial_distance_map)} elements.")
        except Exception:
            log_status(f"[{sim_dir.name}] Built radial-distance map.")

    elout_df = timed_step(f"{sim_dir.name}: parse elout -> {elout_csv.name}", parse_elout, elout_path, elout_csv, element_size, element_radial_distance_map=element_radial_distance_map)
    log_status(f"[{sim_dir.name}] elout rows parsed: {len(elout_df)}")

    timed_step(f"{sim_dir.name}: write pressure matrix -> {pressure_matrix_csv.name}", write_pressure_matrix, elout_df, pressure_matrix_csv)

    nodout_df = pd.DataFrame()
    if nodout_path.exists():
        nodout_df = timed_step(f"{sim_dir.name}: parse nodout -> {nodout_csv.name}", parse_nodout, nodout_path, nodout_csv, element_size)
        log_status(f"[{sim_dir.name}] nodout centerline rows parsed: {len(nodout_df)}")
        timed_step(f"{sim_dir.name}: write velocity matrix -> {velocity_matrix_csv.name}", write_velocity_matrix, nodout_df, velocity_matrix_csv)
    else:
        log_status(f"[{sim_dir.name}] No nodout file found. Writing empty nodout/velocity outputs.")
        pd.DataFrame(columns=["Time (µs)", "Node ID", "X-coor", "Calculated radial distance (cm)", "X-vel"]).to_csv(nodout_csv, index=False, encoding='utf-8-sig')
        timed_step(f"{sim_dir.name}: write empty velocity matrix -> {velocity_matrix_csv.name}", write_velocity_matrix, pd.DataFrame(), velocity_matrix_csv)

    d3plot_df = timed_step(f"{sim_dir.name}: build d3plot state dataframe", try_build_d3plot_state_dataframe, sim_dir)
    log_status(f"[{sim_dir.name}] d3plot state rows built: {len(d3plot_df)}")

    #Use one authoritative timestep list for all exported outputs; prefer elout, then nodout, then fall back to d3plot times
    master_time_order = []
    if not elout_df.empty and "time" in elout_df.columns:
        master_time_order = _unique_numeric_times_in_order(elout_df["time"])
    elif not nodout_df.empty and "time" in nodout_df.columns:
        master_time_order = _unique_numeric_times_in_order(nodout_df["time"])
    elif not d3plot_df.empty and "time" in d3plot_df.columns:
        master_time_order = _unique_numeric_times_in_order(d3plot_df["time"])

    d3plot_df_for_matrices = _align_d3plot_df_to_target_times_by_nearest(d3plot_df, master_time_order)

    if not d3plot_df.empty and master_time_order:
        native_d3_times = sorted(pd.to_numeric(d3plot_df["time"], errors="coerce").dropna().unique().tolist())
        preview_pairs = []
        for t in master_time_order[:5]:
            nearest_t = min(native_d3_times, key=lambda x: abs(float(x) - float(t))) if native_d3_times else np.nan
            preview_pairs.append(f"{float(t):.8g}->{float(nearest_t):.8g}")
        log_status(f"[{sim_dir.name}] d3plot->elout timestep mapping preview: " + ", ".join(preview_pairs))
    timed_step(f"{sim_dir.name}: write fluid density matrix -> {fluid_density_matrix_csv.name}", write_d3plot_state_matrix, d3plot_df_for_matrices, fluid_density_matrix_csv, "density", master_time_order)
    timed_step(f"{sim_dir.name}: write internal-energy-like matrix -> {internal_energy_matrix_csv.name}", write_d3plot_state_matrix, d3plot_df_for_matrices, internal_energy_matrix_csv, "internal energy-like quantity", master_time_order)

    wrote_d3plot_bubble = timed_step(f"{sim_dir.name}: try d3plot bubble-front extraction", try_write_bubble_front_from_d3plot, sim_dir, bubble_front_csv, elout_df, nodout_df, element_size)
    log_status(f"[{sim_dir.name}] d3plot bubble-front writer succeeded: {wrote_d3plot_bubble}")

    if not wrote_d3plot_bubble:
        timed_step(f"{sim_dir.name}: write bubble/shock fronts from elout+d3plot", write_bubble_and_shock_from_elout, elout_df, nodout_df, d3plot_df, bubble_front_csv, shock_front_csv)
    else:
        timed_step(f"{sim_dir.name}: write shock front from elout+d3plot", write_bubble_and_shock_from_elout, elout_df, nodout_df, d3plot_df, outputs_dir / "_temp_bubble_unused.csv", shock_front_csv)
        try:
            (outputs_dir / "_temp_bubble_unused.csv").unlink()
            log_status(f"[{sim_dir.name}] Removed temporary bubble-front file.")
        except Exception as exc:
            log_status(f"[{sim_dir.name}] Could not remove temporary bubble-front file. ({exc})")

    produced = [
        elout_csv.name, nodout_csv.name, pressure_matrix_csv.name, velocity_matrix_csv.name,
        fluid_density_matrix_csv.name, internal_energy_matrix_csv.name,
        bubble_front_csv.name, shock_front_csv.name
    ]
    log_status(f"[{sim_dir.name}] Finished in {time.time() - sim_t0:.1f}s. Outputs expected in {outputs_dir}: {', '.join(produced)}")

#Post-process only sim* folders inside the given experiment folder
def post_process_experiment(exp_dir: Path, element_size: float):
    global SUPPRESS_POSTPROCESS_LOGGING
    sim_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and re.match(r"^sim\d+$", p.name)],
        key=lambda p: int(re.search(r"\d+", p.name).group())
    )
    if not sim_dirs:
        print(f"\nNo sim folders found inside {exp_dir}. Skipping post-processing.")
        return
    print("\nStarting post-processing...")

    SUPPRESS_POSTPROCESS_LOGGING = True
    for idx, sim_dir in enumerate(sim_dirs, start=1):
        process_sim_folder(sim_dir, element_size)

    SUPPRESS_POSTPROCESS_LOGGING = False
    print("Post-processing is complete.")


# -------------------------------
# Main
# -------------------------------
def main():
    master = Path.cwd()

    if POSTPROCESS_ONLY:
        exp_dir = resolve_postprocess_experiment_dir(master, POSTPROCESS_EXPERIMENT_DIR)
        print(f"\nPOSTPROCESS_ONLY=True -> processing existing simulation results in: {exp_dir}")
        post_process_experiment(exp_dir, ELEMENT_SIZE_CM)
        return

    tpl_mesh = master / TEMPLATE_MESH_SCRIPT
    tpl_k = master / TEMPLATE_INPUT_K
    tpl_csv = master / TEMPLATE_MODEL_PARAMS
    script_sh = master / SCRIPT_SH

    for p in (tpl_mesh, tpl_k, tpl_csv, script_sh):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    n = prompt_positive_int("Number of parameter sets: ")
    param_names = list(PARAMETERS.keys())
    samples = latin_hypercube(n, param_names, SEED)
    exp = next_experiment_dir(master)
    exp.mkdir()
    csv_path = exp / "modelParams.csv"
    shutil.copy2(tpl_csv, csv_path)

    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise RuntimeError("TEMPLATE_modelParams.csv must have at least 3 rows.")

    ensure_min_cols(rows, 11)
    rows[1][0] = str(ELEMENT_SIZE_CM)
    rows[1][1] = str(EXPL_RADIUS_CM)
    rows[1][2] = str(REGION_X_DIM)
    rows[1][3] = str(REGION_Y_DIM)
    rows[1][4] = str(REGION_Z_DIM)
    rows[1][5] = str(SIM_ENDTIME)
    rows[1][6] = str(WRITE_D3PLOT)
    rows[1][7] = str(WRITE_GLSTAT)
    rows[1][8] = str(WRITE_ELOUT)
    rows[1][9] = str(WRITE_NODOUT)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    model_row_vals = {
        "ELEMENT_SIZE_CM": float(rows[1][0]),
        "EXPL_RADIUS_CM": float(rows[1][1]),
        "REGION_X_DIM": float(rows[1][2]),
        "REGION_Y_DIM": float(rows[1][3]),
        "REGION_Z_DIM": float(rows[1][4]),
        "SIM_ENDTIME": float(rows[1][5]),
        "WRITE_D3PLOT": float(rows[1][6]),
        "WRITE_GLSTAT": float(rows[1][7]),
        "WRITE_ELOUT": float(rows[1][8]),
        "WRITE_NODOUT": float(rows[1][9]),
    }

    mesh_script = exp / "generateMeshFile.py"
    shutil.copy2(tpl_mesh, mesh_script)
    patch_mesh_script_constants(mesh_script, model_row_vals)
    print("\nGenerating mesh...")

    subprocess.run([sys.executable, mesh_script.name], cwd=exp, check=True)
    print("Completed mesh generation.")

    mesh_outputs = [
        p for p in exp.iterdir()
        if p.is_file() and p.name not in ("generateMeshFile.py", "modelParams.csv")
    ]

    with csv_path.open("r", newline="") as f:
        template_rows = list(csv.reader(f))
    ensure_min_cols(template_rows, 11)
    header = template_rows[2]
    ncols = max(len(header), 11)
    sim_rows = []

    for i in range(n):
        r = [""] * ncols
        r[0] = f"sim{i+1}"
        r[1] = fmt_sci_3(float(samples["A"][i]))
        r[2] = fmt_sci_3(float(samples["B"][i]))
        r[3] = fmt_fix_3(float(samples["R1"][i]))
        r[4] = fmt_fix_3(float(samples["R2"][i]))
        r[5] = fmt_fix_3(float(samples["omega"][i]))
        r[6] = fmt_fix_3(float(samples["rho0"][i]))
        r[7] = fmt_sci_3(float(samples["e0"][i]))
        r[8] = fmt_sci_3(float(samples["Pcj"][i]))
        r[9] = fmt_fix_3(float(samples["D"][i]))
        r[10] = ""
        sim_rows.append(r)
    
    out_rows = template_rows[:3] + sim_rows
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(out_rows)
    sim_dirs = []

    for i in range(n):
        sim_name = f"sim{i+1}"
        sim_dir = exp / sim_name
        sim_dir.mkdir()
        sim_dirs.append(sim_dir)
        input_k = sim_dir / "input.k"
        shutil.copy2(tpl_k, input_k)
        sim_params = {k: float(samples[k][i]) for k in PARAMETERS.keys()}
        patch_input_k(input_k, model_row_vals=model_row_vals, sim_params=sim_params)
        shutil.copy2(script_sh, sim_dir / "script.sh")
        for m in mesh_outputs:
            shutil.copy2(m, sim_dir / m.name)

    for m in mesh_outputs:
        try:
            m.unlink()
        except Exception:
            pass

    print(f"\nCreated {exp}.")
    sim_job_map = {}

    print("\nSubmitting Slurm jobs...")
    for sim_dir in sim_dirs:
        sim_name = sim_dir.name
        job_id = submit_slurm_job(sim_dir)
        sim_job_map[sim_name] = job_id
        print(f"Submitted {sim_name} as Job {job_id}.")

    write_job_ids_to_csv(csv_path, sim_job_map)
    monitor_jobs(sim_job_map.values())

    print("\nAll submitted simulations are complete.")
    post_process_experiment(exp, ELEMENT_SIZE_CM)

if __name__ == "__main__":
    main()

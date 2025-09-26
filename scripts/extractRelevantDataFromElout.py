# Script to parse `elout` file, extract the time step, element ID, and xyz normal stresses, and save as CSV.
# Haena Lee, September 2025

import re
import pandas as pd
from pathlib import Path

def parse_elout(elout_path: str, out_csv: str):
    src = Path(elout_path)
    out = Path(out_csv)

    #regex patterns
    elem_line_re = re.compile(r"^\s*(\d+)\-\s*(\d+)\s*$")  # element - materl
    data_start_re = re.compile(r"^\s*(\d+)\s+(.*)$")       # ipt then floats
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-\+]?\d+)?|\.\d+(?:[EeDd][-\+]?\d+)?")
    time_re = re.compile(r"\(\s*at\s*time\s*([^\)]+)\)", re.IGNORECASE)

    #convert fortran D exponents to float
    def f2(x: str) -> float:
        return float(x.replace("D", "E"))

    rows = []
    current_time = None
    current_element = None

    lines = src.read_text(errors="ignore").splitlines()

    for i, line in enumerate(lines):
        #detect time header
        tm = time_re.search(line)
        if tm:
            tmatch = re.search(r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-\+]?\d+)?",
                               tm.group(1).replace("D", "E"))
            current_time = f2(tmatch.group(0)) if tmatch else None
            continue

        #detect element line
        em = elem_line_re.match(line)
        if em:
            current_element = int(em.group(1))
            #next non-empty line contains the data
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                dm = data_start_re.match(lines[j])
                if dm:
                    floats = [f2(tok) for tok in float_re.findall(dm.group(2))]
                    if len(floats) >= 3:
                        sig_xx, sig_yy, sig_zz = floats[0], floats[1], floats[2]
                        rows.append({
                            "time": current_time,
                            "element": current_element,
                            "sig-xx": sig_xx,
                            "sig-yy": sig_yy,
                            "sig-zz": sig_zz
                        })

   #save as CSV file
    df = pd.DataFrame(rows, columns=["time", "element", "sig-xx", "sig-yy", "sig-zz"])
    df = df.sort_values(["time", "element"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Parsed {len(df)} rows. Saved to {out}")

if __name__ == "__main__":
    parse_elout("elout", "eloutRelevantData.csv")
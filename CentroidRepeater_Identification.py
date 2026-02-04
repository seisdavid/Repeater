#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DG

Identify centroid-repeater families.

Method (Gao et al., 2023, SRL):
1) Sort all events by Mag from large to small.
2) Iterate from the largest event as the reference ("master").
3) Search among the remaining (unassigned) events for centroid-repeater events that satisfy:
   - |ΔMag| <= MAG_TOL
   - Horizontal centroid separation <= DIST_FACTOR * Rmax(mag_master, STRESS_DROP_MPA)
4) Assign matched events (including the master) to a FamilyID.
5) Only keep families with size >= 2.

Input (Excel columns):
  Orig, LatHy, LonHy, DepHy, LatCe, LonCe, DepCe, Mag

Output (Excel columns):
  Orig, LatHy, LonHy, DepHy, LatCe, LonCe, DepCe, Mag, FamilyID
"""

# ============================================================
# USER PARAMETERS (edit here)
# ============================================================
INPUT_XLSX  = "./Input_sequence.xlsx"
OUTPUT_XLSX = "./Centroid_Repeaters.xlsx"

STRESS_DROP_MPA = 38.0   # stress drop in MPa for rupture radius
MD = 0.3                 # centroid-repeater criterion |ΔMag| <= MD
DIST_FACTOR = 0.8        # centroid-repeater criterion: D_centroid <= DIST_FACTOR * Rmax(master)

MAG_MIN = 3.2            # remove events with Mag < MAG_MIN as their source dimensions are smaller than location uncertainty

SHEET_NAME_OUT = "CentroidRepeaters"  # output sheet name

# Magnitude-to-moment constant for Mw (log10 M0 [N·m] = 1.5*Mw + 9.1)
MAG_TO_M0_CONST = 9.1

#%%
# ============================================================
# Imports
# ============================================================
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth


# ============================================================
# Circular crack model
# ============================================================
def mag_to_m0_nm(mag: float, c: float = MAG_TO_M0_CONST) -> float:
    """Convert magnitude (mag) to seismic moment M0 (N·m)."""
    return 10.0 ** (1.5 * float(mag) + float(c))


def rmax_circular_crack_m(
    mag_max: float,
    stress_drop_mpa: float,
    const: float = 7 / 16,
    c: float = MAG_TO_M0_CONST,
) -> float:
    """
    Circular crack model:
      Δσ = (7/16) * M0 / r^3  =>  r = ((7/16) * M0 / Δσ)^(1/3)
    """
    stress_drop_pa = float(stress_drop_mpa) * 1e6  # MPa -> Pa
    m0 = mag_to_m0_nm(mag_max, c=c)
    return float((const * m0 / stress_drop_pa) ** (1.0 / 3.0))


# ============================================================
# Distance (ObsPy)
# ============================================================
def gps_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance (meters) using ObsPy gps2dist_azimuth.
    Returns only distance; azimuth outputs are ignored.
    """
    dist_m, _, _ = gps2dist_azimuth(float(lat1), float(lon1), float(lat2), float(lon2))
    return float(dist_m)


# ============================================================
# Core logic
# ============================================================
def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _renumber_family_ids_by_master_mag(out: pd.DataFrame) -> pd.DataFrame:
    """
    Renumber FamilyID to be consecutive (1..K) in a meaningful order:
    families are ordered by the maximum Mag within each family (master magnitude),
    from large to small.
    """
    if len(out) == 0:
        return out

    fam_master_mag = out.groupby("FamilyID")["Mag"].max().sort_values(ascending=False)
    ordered_old_ids = fam_master_mag.index.to_list()

    remap: Dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids, start=1)}
    out = out.copy()
    out["FamilyID"] = out["FamilyID"].map(remap).astype(int)
    return out


def identify_centroid_repeater_families(
    df: pd.DataFrame,
    md: float,
    dist_factor: float,
    stress_drop_mpa: float,
) -> pd.DataFrame:
    """
    Greedy centroid-repeater identification (SRL-style):
    - Sort by Mag descending
    - For each unassigned event i as master, find unassigned j satisfying criteria
    - If >=2 events match (including master), assign a FamilyID
    - Assigned events do NOT participate in later searches
    - Return only centroid-repeater families with size >= 2
    """
    required = ["Orig", "LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag"]
    _validate_columns(df, required)

    work = df.copy()

    # Coerce numeric columns
    for col in ["LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Drop rows missing essentials for centroid-repeater criteria
    key_ok = work["Mag"].notna() & work["LatCe"].notna() & work["LonCe"].notna()
    work = work.loc[key_ok].reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("No valid rows after removing missing Mag/LatCe/LonCe.")

    # Sort by magnitude descending
    work = work.sort_values("Mag", ascending=False).reset_index(drop=True)

    mags = work["Mag"].to_numpy(float)
    latc = work["LatCe"].to_numpy(float)
    lonc = work["LonCe"].to_numpy(float)

    n = len(work)
    family = np.full(n, fill_value=-1, dtype=int)  # -1 = unassigned
    fam_id = 0

    for i in range(n):
        if family[i] != -1:
            continue

        mag_master = mags[i]

        # Candidates: unassigned and within SRL magnitude window (MD)
        cand_mask = (family == -1) & (np.abs(mags - mag_master) <= md)
        cand_idx = np.where(cand_mask)[0]
        if cand_idx.size == 0:
            continue

        # Distance threshold: dist_factor * Rmax(master)
        rmax = rmax_circular_crack_m(mag_master, stress_drop_mpa=stress_drop_mpa)
        dist_thresh_m = dist_factor * rmax

        # Select those within centroid distance threshold
        sel = []
        for j in cand_idx:
            d_m = gps_distance_m(latc[i], lonc[i], latc[j], lonc[j])
            if d_m <= dist_thresh_m:
                sel.append(j)

        sel_idx = np.array(sel, dtype=int)

        # Create a centroid-repeater family only if master has at least one match
        if sel_idx.size >= 2:
            fam_id += 1
            family[sel_idx] = fam_id

    work["FamilyID"] = family

    out = work.loc[
        work["FamilyID"] > 0,
        ["Orig", "LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag", "FamilyID"],
    ].copy()

    # Renumber FamilyID by master magnitude (descending)
    out = _renumber_family_ids_by_master_mag(out)

    # Sort output so that the same FamilyID is contiguous
    # Within family: Mag descending; then Orig for stable ordering
    out = out.sort_values(["FamilyID", "Mag", "Orig"], ascending=[True, False, True]).reset_index(drop=True)

    # Add IsMaster: exactly one master per family (the first row after sorting)
    out["IsMaster"] = out.groupby("FamilyID").cumcount().eq(0)

    return out


def print_family_summary(out: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print a clear summary table to the screen.

    Columns:
      FamilyID   : centroid-repeater family index (1..K)
      N_events   : number of events in the family
      MasterMag  : magnitude of the master event (largest Mag in the family)
    """
    if len(out) == 0:
        print("No centroid-repeater families found under current criteria.")
        return

    summary = (
        out.groupby("FamilyID")
        .agg(N_events=("FamilyID", "size"), MasterMag=("Mag", "max"))
        .sort_values(["N_events", "MasterMag"], ascending=[False, False])
        .reset_index()
    )

    print(f"Number of centroid-repeater families: {summary.shape[0]}")
    print("Top centroid-repeater family sizes (FamilyID | N_events | MasterMag):")
    print(summary.head(top_n).to_string(index=False))


def main() -> None:
    # Basic parameter checks
    if MD < 0:
        raise ValueError("MD must be >= 0.")
    if DIST_FACTOR <= 0:
        raise ValueError("DIST_FACTOR must be > 0.")
    if STRESS_DROP_MPA <= 0:
        raise ValueError("STRESS_DROP_MPA must be > 0 (MPa).")

    # Read input
    try:
        df = pd.read_excel(INPUT_XLSX)
        df = df[df["Mag"] >= MAG_MIN].copy()
    except Exception as e:
        print(f"[ERROR] Failed to read input Excel: {INPUT_XLSX}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Identify centroid-repeater families
    try:
        out = identify_centroid_repeater_families(
            df=df,
            md=float(MD),
            dist_factor=float(DIST_FACTOR),
            stress_drop_mpa=float(STRESS_DROP_MPA),
        )
    except Exception as e:
        print(f"[ERROR] Centroid-repeater identification failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Save output
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name=SHEET_NAME_OUT)
    except Exception as e:
        print(f"[ERROR] Failed to write output Excel: {OUTPUT_XLSX}\n{e}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved {len(out)} events in centroid-repeater families to: {OUTPUT_XLSX}")
    print_family_summary(out, top_n=10)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 23:25:28 2026

@author: DG

Search hypocenter-repeaters (i.e., co-located large and small earthquake pairs) based on horizontal hypocenter proximity.

Input Excel:
  Orig, LatHy, LonHy, DepHy, LatCe, LonCe, DepCe, Mag

Rule:
  1) Sort all earthquakes by Mag from large to small.
  2) Each earthquake with Mag >= REF_MIN_MAG is a reference (large EQ).
  3) For each reference, search ALL earthquakes with:
       - Mag <= CAND_MAX_MAG
       - horizontal distance (LatHy/LonHy only; depth ignored) <= MAX_DIST_M
     Every (reference, candidate) found is saved as a hypocenter-repeater pair.

IMPORTANT (explicitly allowed):
  - A single large/reference earthquake can match MULTIPLE small earthquakes.
  - A single small/candidate earthquake is allowed to match MULTIPLE references.
    Therefore, the same candidate may appear in multiple rows of the output.
"""

# =========================
# USER SETTINGS
# =========================
INPUT_XLSX  = "./Input_sequence.xlsx"
OUTPUT_XLSX = "./Hypocenter_Repeaters.xlsx"

REF_MIN_MAG  = 4.5    # large EQ
CAND_MAX_MAG = 4.0    # small EQ
MAX_DIST_M   = 100.0  # distance threshold 

#%%
# =========================
from pathlib import Path
import numpy as np
import pandas as pd
# ---- ObsPy distance ----
try:
    from obspy.geodetics.base import gps2dist_azimuth
except Exception as e:
    raise ImportError(
        "ObsPy is required for gps2dist_azimuth. Install it via:\n"
        "  pip install obspy\n"
        f"Original import error: {e}"
    )


def gps2dist_array_m(ref_lat: float, ref_lon: float,
                     lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute distances (meters) from one reference point to many target points
    using obspy.gps2dist_azimuth (not vectorized, so we loop).

    Parameters
    ----------
    ref_lat, ref_lon : float
        Reference latitude/longitude in degrees.
    lats, lons : np.ndarray
        Target latitudes/longitudes in degrees.

    Returns
    -------
    np.ndarray
        Distances in meters (float).
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    dists = np.empty(lats.shape[0], dtype=float)
    for i in range(lats.shape[0]):
        # gps2dist_azimuth returns (dist_m, az12, az21)
        dist_m, _, _ = gps2dist_azimuth(ref_lat, ref_lon, float(lats[i]), float(lons[i]))
        dists[i] = float(dist_m)
    return dists


def main():
    inpath = Path(INPUT_XLSX)
    if not inpath.exists():
        raise FileNotFoundError(f"Input Excel not found: {inpath.resolve()}")

    # Read Excel (single sheet)
    df = pd.read_excel(inpath)

    # Clean column names (remove hidden spaces)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Orig", "LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Your columns are: {list(df.columns)}"
        )

    # Convert numeric columns
    for col in ["LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=["Orig", "LatHy", "LonHy", "Mag"]).copy()

    # Sort by magnitude descending
    df = df.sort_values("Mag", ascending=False).reset_index(drop=True)

    # Reference (large) events and candidate (small) events
    refs = df[df["Mag"] >= REF_MIN_MAG].copy()
    cands = df[df["Mag"] <= CAND_MAX_MAG].copy()

    pair_rows = []

    # Pre-cache candidate arrays for speed
    cand_lat = cands["LatHy"].to_numpy(float)
    cand_lon = cands["LonHy"].to_numpy(float)

    # For each reference, keep ALL matched candidates (no deduplication)
    for _, ref in refs.iterrows():
        ref_lat = float(ref["LatHy"])
        ref_lon = float(ref["LonHy"])

        # Distance in meters using ObsPy
        dists = gps2dist_array_m(ref_lat, ref_lon, cand_lat, cand_lon)

        hit_idx = np.where(dists <= MAX_DIST_M)[0]
        if hit_idx.size == 0:
            continue

        for j in hit_idx:
            cand = cands.iloc[j]
            pair_rows.append(
                {
                    # reference
                    "Ref_Orig": ref["Orig"],
                    "Ref_Mag": float(ref["Mag"]),
                    "Ref_LatHy": float(ref["LatHy"]),
                    "Ref_LonHy": float(ref["LonHy"]),
                    "Ref_DepHy": float(ref["DepHy"]) if pd.notna(ref["DepHy"]) else np.nan,
                    "Ref_LatCe": float(ref["LatCe"]) if pd.notna(ref["LatCe"]) else np.nan,
                    "Ref_LonCe": float(ref["LonCe"]) if pd.notna(ref["LonCe"]) else np.nan,
                    "Ref_DepCe": float(ref["DepCe"]) if pd.notna(ref["DepCe"]) else np.nan,
                    # candidate
                    "Cand_Orig": cand["Orig"],
                    "Cand_Mag": float(cand["Mag"]),
                    "Cand_LatHy": float(cand["LatHy"]),
                    "Cand_LonHy": float(cand["LonHy"]),
                    "Cand_DepHy": float(cand["DepHy"]) if pd.notna(cand["DepHy"]) else np.nan,
                    "Cand_LatCe": float(cand["LatCe"]) if pd.notna(cand["LatCe"]) else np.nan,
                    "Cand_LonCe": float(cand["LonCe"]) if pd.notna(cand["LonCe"]) else np.nan,
                    "Cand_DepCe": float(cand["DepCe"]) if pd.notna(cand["DepCe"]) else np.nan,
                    # distance
                    "Dist_m": float(dists[j]),
                }
            )

    pairs = pd.DataFrame(pair_rows)

    # Add helpful counts (for checking multi-matches); does NOT change pairing
    if len(pairs) > 0:
        cand_counts = pairs.groupby("Cand_Orig")["Ref_Orig"].nunique().rename("Cand_MatchedRefCount")
        ref_counts = pairs.groupby("Ref_Orig")["Cand_Orig"].nunique().rename("Ref_MatchedCandCount")
        pairs = pairs.join(cand_counts, on="Cand_Orig")
        pairs = pairs.join(ref_counts, on="Ref_Orig")

        # Sort output nicely
        pairs = pairs.sort_values(
            ["Ref_Mag", "Ref_Orig", "Dist_m", "Cand_Mag"],
            ascending=[False, True, True, False],
        ).reset_index(drop=True)

    # Summary
    summary_df = pd.DataFrame(
        {
            "N_total_events": [len(df)],
            f"N_reference_events_(Mag>={REF_MIN_MAG:.2f})": [len(refs)],
            f"N_candidate_events_(Mag<={CAND_MAX_MAG:.2f})": [len(cands)],
            "Max_dist_m": [MAX_DIST_M],
            "N_pairs_found": [len(pairs)],
            "Unique_Ref_events_in_pairs": [pairs["Ref_Orig"].nunique() if len(pairs) else 0],
            "Unique_Cand_events_in_pairs": [pairs["Cand_Orig"].nunique() if len(pairs) else 0],
            "Dist_m_min": [pairs["Dist_m"].min() if len(pairs) else np.nan],
            "Dist_m_median": [pairs["Dist_m"].median() if len(pairs) else np.nan],
            "Dist_m_max": [pairs["Dist_m"].max() if len(pairs) else np.nan],
            "Max_Cand_MatchedRefCount": [pairs["Cand_MatchedRefCount"].max() if len(pairs) else 0],
            "Max_Ref_MatchedCandCount": [pairs["Ref_MatchedCandCount"].max() if len(pairs) else 0],
        }
    )

    # Write Excel
    outpath = Path(OUTPUT_XLSX)
    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        pairs.to_excel(writer, sheet_name="Pairs", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"[OK] Saved to: {outpath.resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DG

Rupture Similarity Index (RSI) analysis + Monte Carlo (MC) uncertainty propagation.

This script has two parts:

A) Deterministic RSI summary (same as the original RSI_analysis.py):
   - Compute per-family RSI components (RSIazi, RSIdist, RSIhypo, RSIcent, RSIcomp)
   - Summarize the fraction of families with RSI >= T (T in RSI_THRESHOLDS)
   - Repeat under different HC-distance screening criteria (HC_THRESHOLDS_M)

B) MC uncertainty propagation (added here; following RSI.pdf, section (5)):
   - Perturb hypocenter and centroid locations in the horizontal plane by adding
     independent Gaussian noise to local east/north components:
         e_x, e_y ~ N(0, sigma_xy^2)   (per component)
   - For each family, run N_MC realizations and recompute:
       * per-event HC distance and HC azimuth (from perturbed hypo -> perturbed cent)
       * per-family RSI components and RSIcomp
   - For each threshold T, compute exceedance probability:
         EP(T) = P(RSIcomp >= T)
   - Classify families into three categories at each T:
       * High-confidence repeatable:      EP >= EP_HIGH (default 0.95)
       * Indistinguishable:               EP_LOW < EP < EP_HIGH (default 0.05–0.95)
       * High-confidence not repeatable:  EP <= EP_LOW (default 0.05)
   - Plot the fraction of families in these three categories vs threshold T.

Notes:
  * RSI definitions follow Eq. (24) in RSI.pdf:
      RSIazi  = 1 - SCR/360
      RSIdist = 1 - (dmax - dmin)/dmax
      RSIhypo = 1 - Dmax_hypo/Rmax
      RSIcent = 1 - Dmax_cent/Rmax
      RSIcomp = min(RSIazi, RSIdist, RSIhypo, RSIcent)
  * We only consider horizontal locations (lon/lat) and ignore depth.
  * For RSIhypo/RSIcent, values are clipped to [0, 1] to preserve the bounded RSI definition.
"""

# =========================
# User settings
# =========================
INPUT_XLSX = "./RepeaterCatalog.xlsx"
STRESS_DROP_MPA = 38.0

# Keep only families with size <= MAX_FAMILY_SIZE.
# Set to a very large number (e.g., 10**9) to keep all families.
MAX_FAMILY_SIZE = 100

# Drop a family if it contains ANY event with HC distance < threshold.
HC_THRESHOLDS_M = [0, 100, 200, 300, 400, 500]  # 0 means no screening

# Thresholds T used in summary counts and MC EP(T).
RSI_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# -------------------------
# MC settings 
# -------------------------
RUN_MC = False
SIGMA_XY_M = 40.0   # per component (east & north). Example: 40, 80, 100
N_MC = 10_000

# EP-based classification thresholds
EP_HIGH = 0.95
EP_LOW  = 0.05

# Random seed for MC (set to an int for reproducibility; set to None for random)
MC_RANDOM_SEED = 0

# -------------------------
# Plot styling
# -------------------------
TITLE_FONTSIZE   = 9
TITLE_FONTWEIGHT = "normal"

# Input column names
FAM_COL = "FamilyID"
AZ_COL  = "DirectivityHC"   # degrees in [0, 360)
HCD_COL = "HCdis"           # meters
MAG_COL = "Mag"

# Horizontal locations (lon/lat)
LON_HY_COL = "LonHy"
LAT_HY_COL = "LatHy"
LON_CE_COL = "LonCe"
LAT_CE_COL = "LatCe"

# Mag -> M0 conversion constant used in your workflow
MAG_TO_M0_CONST = 9.1

# =========================
# Imports
# =========================
import os
import shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Tuple, List, Dict, Optional

# Try to use ObsPy's accurate geodesy; fall back to a lightweight implementation if unavailable.
try:
    from obspy.geodetics.base import gps2dist_azimuth  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def gps2dist_azimuth(lat1, lon1, lat2, lon2):
        """
        Fallback replacement for obspy.geodetics.base.gps2dist_azimuth.

        Returns:
            dist_m: great-circle distance in meters (haversine)
            az12: forward azimuth from point 1 -> point 2 (degrees, can be negative)
            az21: back azimuth from point 2 -> point 1 (degrees, can be negative)
        """
        # Convert to radians
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        lam1 = math.radians(float(lon1))
        lam2 = math.radians(float(lon2))

        dphi = phi2 - phi1
        dlam = lam2 - lam1

        # Haversine distance
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        R = 6371000.0  # mean Earth radius (m)
        dist_m = R * c

        # Forward azimuth (bearing)
        y = math.sin(dlam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        az12 = math.degrees(math.atan2(y, x))

        # Back azimuth: swap points
        yb = math.sin(-dlam) * math.cos(phi1)
        xb = math.cos(phi2) * math.sin(phi1) - math.sin(phi2) * math.cos(phi1) * math.cos(-dlam)
        az21 = math.degrees(math.atan2(yb, xb))

        return dist_m, az12, az21

OUT_DIR = f"./RSI_analysis_NumMax_{MAX_FAMILY_SIZE}"
OUTPUT_PDF = os.path.join(OUT_DIR, "RSI_analysis_summary.pdf")
SUMMARY_CSV = os.path.join(OUT_DIR, "RSI_analysis_summary.csv")

# MC outputs
MC_PDF = os.path.join(OUT_DIR, f"RSI_MC_Sigma_{int(SIGMA_XY_M)}m.pdf")
MC_TXT = os.path.join(OUT_DIR, f"RSI_MC_Sigma_{int(SIGMA_XY_M)}m_Summary.txt")

# =========================
# I/O utilities
# =========================
def reset_outdir(path: str) -> None:
    """Delete output directory if it exists, then recreate it."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# =========================
# Circular crack reference radius Rmax
# =========================
def mag_to_m0_nm(mag: float, c: float = MAG_TO_M0_CONST) -> float:
    """Convert magnitude (mag) to seismic moment M0 (N·m)."""
    return 10.0 ** (1.5 * float(mag) + float(c))

def rmax_circular_crack_m(mag_max: float, stress_drop_mpa: float, const: float = 7/16, c: float = MAG_TO_M0_CONST) -> float:
    """
    Circular crack model:
      Δσ = (7/16) * M0 / r^3  =>  r = ((7/16) * M0 / Δσ)^(1/3)
    """
    stress_drop_pa = float(stress_drop_mpa) * 1e6  # MPa -> Pa
    m0 = mag_to_m0_nm(mag_max, c=c)
    return float((const * m0 / stress_drop_pa) ** (1.0 / 3.0))

# =========================
# RSI component metrics
# =========================
def smallest_circular_range_deg(angles_deg: np.ndarray) -> float:
    """Smallest circular range (SCR) in degrees for azimuths in [0, 360)."""
    ang = np.asarray(angles_deg, dtype=float) % 360.0
    ang = ang[np.isfinite(ang)]
    if ang.size == 0:
        return np.nan
    if ang.size == 1:
        return 0.0
    ang_sorted = np.sort(ang)
    gaps = np.diff(ang_sorted)
    wrap_gap = 360.0 - (ang_sorted[-1] - ang_sorted[0])
    gaps = np.append(gaps, wrap_gap)
    return float(360.0 - np.max(gaps))

def max_dist_to_mean_geodesic_m(lons: np.ndarray, lats: np.ndarray) -> float:
    """
    Maximum horizontal distance (meters) from each point to the mean location.

    We compute the distance from (mean lat, mean lon) to each event using
    gps2dist_azimuth, and take the maximum.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    mask = np.isfinite(lons) & np.isfinite(lats)
    if mask.sum() == 0:
        return np.nan
    if mask.sum() == 1:
        return 0.0

    lons = lons[mask]
    lats = lats[mask]
    lon0 = float(np.mean(lons))
    lat0 = float(np.mean(lats))

    dmax = 0.0
    for lon, lat in zip(lons, lats):
        dist_m, _, _ = gps2dist_azimuth(lat0, lon0, float(lat), float(lon))
        dmax = max(dmax, float(dist_m))
    return float(dmax)

def compute_family_rsi_components_from_arrays(
    az_deg: np.ndarray,
    hc_dist_m: np.ndarray,
    lon_hy: np.ndarray,
    lat_hy: np.ndarray,
    lon_ce: np.ndarray,
    lat_ce: np.ndarray,
    mag: np.ndarray,
    stress_drop_mpa: float,
) -> dict:
    """
    Compute RSI components for a family given arrays (optionally perturbed).

    Args:
        az_deg:    HC azimuths in degrees (hypo -> cent), length n_events
        hc_dist_m: HC distances in meters, length n_events
        lon_hy, lat_hy: hypocenter lon/lat arrays
        lon_ce, lat_ce: centroid lon/lat arrays
        mag: magnitude array
    """
    out = {}

    # --- RSIazi: 1 - SCR/360
    scr = smallest_circular_range_deg(az_deg)
    out["SCR_deg"] = scr
    out["RSIazi"] = 1.0 - (scr / 360.0) if np.isfinite(scr) else np.nan

    # --- RSIdist: 1 - (dmax - dmin)/dmax
    d = np.asarray(hc_dist_m, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        out["d_min_m"] = np.nan
        out["d_max_m"] = np.nan
        out["RSIdist"] = np.nan
    else:
        dmin = float(np.min(d))
        dmax = float(np.max(d))
        out["d_min_m"] = dmin
        out["d_max_m"] = dmax
        if dmax > 0:
            out["RSIdist"] = 1.0 - ((dmax - dmin) / dmax)
        else:
            out["RSIdist"] = 1.0 if dmin == 0 else np.nan

    # --- Rmax from Mag_max
    mag = np.asarray(mag, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        rmax = np.nan
    else:
        rmax = rmax_circular_crack_m(float(np.max(mag)), stress_drop_mpa=stress_drop_mpa)
    out["Rmax_m"] = rmax

    # --- RSIhypo
    dmax_hy = max_dist_to_mean_geodesic_m(lon_hy, lat_hy)
    out["Dmax_hypo_m"] = dmax_hy
    if (not np.isfinite(dmax_hy)) or (not np.isfinite(rmax)) or (rmax <= 0):
        out["RSIhypo"] = np.nan
    else:
        out["RSIhypo"] = float(np.clip(1.0 - (dmax_hy / rmax), 0.0, 1.0))

    # --- RSIcent
    dmax_ce = max_dist_to_mean_geodesic_m(lon_ce, lat_ce)
    out["Dmax_cent_m"] = dmax_ce
    if (not np.isfinite(dmax_ce)) or (not np.isfinite(rmax)) or (rmax <= 0):
        out["RSIcent"] = np.nan
    else:
        out["RSIcent"] = float(np.clip(1.0 - (dmax_ce / rmax), 0.0, 1.0))

    # --- RSIcomp
    comps = [out.get("RSIazi"), out.get("RSIdist"), out.get("RSIhypo"), out.get("RSIcent")]
    if any([not np.isfinite(v) for v in comps]):
        out["RSIcomp"] = np.nan
    else:
        out["RSIcomp"] = float(np.min(comps))

    return out

def compute_family_rsi_components(g: pd.DataFrame, stress_drop_mpa: float) -> dict:
    """Deterministic RSI components for a single family (using catalog columns)."""
    az_deg = pd.to_numeric(g[AZ_COL], errors="coerce").to_numpy()
    hc_dist_m = pd.to_numeric(g[HCD_COL], errors="coerce").to_numpy()
    lon_hy = pd.to_numeric(g[LON_HY_COL], errors="coerce").to_numpy()
    lat_hy = pd.to_numeric(g[LAT_HY_COL], errors="coerce").to_numpy()
    lon_ce = pd.to_numeric(g[LON_CE_COL], errors="coerce").to_numpy()
    lat_ce = pd.to_numeric(g[LAT_CE_COL], errors="coerce").to_numpy()
    mag = pd.to_numeric(g[MAG_COL], errors="coerce").to_numpy()

    return compute_family_rsi_components_from_arrays(
        az_deg=az_deg,
        hc_dist_m=hc_dist_m,
        lon_hy=lon_hy, lat_hy=lat_hy,
        lon_ce=lon_ce, lat_ce=lat_ce,
        mag=mag,
        stress_drop_mpa=stress_drop_mpa,
    )

# =========================
# MC utilities
# =========================
EARTH_R_M = 6371000.0

def latlon_to_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate local projection (equirectangular) around (lon0, lat0).

    x: east (m), y: north (m)
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (np.deg2rad(lon - lon0) * EARTH_R_M * np.cos(np.deg2rad(lat0)))
    y = (np.deg2rad(lat - lat0) * EARTH_R_M)
    return x, y

def xy_to_latlon(x: np.ndarray, y: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of latlon_to_xy_m."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lat = lat0 + np.rad2deg(y / EARTH_R_M)
    lon = lon0 + np.rad2deg(x / (EARTH_R_M * np.cos(np.deg2rad(lat0))))
    return lon, lat

def compute_hc_dist_az_arrays(lat_hy: np.ndarray, lon_hy: np.ndarray, lat_ce: np.ndarray, lon_ce: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-event HC distance (m) and azimuth (deg in [0,360)) from hypocenter -> centroid."""
    n = len(lat_hy)
    dist = np.full(n, np.nan, dtype=float)
    az   = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not (np.isfinite(lat_hy[i]) and np.isfinite(lon_hy[i]) and np.isfinite(lat_ce[i]) and np.isfinite(lon_ce[i])):
            continue
        d_m, az12, _ = gps2dist_azimuth(float(lat_hy[i]), float(lon_hy[i]), float(lat_ce[i]), float(lon_ce[i]))
        dist[i] = float(d_m)
        az[i] = (float(az12) + 360.0) % 360.0
    return dist, az

def mc_ep_for_family(g: pd.DataFrame, stress_drop_mpa: float, sigma_xy_m: float, n_mc: int, rng: np.random.Generator) -> dict:
    """
    Monte Carlo: compute EP(T) for RSIcomp for one family.

    Returns:
        dict with keys:
          EP_<T> for each T in RSI_THRESHOLDS,
          plus optional diagnostics (mean RSIcomp etc.).
    """
    lon_hy0 = pd.to_numeric(g[LON_HY_COL], errors="coerce").to_numpy(dtype=float)
    lat_hy0 = pd.to_numeric(g[LAT_HY_COL], errors="coerce").to_numpy(dtype=float)
    lon_ce0 = pd.to_numeric(g[LON_CE_COL], errors="coerce").to_numpy(dtype=float)
    lat_ce0 = pd.to_numeric(g[LAT_CE_COL], errors="coerce").to_numpy(dtype=float)
    mag0    = pd.to_numeric(g[MAG_COL], errors="coerce").to_numpy(dtype=float)

    # Choose a local origin for small perturbations (family mean of all points)
    all_lon = np.concatenate([lon_hy0[np.isfinite(lon_hy0)], lon_ce0[np.isfinite(lon_ce0)]])
    all_lat = np.concatenate([lat_hy0[np.isfinite(lat_hy0)], lat_ce0[np.isfinite(lat_ce0)]])
    if all_lon.size == 0 or all_lat.size == 0:
        return {f"EP_{T:.2f}": np.nan for T in RSI_THRESHOLDS}

    lon0 = float(np.mean(all_lon))
    lat0 = float(np.mean(all_lat))

    # Convert to local meters
    x_hy, y_hy = latlon_to_xy_m(lon_hy0, lat_hy0, lon0=lon0, lat0=lat0)
    x_ce, y_ce = latlon_to_xy_m(lon_ce0, lat_ce0, lon0=lon0, lat0=lat0)

    # Pre-allocate counts
    count_ge_T = {T: 0 for T in RSI_THRESHOLDS}
    valid_realizations = 0

    # Run MC
    for _ in range(n_mc):
        # Independent Gaussian noise per event and per component
        ex_hy = rng.normal(0.0, sigma_xy_m, size=x_hy.shape)
        ey_hy = rng.normal(0.0, sigma_xy_m, size=y_hy.shape)
        ex_ce = rng.normal(0.0, sigma_xy_m, size=x_ce.shape)
        ey_ce = rng.normal(0.0, sigma_xy_m, size=y_ce.shape)

        lon_hy, lat_hy = xy_to_latlon(x_hy + ex_hy, y_hy + ey_hy, lon0=lon0, lat0=lat0)
        lon_ce, lat_ce = xy_to_latlon(x_ce + ex_ce, y_ce + ey_ce, lon0=lon0, lat0=lat0)

        # Recompute per-event HC distance and azimuth
        hc_dist, hc_az = compute_hc_dist_az_arrays(lat_hy, lon_hy, lat_ce, lon_ce)

        met = compute_family_rsi_components_from_arrays(
            az_deg=hc_az,
            hc_dist_m=hc_dist,
            lon_hy=lon_hy, lat_hy=lat_hy,
            lon_ce=lon_ce, lat_ce=lat_ce,
            mag=mag0,
            stress_drop_mpa=stress_drop_mpa,
        )

        rsi_comp = met.get("RSIcomp", np.nan)
        if not np.isfinite(rsi_comp):
            continue

        valid_realizations += 1
        for T in RSI_THRESHOLDS:
            if rsi_comp >= T:
                count_ge_T[T] += 1

    # Convert to EP
    out = {}
    for T in RSI_THRESHOLDS:
        out[f"EP_{T:.2f}"] = (count_ge_T[T] / valid_realizations) if valid_realizations > 0 else np.nan
    out["n_mc_valid"] = int(valid_realizations)
    return out

# =========================
# Plotting
# =========================
def plot_case(ax, proportions_by_T: dict, title: str, title_fontsize: int = 12, title_fontweight: str = "normal") -> None:
    """
    Deterministic summary plot: fraction of families (%) exceeding RSI thresholds
    for each RSI component.
    """
    x = np.arange(1, 6)
    styles = {
        0.80: dict(color="red",     marker="s", linestyle="-"),
        0.85: dict(color="lime",    marker="D", linestyle="--"),
        0.90: dict(color="blue",    marker="v", linestyle="-"),
        0.95: dict(color="magenta", marker="o", linestyle="--"),
    }

    for T in RSI_THRESHOLDS:
        ax.plot(
            x, proportions_by_T[T],
            linewidth=2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=f"T = {T:.2f}",
            **styles[T],
        )

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(["RSIazi", "RSIdist", "RSIhypo", "RSIcent", "RSIcomp"], rotation=0)
    ax.set_ylabel("Fraction of families (%)")
    ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
    ax.axhline(y=80, linestyle="--", color="gray")
    ax.legend(loc="upper right", frameon=True)

def plot_mc_panel(ax, Ts: List[float], frac_rep: List[float], frac_ind: List[float], frac_not: List[float], title: str) -> None:
    """MC plot panel (fraction of families vs threshold T for RSIcomp)."""
    ax.plot(Ts, frac_rep, marker="^", linewidth=2, label=f"High-confidence repeatable (EP ≥ {EP_HIGH:.2f})")
    ax.plot(Ts, frac_ind, marker="o", linewidth=2, label=f"Indistinguishable ({EP_LOW:.2f} < EP < {EP_HIGH:.2f})")
    ax.plot(Ts, frac_not, marker="v", linewidth=2, label=f"High-confidence not repeatable (EP ≤ {EP_LOW:.2f})")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(Ts) - 0.01, max(Ts) + 0.01)
    ax.set_xticks(Ts)
    ax.set_xlabel("Threshold for RSI$_{comp}$")
    ax.set_ylabel("Fraction of families")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight=TITLE_FONTWEIGHT)

# =========================
# Main
# =========================
def main() -> None:
    reset_outdir(OUT_DIR)

    df = pd.read_excel(INPUT_XLSX)

    # ---- Column handling / backward compatibility ----
    base_required = [FAM_COL, MAG_COL, LON_HY_COL, LAT_HY_COL, LON_CE_COL, LAT_CE_COL]
    missing_base = [c for c in base_required if c not in df.columns]
    if missing_base:
        raise KeyError(
            f"Missing required columns: {missing_base}\n"
            f"Available columns: {list(df.columns)}"
        )

    # If HC distance/azimuth columns are missing, compute them.
    if HCD_COL not in df.columns or AZ_COL not in df.columns:
        def _hc_dist_az(row):
            dist_m, az_deg, _ = gps2dist_azimuth(
                float(row[LAT_HY_COL]), float(row[LON_HY_COL]),
                float(row[LAT_CE_COL]), float(row[LON_CE_COL])
            )
            az_deg = (az_deg + 360.0) % 360.0
            return dist_m, az_deg

        out = df.apply(_hc_dist_az, axis=1, result_type="expand")
        if HCD_COL not in df.columns:
            df[HCD_COL] = out[0]
        if AZ_COL not in df.columns:
            df[AZ_COL] = out[1]

    # Ensure numeric
    df[HCD_COL] = pd.to_numeric(df[HCD_COL], errors="coerce")
    df[AZ_COL]  = pd.to_numeric(df[AZ_COL], errors="coerce")
    df[MAG_COL] = pd.to_numeric(df[MAG_COL], errors="coerce")

    summary_rows = []

    # --- Family-size filter (keep <= MAX_FAMILY_SIZE)
    fam_size = df.groupby(FAM_COL).size()
    keep_by_size = fam_size[fam_size <= MAX_FAMILY_SIZE].index
    df_size_ok = df[df[FAM_COL].isin(keep_by_size)].copy()
    n_fams_after_size = int(df_size_ok[FAM_COL].nunique())

    plot_payloads = []  # deterministic plots
    mc_payloads = []    # MC plots

    # Setup RNG for MC
    rng = np.random.default_rng(MC_RANDOM_SEED) if MC_RANDOM_SEED is not None else np.random.default_rng()

    # -------------------------
    # Part A: deterministic RSI summary
    # -------------------------
    with PdfPages(OUTPUT_PDF) as pdf:
        for thr_m in HC_THRESHOLDS_M:

            # Drop a family if ANY event has HC distance < threshold.
            bad_fams = df_size_ok.loc[df_size_ok[HCD_COL] < thr_m, FAM_COL].dropna().unique()
            keep_fams = pd.Index(keep_by_size).difference(pd.Index(bad_fams))

            df_use = df_size_ok[df_size_ok[FAM_COL].isin(keep_fams)].copy()
            total_fams = int(df_use[FAM_COL].nunique())

            # --- Per-family RSI metrics
            records = []
            for fid, g in df_use.groupby(FAM_COL):
                met = compute_family_rsi_components(g, stress_drop_mpa=STRESS_DROP_MPA)
                met["FamilyID"] = fid
                met["family_size"] = int(len(g))
                records.append(met)

            fam_df = pd.DataFrame(records)

            fam_csv = os.path.join(OUT_DIR, f"family_metrics_thr{thr_m}m.csv")
            fam_df.to_csv(fam_csv, index=False)

            # --- Fractions of families exceeding each threshold T
            proportions_by_T = {}
            for T in RSI_THRESHOLDS:
                if total_fams == 0:
                    proportions_by_T[T] = [0, 0, 0, 0, 0]
                    continue

                c1 = int(np.sum(fam_df["RSIazi"] >= T))
                c2 = int(np.sum(fam_df["RSIdist"] >= T))
                c3 = int(np.sum(fam_df["RSIhypo"] >= T))
                c4 = int(np.sum(fam_df["RSIcent"] >= T))
                c5 = int(np.sum(fam_df["RSIcomp"] >= T))

                proportions_by_T[T] = [
                    100.0 * c1 / total_fams,
                    100.0 * c2 / total_fams,
                    100.0 * c3 / total_fams,
                    100.0 * c4 / total_fams,
                    100.0 * c5 / total_fams,
                ]

            title = f"HC distance ≥ {thr_m/1000.0:.1f} km (N = {total_fams} families)"
            plot_payloads.append((thr_m, proportions_by_T, title))

            summary_rows.append({
                "HC_threshold_m": thr_m,
                "stress_drop_MPa": STRESS_DROP_MPA,
                "max_family_size": MAX_FAMILY_SIZE,
                "families_after_size_filter": n_fams_after_size,
                "families_dropped_by_HCdis": int(len(bad_fams)),
                "families_used_for_RSI": total_fams,
                "events_used_for_RSI": int(len(df_use)),
                "family_metrics_csv": os.path.basename(fam_csv),
                "note": "RSIhypo/RSIcent are clipped to [0,1]; RSIcomp = min(RSIazi, RSIdist, RSIhypo, RSIcent).",
            })

        # --- One page with one panel per HC threshold
        nrows = len(HC_THRESHOLDS_M)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(5.0, 3.8 * nrows), sharex=True, sharey=True)
        if nrows == 1:
            axes = [axes]

        for ax, (thr_m, proportions_by_T, ttl) in zip(axes, plot_payloads):
            panel_title = f"SD = {STRESS_DROP_MPA:.0f} MPa, family size ≤ {MAX_FAMILY_SIZE}, {ttl}"
            plot_case(ax, proportions_by_T, panel_title, title_fontsize=TITLE_FONTSIZE, title_fontweight=TITLE_FONTWEIGHT)

        for ax in axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    # -------------------------
    # Part B: Monte Carlo uncertainty propagation (EP-based)
    # -------------------------
    if RUN_MC:
        # For each HC threshold, compute EP per family and then fractions by category for each T
        txt_lines = []
        txt_lines.append("Monte Carlo uncertainty propagation for RSIcomp\n")
        txt_lines.append(f"sigma_xy (per component) = {SIGMA_XY_M:.1f} m\n")
        txt_lines.append(f"N_MC = {N_MC}\n")
        txt_lines.append(f"EP thresholds: repeatable if EP >= {EP_HIGH:.2f}, not repeatable if EP <= {EP_LOW:.2f}\n")
        txt_lines.append(f"Stress drop = {STRESS_DROP_MPA:.1f} MPa\n")
        txt_lines.append(f"MAX_FAMILY_SIZE = {MAX_FAMILY_SIZE}\n")
        txt_lines.append("HC screening rule: drop a family if ANY event has HC distance < threshold\n")
        txt_lines.append("\n")

        # Prepare multi-panel layout similar to RSI_Sigma_40m.pdf (3x2 if 6 thresholds)
        n_panels = len(HC_THRESHOLDS_M)
        ncols = 2 if n_panels > 1 else 1
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.0, 9.0))
        axes = np.atleast_1d(axes).ravel()

        # Place legend once (bottom)
        legend_handles = None
        legend_labels = None

        for i, thr_m in enumerate(HC_THRESHOLDS_M):
            ax = axes[i]

            bad_fams = df_size_ok.loc[df_size_ok[HCD_COL] < thr_m, FAM_COL].dropna().unique()
            keep_fams = pd.Index(keep_by_size).difference(pd.Index(bad_fams))
            df_use = df_size_ok[df_size_ok[FAM_COL].isin(keep_fams)].copy()
            total_fams = int(df_use[FAM_COL].nunique())

            # Per-family EP table
            ep_records = []
            for fid, g in df_use.groupby(FAM_COL):
                ep = mc_ep_for_family(g, stress_drop_mpa=STRESS_DROP_MPA, sigma_xy_m=SIGMA_XY_M, n_mc=N_MC, rng=rng)
                ep["FamilyID"] = fid
                ep["family_size"] = int(len(g))
                ep_records.append(ep)

            ep_df = pd.DataFrame(ep_records)
            ep_csv = os.path.join(OUT_DIR, f"family_EP_thr{thr_m}m_sigma{int(SIGMA_XY_M)}m.csv")
            ep_df.to_csv(ep_csv, index=False)

            # Fractions in each category vs T
            Ts = [float(t) for t in RSI_THRESHOLDS]
            frac_rep, frac_ind, frac_not = [], [], []

            for T in Ts:
                col = f"EP_{T:.2f}"
                if total_fams == 0 or col not in ep_df.columns:
                    frac_rep.append(0.0); frac_ind.append(0.0); frac_not.append(0.0)
                    continue
                epv = pd.to_numeric(ep_df[col], errors="coerce")
                # Only use families with finite EP (should be most of them)
                epv = epv[np.isfinite(epv)]
                n = len(epv)
                if n == 0:
                    frac_rep.append(0.0); frac_ind.append(0.0); frac_not.append(0.0)
                else:
                    n_rep = int(np.sum(epv >= EP_HIGH))
                    n_not = int(np.sum(epv <= EP_LOW))
                    n_ind = int(np.sum((epv > EP_LOW) & (epv < EP_HIGH)))
                    frac_rep.append(n_rep / n)
                    frac_ind.append(n_ind / n)
                    frac_not.append(n_not / n)

            title = f"SD = {STRESS_DROP_MPA:.0f} MPA\nHC distance ≥ {thr_m/1000.0:.1f} km"
            plot_mc_panel(ax, Ts, frac_rep, frac_ind, frac_not, title)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            # Write text summary for this HC threshold
            txt_lines.append(f"HC distance ≥ {thr_m/1000.0:.1f} km (N = {total_fams} families)\n")
            for T, r, ind, nr in zip(Ts, frac_rep, frac_ind, frac_not):
                txt_lines.append(f"  T = {T:.2f}: repeatable = {r:.3f}, indistinguishable = {ind:.3f}, not repeatable = {nr:.3f}\n")
            txt_lines.append(f"  EP table CSV: {os.path.basename(ep_csv)}\n")
            txt_lines.append("\n")

        # Turn off extra axes if any
        for j in range(n_panels, len(axes)):
            axes[j].axis("off")

        if legend_handles is not None:
            fig.legend(
                legend_handles, legend_labels,
                loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 0.02),
                fontsize=7
            )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(MC_PDF)
        plt.close(fig)

        with open(MC_TXT, "w", encoding="utf-8") as f:
            f.writelines(txt_lines)

    print("All results saved in:", OUT_DIR)
    print("Deterministic PDF:", OUTPUT_PDF)
    print("Deterministic summary CSV:", SUMMARY_CSV)
    if RUN_MC:
        print("MC PDF:", MC_PDF)
        print("MC summary TXT:", MC_TXT)

if __name__ == "__main__":
    main()

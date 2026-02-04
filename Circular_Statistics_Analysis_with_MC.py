#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:23:21 2026

@author: DG

HC azimuth circular statistics analysis 

Definition
----------
HC azimuth = azimuth/bearing FROM hypocenter TO centroid for each event:
- Computed by ObsPy: obspy.geodetics.gps2dist_azimuth(latHy, lonHy, latCe, lonCe)
- Use the forward azimuth az12 (degrees clockwise from North, range [0, 360))

IMPORTANT: Depth is ignored
---------------------------
DepHy/DepCe are NOT used for HC azimuth. The direction is defined purely in the horizontal plane.

What is computed (using astropy)
--------------------------------
1) Rayleigh test p-value (test against circular uniformity)
2) Circular mean direction (degrees)
3) Circular variance (0..1)

Robustness analyses
-------------------
A) Bootstrap (resample events with replacement):
   - Sector fractions + 95% confidence intervals
   - Bin-wise 95% confidence envelope for rose diagram

B) Monte Carlo (horizontal location uncertainty propagation):
   - Add independent Gaussian errors to EAST/NORTH components of BOTH Hy and Ce:
       e_x, e_y ~ N(0, sigma_xy^2)  (meters per component)
   - Convert perturbed EN back to lon/lat (via local AEQD inverse transform if pyproj is available;
     otherwise approximate meters<->degrees).
   - Recompute azimuths using ObsPy gps2dist_azimuth for each realization.

Dependencies
------------
pip install numpy pandas openpyxl matplotlib astropy obspy pyproj

Input (Excel)
-------------
Required columns:
  LatHy, LonHy, DepHy, LatCe, LonCe, DepCe, Mag

Outputs
-------
- CSV:
  stats_observed.csv
  stats_bootstrap.csv
  stats_montecarlo.csv
- PDF:
  rose_observed_with_bootstrap_CI.pdf
  rayleigh_pvalue_hist_mc_sigma_XXm.pdf
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- silence known noisy warnings (optional) ----
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`alltrue` is deprecated as of NumPy 1.25.0.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)

# ---- circular stats (astropy) ----
from astropy.stats.circstats import circmean, circvar, rayleightest

# ---- azimuth & distance (ObsPy) ----
from obspy.geodetics import gps2dist_azimuth

# ---- projection helpers (recommended for MC) ----
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False


# =============================================================================
# USER PARAMETERS (edit here)
# =============================================================================

EXCEL_PATH = "RepeaterCatalog.xlsx"     # Input Excel file containing the earthquake catalog
SHEET_NAME = 0                          # Sheet index (int) or sheet name (str) in the Excel file
OUTDIR = "Out_Circular_Stats"           # Output folder where CSV/TXT/PDF products will be saved

# -------------------------
# Quality control (QC)
# -------------------------
DROP_ZERO_LENGTH_VECTORS = True         # If True, drop rows where Hy and Ce have identical lat/lon (azimuth undefined)
MIN_EVENTS_REQUIRED = 5                 # Minimum number of valid events required to run stats/bootstrap/MC

# Family-level QC by HC distance
ENABLE_FAMILY_HC_FILTER = True          # If True, drop any family that contains an event with HCdist_m < threshold
FAMILY_ID_COL = "FamilyID"              # Column name for family identifier in Excel (required if ENABLE_FAMILY_HC_FILTER=True)
HC_DIST_THRESHOLD_M = 100.0             # Threshold in meters; if any event in a family is below => drop family

# -------------------------
# Rose diagram (polar histogram)
# -------------------------
BIN_WIDTH_DEG = 10.0                    # Bin width (degrees) for the rose diagram (e.g., 5, 10, 15)
ROSE_NORMALIZE = True                   # If True, show fraction per bin; if False, show counts per bin

# -------------------------
# Bootstrap
# -------------------------
N_BOOT = 10_000                         # Number of bootstrap resamples (larger => smoother CI, but slower)
BOOT_RANDOM_SEED = 0                    # Random seed for bootstrap reproducibility

# -------------------------
# Directional sectors (summary metrics)
# -------------------------
DOWNDIP_MIN_DEG, DOWNDIP_MAX_DEG = 245.0, 335.0   # "Downdip" azimuth sector bounds (deg, inclusive)
UPDIP_MIN_DEG,  UPDIP_MAX_DEG  = 65.0,  155.0     # "Updip" azimuth sector bounds (deg, inclusive)

# -------------------------
# Monte Carlo (horizontal location uncertainty propagation)
# -------------------------
N_MC = 10_000                           # Number of Monte Carlo realizations (larger => more stable, but slower)
MC_RANDOM_SEED = 0                      # Random seed for Monte Carlo reproducibility
SIGMA_XY_M = 40.0                       # Std dev (m) for EACH horizontal component (E and N) of Hy and Ce perturbations
MC_REPORT_ALPHA = 0.05                  # Alpha used to report fraction of realizations with p-value < alpha

EXTRA_SIGMAS_M = [80.0, 100.0]          # Optional: additional sigma values to test, e.g., [80.0, 100.0]

# =============================================================================
# End user parameters
# =============================================================================


@dataclass
class CircularStatsResult:
    n: int
    rayleigh_p: float
    mean_deg: float
    var: float
    frac_downdip: float
    frac_updip: float


# =============================================================================
# Utility helpers
# =============================================================================

def _ensure_outdir(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _read_excel(path: str | Path, sheet=SHEET_NAME) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    required = ["LatHy", "LonHy", "DepHy", "LatCe", "LonCe", "DepCe", "Mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")
    return df.copy()


def _q025_q50_q975(x: np.ndarray) -> np.ndarray:
    """Return [2.5%, 50%, 97.5%] quantiles as a 1D numpy array."""
    x = np.asarray(x, dtype=float)
    return np.quantile(x, [0.025, 0.50, 0.975])


def _fmt_ci3(ci3: np.ndarray, precision: int = 8) -> str:
    """Format a 3-quantile CI array like numpy printing, e.g., [0.12 0.34 0.56]."""
    ci3 = np.asarray(ci3, dtype=float)
    return np.array2string(ci3, precision=precision, separator=" ", max_line_width=10**9)


# =============================================================================
# Projection helpers (for MC perturbations in meters)
# =============================================================================

def _build_local_transformers(lat0: float, lon0: float):
    """
    Build forward/inverse transformers for a local Azimuthal Equidistant projection centered at (lat0, lon0).
    Forward:  (lon,lat) -> (x,y) meters
    Inverse:  (x,y) meters -> (lon,lat)
    """
    if not _HAS_PYPROJ:
        return None, None, False

    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True)   # lon,lat -> x,y
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True)   # x,y -> lon,lat
    return fwd, inv, True


def _ll_to_en_m(lon: np.ndarray, lat: np.ndarray, fwd, use_pyproj: bool,
                lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat (deg) to local East/North (meters).
    If pyproj is unavailable, use a small-angle approximation (best for small regions).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if use_pyproj:
        e, n = fwd.transform(lon, lat)
        return np.asarray(e, float), np.asarray(n, float)

    lat0_rad = np.deg2rad(lat0)
    m_per_deg_lat = 111_132.92
    m_per_deg_lon = 111_412.84 * np.cos(lat0_rad)

    dlat = lat - lat0
    dlon = lon - lon0
    n = dlat * m_per_deg_lat
    e = dlon * m_per_deg_lon
    return e, n


def _en_to_ll_deg(e: np.ndarray, n: np.ndarray, inv, use_pyproj: bool,
                  lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert local East/North (meters) back to lon/lat (deg).
    If pyproj is unavailable, use an inverse small-angle approximation.
    """
    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)

    if use_pyproj:
        lon, lat = inv.transform(e, n)
        return np.asarray(lon, float), np.asarray(lat, float)

    lat0_rad = np.deg2rad(lat0)
    m_per_deg_lat = 111_132.92
    m_per_deg_lon = 111_412.84 * np.cos(lat0_rad)

    dlat = n / m_per_deg_lat
    dlon = e / m_per_deg_lon
    lat = lat0 + dlat
    lon = lon0 + dlon
    return lon, lat


# =============================================================================
# Core computations
# =============================================================================

def _gps_azimuth_deg(lat1: np.ndarray, lon1: np.ndarray,
                     lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Compute azimuths (degrees) using ObsPy gps2dist_azimuth for many point pairs.
    We use az12 as the HC azimuth (Hy -> Ce). Range mapped to [0, 360).
    """
    lat1 = np.asarray(lat1, float)
    lon1 = np.asarray(lon1, float)
    lat2 = np.asarray(lat2, float)
    lon2 = np.asarray(lon2, float)

    if not (len(lat1) == len(lon1) == len(lat2) == len(lon2)):
        raise ValueError("Lat/Lon arrays must have the same length.")

    az = np.empty(len(lat1), dtype=float)
    for i in range(len(lat1)):
        _, az12, _ = gps2dist_azimuth(lat1[i], lon1[i], lat2[i], lon2[i])
        az[i] = az12 % 360.0
    return az


def _gps_distance_m(lat1: np.ndarray, lon1: np.ndarray,
                    lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Compute geodesic distance (meters) using ObsPy gps2dist_azimuth for many point pairs.
    This is used as HC distance (Hy -> Ce). Depth is ignored.
    """
    lat1 = np.asarray(lat1, float)
    lon1 = np.asarray(lon1, float)
    lat2 = np.asarray(lat2, float)
    lon2 = np.asarray(lon2, float)

    if not (len(lat1) == len(lon1) == len(lat2) == len(lon2)):
        raise ValueError("Lat/Lon arrays must have the same length.")

    dist = np.empty(len(lat1), dtype=float)
    for i in range(len(lat1)):
        d, _, _ = gps2dist_azimuth(lat1[i], lon1[i], lat2[i], lon2[i])
        dist[i] = float(d)
    return dist


def _sector_fraction_deg(az_deg: np.ndarray, amin: float, amax: float) -> float:
    """Fraction of azimuths within [amin, amax] degrees (non-wrapping sector)."""
    az = np.asarray(az_deg, dtype=float)
    return float(np.mean((az >= amin) & (az <= amax)))


def _circular_stats_astropy(az_deg: np.ndarray) -> tuple[float, float, float]:
    """
    Compute (Rayleigh p-value, circular mean direction [deg], circular variance [0..1]).

    Compatible across different Astropy versions:
      - Some versions accept circmean(..., high=..., low=...)
      - Some versions do not.
    Fallback to a manual implementation if needed.

    Manual variance definition: V = 1 - R, where R = |mean(exp(i*theta))|.
    """
    az_deg = np.asarray(az_deg, dtype=float)
    az_rad = np.deg2rad(az_deg)

    # Rayleigh test (astropy): may return p-value or (z, p) depending on version
    rt = rayleightest(az_rad)
    if isinstance(rt, (tuple, list, np.ndarray)):
        p = float(rt[-1])
    else:
        p = float(rt)

    # circular mean & variance with version compatibility
    try:
        mean_rad = float(circmean(az_rad, high=2*np.pi, low=0.0))
        var = float(circvar(az_rad, high=2*np.pi, low=0.0))
    except TypeError:
        try:
            mean_rad = float(circmean(az_rad))
            var = float(circvar(az_rad))
            mean_rad = mean_rad % (2*np.pi)
        except TypeError:
            C = np.mean(np.cos(az_rad))
            S = np.mean(np.sin(az_rad))
            mean_rad = float(np.arctan2(S, C) % (2*np.pi))
            R = float(np.hypot(C, S))
            var = float(1.0 - R)

    mean_deg = float(np.rad2deg(mean_rad) % 360.0)
    return p, mean_deg, var


def compute_observed_stats(az_deg: np.ndarray) -> CircularStatsResult:
    p, mu, v = _circular_stats_astropy(az_deg)
    f_down = _sector_fraction_deg(az_deg, DOWNDIP_MIN_DEG, DOWNDIP_MAX_DEG)
    f_up = _sector_fraction_deg(az_deg, UPDIP_MIN_DEG, UPDIP_MAX_DEG)
    return CircularStatsResult(
        n=int(len(az_deg)),
        rayleigh_p=p,
        mean_deg=mu,
        var=v,
        frac_downdip=f_down,
        frac_updip=f_up
    )


def _rose_histogram(az_deg: np.ndarray, bin_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges_rad, hist_values). Hist can be fractions or counts depending on ROSE_NORMALIZE."""
    edges_deg = np.arange(0.0, 360.0 + bin_width_deg, bin_width_deg)
    counts, _ = np.histogram(np.asarray(az_deg) % 360.0, bins=edges_deg)

    if ROSE_NORMALIZE:
        hist = counts / max(counts.sum(), 1)
    else:
        hist = counts.astype(float)

    edges_rad = np.deg2rad(edges_deg)
    return edges_rad, hist


def bootstrap_analysis(az_deg: np.ndarray, n_boot: int, seed: int, bin_width_deg: float) -> dict:
    """
    Nonparametric bootstrap over events:
      - resample azimuths with replacement (same sample size)
      - compute sector fractions and rose histogram each time
      - return sector-fraction CI as [2.5%, 50%, 97.5%]
      - return per-bin 2.5% and 97.5% envelope for rose
    """
    rng = np.random.default_rng(seed)
    az = np.asarray(az_deg, dtype=float)
    n = len(az)
    if n < MIN_EVENTS_REQUIRED:
        raise ValueError(f"Need at least {MIN_EVENTS_REQUIRED} events for bootstrap; got {n}.")

    f_down = np.empty(n_boot, float)
    f_up = np.empty(n_boot, float)

    edges_deg = np.arange(0.0, 360.0 + bin_width_deg, bin_width_deg)
    nbins = len(edges_deg) - 1
    rose_mat = np.empty((n_boot, nbins), float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = az[idx]

        f_down[i] = _sector_fraction_deg(sample, DOWNDIP_MIN_DEG, DOWNDIP_MAX_DEG)
        f_up[i] = _sector_fraction_deg(sample, UPDIP_MIN_DEG, UPDIP_MAX_DEG)

        _, hist = _rose_histogram(sample, bin_width_deg)
        rose_mat[i, :] = hist

    return {
        "f_down_ci": _q025_q50_q975(f_down),                # [2.5%, 50%, 97.5%]
        "f_up_ci": _q025_q50_q975(f_up),                    # [2.5%, 50%, 97.5%]
        "rose_edges_deg": edges_deg,                        # for plotting
        "rose_ci_lo": np.quantile(rose_mat, 0.025, axis=0), # per-bin 2.5%
        "rose_ci_hi": np.quantile(rose_mat, 0.975, axis=0), # per-bin 97.5%
    }


def plot_rose_with_bootstrap_envelope(az_deg: np.ndarray, boot: dict, outpath: Path, title: str):
    """
    Plot observed rose histogram (orange) over a bootstrap 95% envelope (light blue),
    styled like the example figure (envelope shown as translucent background bars).
    """
    edges_deg = boot["rose_edges_deg"]
    edges_rad = np.deg2rad(edges_deg)
    centers_rad = (edges_rad[:-1] + edges_rad[1:]) / 2.0
    width_rad = np.diff(edges_rad)

    _, hist_obs = _rose_histogram(az_deg, BIN_WIDTH_DEG)
    lo = np.asarray(boot["rose_ci_lo"], float)
    hi = np.asarray(boot["rose_ci_hi"], float)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # background envelope as translucent bars (95% CI band: [lo, hi])
    lo_clip = np.clip(lo, 0.0, None)
    band = np.clip(hi - lo_clip, 0.0, None)
    ax.bar(
        centers_rad, band,
        bottom=lo_clip,
        width=width_rad,
        align="center",
        alpha=0.25,
        edgecolor="none",
        linewidth=0.0,
        zorder=1
    )

    # observed histogram on top
    ax.bar(
        centers_rad, hist_obs,
        width=width_rad,
        align="center",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.8,
        zorder=2
    )

    ax.set_title(title, pad=18)

    rmax = max(np.max(hist_obs), np.max(hi))
    ax.set_ylim(0, rmax * 1.05)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def monte_carlo_analysis(lat_hy, lon_hy, lat_ce, lon_ce,
                         fwd, inv, use_pyproj: bool,
                         lat0: float, lon0: float,
                         n_mc: int, sigma_xy_m: float, seed: int) -> dict:
    """
    Monte Carlo uncertainty propagation:
      1) Convert lon/lat to local EN meters
      2) Add Gaussian noise in EN (sigma_xy_m per component)
      3) Convert perturbed EN back to lon/lat
      4) Compute azimuths using ObsPy gps2dist_azimuth
      5) Compute Rayleigh p-values & sector fractions
    """
    rng = np.random.default_rng(seed)

    lat_hy = np.asarray(lat_hy, float)
    lon_hy = np.asarray(lon_hy, float)
    lat_ce = np.asarray(lat_ce, float)
    lon_ce = np.asarray(lon_ce, float)

    n = len(lat_hy)
    if n < MIN_EVENTS_REQUIRED:
        raise ValueError(f"Need at least {MIN_EVENTS_REQUIRED} events for Monte Carlo; got {n}.")

    # baseline EN (meters)
    e_hy, n_hy = _ll_to_en_m(lon_hy, lat_hy, fwd, use_pyproj, lat0, lon0)
    e_ce, n_ce = _ll_to_en_m(lon_ce, lat_ce, fwd, use_pyproj, lat0, lon0)

    pvals = np.empty(n_mc, float)
    f_down = np.empty(n_mc, float)
    f_up = np.empty(n_mc, float)

    for i in range(n_mc):
        eh = e_hy + rng.normal(0.0, sigma_xy_m, size=n)
        nh = n_hy + rng.normal(0.0, sigma_xy_m, size=n)
        ec = e_ce + rng.normal(0.0, sigma_xy_m, size=n)
        nc = n_ce + rng.normal(0.0, sigma_xy_m, size=n)

        lon_hy_p, lat_hy_p = _en_to_ll_deg(eh, nh, inv, use_pyproj, lat0, lon0)
        lon_ce_p, lat_ce_p = _en_to_ll_deg(ec, nc, inv, use_pyproj, lat0, lon0)

        az = _gps_azimuth_deg(lat_hy_p, lon_hy_p, lat_ce_p, lon_ce_p)
        p, _, _ = _circular_stats_astropy(az)

        pvals[i] = p
        f_down[i] = _sector_fraction_deg(az, DOWNDIP_MIN_DEG, DOWNDIP_MAX_DEG)
        f_up[i] = _sector_fraction_deg(az, UPDIP_MIN_DEG, UPDIP_MAX_DEG)

    return {
        "sigma_xy_m": float(sigma_xy_m),
        "pvals": pvals,
        "f_down": f_down,
        "f_up": f_up,
        "frac_p_lt_alpha": float(np.mean(pvals < MC_REPORT_ALPHA)),
        "p_q025_q50_q975": _q025_q50_q975(pvals),
        "f_down_q025_q50_q975": _q025_q50_q975(f_down),
        "f_up_q025_q50_q975": _q025_q50_q975(f_up),
    }


def plot_mc_pvalue_hist(pvals: np.ndarray, outpath: Path, title: str):
    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(pvals, bins=50)
    ax.set_xlabel("Rayleigh test p-value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    outdir = _ensure_outdir(OUTDIR)

    # ---------------- read data ----------------
    df = _read_excel(EXCEL_PATH, sheet=SHEET_NAME)

    df = df.dropna(subset=["LatHy", "LonHy", "LatCe", "LonCe"]).reset_index(drop=True)
    if len(df) < MIN_EVENTS_REQUIRED:
        raise ValueError(f"After dropping NaNs, only {len(df)} rows remain (<{MIN_EVENTS_REQUIRED}).")

    # QC: drop Hy==Ce (undefined azimuth). Use exact lat/lon equality (fast).
    if DROP_ZERO_LENGTH_VECTORS:
        keep = ~((df["LatHy"].values == df["LatCe"].values) & (df["LonHy"].values == df["LonCe"].values))
        df = df.loc[keep].reset_index(drop=True)

    if len(df) < MIN_EVENTS_REQUIRED:
        raise ValueError(f"After QC (zero-length), only {len(df)} rows remain (<{MIN_EVENTS_REQUIRED}).")

    # ---------------- compute HC distance for every event ----------------
    df["HCdist_m"] = _gps_distance_m(
        df["LatHy"].values, df["LonHy"].values,
        df["LatCe"].values, df["LonCe"].values
    )

    # ---------------- family-level QC filter ----------------
    qc_info = {
        "enabled": ENABLE_FAMILY_HC_FILTER,
        "threshold_m": HC_DIST_THRESHOLD_M,
        "family_col": FAMILY_ID_COL,
        "events_before": int(len(df)),
        "families_before": None,
        "events_after": None,
        "families_after": None,
        "families_removed": 0,
    }

    if ENABLE_FAMILY_HC_FILTER:
        if FAMILY_ID_COL not in df.columns:
            raise ValueError(
                f"ENABLE_FAMILY_HC_FILTER=True but column '{FAMILY_ID_COL}' not found in Excel.\n"
                f"Either add '{FAMILY_ID_COL}' column or set ENABLE_FAMILY_HC_FILTER=False."
            )

        df = df.dropna(subset=[FAMILY_ID_COL]).reset_index(drop=True)
        qc_info["families_before"] = int(df[FAMILY_ID_COL].astype(str).nunique())

        fam_min = df.groupby(FAMILY_ID_COL)["HCdist_m"].min()
        bad_fams = fam_min.index[fam_min < HC_DIST_THRESHOLD_M].tolist()
        df = df.loc[~df[FAMILY_ID_COL].isin(bad_fams)].reset_index(drop=True)

        qc_info["events_after"] = int(len(df))
        qc_info["families_after"] = int(df[FAMILY_ID_COL].astype(str).nunique())
        qc_info["families_removed"] = int(len(set(bad_fams)))
    else:
        qc_info["events_after"] = int(len(df))
        if FAMILY_ID_COL in df.columns:
            qc_info["families_before"] = int(df[FAMILY_ID_COL].astype(str).nunique())
            qc_info["families_after"] = qc_info["families_before"]

    if len(df) < MIN_EVENTS_REQUIRED:
        raise ValueError(
            f"After family-level HC distance filtering, only {len(df)} events remain (<{MIN_EVENTS_REQUIRED}).\n"
            f"Consider lowering HC_DIST_THRESHOLD_M or disabling ENABLE_FAMILY_HC_FILTER."
        )

    # ---------------- observed azimuths & stats ----------------
    az_deg = _gps_azimuth_deg(df["LatHy"].values, df["LonHy"].values,
                              df["LatCe"].values, df["LonCe"].values)

    obs = compute_observed_stats(az_deg)

    # ---------------- bootstrap ----------------
    boot = bootstrap_analysis(az_deg, n_boot=N_BOOT, seed=BOOT_RANDOM_SEED, bin_width_deg=BIN_WIDTH_DEG)

    # ---------------- plot rose ----------------
    plot_rose_with_bootstrap_envelope(
        az_deg=az_deg,
        boot=boot,
        outpath=outdir / "rose_observed_with_bootstrap_CI.pdf",
        title="HC azimuth rose (ObsPy) + bootstrap 95% CI envelope"
    )

    # ---------------- Monte Carlo ----------------
    lat0 = float(np.mean(np.r_[df["LatHy"].values, df["LatCe"].values]))
    lon0 = float(np.mean(np.r_[df["LonHy"].values, df["LonCe"].values]))
    fwd, inv, use_pyproj = _build_local_transformers(lat0, lon0)

    if not use_pyproj:
        print("\n[Warning] pyproj not found. MC uses an approximate meters<->degrees conversion.")
        print("          For best accuracy, install pyproj: pip install pyproj")

    sigmas_to_run = [SIGMA_XY_M] + list(EXTRA_SIGMAS_M)

    mc_rows = []        # for CSV
    mc_summaries = []   # for TXT

    for sigma in sigmas_to_run:
        mc = monte_carlo_analysis(
            lat_hy=df["LatHy"].values, lon_hy=df["LonHy"].values,
            lat_ce=df["LatCe"].values, lon_ce=df["LonCe"].values,
            fwd=fwd, inv=inv, use_pyproj=use_pyproj,
            lat0=lat0, lon0=lon0,
            n_mc=N_MC, sigma_xy_m=sigma, seed=MC_RANDOM_SEED
        )

        p_q = mc["p_q025_q50_q975"]
        fd_q = mc["f_down_q025_q50_q975"]
        fu_q = mc["f_up_q025_q50_q975"]

        mc_rows.append({
            "sigma_xy_m": mc["sigma_xy_m"],
            "n_mc": N_MC,
            "alpha": MC_REPORT_ALPHA,
            "frac_p_lt_alpha": mc["frac_p_lt_alpha"],
            "p_q025": p_q[0],
            "p_q50": p_q[1],
            "p_q975": p_q[2],
            "downdip_q025": fd_q[0],
            "downdip_q50": fd_q[1],
            "downdip_q975": fd_q[2],
            "updip_q025": fu_q[0],
            "updip_q50": fu_q[1],
            "updip_q975": fu_q[2],
        })

        mc_summaries.append({
            "sigma_xy_m": float(sigma),
            "p_q": p_q,
            "frac_p_lt_alpha": mc["frac_p_lt_alpha"],
            "fdown_q": fd_q,
            "fup_q": fu_q,
        })

        plot_mc_pvalue_hist(
            pvals=mc["pvals"],
            outpath=outdir / f"rayleigh_pvalue_hist_mc_sigma_{sigma:.0f}m.pdf",
            title=f"MC Rayleigh p-value (ObsPy azimuth; sigma_xy={sigma:.0f} m)"
        )

    # ---------------- save CSV outputs ----------------
    pd.DataFrame([{
        "n": obs.n,
        "rayleigh_p": obs.rayleigh_p,
        "circular_mean_deg": obs.mean_deg,
        "circular_variance": obs.var,
        "frac_downdip": obs.frac_downdip,
        "frac_updip": obs.frac_updip,
        "downdip_sector_deg": f"[{DOWNDIP_MIN_DEG}, {DOWNDIP_MAX_DEG}]",
        "updip_sector_deg": f"[{UPDIP_MIN_DEG}, {UPDIP_MAX_DEG}]",
        "bin_width_deg": BIN_WIDTH_DEG,
        "rose_normalize": ROSE_NORMALIZE,
        "note": "Azimuth computed by ObsPy gps2dist_azimuth; depth ignored."
    }]).to_csv(outdir / "stats_observed.csv", index=False)

    pd.DataFrame([{
        "n_boot": N_BOOT,
        "downdip_ci_q025": boot["f_down_ci"][0],
        "downdip_ci_q50": boot["f_down_ci"][1],
        "downdip_ci_q975": boot["f_down_ci"][2],
        "updip_ci_q025": boot["f_up_ci"][0],
        "updip_ci_q50": boot["f_up_ci"][1],
        "updip_ci_q975": boot["f_up_ci"][2],
        "note": "Sector-fraction CI reported as [2.5%, 50%, 97.5%] from bootstrap."
    }]).to_csv(outdir / "stats_bootstrap.csv", index=False)

    pd.DataFrame(mc_rows).to_csv(outdir / "stats_montecarlo.csv", index=False)

    # ---------------- write TXT summary (includes ALL sigmas) ----------------
    down_ci = boot["f_down_ci"]
    up_ci = boot["f_up_ci"]

    txt_lines = []
    txt_lines.append("Quality control:")
    txt_lines.append(f"  ENABLE_FAMILY_HC_FILTER = {ENABLE_FAMILY_HC_FILTER}")
    if ENABLE_FAMILY_HC_FILTER:
        txt_lines.append(f"  HC_DIST_THRESHOLD_M     = {HC_DIST_THRESHOLD_M:.1f} m")
        txt_lines.append(f"  FAMILY_ID_COL           = {FAMILY_ID_COL}")
    txt_lines.append(f"  events before QC        = {qc_info['events_before']}")
    if qc_info["families_before"] is not None:
        txt_lines.append(f"  families before QC      = {qc_info['families_before']}")
    txt_lines.append(f"  events after QC         = {qc_info['events_after']}")
    if qc_info["families_after"] is not None:
        txt_lines.append(f"  families after QC       = {qc_info['families_after']}")
    if ENABLE_FAMILY_HC_FILTER:
        txt_lines.append(f"  families removed        = {qc_info['families_removed']}")
    txt_lines.append("")

    txt_lines.append("Observed circular stats (DirectivityHC):")
    txt_lines.append(f"  Rayleigh p        = {obs.rayleigh_p:.3e}")
    txt_lines.append(f"  circmean (deg)    = {obs.mean_deg:.2f}")
    txt_lines.append(f"  circvar           = {obs.var:.3f}")
    txt_lines.append(f"  downdip fraction  = {obs.frac_downdip:.3f}   95%CI={_fmt_ci3(down_ci)}")
    txt_lines.append(f"  updip fraction    = {obs.frac_updip:.3f}   95%CI={_fmt_ci3(up_ci)}")
    txt_lines.append("")

    txt_lines.append("Monte Carlo robustness:")
    for s in mc_summaries:
        p_q = s["p_q"]
        fdown_q = s["fdown_q"]
        fup_q = s["fup_q"]
        txt_lines.append(f"  sigma_xy = {s['sigma_xy_m']:.1f} m (per component):")
        txt_lines.append(f"    MC Rayleigh p median     = {p_q[1]:.3e}")
        txt_lines.append(f"    MC Rayleigh p 97.5%ile   = {p_q[2]:.3e}")
        txt_lines.append(f"    fraction(p<{MC_REPORT_ALPHA:.2f})         = {s['frac_p_lt_alpha']:.3f}")
        txt_lines.append(f"    downdip frac 95% CI      = {_fmt_ci3(fdown_q)}")
        txt_lines.append(f"    updip   frac 95% CI      = {_fmt_ci3(fup_q)}")

    txt_path = outdir / "Circular_Stats_Summary.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    print(f"\nAll outputs saved to: {outdir.resolve()}")
    print(f"TXT summary saved to: {txt_path.resolve()}")


if __name__ == "__main__":
    main()

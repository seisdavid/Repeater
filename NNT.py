#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:34:27 2026

@author: DG

Nearest-Neighbor Test (NNT) for earthquake clustering in 2-D (horizontal only).

Implements the classical Clark–Evans nearest-neighbor framework:
- Observed mean nearest-neighbor distance: r_obs
- Expected mean distance under 2-D Poisson: r_exp = 1/(2*sqrt(lambda))
- Density: lambda = N / S
- Standard error (Clark–Evans): se = 0.26136 / sqrt(N*lambda)
- Ratio: R = r_obs / r_exp
- Z-score: Z = (r_exp - r_obs)/se  (so clustering -> Z > 0)
Significant clustering (95%): (R < 1) and (Z > 1.96)

Study area S is estimated in two ways:
1) Minimum Enclosing Circle (MEC): S = pi * R_mec^2
2) Minimum Volume Enclosing Ellipse (MVEE): S = pi * a * b

Input Excel must contain columns:
  LatHy, LonHy, DepHy, LatCe, LonCe, DepCe, Mag
Optionally a grouping column (e.g., FamilyID / SeqID) to run NNT per group.

Outputs a TXT summary.

"""

from __future__ import annotations
from pathlib import Path
import shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

# ============================================================
# USER PARAMETERS
# ============================================================

INPUT_XLSX = "Input_sequence.xlsx"
SHEET_NAME = 0  # sheet index or name

# Column names
LAT_HY, LON_HY = "LatHy", "LonHy"
LAT_CE, LON_CE = "LatCe", "LonCe"
MAG_COL = "Mag"

# Minimum points required to run NNT
MIN_N = 3

# Earth radius for local projection (meters)
EARTH_R_M = 6371000.0

# Output folder
OUT_DIR = Path("Out_NNT")

# Output filenames
OUT_TXT = "NNT_Summary.txt"
FIG_A_PDF = "FigA_MEC.pdf"
FIG_B_PDF = "FigB_MVEE.pdf"

# NNT threshold for "clustered"
Z_CRIT = 1.96

# MVEE settings
MVEE_TOL = 1e-6
MVEE_MAX_ITER = 10_000

# =========================
# Circular crack radius
# =========================
MAG_TO_M0_CONST = 9.1     # set to your preferred constant
STRESS_DROP_MPA = 38.0    # assumed stress drop (MPa)

def mag_to_m0_nm(mag: float, c: float = MAG_TO_M0_CONST) -> float:
    """Convert magnitude (mag) to seismic moment M0 (N·m)."""
    return 10.0 ** (1.5 * float(mag) + float(c))

def rmax_circular_crack_m(
    mag_max: float,
    stress_drop_mpa: float,
    const: float = 7/16,
    c: float = MAG_TO_M0_CONST
) -> float:
    """
    Circular crack model:
      Δσ = (7/16) * M0 / r^3  =>  r = ((7/16) * M0 / Δσ)^(1/3)
    """
    stress_drop_pa = float(stress_drop_mpa) * 1e6  # MPa -> Pa
    m0 = mag_to_m0_nm(mag_max, c=c)
    return float((const * m0 / stress_drop_pa) ** (1.0 / 3.0))

# ============================================================
# Local projection utilities (common origin for hypo+centroid)
# ============================================================

def _latlon_to_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_rad: float, lon0_rad: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg.astype(float))
    lon = np.deg2rad(lon_deg.astype(float))
    x = EARTH_R_M * (lon - lon0_rad) * math.cos(lat0_rad)
    y = EARTH_R_M * (lat - lat0_rad)
    return np.column_stack([x, y])

def latlon_to_local_xy_m_common_origin(
    lat_hy: np.ndarray, lon_hy: np.ndarray,
    lat_ce: np.ndarray, lon_ce: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project both hypocenters and centroids into one consistent local XY system,
    using the mean lat/lon of all available points as the origin.
    """
    lat_all = np.concatenate([lat_hy.astype(float), lat_ce.astype(float)])
    lon_all = np.concatenate([lon_hy.astype(float), lon_ce.astype(float)])
    lat0_rad = float(np.deg2rad(np.mean(lat_all)))
    lon0_rad = float(np.deg2rad(np.mean(lon_all)))

    xy_hy = _latlon_to_xy_m(lat_hy, lon_hy, lat0_rad, lon0_rad)
    xy_ce = _latlon_to_xy_m(lat_ce, lon_ce, lat0_rad, lon0_rad)
    return xy_hy, xy_ce

# ============================================================
# NNT core
# ============================================================

def mean_nearest_neighbor_distance(points_xy: np.ndarray) -> float:
    n = points_xy.shape[0]
    if n < 2:
        return float("nan")
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(nn))

def nnt_stats(points_xy: np.ndarray, area_S: float) -> dict:
    n = int(points_xy.shape[0])
    if n < 2 or (not np.isfinite(area_S)) or area_S <= 0:
        return dict(N=n, S=area_S, lambda_=np.nan, r_obs=np.nan, r_exp=np.nan, R=np.nan, Z=np.nan,
                    clustered=False)
    r_obs = mean_nearest_neighbor_distance(points_xy)
    lam = n / area_S
    r_exp = 1.0 / (2.0 * math.sqrt(lam))
    se = 0.26136 / math.sqrt(n * lam)  # Clark–Evans
    R = r_obs / r_exp
    Z = (r_exp - r_obs) / se  # clustering -> Z > 0
    clustered = (R < 1.0) and (Z > Z_CRIT)
    return dict(N=n, S=area_S, lambda_=lam, r_obs=r_obs, r_exp=r_exp, R=R, Z=Z,
                clustered=clustered, se=se)

# ============================================================
# Minimum Enclosing Circle (randomized incremental)
# ============================================================

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(a - b)))

def _circle_from_2pts(a: np.ndarray, b: np.ndarray):
    c = (a + b) / 2.0
    r = _dist(a, c)
    return c, r

def _circle_from_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax*ax + ay*ay)*(by-cy) + (bx*bx + by*by)*(cy-ay) + (cx*cx + cy*cy)*(ay-by)) / d
    uy = ((ax*ax + ay*ay)*(cx-bx) + (bx*bx + by*by)*(ax-cx) + (cx*cx + cy*cy)*(bx-ax)) / d
    center = np.array([ux, uy], dtype=float)
    r = _dist(center, a)
    return center, r

def _is_in_circle(p: np.ndarray, c: np.ndarray, r: float, eps=1e-10) -> bool:
    return _dist(p, c) <= r + eps

def minimum_enclosing_circle(points_xy: np.ndarray, seed: int = 0):
    pts = np.array(points_xy, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return np.array([np.nan, np.nan]), np.nan
    if n == 1:
        return pts[0].copy(), 0.0

    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(n)]

    c = pts[0].copy()
    r = 0.0
    for i in range(n):
        p = pts[i]
        if not _is_in_circle(p, c, r):
            c = p.copy()
            r = 0.0
            for j in range(i):
                q = pts[j]
                if not _is_in_circle(q, c, r):
                    c, r = _circle_from_2pts(p, q)
                    for k in range(j):
                        t = pts[k]
                        if not _is_in_circle(t, c, r):
                            out = _circle_from_3pts(p, q, t)
                            if out is not None:
                                c, r = out
                            else:
                                trio = [p, q, t]
                                maxd = -1.0
                                best = None
                                for u in range(3):
                                    for v in range(u+1, 3):
                                        d_uv = _dist(trio[u], trio[v])
                                        if d_uv > maxd:
                                            maxd = d_uv
                                            best = (trio[u], trio[v])
                                c, r = _circle_from_2pts(best[0], best[1])
    return c, float(r)

# ============================================================
# MVEE (Khachiyan) + ellipse drawing helpers
# ============================================================

def mvee(points_xy: np.ndarray, tol: float = MVEE_TOL, max_iter: int = MVEE_MAX_ITER):
    P = np.array(points_xy, dtype=float)
    n, d = P.shape
    if n == 0:
        return np.array([np.nan, np.nan]), np.full((2, 2), np.nan)
    if n == 1:
        c = P[0].copy()
        A = np.eye(2) / 1e-12
        return c, A

    Q = np.column_stack([P, np.ones(n)])  # (n, d+1)
    u = np.ones(n) / n

    for _ in range(max_iter):
        X = Q.T @ np.diag(u) @ Q
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            X_inv = np.linalg.pinv(X)

        M = np.einsum("ij,jk,ik->i", Q, X_inv, Q)
        j = int(np.argmax(M))
        max_m = M[j]

        step = (max_m - (d + 1)) / ((d + 1) * (max_m - 1))
        new_u = (1 - step) * u
        new_u[j] += step

        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    center = P.T @ u
    PuP = (P - center).T @ np.diag(u) @ (P - center)
    try:
        A = np.linalg.inv(PuP) / d
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(PuP) / d
    return center, A

def ellipse_params_from_A(A: np.ndarray) -> tuple[float, float, float]:
    """
    For (x-c)^T A (x-c) <= 1:
      semi-axes = 1/sqrt(eigvals(A))
      angle = angle of eigenvector for major axis (deg, CCW)
    Returns (width, height, angle_deg) where width/height are full lengths (2a, 2b).
    """
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-300)
    a = 1.0 / math.sqrt(float(w[0]))  # larger semi-axis
    b = 1.0 / math.sqrt(float(w[1]))  # smaller semi-axis
    vx, vy = V[:, 0]
    angle = math.degrees(math.atan2(vy, vx))
    width = 2.0 * a
    height = 2.0 * b
    return float(width), float(height), float(angle)

def ellipse_area_from_A(A: np.ndarray) -> tuple[float, float, float]:
    A = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(A)
    w = np.maximum(w, 1e-300)
    a = 1.0 / math.sqrt(float(w[0]))
    b = 1.0 / math.sqrt(float(w[1]))
    area = math.pi * a * b
    return float(area), float(a), float(b)

def ellipse_bbox_halfwidth_halfheight(width: float, height: float, angle_deg: float) -> tuple[float, float]:
    """
    Axis-aligned bounding box half-width/half-height for a rotated ellipse.
    width/height are full lengths (2a, 2b).
    """
    a = 0.5 * float(width)
    b = 0.5 * float(height)
    th = math.radians(float(angle_deg))
    c = math.cos(th)
    s = math.sin(th)
    hw = math.sqrt((a * c) ** 2 + (b * s) ** 2)
    hh = math.sqrt((a * s) ** 2 + (b * c) ** 2)
    return float(hw), float(hh)

# ============================================================
# Plotting helpers
# ============================================================

def mag_to_color(mag: float) -> str:
    """
    Color bins:
      Mag >= 4.5           -> red
      3.2 <= Mag < 4.5     -> limegreen (lemon green)
      Mag < 3.2            -> blue
    """
    if mag >= 4.5:
        return "red"
    elif mag >= 3.2:
        return "limegreen"
    else:
        return "blue"

def add_scale_bar_right_bottom(ax, length_m: float = 500.0, label: str = "0.5 km"):
    """
    Draw scale bar at bottom-right of current axes limits.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    margin_x = 0.08 * (x1 - x0)
    margin_y = 0.08 * (y1 - y0)

    sx = x1 - margin_x - length_m  # left end x
    sy = y0 + margin_y             # y position
    ax.plot([sx, sx + length_m], [sy, sy], linewidth=2, color="black")
    ax.text(sx + 0.5 * length_m, sy - 0.03 * length_m, label, ha="center", va="top", fontsize=10, color="black")

def set_equal_aspect_with_bounds(ax, xmin, xmax, ymin, ymax, pad_frac: float = 0.12):
    dx = xmax - xmin
    dy = ymax - ymin
    pad = pad_frac * max(dx, dy, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")

def compute_plot_bounds_for_mec(
    xy_hy: np.ndarray,
    xy_ce: np.ndarray,
    mags: np.ndarray,
    mec_hy: tuple[np.ndarray, float],
    mec_ce: tuple[np.ndarray, float]
) -> tuple[float, float, float, float]:
    # Points
    xmin = float(np.min([xy_hy[:, 0].min(), xy_ce[:, 0].min()]))
    xmax = float(np.max([xy_hy[:, 0].max(), xy_ce[:, 0].max()]))
    ymin = float(np.min([xy_hy[:, 1].min(), xy_ce[:, 1].min()]))
    ymax = float(np.max([xy_hy[:, 1].max(), xy_ce[:, 1].max()]))

    # Rupture circles (centroid-centered)
    rr = np.array([rmax_circular_crack_m(float(m), STRESS_DROP_MPA) for m in mags], dtype=float)
    cx = xy_ce[:, 0]
    cy = xy_ce[:, 1]
    xmin = min(xmin, float(np.min(cx - rr)))
    xmax = max(xmax, float(np.max(cx + rr)))
    ymin = min(ymin, float(np.min(cy - rr)))
    ymax = max(ymax, float(np.max(cy + rr)))

    # MEC envelopes
    (c_hy, r_hy) = mec_hy
    (c_ce, r_ce) = mec_ce
    xmin = min(xmin, float(c_hy[0] - r_hy), float(c_ce[0] - r_ce))
    xmax = max(xmax, float(c_hy[0] + r_hy), float(c_ce[0] + r_ce))
    ymin = min(ymin, float(c_hy[1] - r_hy), float(c_ce[1] - r_ce))
    ymax = max(ymax, float(c_hy[1] + r_hy), float(c_ce[1] + r_ce))

    return xmin, xmax, ymin, ymax

def compute_plot_bounds_for_mvee(
    xy_hy: np.ndarray,
    xy_ce: np.ndarray,
    mags: np.ndarray,
    mvee_hy: tuple[np.ndarray, np.ndarray],
    mvee_ce: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float, float, float]:
    # Points
    xmin = float(np.min([xy_hy[:, 0].min(), xy_ce[:, 0].min()]))
    xmax = float(np.max([xy_hy[:, 0].max(), xy_ce[:, 0].max()]))
    ymin = float(np.min([xy_hy[:, 1].min(), xy_ce[:, 1].min()]))
    ymax = float(np.max([xy_hy[:, 1].max(), xy_ce[:, 1].max()]))

    # Rupture circles
    rr = np.array([rmax_circular_crack_m(float(m), STRESS_DROP_MPA) for m in mags], dtype=float)
    cx = xy_ce[:, 0]
    cy = xy_ce[:, 1]
    xmin = min(xmin, float(np.min(cx - rr)))
    xmax = max(xmax, float(np.max(cx + rr)))
    ymin = min(ymin, float(np.min(cy - rr)))
    ymax = max(ymax, float(np.max(cy + rr)))

    # MVEE ellipses via axis-aligned bounding boxes
    (cen_hy, A_hy) = mvee_hy
    (cen_ce, A_ce) = mvee_ce

    w_hy, h_hy, ang_hy = ellipse_params_from_A(A_hy)
    hw_hy, hh_hy = ellipse_bbox_halfwidth_halfheight(w_hy, h_hy, ang_hy)
    xmin = min(xmin, float(cen_hy[0] - hw_hy))
    xmax = max(xmax, float(cen_hy[0] + hw_hy))
    ymin = min(ymin, float(cen_hy[1] - hh_hy))
    ymax = max(ymax, float(cen_hy[1] + hh_hy))

    w_ce, h_ce, ang_ce = ellipse_params_from_A(A_ce)
    hw_ce, hh_ce = ellipse_bbox_halfwidth_halfheight(w_ce, h_ce, ang_ce)
    xmin = min(xmin, float(cen_ce[0] - hw_ce))
    xmax = max(xmax, float(cen_ce[0] + hw_ce))
    ymin = min(ymin, float(cen_ce[1] - hh_ce))
    ymax = max(ymax, float(cen_ce[1] + hh_ce))

    return xmin, xmax, ymin, ymax

def annotate_nnt(ax, stats_hy: dict, stats_ce: dict):
    txt = (
        f"Hypocenter NNT (N={stats_hy['N']}):\n"
        f"  r_obs={stats_hy['r_obs']:.3f} m\n"
        f"  r_exp={stats_hy['r_exp']:.3f} m\n"
        f"  R={stats_hy['R']:.3f}, Z={stats_hy['Z']:.3f}\n\n"
        f"Centroid NNT (N={stats_ce['N']}):\n"
        f"  r_obs={stats_ce['r_obs']:.3f} m\n"
        f"  r_exp={stats_ce['r_exp']:.3f} m\n"
        f"  R={stats_ce['R']:.3f}, Z={stats_ce['Z']:.3f}"
    )
    ax.text(
        0.02, 0.02, txt,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, linewidth=0.8)
    )

# ============================================================
# Summary TXT
# ============================================================

def fmt(x) -> str:
    if x is None:
        return "nan"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "nan"
        return f"{xf:.6g}"
    except Exception:
        return str(x)

def write_summary_txt(path: Path, mec_hy, mec_ce, mvee_hy, mvee_ce,
                      stats_mec_hy, stats_mec_ce, stats_mvee_hy, stats_mvee_ce):
    lines = []
    lines.append("Nearest-Neighbor Test (NNT) Summary (2-D horizontal only)")
    lines.append("=" * 72)
    lines.append(f"Input Excel         : {Path(INPUT_XLSX).resolve()}")
    lines.append(f"Sheet               : {SHEET_NAME}")
    lines.append(f"MIN_N               : {MIN_N}")
    lines.append(f"Cluster criterion   : (R < 1) AND (Z > {Z_CRIT})")
    lines.append("")
    lines.append("Circular crack model for dashed centroid circles:")
    lines.append(f"  MAG_TO_M0_CONST   : {MAG_TO_M0_CONST}")
    lines.append(f"  STRESS_DROP_MPA   : {STRESS_DROP_MPA}")
    lines.append("")

    def block(title, stats, area_meta):
        lines.append("-" * 72)
        lines.append(title)
        lines.append(f"  N        = {stats['N']}")
        lines.append(f"  S (m^2)  = {fmt(stats['S'])}")
        lines.append(f"  lambda   = {fmt(stats['lambda_'])} (1/m^2)")
        lines.append(f"  r_obs(m) = {fmt(stats['r_obs'])}")
        lines.append(f"  r_exp(m) = {fmt(stats['r_exp'])}")
        lines.append(f"  R        = {fmt(stats['R'])}")
        lines.append(f"  Z        = {fmt(stats['Z'])}")
        lines.append(f"  clustered= {stats['clustered']}")
        for k, v in area_meta.items():
            lines.append(f"  {k} = {fmt(v)}")

    c_hy, r_hy = mec_hy
    c_ce, r_ce = mec_ce
    block("MEC (Minimum Enclosing Circle) — Hypocenter",
          stats_mec_hy,
          {"MEC_center_x(m)": c_hy[0], "MEC_center_y(m)": c_hy[1], "MEC_radius_m": r_hy})
    block("MEC (Minimum Enclosing Circle) — Centroid",
          stats_mec_ce,
          {"MEC_center_x(m)": c_ce[0], "MEC_center_y(m)": c_ce[1], "MEC_radius_m": r_ce})

    cen_hy, A_hy, area_hy, a_hy, b_hy = mvee_hy
    cen_ce, A_ce, area_ce, a_ce, b_ce = mvee_ce
    block("MVEE (Minimum Volume Enclosing Ellipse) — Hypocenter",
          stats_mvee_hy,
          {"MVEE_center_x(m)": cen_hy[0], "MVEE_center_y(m)": cen_hy[1], "a_m": a_hy, "b_m": b_hy})
    block("MVEE (Minimum Volume Enclosing Ellipse) — Centroid",
          stats_mvee_ce,
          {"MVEE_center_x(m)": cen_ce[0], "MVEE_center_y(m)": cen_ce[1], "a_m": a_ce, "b_m": b_ce})

    path.write_text("\n".join(lines), encoding="utf-8")

# ============================================================
# Figures
# ============================================================

def plot_fig_A_mec(out_pdf: Path, xy_hy: np.ndarray, xy_ce: np.ndarray, mags: np.ndarray,
                   mec_hy, mec_ce, stats_hy: dict, stats_ce: dict):
    hy_state = "clustered" if stats_hy["clustered"] else "unclustered"
    ce_state = "clustered" if stats_ce["clustered"] else "unclustered"
    title = f"Hypocenter: {hy_state};  Centroid: {ce_state}"

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Event drawings
    for (hx, hy), (cx, cy), m in zip(xy_hy, xy_ce, mags):
        col = mag_to_color(float(m))
        ax.plot([hx, cx], [hy, cy], linewidth=2.0, linestyle="-", color=col, zorder=2)
        ax.scatter([hx], [hy], s=30, facecolors="none", edgecolors=col, linewidths=2.0, zorder=3)
        ax.scatter([cx], [cy], s=16, facecolors=col, edgecolors=col, linewidths=0.8, zorder=4)

        rr = rmax_circular_crack_m(float(m), STRESS_DROP_MPA)
        ax.add_patch(Circle((cx, cy), rr, fill=False, linestyle="--",
                            linewidth=1.6, edgecolor=col, zorder=1))

    # MEC envelopes
    c_hy, r_hy = mec_hy
    c_ce, r_ce = mec_ce

    ax.scatter([c_hy[0]], [c_hy[1]], marker="*", s=180, color="black", zorder=6)
    ax.add_patch(Circle((c_hy[0], c_hy[1]), r_hy, fill=False, linestyle="-",
                        linewidth=2.2, edgecolor="black", zorder=5))

    ax.scatter([c_ce[0]], [c_ce[1]], marker="*", s=180, color="purple", zorder=6)
    ax.add_patch(Circle((c_ce[0], c_ce[1]), r_ce, fill=False, linestyle="-",
                        linewidth=2.2, edgecolor="purple", zorder=5))

    annotate_nnt(ax, stats_hy, stats_ce)

    # Bounds include points + rupture circles + MEC circles
    xmin, xmax, ymin, ymax = compute_plot_bounds_for_mec(xy_hy, xy_ce, mags, mec_hy, mec_ce)
    set_equal_aspect_with_bounds(ax, xmin, xmax, ymin, ymax, pad_frac=0.12)

    # Scale bar at bottom-right
    add_scale_bar_right_bottom(ax, length_m=500.0, label="0.5 km")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

def plot_fig_B_mvee(out_pdf: Path, xy_hy: np.ndarray, xy_ce: np.ndarray, mags: np.ndarray,
                    mvee_hy: tuple[np.ndarray, np.ndarray], mvee_ce: tuple[np.ndarray, np.ndarray],
                    stats_hy: dict, stats_ce: dict):
    hy_state = "clustered" if stats_hy["clustered"] else "unclustered"
    ce_state = "clustered" if stats_ce["clustered"] else "unclustered"
    title = f"Hypocenter: {hy_state};  Centroid: {ce_state}"

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Event drawings
    for (hx, hy), (cx, cy), m in zip(xy_hy, xy_ce, mags):
        col = mag_to_color(float(m))
        ax.plot([hx, cx], [hy, cy], linewidth=2.0, linestyle="-", color=col, zorder=2)
        ax.scatter([hx], [hy], s=30, facecolors="none", edgecolors=col, linewidths=2.0, zorder=3)
        ax.scatter([cx], [cy], s=16, facecolors=col, edgecolors=col, linewidths=0.8, zorder=4)

        rr = rmax_circular_crack_m(float(m), STRESS_DROP_MPA)
        ax.add_patch(Circle((cx, cy), rr, fill=False, linestyle="--",
                            linewidth=1.6, edgecolor=col, zorder=1))

    # MVEE envelopes
    cen_hy, A_hy = mvee_hy
    cen_ce, A_ce = mvee_ce

    ax.scatter([cen_hy[0]], [cen_hy[1]], marker="*", s=180, color="black", zorder=6)
    w_hy, h_hy, ang_hy = ellipse_params_from_A(A_hy)
    ax.add_patch(Ellipse((cen_hy[0], cen_hy[1]), width=w_hy, height=h_hy, angle=ang_hy,
                         fill=False, edgecolor="black", linewidth=2.2, linestyle="-", zorder=5))

    ax.scatter([cen_ce[0]], [cen_ce[1]], marker="*", s=180, color="purple", zorder=6)
    w_ce, h_ce, ang_ce = ellipse_params_from_A(A_ce)
    ax.add_patch(Ellipse((cen_ce[0], cen_ce[1]), width=w_ce, height=h_ce, angle=ang_ce,
                         fill=False, edgecolor="purple", linewidth=2.2, linestyle="-", zorder=5))

    annotate_nnt(ax, stats_hy, stats_ce)

    # Bounds include points + rupture circles + MVEE ellipses
    xmin, xmax, ymin, ymax = compute_plot_bounds_for_mvee(xy_hy, xy_ce, mags, (cen_hy, A_hy), (cen_ce, A_ce))
    set_equal_aspect_with_bounds(ax, xmin, xmax, ymin, ymax, pad_frac=0.12)

    # Scale bar at bottom-right
    add_scale_bar_right_bottom(ax, length_m=500.0, label="0.5 km")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    # Output folder: delete then recreate
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read Excel
    in_path = Path(INPUT_XLSX)
    if not in_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {in_path.resolve()}")
    df = pd.read_excel(in_path, sheet_name=SHEET_NAME)

    # Check required columns
    for c in [LAT_HY, LON_HY, LAT_CE, LON_CE, MAG_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Drop rows missing coords or Mag (Mag needed for colors & rupture circles)
    df2 = df.dropna(subset=[LAT_HY, LON_HY, LAT_CE, LON_CE, MAG_COL]).copy()
    if len(df2) < MIN_N:
        raise ValueError(f"Not enough events after dropping NaNs: N={len(df2)} (need >= {MIN_N})")

    lat_hy = df2[LAT_HY].to_numpy(dtype=float)
    lon_hy = df2[LON_HY].to_numpy(dtype=float)
    lat_ce = df2[LAT_CE].to_numpy(dtype=float)
    lon_ce = df2[LON_CE].to_numpy(dtype=float)
    mags = df2[MAG_COL].to_numpy(dtype=float)

    # Project to common local XY
    xy_hy, xy_ce = latlon_to_local_xy_m_common_origin(lat_hy, lon_hy, lat_ce, lon_ce)

    # ========== MEC ==========
    c_hy_mec, r_hy_mec = minimum_enclosing_circle(xy_hy, seed=0)
    c_ce_mec, r_ce_mec = minimum_enclosing_circle(xy_ce, seed=0)
    S_hy_mec = math.pi * (r_hy_mec ** 2)
    S_ce_mec = math.pi * (r_ce_mec ** 2)
    stats_hy_mec = nnt_stats(xy_hy, S_hy_mec)
    stats_ce_mec = nnt_stats(xy_ce, S_ce_mec)

    # ========== MVEE ==========
    cen_hy, A_hy = mvee(xy_hy)
    area_hy, a_hy, b_hy = ellipse_area_from_A(A_hy)
    cen_ce, A_ce = mvee(xy_ce)
    area_ce, a_ce, b_ce = ellipse_area_from_A(A_ce)
    stats_hy_mvee = nnt_stats(xy_hy, area_hy)
    stats_ce_mvee = nnt_stats(xy_ce, area_ce)

    # TXT
    write_summary_txt(
        OUT_DIR / OUT_TXT,
        mec_hy=(c_hy_mec, r_hy_mec),
        mec_ce=(c_ce_mec, r_ce_mec),
        mvee_hy=(cen_hy, A_hy, area_hy, a_hy, b_hy),
        mvee_ce=(cen_ce, A_ce, area_ce, a_ce, b_ce),
        stats_mec_hy=stats_hy_mec,
        stats_mec_ce=stats_ce_mec,
        stats_mvee_hy=stats_hy_mvee,
        stats_mvee_ce=stats_ce_mvee
    )

    # Figures
    plot_fig_A_mec(
        OUT_DIR / FIG_A_PDF,
        xy_hy=xy_hy, xy_ce=xy_ce, mags=mags,
        mec_hy=(c_hy_mec, r_hy_mec),
        mec_ce=(c_ce_mec, r_ce_mec),
        stats_hy=stats_hy_mec, stats_ce=stats_ce_mec
    )

    plot_fig_B_mvee(
        OUT_DIR / FIG_B_PDF,
        xy_hy=xy_hy, xy_ce=xy_ce, mags=mags,
        mvee_hy=(cen_hy, A_hy),
        mvee_ce=(cen_ce, A_ce),
        stats_hy=stats_hy_mvee, stats_ce=stats_ce_mvee
    )

    print("[OK] Outputs written to:", OUT_DIR.resolve())
    print(" -", (OUT_DIR / OUT_TXT).name)
    print(" -", (OUT_DIR / FIG_A_PDF).name)
    print(" -", (OUT_DIR / FIG_B_PDF).name)

if __name__ == "__main__":
    main()


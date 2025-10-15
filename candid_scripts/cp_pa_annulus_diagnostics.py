#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CP diagnostics (standalone): (1) chromatic model selection per triangle; (2) PA-coherent CP in annuli.

- Loads OIFITS via CANDID
- Restricts observables to ['v2','cp'] (no T3amp)
- Optional CP sigma floor (deg) in quadrature
- Picks |u,v| annuli from quantiles (or explicit L0 list), then:
  (1) fits CONST/LIN/SIN models of CP vs s (s := longest-edge length, ∝ 1/λ)
      -> BIC table, ΔBIC, per-triangle plots (top-N)
  (2) circular-mean CP vs baseline PA (longest edge), with bootstrap SE + Rayleigh p
      -> plot + CSV

Install:
    pip install "git+https://github.com/amerand/CANDID.git#egg=candid"

Example:
    python cp_pa_annulus_diagnostics.py \
        --files "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*fits" \
        --out_dir ./diag_out --ins pionier \
        --cp_floor_deg 1.0 --annulus_quantiles 65 80 90 --frac_width 0.10 \
        --bins 12 --min_channels 8 --plot_top 12 --nboot 800


"""
import argparse, glob, json, os, re, fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit

import candid

# -------------------- misc helpers --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _angle_deg(u, v):
    # PA (deg East of North), folded to [0,180)
    return (np.degrees(np.arctan2(u, v)) % 180.0)

def _unwrap_sorted(x, y_rad):
    i = np.argsort(x)
    xs = x[i]
    ys = np.unwrap(y_rad[i])
    return xs, ys

# def _bic(chi2, k, n):
#     return chi2 + k*np.log(max(n, 1))

def _sin_model(x, a, A, w, phi):
    return a + A*np.sin(w*x + phi)

def _initial_sine_guess(x, y):
    a0 = np.nanmedian(y)
    rng = x.max() - x.min()
    if not (np.isfinite(rng) and rng > 0):
        return a0, 0.0, 1.0, 0.0
    w_grid = 2*np.pi * np.array([0.5, 1.0, 2.0, 3.0, 4.0]) / max(rng, 1e-9)
    phi_grid = np.array([0.0, 0.5*np.pi, np.pi, 1.5*np.pi])
    best = None
    for w in w_grid:
        for phi in phi_grid:
            s = np.sin(w*x + phi)
            den = np.dot(s, s)
            A0 = 0.0 if den == 0 else np.dot(y - a0, s)/den
            rss = np.nanmean((y - (a0 + A0*s))**2)
            if best is None or rss < best[0]:
                best = (rss, a0, A0, w, phi)
    _, a0, A0, w0, phi0 = best
    return a0, A0, w0, phi0

def _gather_v2_baselines(o):
    Ls = []
    for c in getattr(o, "_chi2Data", []):
        if c[0].split(';')[0].lower() in ('v2', 'vis2', 'v2amp'):
            u = np.asarray(c[1]).ravel(); v = np.asarray(c[2]).ravel()
            Ls.append(np.hypot(u, v))
    if not Ls: return np.array([])
    return np.concatenate(Ls)

def _gather_cp_longest_edge(o):
    Lmaxs = []
    for c in getattr(o, "_chi2Data", []):
        if c[0].split(';')[0].lower() in ('cp','t3','icp','scp','ccp'):
            u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
            u2 = np.asarray(c[3]).ravel(); v2 = np.asarray(c[4]).ravel()
            u3, v3 = -(u1+u2), -(v1+v2)
            L1 = np.hypot(u1, v1); L2 = np.hypot(u2, v2); L3 = np.hypot(u3, v3)
            Lmaxs.append(np.maximum(L1, np.maximum(L2, L3)))
    if not Lmaxs: return np.array([])
    return np.concatenate(Lmaxs)

def apply_cp_sigma_floor(o, cp_floor_deg=None):
    if cp_floor_deg is None: return
    f = float(cp_floor_deg)
    for c in getattr(o, "_chi2Data", []):
        kind = c[0].split(';')[0].lower()
        if kind in ('cp','t3','icp','scp','ccp'):
            err = np.asarray(c[-1], dtype=float)
            c[-1] = np.hypot(err, np.deg2rad(f))

# -------------------- (1) CP chromatic model selection --------------------
def _guess_lambda_from_tuple(c):
    """
    Return wavelength array (meters) for this _chi2Data tuple, or None.
    Handles meters, microns, nanometers, and wavenumber (1/m).
    Then sanity-checks against band hint in the label to fix x10 slips.
    """
    # 1) raw guess from numeric scale
    def _raw_guess(arr):
        med = float(np.nanmedian(np.abs(arr)))
        # meters
        if 5e-7 < med < 3e-5:
            return arr.astype(float)
        # microns
        if 0.3 < med < 30.0:
            return (arr.astype(float) * 1e-6)
        # nanometers
        if 300.0 < med < 30000.0:
            return (arr.astype(float) * 1e-9)
        # wavenumber (1/m)
        if 2e4 < med < 2e7:
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = 1.0 / arr.astype(float)
            return lam
        return None

    n = np.asarray(c[-2]).size
    lam = None
    for j in range(1, len(c)-1):  # scan fields
        arr = np.asarray(c[j]).ravel()
        if arr.size != n or not np.isfinite(arr).any():
            continue
        lam = _raw_guess(arr)
        if lam is not None:
            break
    if lam is None:
        return None

    # 2) band hint from label (if present)
    # example label: 'cp;PIONIER_Pnat(1.5160680/1.7606205)'
    lbl = str(c[0])
    hint = None
    if "(" in lbl and ")" in lbl:
        try:
            inside = lbl.split("(",1)[1].split(")",1)[0]
            parts = [p for p in re.split(r"[/,; ]+", inside) if p]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    pass
            if vals:
                # numbers are in microns for PIONIER labels
                hint = np.median(vals) * 1e-6  # meters
        except Exception:
            pass

    if hint and np.isfinite(hint):
        gmed = float(np.nanmedian(lam))
        # if they disagree by >3×, try to fix by power-of-10 snap
        ratio = gmed / hint
        if not (0.3 <= ratio <= 3.0):
            # nearest power-of-10 factor
            pow10 = 10.0**round(np.log10(max(ratio, 1e-12)))
            lam = lam / pow10  # correct the scale
    return lam
# def _guess_lambda_from_tuple(c):
#     """
#     Try to find the wavelength array inside a CANDID _chi2Data tuple.
#     Return λ in **meters** (1D array) or None.

#     We look for an array with the same length as the data and whose
#     scale matches one of: meters, microns, nanometers, or wavenumber (1/m).
#     """
#     n = np.asarray(c[-2]).size  # same length as data array
#     for j in range(1, len(c)-1):  # skip label at 0; last item is the error
#         arr = np.asarray(c[j]).ravel()
#         if arr.size != n or not np.isfinite(arr).any():
#             continue

#         med = float(np.nanmedian(np.abs(arr)))
#         # meters (e.g., 1.6e-6)
#         if 5e-7 < med < 3e-5:
#             return arr.astype(float)
#         # microns (e.g., 1.5 ... 2.2)
#         if 0.3 < med < 30.0:
#             return (arr.astype(float) * 1e-6)
#         # nanometers (e.g., 1500 ... 2200)
#         if 300.0 < med < 30000.0:
#             return (arr.astype(float) * 1e-9)
#         # wavenumber 1/m (e.g., ~6e5 for H-band)
#         if 2e4 < med < 2e7:
#             with np.errstate(divide="ignore", invalid="ignore"):
#                 lam = 1.0 / arr.astype(float)
#             return lam

#     return None

# def _guess_lambda_from_tuple(c):
#     """
#     Try to find the wavelength array (meters) inside a CANDID _chi2Data tuple.
#     Returns a 1D array or None.
#     Heuristic: same length as data, values ~ 0.5–20 micron (=5e-7..2e-5 m).
#     """
#     n = np.asarray(c[-2]).size  # same length as data array
#     for j in range(1, len(c)-2):  # skip label at 0 and [data,err] at the end
#         arr = np.asarray(c[j]).ravel()
#         if arr.size != n or not np.isfinite(arr).any():
#             continue
#         med = float(np.nanmedian(np.abs(arr)))
#         if 5e-7 < med < 2e-5:     # meters; works for H/K/L/M/N bands
#             return arr
#     return None

def _global_median_lambda(o):
    w = []
    for c in getattr(o, "_chi2Data", []):
        lam = _guess_lambda_from_tuple(c)
        if lam is not None:
            w.append(lam)
    if not w:
        return None
    w = np.concatenate(w)
    w = w[np.isfinite(w)]
    return float(np.nanmedian(w)) if w.size else None


def _group_cp_by_triangle_in_annulus(o, L0, frac_width=0.10, tol_deg=2.0):
    """
    Use the annulus to SELECT triangles, but once selected, keep ALL channels
    for those triangles so CP vs s has leverage.

    We form a triangle key by rounding the three edge PAs (deg E of N) using
    the samples that lie inside the annulus, then we push *all* channels from
    that triangle (no s-clip) into the group arrays.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"s": [], "cp_deg": [], "sig_deg": [], "pa_long_deg": []})

    rPA = lambda a: np.round((np.asarray(a, float) % 180.0)/tol_deg)*tol_deg

    lam_med = _global_median_lambda(o)  # meters (None if we can’t find it)
    if lam_med is None:
        print("[WARN] Could not find wavelengths; assuming |u,v| already include 1/λ scaling.")

    for c in o._chi2Data:
        kind = c[0].split(';')[0].lower()
        if kind not in ('cp','t3','icp','scp','ccp'):
            continue

        # geometry
        u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
        u2 = np.asarray(c[3]).ravel(); v2 = np.asarray(c[4]).ravel()
        u3, v3 = -(u1+u2), -(v1+v2)
        a1 = _angle_deg(u1, v1); a2 = _angle_deg(u2, v2); a3 = _angle_deg(u3, v3)
        L1 = np.hypot(u1, v1);   L2 = np.hypot(u2, v2);   L3 = np.hypot(u3, v3)
        Lmax = np.maximum(L1, np.maximum(L2, L3))
        imax = np.argmax(np.vstack([L1, L2, L3]), axis=0)           # per-channel longest edge
        pa_long = np.choose(imax, [a1, a2, a3])

        # data
        cp  = np.asarray(c[-2]).ravel()   # rad
        sig = np.asarray(c[-1]).ravel()   # rad
        good = np.isfinite(cp) & np.isfinite(sig)

        # selection BY annulus, using Smax
        if lam_med is None:
            Smax = Lmax   # often already in wavelengths for many OIFITS
        else:
            # if u,v in meters, promote to spatial frequency
            # we don't have per-channel λ; lam_med gives a broad center, OK for selecting triangles
            Smax = Lmax / lam_med
        sel = good & (Smax >= L0*(1-frac_width)) & (Smax <= L0*(1+frac_width))
        if not np.any(sel):
            continue  # this triangle never hits the annulus

        # define the triangle key from the sel samples
        k1 = rPA(a1[sel]); k2 = rPA(a2[sel]); k3 = rPA(a3[sel])
        # take the mode (most common) rounded PA triplet across the selected samples
        keys = np.vstack([k1, k2, k3]).T
        # canonicalize ordering
        keys = np.array([tuple(sorted(t.tolist())) for t in keys])
        # pick the most frequent key; if tie, first
        if keys.size == 0:
            continue
        # simple mode:
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        tri_key = tuple(uniq[np.argmax(counts)].tolist())

        # now push ALL channels of this triangle (not just sel)
        # build s per channel
        if lam_med is None:
            s1, s2, s3 = L1, L2, L3
        else:
            s1, s2, s3 = L1/lam_med, L2/lam_med, L3/lam_med
        s_long = np.choose(imax, [s1, s2, s3])

        # If s_long still has tiny dynamic range, fall back to a normalized channel index
        if np.nanmax(s_long) - np.nanmin(s_long) < 1e-6:
            idx = np.arange(s_long.size, dtype=float)
            idx = (idx - idx.min()) / max(idx.ptp(), 1.0)
            s_long = idx

        # append
        groups[tri_key]["s"].extend(np.asarray(s_long[good], float).tolist())
        groups[tri_key]["cp_deg"].extend(np.degrees(cp[good]).tolist())
        groups[tri_key]["sig_deg"].extend(np.degrees(sig[good]).tolist())
        groups[tri_key]["pa_long_deg"].extend(np.asarray(pa_long[good], float).tolist())

    # finalize arrays
    for k, g in groups.items():
        for nm in ("s","cp_deg","sig_deg","pa_long_deg"):
            g[nm] = np.asarray(g[nm], dtype=float)
    return groups

# def _group_cp_by_triangle_in_annulus(o, L0, frac_width=0.10, tol_deg=2.0):
#     """
#     Group CP samples by (rounded) triangle PAs, but *select* and *plot* them
#     in a spatial-frequency annulus s = B/λ around s0 ≈ (L0 / median(λ)).

#     L0: the center you passed before (in baseline units). We convert it to s0.
#     """
#     from collections import defaultdict
#     groups = defaultdict(lambda: {"s": [], "cp_deg": [], "sig_deg": [], "pa_long_deg": []})

#     rPA = lambda a: np.round((np.asarray(a, float) % 180.0)/tol_deg)*tol_deg

#     lam_med = _global_median_lambda(o)  # meters
#     if lam_med is None:
#         print("[WARN] Could not find wavelengths; CP chromatic test will use baseline only (s flat).")
#     s0 = (L0 / lam_med) if lam_med else L0  # fall back if no λ

#     for c in o._chi2Data:
#         kind = c[0].split(';')[0].lower()
#         if kind not in ('cp','t3','icp','scp','ccp'):
#             continue

#         u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
#         u2 = np.asarray(c[3]).ravel(); v2 = np.asarray(c[4]).ravel()
#         u3, v3 = -(u1+u2), -(v1+v2)

#         L1 = np.hypot(u1, v1); L2 = np.hypot(u2, v2); L3 = np.hypot(u3, v3)
#         Lmax = np.maximum(L1, np.maximum(L2, L3))

#         lam = _guess_lambda_from_tuple(c)   # meters per channel (vector)
#         if lam is None:
#             # no λ available: treat s ≡ L (will look flat in x)
#             S1, S2, S3, Smax = L1, L2, L3, Lmax
#         else:
#             lam = lam.astype(float)
#             S1, S2, S3 = L1/lam, L2/lam, L3/lam
#             Smax = Lmax/lam

#         # select by spatial frequency annulus
#         band = (Smax >= s0*(1-frac_width)) & (Smax <= s0*(1+frac_width))
#         if not np.any(band):
#             continue

#         a1 = _angle_deg(u1, v1); a2 = _angle_deg(u2, v2); a3 = _angle_deg(u3, v3)

#         k1 = rPA(a1[band]); k2 = rPA(a2[band]); k3 = rPA(a3[band])
#         keys = np.vstack([k1, k2, k3]).T

#         # longest-edge PA and s for plotting
#         imax = np.argmax(np.vstack([L1[band], L2[band], L3[band]]), axis=0)
#         pa_long = np.choose(imax, [a1[band], a2[band], a3[band]])
#         s_long  = np.choose(imax, [S1[band], S2[band], S3[band]])

#         cp_deg = np.degrees(np.asarray(c[-2]).ravel()[band])
#         sig_deg = np.degrees(np.asarray(c[-1]).ravel()[band])

#         for k_trip, s_val, cpv, sev, paL in zip(keys, s_long, cp_deg, sig_deg, pa_long):
#             k = tuple(sorted(k_trip.tolist()))
#             groups[k]["s"].append(float(s_val))
#             groups[k]["cp_deg"].append(float(cpv))
#             groups[k]["sig_deg"].append(float(sev))
#             groups[k]["pa_long_deg"].append(float(paL))

#     for k, g in groups.items():
#         for nm in ("s","cp_deg","sig_deg","pa_long_deg"):
#             g[nm] = np.asarray(g[nm], dtype=float)
#     return groups



# def run_cp_chromatic_model_selection(o, out_dir, L0, frac_width=0.10,
#                                      tol_deg=2.0, min_channels=8,
#                                      make_plots=True, plot_top=12, tag="CP_chromatic"):
#     ensure_dir(out_dir)
#     groups = _group_cp_by_triangle_in_annulus(o, L0, frac_width=frac_width, tol_deg=tol_deg)
#     rows, figs_made = [], 0

#     for ktri, g in groups.items():
#         s = g["s"]; y_deg = g["cp_deg"]; sig_deg = g["sig_deg"]
#         m = np.isfinite(s) & np.isfinite(y_deg) & np.isfinite(sig_deg) & (sig_deg > 0)
#         if np.sum(m) < min_channels:
#             continue
#         s = s[m]; y_deg = y_deg[m]; sig_deg = sig_deg[m]

#         # unwrap CP in radians after sorting by s
#         s, y_rad = _unwrap_sorted(s, np.radians(y_deg))
#         sig_rad = np.radians(sig_deg)
#         w = 1.0/np.maximum(sig_rad, 1e-6)**2
#         N = s.size

#         # CONST
#         c0 = np.average(y_rad, weights=w)
#         chi2_const = np.sum(((y_rad - c0)**2) * w)
#         bic_const = _bic(chi2_const, k=1, n=N)

#         # LIN (weighted LS)
#         A = np.vstack([np.ones_like(s), s]).T
#         Aw = A*np.sqrt(w[:,None]); yw = y_rad*np.sqrt(w)
#         try:
#             beta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
#             y_lin = beta[0] + beta[1]*s
#             chi2_lin = np.sum(((y_rad - y_lin)**2) * w)
#             bic_lin = _bic(chi2_lin, k=2, n=N)
#         except Exception:
#             chi2_lin = np.inf; bic_lin = np.inf

#         # SIN
#         a0, A0, w0, phi0 = _initial_sine_guess(s, y_rad)
#         p0 = [a0, A0, w0, phi0]
#         bounds = ([-np.inf, -np.inf, 0.0, -2*np.pi], [np.inf, np.inf, np.inf, 2*np.pi])
#         try:
#             popt, pcov = curve_fit(_sin_model, s, y_rad, p0=p0, sigma=np.sqrt(1.0/w),
#                                    absolute_sigma=True, maxfev=20000, bounds=bounds)
#             y_sin = _sin_model(s, *popt)
#             chi2_sin = np.sum(((y_rad - y_sin)**2) * w)
#             bic_sin = _bic(chi2_sin, k=4, n=N)
#         except Exception:
#             popt = [np.nan]*4
#             chi2_sin = np.inf; bic_sin = np.inf

#         bics = {"CONST": bic_const, "LIN": bic_lin, "SIN": bic_sin}
#         best_m = min(bics, key=bics.get)
#         rows.append({
#             "tri_key": ktri, "Nchan": int(N),
#             "L0": float(L0), "frac_width": float(frac_width),
#             "pa_long_mean_deg": float(np.nanmedian(g["pa_long_deg"])),
#             "BIC_CONST": float(bic_const), "BIC_LIN": float(bic_lin), "BIC_SIN": float(bic_sin),
#             "best_model": best_m,
#             "ΔBIC(SIN−LIN)": float(bic_sin - bic_lin),
#             "ΔBIC(SIN−CONST)": float(bic_sin - bic_const)
#         })

#         # optional plot
#         if make_plots and figs_made < plot_top and np.isfinite(bic_sin):
#             plt.figure(figsize=(7.2, 3.6))
#             plt.errorbar(s, np.degrees(y_rad), yerr=np.degrees(sig_rad),
#                          fmt='o', ms=3.5, capsize=2, label='CP data')
#             xs = np.linspace(s.min(), s.max(), 400)
#             plt.plot(xs, np.degrees(c0 + 0*xs), lw=1.2, label=f'CONST BIC={bic_const:.1f}')
#             if np.isfinite(bic_lin):
#                 plt.plot(xs, np.degrees(beta[0] + beta[1]*xs), lw=1.2, label=f'LIN BIC={bic_lin:.1f}')
#             if np.isfinite(bic_sin):
#                 plt.plot(xs, np.degrees(_sin_model(xs, *popt)), lw=1.8, label=f'SIN BIC={bic_sin:.1f}')
#             plt.xlabel("s ≡ L$_{max}$ (∝ 1/λ)"); plt.ylabel("Closure phase (deg)")
#             ktxt = " / ".join([f"{a:.0f}°" for a in ktri])
#             plt.title(f"Triangle PAs≈{ktxt} | |u,v|≈{L0:.1f}±{100*frac_width:.0f}% | N={N}")
#             plt.legend(); plt.grid(alpha=0.25)
#             outp = os.path.join(out_dir, f"{tag}_tri_{figs_made+1:02d}_L0_{L0:.1f}.png")
#             plt.tight_layout(); plt.savefig(outp, dpi=180); plt.close(); print("[SAVE]", outp)
#             figs_made += 1

#     if rows:
#         df = pd.DataFrame(rows).sort_values(["BIC_SIN", "BIC_LIN"])
#         csvp = os.path.join(out_dir, f"{tag}_summary_L0_{L0:.1f}.csv")
#         df.to_csv(csvp, index=False); print("[SAVE]", csvp)
#         # quick console digest
#         best_counts = df["best_model"].value_counts().to_dict()
#         print(f"[{tag}] best-model counts:", best_counts)
#         print(f"[{tag}] median ΔBIC(SIN−LIN)={np.nanmedian(df['ΔBIC(SIN−LIN)']):.2f}")
#         return df
#     else:
#         print(f"[{tag}] No usable triangles (≥{min_channels} channels).")
#         return None

# --- STEP 3: fit three models vs spatial frequency s = B/λ and plot ---

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def _neg2loglike_gauss(y, yhat, sig):
    """-2 ln L for independent Gaussian errors (absolute_sigma)."""
    r = (y - yhat) / sig
    return np.sum(r*r + np.log(2*np.pi*sig*sig))

# def _bic(n, k, neg2loglike):
#     """Bayesian Information Criterion."""
#     return k*np.log(n) + neg2loglike

def _bic_bayes(n, k, neg2loglike):
    return k*np.log(n) + neg2loglike

def _fit_cp_models_on_group(group, min_channels=6):
    """
    Fit CONST, LIN(s), SIN(s) to closure phase vs s=B/λ for one triangle group.
    Returns dict {name: {popt, bic, yhat_fn}} or None if not enough data.
    """
    s   = np.asarray(group["s"], float)
    y   = np.asarray(group["cp_deg"], float)
    sig = np.asarray(group["sig_deg"], float)

    m = np.isfinite(s) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    s, y, sig = s[m], y[m], sig[m]
    n = y.size
    if n < max(4, int(min_channels)):
        return None

    def const_model(x, c): return np.full_like(x, c, dtype=float)
    def lin_model(x, a, b): return a + b*x
    def sin_model(x, c, A, f, phi): return c + A*np.sin(2*np.pi*f*x + phi)

    out = {}

    # CONST
    p0 = [np.nanmedian(y)]
    popt, _ = curve_fit(const_model, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
    yhat = const_model(s, *popt)
    out["CONST"] = {
        "popt": popt,
        "bic":  (1*np.log(n) + np.sum(((y - yhat)/sig)**2 + np.log(2*np.pi*sig*sig))),
        "yhat_fn": (lambda xs, _p=popt: const_model(xs, *_p))
    }

    # LIN
    p0 = [np.nanmedian(y), 0.0]
    popt, _ = curve_fit(lin_model, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
    yhat = lin_model(s, *popt)
    out["LIN"] = {
        "popt": popt,
        "bic":  (2*np.log(n) + np.sum(((y - yhat)/sig)**2 + np.log(2*np.pi*sig*sig))),
        "yhat_fn": (lambda xs, _p=popt: lin_model(xs, *_p))
    }

    # SIN
    try:
        s_rng = max(s.max() - s.min(), 1e-9)
        f0    = 0.5/s_rng
        A0    = 0.5*(np.nanpercentile(y,84) - np.nanpercentile(y,16))
        p0    = [np.nanmedian(y), max(A0, 0.5), f0, 0.0]
        bounds = ([-180.0, 0.0, 0.0, -2*np.pi],
                  [ +180.0, 180.0, np.inf, +2*np.pi])
        popt, _ = curve_fit(sin_model, s, y, p0=p0, bounds=bounds,
                            sigma=sig, absolute_sigma=True, maxfev=20000)
        yhat = sin_model(s, *popt)
        out["SIN"] = {
            "popt": popt,
            "bic":  (4*np.log(n) + np.sum(((y - yhat)/sig)**2 + np.log(2*np.pi*sig*sig))),
            "yhat_fn": (lambda xs, _p=popt: sin_model(xs, *_p))
        }
    except Exception:
        pass

    return out

# def _fit_cp_models_on_group(group, min_channels=6):
#     """
#     Fit CONST, LIN(s), SIN(s) to closure phase vs s=B/λ for one triangle group.
#     Returns dict {name: {popt, bic, yhat}} or None if not enough data.
#     """
#     s   = np.asarray(group["s"], float)
#     y   = np.asarray(group["cp_deg"], float)
#     sig = np.asarray(group["sig_deg"], float)

#     m = np.isfinite(s) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
#     s, y, sig = s[m], y[m], sig[m]
#     n = y.size
#     if n < max(4, int(min_channels)):
#         return None

#     out = {}

#     # CONST
#     def const_model(x, c): return np.full_like(x, c, dtype=float)
#     p0 = [np.nanmedian(y)]
#     popt, _ = curve_fit(const_model, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
#     yhat = const_model(s, *popt)
#     out["CONST"] = {
#         "popt": popt,
#         "bic": _bic_bayes(n, 1, _neg2loglike_gauss(y, yhat, sig)), # _bic(n, 1, _neg2loglike_gauss(y, yhat, sig)),
#         "yhat_fn": lambda xs: const_model(xs, *popt)
#     }

#     # LIN
#     def lin_model(x, a, b): return a + b*x
#     p0 = [np.nanmedian(y), 0.0]
#     popt, _ = curve_fit(lin_model, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
#     yhat = lin_model(s, *popt)
#     out["LIN"] = {
#         "popt": popt,
#         "bic": _bic_bayes(n, 2, _neg2loglike_gauss(y, yhat, sig)), #_bic(n, 2, _neg2loglike_gauss(y, yhat, sig)),
#         "yhat_fn": lambda xs: lin_model(xs, *popt)
#     }

#     # SIN (chromatic companion-like)
#     # y = c + A * sin(2π f s + φ)
#     def sin_model(x, c, A, f, phi): return c + A*np.sin(2*np.pi*f*x + phi)
#     s_rng = max(s.max() - s.min(), 1e-9)
#     f0    = 0.5/s_rng        # ~half a cycle across the covered s-range
#     A0    = 0.5*(np.nanpercentile(y, 84) - np.nanpercentile(y, 16))
#     p0    = [np.nanmedian(y), max(A0, 0.5), f0, 0.0]
#     bounds = ([-180.0, 0.0, 0.0, -2*np.pi],
#               [ +180.0, 180.0, np.inf, +2*np.pi])
#     try:
#         popt, _ = curve_fit(sin_model, s, y, p0=p0, bounds=bounds,
#                             sigma=sig, absolute_sigma=True, maxfev=20000)
#         yhat = sin_model(s, *popt)
#         out["SIN"] = {
#             "popt": popt,
#             "bic": _bic_bayes(n, 2, _neg2loglike_gauss(y, yhat, sig)), #_bic(n, 4, _neg2loglike_gauss(y, yhat, sig)),
#             "yhat_fn": lambda xs: sin_model(xs, *popt)
#         }
#     except Exception:
#         # If the sinusoid fails to converge, just omit it
#         pass

#     return out

def _plot_cp_group_models(group, fitres, out_png, title_prefix="", s0=None, frac_width=None):
    """
    Make a single panel showing CP vs s with model overlays and BICs.
    """
    s   = np.asarray(group["s"], float)
    y   = np.asarray(group["cp_deg"], float)
    sig = np.asarray(group["sig_deg"], float)

    m = np.isfinite(s) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    s, y, sig = s[m], y[m], sig[m]
    if s.size == 0:
        return

    xs = np.linspace(s.min(), s.max(), 600)

    # show spatial frequency in Mλ if large
    scale = 1e6 if np.nanpercentile(xs, 90) > 3e5 else 1.0
    xlabel = r"$s \equiv B/\lambda$ (M$\lambda$)" if scale == 1e6 else r"$s \equiv B/\lambda$"

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.errorbar(s/scale, y, yerr=sig, fmt='o', ms=4, alpha=0.8, label="CP data")

    for name in ("CONST", "LIN", "SIN"):
        if name in fitres:
            bic = fitres[name]["bic"]
            yhat_fn = fitres[name]["yhat_fn"]
            ax.plot(xs/scale, yhat_fn(xs), lw=2, label=f"{name} BIC={bic:.1f}")

    if s0 is not None and frac_width is not None:
        ax.axvspan((s0*(1-frac_width))/scale, (s0*(1+frac_width))/scale, color='k', alpha=0.05, lw=0)

    ax.set_xlabel(xlabel)

    # xs = np.linspace(s.min(), s.max(), 600)

    # fig, ax = plt.subplots(figsize=(9.0, 4.2))
    # ax.errorbar(s, y, yerr=sig, fmt='o', ms=4, alpha=0.8, label="CP data")

    # # Overlay models
    # order = ("CONST", "LIN", "SIN")
    # for name in order:
    #     if name in fitres:
    #         bic = fitres[name]["bic"]
    #         yhat_fn = fitres[name]["yhat_fn"]
    #         ax.plot(xs, yhat_fn(xs), lw=2, label=f"{name} BIC={bic:.1f}")

    # if s0 is not None and frac_width is not None:
    #     ax.axvspan(s0*(1-frac_width), s0*(1+frac_width), color='k', alpha=0.05, lw=0)

    # # Title helper: show the rounded triangle PAs if you have them
    # if "pa_long_deg" in group and group["pa_long_deg"].size:
    #     # Use the first sample’s 3 PAs if you stored them; otherwise skip.
    #     pass

    N = s.size
    ttl = f"{title_prefix} | N={N}"
    ax.set_title(ttl)
    ax.set_xlabel(r"$s \equiv B/\lambda$ (spatial frequency)")
    ax.set_ylabel("Closure phase (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_cp_chromatic_model_selection(o, L0, out_dir, frac_width=0.10,
                                     tol_deg=2.0, min_channels=8,
                                     plot_top=12, prefix="cp_chromatic",
                                     return_dataframe=True):
    """
    1) select triangles via annulus, then keep ALL channels
    2) fit CONST/LIN/SIN vs s=B/λ (or |u,v| / channel index fallback)
    3) write CSV; return DataFrame with best_model and dBIC columns
    """
    ensure_dir(out_dir)

    # s0 printed only for titles; selection already handled in the grouper
    lam_med = _global_median_lambda(o)
    s0 = (L0 / lam_med) if lam_med else L0

    print(f"[INFO] λ_median = {lam_med:.3e} m  →  s₀ = {s0/1e6:.2f} Mλ  (from L0={L0:.2f})")

    groups = _group_cp_by_triangle_in_annulus(o, L0, frac_width=frac_width, tol_deg=tol_deg)
    rows = []

    for tri_key, g in groups.items():
        fits = _fit_cp_models_on_group(g, min_channels=min_channels)
        if not fits:
            continue

        # make sure all BIC keys exist
        for nm in ("CONST","LIN","SIN"):
            if nm not in fits:
                fits[nm] = {"bic": np.inf, "popt": None, "yhat_fn": (lambda xs: np.nan*np.ones_like(xs))}

        rows.append({
            "triangle_key": str(tri_key),
            "N": int(np.isfinite(g["cp_deg"]).sum()),
            "bic_CONST": float(fits["CONST"]["bic"]),
            "bic_LIN":   float(fits["LIN"]["bic"]),
            "bic_SIN":   float(fits["SIN"]["bic"]),
            "dBIC_SIN_vs_CONST": float(fits["SIN"]["bic"] - fits["CONST"]["bic"]),
            "dBIC_SIN_vs_LIN":   float(fits["SIN"]["bic"] - fits["LIN"]["bic"]),
            "dBIC_LIN_vs_CONST": float(fits["LIN"]["bic"] - fits["CONST"]["bic"]),
            "s0_center": float(s0),
            "frac_width": float(frac_width),
        })

    if not rows:
        print("[CP chromatic] No usable triangle groups in this annulus.")
        return None if not return_dataframe else pd.DataFrame()

    df = pd.DataFrame(rows)
    # best model by min BIC
    cols = ["bic_CONST","bic_LIN","bic_SIN"]
    for c in cols:
        if c not in df: df[c] = np.inf
    df["best_model"] = df[cols].idxmin(axis=1).str.replace("bic_","").str.upper()

    # sort with strongest evidence for sinusoid first
    df = df.sort_values("dBIC_SIN_vs_CONST")

    # save CSV
    csvp = os.path.join(out_dir, f"{prefix}_summary_L0_{L0:.1f}.csv")
    df.to_csv(csvp, index=False); print("[SAVE]", csvp)

    # plots for the top K
    top = df.head(int(plot_top))
    for _, row in top.iterrows():
        key = row["triangle_key"]
        try:
            tri_key = eval(key)
        except Exception:
            tri_key = key
        g = groups[tri_key]
        fits = _fit_cp_models_on_group(g, min_channels=min_channels)
        out_png = os.path.join(out_dir, f"{prefix}_s_annulus_{s0:.2f}_key_{hash(key)%10**6}.png")
        title = f"Triangle PAs≈{key} | selected by |u,v|≈{L0:.1f}±{100*frac_width:.0f}%, fit uses ALL channels"
        _plot_cp_group_models(g, fits, out_png, title_prefix=title, s0=s0, frac_width=frac_width)

    return df if return_dataframe else None
# def run_cp_chromatic_model_selection(o, L0, out_dir, frac_width=0.10,
#                                      tol_deg=2.0, min_channels=8,
#                                      plot_top=12, prefix="cp_chromatic",
#                                      return_dataframe=True):
#     """
#     Wrapper:
#       1) groups CP by triangle (PAs), *selects* samples in an s-annulus
#       2) fits CONST/LIN/SIN vs s=B/λ
#       3) saves per-triangle plots with BICs
#       4) returns a table sorted by ΔBIC (best evidence for chromaticity)
#     """
#     import pandas as pd

#     # s0 to display (the grouper itself already selects by s)
#     lam_med = _global_median_lambda(o)
#     s0 = (L0 / lam_med) if lam_med else L0

#     groups = _group_cp_by_triangle_in_annulus(o, L0, frac_width=frac_width, tol_deg=tol_deg)
#     rows = []

#     # Score groups by how much SIN beats CONST (ΔBIC < 0 is "better than")
#     for tri_key, g in groups.items():
#         fits = _fit_cp_models_on_group(g, min_channels=min_channels)
#         if not fits:
#             continue

#         # ensure consistent keys
#         for nm in ("CONST", "LIN", "SIN"):
#             if nm not in fits:
#                 # sentinel large BIC so it sorts poorly
#                 fits[nm] = {"bic": np.inf, "popt": None, "yhat_fn": lambda x: np.nan*np.ones_like(x)}

#         drow = {
#             "triangle_key": str(tri_key),
#             "N": int(np.isfinite(g["cp_deg"]).sum()),
#             "bic_CONST": float(fits["CONST"]["bic"]),
#             "bic_LIN":   float(fits["LIN"]["bic"]),
#             "bic_SIN":   float(fits["SIN"]["bic"]),
#             "dBIC_SIN_vs_CONST": float(fits["SIN"]["bic"] - fits["CONST"]["bic"]),
#             "dBIC_LIN_vs_CONST": float(fits["LIN"]["bic"] - fits["CONST"]["bic"]),
#             "s0_center": float(s0),
#             "frac_width": float(frac_width),
#         }
#         rows.append(drow)

#     if not rows:
#         print("[CP chromatic] No usable triangle groups in this annulus.")
#         return None if not return_dataframe else pd.DataFrame()

#     # Build dataframe from rows
#     df = pd.DataFrame(rows)

#     # Add the winning model per triangle by minimum BIC
#     cols = ["bic_CONST", "bic_LIN", "bic_SIN"]
#     for c in cols:
#         if c not in df:
#             df[c] = np.inf
#     df["best_model"] = df[cols].idxmin(axis=1).str.replace("bic_", "").str.upper()

#     # Ensure ΔBIC column exists, then sort (most negative = strongest SIN preference)
#     if "dBIC_SIN_vs_CONST" not in df.columns:
#         df["dBIC_SIN_vs_CONST"] = df["bic_SIN"] - df["bic_CONST"]
#     df = df.sort_values("dBIC_SIN_vs_CONST")

#     # plots for the top K groups
#     top = df.head(int(plot_top))
#     for _, row in top.iterrows():
#         key = row["triangle_key"]
#         # convert string back to tuple for dict access if needed
#         try:
#             tri_key = eval(key)
#         except Exception:
#             tri_key = key
#         g = groups[tri_key]
#         fits = _fit_cp_models_on_group(g, min_channels=min_channels)
#         out_png = os.path.join(out_dir, f"{prefix}_s_annulus_{s0:.2f}_key_{hash(key) % 10**6}.png")
#         title = f"Triangle PAs≈{key} | |u,v|→s≈{s0:.1f}±{100*frac_width:.0f}%"
#         _plot_cp_group_models(g, fits, out_png, title_prefix=title, s0=s0, frac_width=frac_width)

#     return df if return_dataframe else None

# -------------------- (2) Circular-mean CP vs PA in annulus --------------------

def _extract_cp_pa_sig_annulus(o, L0, frac_width=0.10):
    pas, cps_deg, sig_deg = [], [], []
    for c in o._chi2Data:
        kind = c[0].split(';')[0].lower()
        if kind not in ('cp','t3','icp','scp','ccp'):
            continue
        u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
        u2 = np.asarray(c[3]).ravel(); v2 = np.asarray(c[4]).ravel()
        cp  = np.asarray(c[-2]).ravel()   # rad
        sig = np.asarray(c[-1]).ravel()   # rad
        u3, v3 = -(u1+u2), -(v1+v2)

        L1 = np.hypot(u1, v1); L2 = np.hypot(u2, v2); L3 = np.hypot(u3, v3)
        idx = np.argmax(np.vstack([L1, L2, L3]), axis=0)
        Lmax = np.maximum(L1, np.maximum(L2, L3))
        pa_long = np.choose(idx, [_angle_deg(u1, v1), _angle_deg(u2, v2), _angle_deg(u3, v3)])

        m = (Lmax >= L0*(1-frac_width)) & (Lmax <= L0*(1+frac_width)) & np.isfinite(cp) & np.isfinite(sig)
        if np.any(m):
            pas.append(pa_long[m]); cps_deg.append(np.degrees(cp[m])); sig_deg.append(np.degrees(sig[m]))
    if not pas: return np.array([]), np.array([]), np.array([])
    return np.concatenate(pas), np.concatenate(cps_deg), np.concatenate(sig_deg)

def _circ_mean_deg(alpha_deg, w=None):
    th = np.radians(alpha_deg)
    if w is None: w = np.ones_like(th)
    C = np.sum(w*np.cos(th)); S = np.sum(w*np.sin(th))
    mu = np.degrees(np.arctan2(S, C))
    R = np.hypot(C, S)/np.sum(w)
    return mu, R

def _rayleigh_p(Rbar, Neff):
    z = Neff*(Rbar**2)
    p = np.exp(-z)*(1 + (2*z - z*z)/(4*Neff))
    return max(min(p, 1.0), 0.0)

def cp_pa_circular_stats_annulus(o, out_dir, L0, frac_width=0.10, bins=12, Nboot=800, tag="CP_circmean"):
    ensure_dir(out_dir)
    pa, cp_deg, sig_deg = _extract_cp_pa_sig_annulus(o, L0, frac_width)
    if pa.size == 0:
        print(f"[{tag}] No CP points in annulus."); return None

    edges = np.linspace(0, 180, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    rows = []
    rng = np.random.default_rng(0)
    means = []; ses = []

    for a0, a1, cen in zip(edges[:-1], edges[1:], centers):
        m = (pa >= a0) & (pa < a1)
        if not np.any(m):
            rows.append({"PA_center_deg": cen, "N": 0, "mean_cp_deg": np.nan,
                         "se_deg": np.nan, "Rbar": np.nan, "Rayleigh_p": np.nan})
            means.append(np.nan); ses.append(np.nan)
            continue

        th = cp_deg[m]; w  = 1.0/np.maximum(sig_deg[m], 1e-3)**2
        mu, Rbar = _circ_mean_deg(th, w=w)
        Neff = (w.sum()**2) / (np.sum(w*w) + 1e-12)
        p = _rayleigh_p(Rbar, Neff)

        # bootstrap SE of circular mean
        idx_all = np.arange(th.size)
        mu_bs = []
        prob = w/w.sum()
        for _ in range(Nboot):
            idx = rng.choice(idx_all, size=idx_all.size, replace=True, p=prob)
            mu_b, _ = _circ_mean_deg(th[idx], w=w[idx])
            mu_bs.append(mu_b)
        mu_bs = np.unwrap(np.radians(mu_bs))
        se = np.degrees(np.nanstd(mu_bs, ddof=1))

        rows.append({"PA_center_deg": cen, "N": int(th.size), "mean_cp_deg": float(mu),
                     "se_deg": float(se), "Rbar": float(Rbar), "Rayleigh_p": float(p)})
        means.append(mu); ses.append(se)

    df = pd.DataFrame(rows)
    csvp = os.path.join(out_dir, f"{tag}_summary_L0_{L0:.1f}.csv")
    df.to_csv(csvp, index=False); print("[SAVE]", csvp)

    # plot
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    mvalid = np.isfinite(df["mean_cp_deg"])
    ax.errorbar(df["PA_center_deg"][mvalid], df["mean_cp_deg"][mvalid],
                yerr=df["se_deg"][mvalid], fmt='o', ms=4, capsize=2,
                label="circular mean ± boot.SE")
    ax.axhline(0, ls='--', lw=1, alpha=0.7)
    ax.set_xlim(0, 180); ax.set_xlabel("Baseline PA (deg, E of N)")
    ax.set_ylabel("Closure phase (deg)")
    ax.set_title(f"CP circular mean vs PA | |u,v|≈{L0:.1f}±{100*frac_width:.0f}%")
    ax.grid(alpha=0.25); ax.legend()
    outp = os.path.join(out_dir, f"{tag}_L0_{L0:.1f}.png")
    fig.tight_layout(); fig.savefig(outp, dpi=180); plt.close(fig); print("[SAVE]", outp)
    return df

# -------------------- I/O and orchestration --------------------

def resolve_files(args):
    if args.files:
        pats = []
        for pat in args.files:
            pats.extend(glob.glob(pat))
        return sorted(set(pats))

    # Optional: emulate your paths.json + instrument logic
    if not args.paths_json or not os.path.exists(args.paths_json):
        raise SystemExit("Provide --files GLOB(s) or a valid --paths_json.")

    path_dict = json.load(open(args.paths_json))
    data_root = path_dict[args.comp_loc]["data"]

    if args.ins == "pionier":
        return glob.glob(os.path.join(data_root, "pionier/data/*.fits"))

    if args.ins == "gravity" or fnmatch.fnmatch(args.ins, "gravity_line_*"):
        return glob.glob(os.path.join(data_root, "gravity/data/*.fits"))

    if args.ins in ("matisse_LM","matisse_L","matisse_M"):
        # adapt to your layout; default to *_L directory if present
        pths = []
        pths += glob.glob(os.path.join(data_root, "matisse_wvl_filtered_L/*.fits"))
        pths += glob.glob(os.path.join(data_root, "matisse_wvl_filtered_M/*.fits"))
        pths += glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_chopped_L/*.fits"))
        return pths

    if args.ins in ("matisse_N","matisse_N_short","matisse_N_mid","matisse_N_long") or fnmatch.fnmatch(args.ins, "matisse_N_*um"):
        return glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*.fits"))

    raise SystemExit("Could not infer file pattern; use --files.")

def choose_annuli(o, L0_list, annulus_quantiles, use_v2=True):
    if L0_list: return [float(x) for x in L0_list]
    arr = _gather_v2_baselines(o) if use_v2 else _gather_cp_longest_edge(o)
    if not arr.size:
        # fallback to the other
        arr = _gather_cp_longest_edge(o) if use_v2 else _gather_v2_baselines(o)
    if not arr.size:
        raise SystemExit("Could not determine baseline-length distribution from data.")
    return [float(np.nanpercentile(arr, q)) for q in annulus_quantiles]


def debug_uv_lambda(o, limit=3):
    """
    Print a few CP blocks showing whether λ was found and how s=B/λ looks.
    """
    seen = 0
    print("\n[DEBUG] Inspecting a few CP blocks for λ arrays...")
    for c in getattr(o, "_chi2Data", []):
        kind = c[0].split(';')[0].lower()
        if kind not in ('cp','t3','icp','scp','ccp'):
            continue

        u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
        L1 = np.hypot(u1, v1)
        lam = _guess_lambda_from_tuple(c)

        print(f"  · block {seen+1}: kind={c[0]}  N={L1.size}")
        print(f"    median |u1,v1| (meters?) = {np.nanmedian(L1):.3g}")

        if lam is None:
            print("    λ: NOT FOUND → using s≡|u,v| (baseline in meters)")
        else:
            print(f"    λ median = {np.nanmedian(lam):.6g} m "
                  f"(span {np.nanmin(lam):.6g}..{np.nanmax(lam):.6g})")
            S1 = L1/lam
            print(f"    s=B/λ (first edge) median = {np.nanmedian(S1):.3g} "
                  f"(span {np.nanmin(S1):.3g}..{np.nanmax(S1):.3g})")
        seen += 1
        if seen >= limit:
            break
    print("[DEBUG] Done.\n")

def main():
    ap = argparse.ArgumentParser(description="CP diagnostics: chromatic model selection + PA coherence in annuli")
    ap.add_argument("--files", nargs="+", help="OIFITS glob(s). If omitted, use --paths_json + --ins to resolve.")
    ap.add_argument("--paths_json", type=str, default=None)
    ap.add_argument("--comp_loc", type=str, default="ANU")
    ap.add_argument("--ins", type=str, default="pionier", help="pionier|gravity|matisse_*")
    ap.add_argument("--gravity_channel", type=str, default="SPECTRO_FT")

    ap.add_argument("--out_dir", type=str, default="./diag_out")
    ap.add_argument("--cp_floor_deg", type=float, default=None, help="σ floor (deg) added in quadrature to CP-like data")

    ap.add_argument("--L0_list", type=float, nargs="*", default=None, help="Explicit annulus centers (|u,v| units of data)")
    ap.add_argument("--annulus_quantiles", type=float, nargs="*", default=[65, 80, 90], help="Quantiles to pick L0 if not provided")
    ap.add_argument("--frac_width", type=float, default=0.10, help="±fractional width of annulus")

    ap.add_argument("--bins", type=int, default=12, help="PA bins (0..180)")
    ap.add_argument("--min_channels", type=int, default=8, help="Min spectral channels per triangle for (1)")
    ap.add_argument("--plot_top", type=int, default=12, help="Max triangles to plot in (1)")
    ap.add_argument("--nboot", type=int, default=800, help="Bootstrap reps for (2)")
    ap.add_argument("--tol_deg", type=float, default=2.0, help="Triangle-PA rounding tolerance for grouping")
    ap.add_argument(
        "--annuli_from", choices=["v2", "cp"], default="cp",
        help="Observable to derive L0 quantiles from (default: cp)."
    )
    args = ap.parse_args()

    out_root = ensure_dir(args.out_dir)
    out_dir = ensure_dir(os.path.join(out_root, args.ins))

    files = resolve_files(args)
    if not files: raise SystemExit("No OIFITS matched.")

    # CANDID setup
    candid.CONFIG['Ncores'] = None
    candid.CONFIG['long exec warning'] = None

    o = candid.Open(files)
    o.observables = ['v2','cp']  # force only these
    if "gravity" in args.ins.lower():
        o.instruments = [args.gravity_channel]

    debug_uv_lambda(o, limit=3)

    if args.cp_floor_deg is not None:
        apply_cp_sigma_floor(o, args.cp_floor_deg)
        print(f"[INFO] Applied CP σ-floor of {args.cp_floor_deg:.3f} deg (quadrature).")

    # Pick annuli
    # L0s = choose_annuli(o, args.L0_list, args.annulus_quantiles, use_v2=True)
    # print("[INFO] Using annuli (L0):", ", ".join(f"{x:.2f}" for x in L0s),
    #       f"with ±{100*args.frac_width:.0f}% width")

    L0s = choose_annuli(
        o,
        args.L0_list,
        args.annulus_quantiles,
        use_v2=(args.annuli_from == "v2")
    )
    # Run diagnostics per annulus
    summaries = []
    for L0 in L0s:
        print(f"\n=== Annulus |u,v|≈{L0:.2f} ± {100*args.frac_width:.0f}% ===")
        # df_bic = run_cp_chromatic_model_selection(
        #     o, out_dir, L0, frac_width=args.frac_width, tol_deg=args.tol_deg,
        #     min_channels=args.min_channels, make_plots=True,
        #     plot_top=args.plot_top, tag="CP_chromatic")

        
        df_bic = run_cp_chromatic_model_selection(
            o, L0, out_dir,
            frac_width=args.frac_width,
            tol_deg=args.tol_deg,
            min_channels=args.min_channels,
            plot_top=args.plot_top,
            prefix="CP_chromatic",
            return_dataframe=True
        )

        df_pa = cp_pa_circular_stats_annulus(
            o, out_dir, L0, frac_width=args.frac_width, bins=args.bins,
            Nboot=args.nboot, tag="CP_circmean")

        # quick combined digest line
        digest = {"L0": L0}
        if df_bic is not None and not df_bic.empty:
            vc = df_bic["best_model"].value_counts()
            digest.update({f"best_{k}": int(v) for k, v in vc.items()})
            digest["median_dBIC_SIN_vs_LIN"]   = float(np.nanmedian(df_bic["dBIC_SIN_vs_LIN"]))
            digest["median_dBIC_SIN_vs_CONST"] = float(np.nanmedian(df_bic["dBIC_SIN_vs_CONST"]))

        # digest = {"L0": L0}
        # if df_bic is not None:
        #     vc = df_bic["best_model"].value_counts()
        #     digest.update({f"best_{k}": int(v) for k, v in vc.items()})
        #     digest["median_dBIC_sin_lin"] = float(np.nanmedian(df_bic["ΔBIC(SIN−LIN)"]))
        if df_pa is not None:
            # share median |mean CP| and min Rayleigh p across bins
            digest["median_|meanCP|_deg"] = float(np.nanmedian(np.abs(df_pa["mean_cp_deg"])))
            digest["min_Rayleigh_p"] = float(np.nanmin(df_pa["Rayleigh_p"]))
        summaries.append(digest)

    if summaries:
        df_sum = pd.DataFrame(summaries)
        csvp = os.path.join(out_dir, "CP_diagnostics_overview.csv")
        df_sum.to_csv(csvp, index=False)
        print("[SAVE]", csvp)

    print("\nDone. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
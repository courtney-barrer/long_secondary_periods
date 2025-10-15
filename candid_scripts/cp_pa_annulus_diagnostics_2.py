#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CP diagnostics from OIFITS (no CANDID):
(1) CP circular-mean vs PA in |B| (or s) annuli
(2) Chromatic model selection CP(s) per triangle (CONST / LIN / SIN), with BIC.

- Directly parses OI_T3 (closure phase) + OI_WAVELENGTH
- Groups per triangle by rounded PAs of the three baselines
- Annulus selection by longest-edge spatial frequency s = B/λ (default)
  or by |B| if --annulus_in_s is False.
- Optionally add CP σ-floor (deg) in quadrature

Outputs:
  out_dir/
    CP_circmean_summary_L0_*.csv
    CP_circmean_L0_*.png
    CP_chromatic_s_annulus_*.png
    CP_chromatic_summary_L0_*.csv
    CP_diagnostics_overview.csv

Author: patched for direct OIFITS workflow
"""

from __future__ import annotations
import argparse, glob, os, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import re
from scipy.optimize import curve_fit
from scipy.special import j1

MAS2RAD = 4.848136811e-9  # mas → rad
# --------------------- small utils ---------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _pa_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Position angle East of North, folded to [0,180)."""
    return (np.degrees(np.arctan2(u, v)) % 180.0)

def _bic(n: int, k: int, neg2loglike: float) -> float:
    return k * np.log(max(n, 1)) + neg2loglike

def _neg2loglike_gauss(y, yhat, sig):
    r = (y - yhat) / sig
    return float(np.sum(r*r + np.log(2*np.pi*sig*sig)))

# --------------------- OIFITS reader -------------------

def _col(data, *names):
    """Return first matching column (case-insensitive)."""
    dn = {n.upper(): n for n in data.names}
    for nm in names:
        key = nm.upper()
        if key in dn:
            return data[dn[key]]
    raise KeyError(f"None of {names} found (have: {list(data.names)})")

@dataclass
class Samples:
    u1: np.ndarray
    v1: np.ndarray
    u2: np.ndarray
    v2: np.ndarray
    lam: np.ndarray      # meters
    cp_deg: np.ndarray   # degrees
    sig_deg: np.ndarray  # degrees

def read_cp_from_oifits(file_patterns) -> Samples:
    """Parse closure phases + geometry from OI_T3; wavelength from OI_WAVELENGTH."""
    paths = []
    for pat in (file_patterns if isinstance(file_patterns,(list,tuple)) else [file_patterns]):
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No FITS files matched --files pattern(s).")

    rec_u1, rec_v1, rec_u2, rec_v2, rec_lam, rec_cp, rec_sig = [], [], [], [], [], [], []
    n_files, n_rows_total = 0, 0

    for f in paths:
        try:
            hdul = fits.open(f, memmap=False)
        except Exception as e:
            print(f"[WARN] Could not open {os.path.basename(f)}: {e}")
            continue
        n_files += 1

        # map INSNAME -> EFF_WAVE (meters)
        waves_by_ins = {}
        for h in hdul:
            if isinstance(h, fits.BinTableHDU) and h.header.get("EXTNAME","").strip().upper() == "OI_WAVELENGTH":
                ins = h.header.get("INSNAME","")
                try:
                    eff = np.asarray(_col(h.data, "EFF_WAVE"), dtype=float)
                except Exception:
                    eff = None
                if eff is not None and eff.size:
                    waves_by_ins[ins] = eff

        # read all OI_T3 rows
        for h in hdul:
            if not isinstance(h, fits.BinTableHDU): continue
            if h.header.get("EXTNAME","").strip().upper() != "OI_T3": continue

            ins = h.header.get("INSNAME","")
            waves = waves_by_ins.get(ins)
            if waves is None and waves_by_ins:
                # fallback if INSNAME mismatch
                waves = next(iter(waves_by_ins.values()), None)

            dat = h.data
            if dat is None or len(dat) == 0:
                continue

            try:
                u1 = np.asarray(_col(dat, "U1COORD"), dtype=float)
                v1 = np.asarray(_col(dat, "V1COORD"), dtype=float)
                u2 = np.asarray(_col(dat, "U2COORD"), dtype=float)
                v2 = np.asarray(_col(dat, "V2COORD"), dtype=float)
                cp = _col(dat, "T3PHI")      # deg [NWAVE]
                se = _col(dat, "T3PHIERR")   # deg [NWAVE]
                fl = _col(dat, "FLAG")       # bool [NWAVE]
            except KeyError as e:
                print(f"[WARN] {os.path.basename(f)} missing columns {e}; skipping this OI_T3")
                continue

            n_rows_total += len(dat)

            for i in range(len(dat)):
                if waves is not None:
                    nw = len(waves)
                    lam_i = np.asarray(waves, float)[:nw]
                else:
                    # If no wavelength table: infer NWAVE from row; set λ=1 to allow PA-only stats
                    nw = len(cp[i])
                    lam_i = np.ones(nw, float)

                cp_i = np.asarray(cp[i], float)[:nw]
                se_i = np.asarray(se[i], float)[:nw]
                fl_i = np.asarray(fl[i], bool)[:nw]

                m = (~fl_i) & np.isfinite(cp_i) & np.isfinite(se_i) & (se_i > 0)
                if not np.any(m):
                    continue

                rec_u1.extend([float(u1[i])]*int(m.sum()))
                rec_v1.extend([float(v1[i])]*int(m.sum()))
                rec_u2.extend([float(u2[i])]*int(m.sum()))
                rec_v2.extend([float(v2[i])]*int(m.sum()))
                rec_lam.extend(lam_i[m].tolist())
                rec_cp.extend(cp_i[m].tolist())
                rec_sig.extend(se_i[m].tolist())

        hdul.close()

    arr = Samples(
        u1=np.asarray(rec_u1, float),
        v1=np.asarray(rec_v1, float),
        u2=np.asarray(rec_u2, float),
        v2=np.asarray(rec_v2, float),
        lam=np.asarray(rec_lam, float),
        cp_deg=np.asarray(rec_cp, float),
        sig_deg=np.asarray(rec_sig, float),
    )
    if arr.cp_deg.size == 0:
        print("[ERROR] No CP/T3 samples found after parsing OIFITS.")
    else:
        print(f"[INFO] Loaded {arr.cp_deg.size} CP channels from {n_rows_total} OI_T3 rows in {n_files} file(s).")
        if np.all(arr.lam == 1.0):
            print("[WARN] No OI_WAVELENGTH found; λ set to 1.0 → s=B/λ will be flat.")
    return arr

# --------------------- geometry from samples ---------------------

@dataclass
class Geo:
    L1: np.ndarray; L2: np.ndarray; L3: np.ndarray
    S1: np.ndarray; S2: np.ndarray; S3: np.ndarray
    Lmax: np.ndarray; Smax: np.ndarray
    PA1: np.ndarray; PA2: np.ndarray; PA3: np.ndarray
    PA_long: np.ndarray
    imax: np.ndarray

def build_geometry(samples: Samples) -> Geo:
    u1, v1 = samples.u1, samples.v1
    u2, v2 = samples.u2, samples.v2
    u3, v3 = -(u1+u2), -(v1+v2)

    L1 = np.hypot(u1, v1)
    L2 = np.hypot(u2, v2)
    L3 = np.hypot(u3, v3)
    Lmax = np.maximum(L1, np.maximum(L2, L3))

    lam = samples.lam
    S1, S2, S3 = L1/lam, L2/lam, L3/lam
    Smax = Lmax/lam

    PA1 = _pa_deg(u1, v1)
    PA2 = _pa_deg(u2, v2)
    PA3 = _pa_deg(u3, v3)

    imax = np.argmax(np.vstack([L1,L2,L3]), axis=0)
    PA_long = np.choose(imax, [PA1, PA2, PA3])

    return Geo(L1,L2,L3,S1,S2,S3,Lmax,Smax,PA1,PA2,PA3,PA_long,imax)

# --------------------- CP σ-floor ---------------------

def apply_cp_sigma_floor_inplace(samples: Samples, cp_floor_deg: float | None):
    if cp_floor_deg is None: return
    f = float(cp_floor_deg)
    samples.sig_deg = np.hypot(samples.sig_deg, f)

# --------------------- (1) CP vs PA circ-mean -------------------

def cp_pa_circular_stats_annulus(samples: Samples, geo: Geo, out_dir: str,
                                 L0: float, s0: float, frac_width: float,
                                 use_s_annulus: bool, bins: int, Nboot: int,
                                 tag="CP_circmean") -> pd.DataFrame | None:
    ensure_dir(out_dir)
    PA = geo.PA_long
    cp = samples.cp_deg
    sig = samples.sig_deg

    if use_s_annulus:
        band = (geo.Smax >= s0*(1-frac_width)) & (geo.Smax <= s0*(1+frac_width))
        ann_txt = f"s≈{s0/1e6:.1f} Mλ"
    else:
        band = (geo.Lmax >= L0*(1-frac_width)) & (geo.Lmax <= L0*(1+frac_width))
        ann_txt = f"|u,v|≈{L0:.1f}"

    if not np.any(band):
        print(f"[{tag}] No points in annulus.")
        return None

    pa = PA[band]; cp_deg = cp[band]; sig_deg = sig[band]

    edges = np.linspace(0, 180, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    rng = np.random.default_rng(0)

    rows = []
    for a0,a1,cen in zip(edges[:-1], edges[1:], centers):
        m = (pa >= a0) & (pa < a1)
        if not np.any(m):
            rows.append({"PA_center_deg":cen,"N":0,"mean_cp_deg":np.nan,"se_deg":np.nan,
                         "Rbar":np.nan,"Rayleigh_p":np.nan})
            continue

        th = np.radians(cp_deg[m])
        w  = 1.0/np.maximum(np.radians(sig_deg[m]), 1e-6)**2
        C = np.sum(w*np.cos(th)); S = np.sum(w*np.sin(th))
        mu = np.degrees(np.arctan2(S, C))
        Rbar = np.hypot(C,S)/np.sum(w)
        Neff = (w.sum()**2) / (np.sum(w*w) + 1e-16)
        z = Neff*(Rbar**2)
        p = float(np.exp(-z)*(1 + (2*z - z*z)/(4*Neff)))

        # bootstrap SE
        idx = np.arange(th.size); prob = w/w.sum()
        mu_bs = []
        for _ in range(Nboot):
            jj = rng.choice(idx, size=idx.size, replace=True, p=prob)
            Cb = np.sum(w[jj]*np.cos(th[jj])); Sb = np.sum(w[jj]*np.sin(th[jj]))
            mu_bs.append(np.degrees(np.arctan2(Sb, Cb)))
        se = np.nanstd(mu_bs, ddof=1)

        rows.append({"PA_center_deg":cen,"N":int(th.size),
                     "mean_cp_deg":float(mu),"se_deg":float(se),
                     "Rbar":float(Rbar),"Rayleigh_p":float(max(min(p,1.0),0.0))})

    df = pd.DataFrame(rows)
    csvp = os.path.join(out_dir, f"{tag}_summary_L0_{L0:.1f}.csv")
    df.to_csv(csvp, index=False); print("[SAVE]", csvp)

    # plot
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    mvalid = np.isfinite(df["mean_cp_deg"])
    ax.errorbar(df["PA_center_deg"][mvalid], df["mean_cp_deg"][mvalid],
                yerr=df["se_deg"][mvalid], fmt='o', ms=4, capsize=2,
                label="circular mean ± boot.SE")
    ax.axhline(0, ls='--', lw=1, alpha=0.7)
    ax.set_xlim(0,180)
    ax.set_xlabel("Baseline PA (deg, E of N)")
    ax.set_ylabel("Closure phase (deg)")
    ax.set_title(f"CP circular mean vs PA | {ann_txt} ±{100*frac_width:.0f}%")
    ax.grid(alpha=0.25); ax.legend()
    outp = os.path.join(out_dir, f"{tag}_L0_{L0:.1f}.png")
    fig.tight_layout(); fig.savefig(outp, dpi=180); plt.close(fig); print("[SAVE]", outp)
    return df

# --------------------- (2) Chromatic model per triangle --------
def _triangle_keys(geo, tol_deg: float):
    """
    Return a 1-D object array of length N where each element is a tuple
    of the 3 rounded edge PAs (deg) for that triangle sample.
    """
    # Use already-computed edge PAs if present
    if all(hasattr(geo, a) for a in ("PA1", "PA2", "PA3")):
        pa1, pa2, pa3 = geo.PA1, geo.PA2, geo.PA3
    else:
        # Fall back to computing from u,v if needed
        required = ("u1","v1","u2","v2")
        if not all(hasattr(geo, a) for a in required):
            present = [k for k in dir(geo) if k.isupper() or k.startswith(("u","v","PA"))]
            raise AttributeError(
                "Geo is missing fields to compute triangle PAs. "
                f"Need PA1/PA2/PA3 or {required}. Present: {present}"
            )
        def _pa(u, v): return (np.degrees(np.arctan2(u, v)) % 180.0)
        pa1 = _pa(geo.u1, geo.v1)
        pa2 = _pa(geo.u2, geo.v2)
        pa3 = _pa(-(geo.u1+geo.u2), -(geo.v1+geo.v2))

    # round to tolerance
    r = lambda a: (np.round(a / tol_deg) * tol_deg) % 180.0
    pa1r, pa2r, pa3r = r(pa1), r(pa2), r(pa3)

    # build 1D object array of tuples
    tuples = [tuple(sorted([float(a), float(b), float(c)]))
              for a, b, c in zip(pa1r, pa2r, pa3r)]
    return np.array(tuples, dtype=object)
# def _triangle_keys(geo, tol_deg):
#     """
#     Return a 1-D object array of (pa1, pa2, pa3) tuples (deg), each PA rounded
#     to tol_deg and folded to [0,180). Prefers existing PA arrays (PA1/PA2/PA3),
#     else computes them from u/v if available.
#     """
#     import numpy as np

#     def _ang(u, v):
#         return (np.degrees(np.arctan2(u, v)) % 180.0)

#     gdict = {k.lower(): v for k, v in vars(geo).items()}

#     def _pick(d, *names):
#         for name in names:
#             if name in d and d[name] is not None:
#                 return d[name]
#         return None

#     # try precomputed PAs (case-insensitive)
#     pa1 = _pick(gdict, "pa1", "pa12")
#     pa2 = _pick(gdict, "pa2", "pa23")
#     pa3 = _pick(gdict, "pa3", "pa31")

#     if pa1 is None or pa2 is None or pa3 is None:
#         # compute from u/v
#         u1 = _pick(gdict, "u1", "u12"); v1 = _pick(gdict, "v1", "v12")
#         u2 = _pick(gdict, "u2", "u23"); v2 = _pick(gdict, "v2", "v23")
#         if u1 is None or v1 is None or u2 is None or v2 is None:
#             have = ", ".join(sorted(vars(geo).keys()))
#             raise AttributeError(
#                 "Geo is missing fields to compute triangle PAs "
#                 "(need PA1/PA2/PA3 or u1/v1/u2/v2). "
#                 f"Present: {have}"
#             )
#         u3, v3 = -(u1 + u2), -(v1 + v2)
#         pa1, pa2, pa3 = _ang(u1, v1), _ang(u2, v2), _ang(u3, v3)

#     # round & fold to [0,180)
#     pa1 = (np.round(np.asarray(pa1, float)/tol_deg)*tol_deg) % 180.0
#     pa2 = (np.round(np.asarray(pa2, float)/tol_deg)*tol_deg) % 180.0
#     pa3 = (np.round(np.asarray(pa3, float)/tol_deg)*tol_deg) % 180.0

#     tri = np.sort(np.vstack([pa1, pa2, pa3]).T, axis=1)  # (N,3) numeric
#     keys = np.array([tuple(row.tolist()) for row in tri], dtype=object)  # (N,) object

#     if hasattr(geo, "Smax"):
#         assert keys.shape[0] == np.asarray(geo.Smax).shape[0], \
#             f"_triangle_keys produced {keys.shape[0]} keys, but Smax has {np.asarray(geo.Smax).shape[0]}"

#     return keys

# def _triangle_keys(geo: Geo, tol_deg: float) -> np.ndarray:
#     """
#     Return a 1-D object array of per-sample triangle keys as tuples of rounded PAs.
#     Each key looks like (pa1, pa2, pa3) sorted ascending, values in [0,180).
#     """
#     r = lambda a: np.round((a % 180.0)/tol_deg)*tol_deg
#     k1, k2, k3 = r(geo.PA1), r(geo.PA2), r(geo.PA3)
#     keys_2d = np.sort(np.vstack([k1, k2, k3]).T, axis=1)  # shape (N,3) float
#     # convert to 1-D object array of tuples
#     keys_obj = np.array([tuple(row.tolist()) for row in keys_2d], dtype=object)
#     return keys_obj

# def _triangle_keys(geo: Geo, tol_deg: float) -> np.ndarray:
#     r = lambda a: np.round((a % 180.0)/tol_deg)*tol_deg
#     k1, k2, k3 = r(geo.PA1), r(geo.PA2), r(geo.PA3)
#     keys = np.sort(np.vstack([k1,k2,k3]).T, axis=1)
#     # convert to strings for hashing (easier as dict keys)
#     return np.array([tuple(x.tolist()) for x in keys], dtype=object)

def _fit_models_s(s, y, sig_deg):
    """Fit CONST / LIN / SIN to (s, CP_deg) with Gaussian errors; return dict."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    sig = np.asarray(sig_deg, float)
    m = np.isfinite(s) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    s, y, sig = s[m], y[m], sig[m]
    n = y.size
    if n < 3:
        return {}

    out = {}

    # CONST
    def const_m(x, c): return np.full_like(x, c, dtype=float)
    p0 = [np.nanmedian(y)]
    popt, _ = curve_fit(const_m, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
    yhat = const_m(s, *popt)
    out["CONST"] = {"popt":popt,
                    "bic":_bic(n,1,_neg2loglike_gauss(y,yhat,sig)),
                    "yhat_fn":lambda xs,_p=popt: const_m(xs,*_p)}

    # LIN
    def lin_m(x,a,b): return a + b*x
    p0 = [np.nanmedian(y), 0.0]
    popt, _ = curve_fit(lin_m, s, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=10000)
    yhat = lin_m(s, *popt)
    out["LIN"] = {"popt":popt,
                  "bic":_bic(n,2,_neg2loglike_gauss(y,yhat,sig)),
                  "yhat_fn":lambda xs,_p=popt: lin_m(xs,*_p)}

    # SIN
    def sin_m(x,c,A,f,phi): return c + A*np.sin(2*np.pi*f*x + phi)
    s_rng = max(s.max()-s.min(), 1e-9)
    f0 = 0.5/s_rng
    A0 = 0.5*(np.nanpercentile(y,84) - np.nanpercentile(y,16))
    p0 = [np.nanmedian(y), max(A0, 0.5), f0, 0.0]
    bounds = ([-180.0, 0.0, 0.0, -2*np.pi], [180.0, 180.0, np.inf, 2*np.pi])
    try:
        popt, _ = curve_fit(sin_m, s, y, p0=p0, bounds=bounds,
                            sigma=sig, absolute_sigma=True, maxfev=20000)
        yhat = sin_m(s, *popt)
        out["SIN"] = {"popt":popt,
                      "bic":_bic(n,4,_neg2loglike_gauss(y,yhat,sig)),
                      "yhat_fn":lambda xs,_p=popt: sin_m(xs,*_p)}
    except Exception:
        pass

    return out

def run_chromatic_selection(samples: Samples, geo: Geo, out_dir: str,
                            L0: float, s0: float, frac_width: float,
                            tol_deg: float, min_channels: int,
                            plot_top: int, fit_uses_all_channels: bool=True,
                            prefix="CP_chromatic", return_dataframe=True):

    ensure_dir(out_dir)

    # --- Coerce triangle keys into 1-D object array of 3-tuples ---
    keys_raw = _triangle_keys(geo, tol_deg)
    Smax = geo.Smax
    y    = samples.cp_deg
    sig  = samples.sig_deg

    kr = np.asarray(keys_raw)
    if kr.ndim == 2 and kr.shape[1] == 3:
        keys = np.empty(kr.shape[0], dtype=object)
        for i in range(kr.shape[0]):
            row = np.asarray(list(kr[i]), dtype=float).ravel()
            keys[i] = tuple(sorted(map(float, row)))
    elif kr.ndim == 1 and kr.dtype == object and kr.size == Smax.size:
        keys = kr
    elif kr.ndim == 1 and kr.size == Smax.size:
        keys = np.array([(float(v), float(v), float(v)) for v in kr], dtype=object)
    else:
        raise AssertionError(f"Unexpected keys shape from _triangle_keys: shape={kr.shape}, dtype={kr.dtype}, Smax={Smax.shape}")

    assert keys.ndim == 1 and keys.shape[0] == Smax.shape[0], \
        f"keys shape {keys.shape} incompatible with Smax {Smax.shape}"

    # Annulus mask (in s by default)
    band = (Smax >= s0*(1-frac_width)) & (Smax <= s0*(1+frac_width))
    assert band.shape == Smax.shape == keys.shape == y.shape == sig.shape

    # --- optional debug: show top triangles in the annulus ---
    uniq, counts = np.unique(keys[band], return_counts=True)
    print("[DEBUG] Top triangles in annulus (by #channels):")
    if uniq.size:
        for k, c in sorted(zip(uniq, counts), key=lambda t: -t[1])[:10]:
            print(f"  · {k}: {c} ch")
    else:
        print("  (none)")

    # Choose triangles by #channels inside annulus (in s by default)
    df_rows = []
    unique_keys = np.unique(keys)
    for k in unique_keys:
        # --- normalize k to a 3-tuple of floats ---
        if isinstance(k, np.ndarray):
            k_tuple = tuple(map(float, np.ravel(k).tolist()))
        elif isinstance(k, (list, tuple)):
            k_tuple = tuple(map(float, k))
        else:
            k_tuple = (float(k), float(k), float(k))

        # --- elementwise equality without NumPy broadcasting ---
        kk = np.array([ki == k_tuple for ki in keys], dtype=bool)

        # channels "selected" (inside annulus) for this key
        sel = kk & band
        if sel.sum() < max(2, int(min_channels)):
            continue

        if fit_uses_all_channels:
            fit_idx = kk   # all channels for this triangle across s
        else:
            fit_idx = sel  # only annulus channels

        fits = _fit_models_s(Smax[fit_idx], y[fit_idx], sig[fit_idx])
        if not fits:
            continue

        # fill missing with +inf to allow argmin
        bic_const = fits.get("CONST", {"bic": np.inf})["bic"]
        bic_lin   = fits.get("LIN",   {"bic": np.inf})["bic"]
        bic_sin   = fits.get("SIN",   {"bic": np.inf})["bic"]

        df_rows.append({
            "triangle_key": str(k_tuple),
            "N_total": int(kk.sum()),
            "N_in_annulus": int(sel.sum()),
            "bic_CONST": float(bic_const),
            "bic_LIN": float(bic_lin),
            "bic_SIN": float(bic_sin),
            "dBIC_SIN_vs_CONST": float(bic_sin - bic_const),
            "dBIC_LIN_vs_CONST": float(bic_lin - bic_const),
        })

    if not df_rows:
        print("[CP chromatic] No usable triangle groups in this annulus.")
        return None if return_dataframe else None

    df = pd.DataFrame(df_rows)
    cols = ["bic_CONST","bic_LIN","bic_SIN"]
    for c in cols:
        if c not in df: df[c] = np.inf
    df["best_model"] = df[cols].idxmin(axis=1).str.replace("bic_","").str.upper()
    df = df.sort_values("dBIC_SIN_vs_CONST")

    # plots for top K
    top = df.head(int(plot_top))
    for _, row in top.iterrows():
        key = row["triangle_key"]
        try:
            key_tuple = tuple(eval(key))  # string -> tuple
        except Exception:
            # fall back defensively
            if isinstance(key, (list, tuple)): key_tuple = tuple(key)
            elif isinstance(key, np.ndarray):  key_tuple = tuple(np.ravel(key).tolist())
            else:                               key_tuple = (float(key),)*3

        kk = np.array([ki == key_tuple for ki in keys], dtype=bool)
        if fit_uses_all_channels:
            fit_idx = kk
            fit_note = "fit uses ALL channels"
        else:
            fit_idx = kk & band
            fit_note = "fit uses ANNULUS channels"

        fits = _fit_models_s(Smax[fit_idx], y[fit_idx], sig[fit_idx])
        if not fits: 
            continue

        xs = np.linspace(Smax[fit_idx].min(), Smax[fit_idx].max(), 600)
        fig, ax = plt.subplots(figsize=(9.2, 4.3))
        ax.errorbar(Smax[fit_idx]/1e6, y[fit_idx], yerr=sig[fit_idx],
                    fmt='o', ms=3.5, alpha=0.8, label="CP data")

        for name in ("CONST","LIN","SIN"):
            if name in fits:
                bic = fits[name]["bic"]
                yhat = fits[name]["yhat_fn"](xs)
                ax.plot(xs/1e6, yhat, lw=2, label=f"{name} BIC={bic:.1f}")

        ax.axvspan(s0*(1-frac_width)/1e6, s0*(1+frac_width)/1e6, color='k', alpha=0.06, lw=0)
        ax.set_xlabel(r"$s \equiv B/\lambda$  (M$\lambda$)")
        ax.set_ylabel("Closure phase (deg)")
        ax.grid(alpha=0.25)
        ax.legend()
        ttl = f"Triangle PAs≈{key_tuple} | selected by |u,v|≈{L0:.1f}±{100*frac_width:.0f}% | {fit_note} | N={fit_idx.sum()}"
        ax.set_title(ttl)
        outp = os.path.join(out_dir, f"{prefix}_s_annulus_{s0/1e6:.2f}_key_{hash(key)%10**6}.png")
        fig.tight_layout(); fig.savefig(outp, dpi=180); plt.close(fig); print("[SAVE]", outp)

    csvp = os.path.join(out_dir, f"{prefix}_summary_L0_{L0:.1f}.csv")
    df.to_csv(csvp, index=False); print("[SAVE]", csvp)
    return df if return_dataframe else None

# def run_chromatic_selection(samples: Samples, geo: Geo, out_dir: str,
#                             L0: float, s0: float, frac_width: float,
#                             tol_deg: float, min_channels: int,
#                             plot_top: int, fit_uses_all_channels: bool=True,
#                             prefix="CP_chromatic", return_dataframe=True):

#     ensure_dir(out_dir)

#     # --- Coerce triangle keys into 1-D object array of 3-tuples ---
#     keys_raw = _triangle_keys(geo, tol_deg)
#     Smax = geo.Smax
#     y    = samples.cp_deg
#     sig  = samples.sig_deg

#     kr = np.asarray(keys_raw)

#     if kr.ndim == 2 and kr.shape[1] == 3:
#         # (N,3) of numbers or object → make sorted 3-tuples
#         keys = np.empty(kr.shape[0], dtype=object)
#         for i in range(kr.shape[0]):
#             # handle lists/tuples/objects; force to float array
#             row = np.asarray(list(kr[i]), dtype=float).ravel()
#             keys[i] = tuple(sorted(map(float, row)))
#     elif kr.ndim == 1 and kr.dtype == object and kr.size == Smax.size:
#         # already 1-D object array (likely tuples)
#         keys = kr
#     elif kr.ndim == 1 and kr.size == Smax.size:
#         # 1-D numeric (rare): promote to 3-tuples with identical PA
#         keys = np.array([(float(v), float(v), float(v)) for v in kr], dtype=object)
#     else:
#         raise AssertionError(f"Unexpected keys shape from _triangle_keys: shape={kr.shape}, dtype={kr.dtype}, Smax={Smax.shape}")

#     assert keys.ndim == 1 and keys.shape[0] == Smax.shape[0], \
#         f"keys shape {keys.shape} incompatible with Smax {Smax.shape}"

#     # Annulus mask (in s by default)
#     band = (Smax >= s0*(1-frac_width)) & (Smax <= s0*(1+frac_width))
#     assert band.shape == Smax.shape == keys.shape == y.shape == sig.shape

#     # --- optional debug: show top triangles in the annulus ---
#     uniq, counts = np.unique(keys[band], return_counts=True)
#     print("[DEBUG] Top triangles in annulus (by #channels):")
#     if uniq.size:
#         for k, c in sorted(zip(uniq, counts), key=lambda t: -t[1])[:10]:
#             print(f"  · {k}: {c} ch")
#     else:
#         print("  (none)")

#     # build per-key indices
#     df_rows = []
#     unique_keys = sorted(set(keys))   # robust for object arrays
#     for k in unique_keys:
#         kk = (keys == k)              # 1-D boolean mask

#         # channels "selected" (inside annulus) for this key
#         sel = kk & band               # no more broadcasting errors
#         if sel.sum() < max(2, int(min_channels)):
#             continue

#         if fit_uses_all_channels:
#             fit_idx = kk   # all channels for this triangle across s
#         else:
#             fit_idx = sel  # only annulus channels

#         fits = _fit_models_s(Smax[fit_idx], y[fit_idx], sig[fit_idx])
#         if not fits:
#             continue

#         # fill missing with +inf to allow argmin
#         bic_const = fits.get("CONST", {"bic": np.inf})["bic"]
#         bic_lin   = fits.get("LIN",   {"bic": np.inf})["bic"]
#         bic_sin   = fits.get("SIN",   {"bic": np.inf})["bic"]

#         df_rows.append({
#             "triangle_key": str(k),
#             "N_total": int(kk.sum()),
#             "N_in_annulus": int(sel.sum()),
#             "bic_CONST": float(bic_const),
#             "bic_LIN": float(bic_lin),
#             "bic_SIN": float(bic_sin),
#             "dBIC_SIN_vs_CONST": float(bic_sin - bic_const),
#             "dBIC_LIN_vs_CONST": float(bic_lin - bic_const),
#         })

#     if not df_rows:
#         print("[CP chromatic] No usable triangle groups in this annulus.")
#         return None if return_dataframe else None

#     df = pd.DataFrame(df_rows)
#     # Best model name
#     cols = ["bic_CONST","bic_LIN","bic_SIN"]
#     for c in cols:
#         if c not in df: df[c] = np.inf
#     df["best_model"] = df[cols].idxmin(axis=1).str.replace("bic_","").str.upper()
#     df = df.sort_values("dBIC_SIN_vs_CONST")

#     # plots for top K
#     top = df.head(int(plot_top))
#     for _, row in top.iterrows():
#         key = row["triangle_key"]
#         try:
#             key_tuple = eval(key)
#         except Exception:
#             key_tuple = key

#         kk = (keys == key_tuple)
#         if fit_uses_all_channels:
#             fit_idx = kk
#             fit_note = "fit uses ALL channels"
#         else:
#             fit_idx = kk & band
#             fit_note = "fit uses ANNULUS channels"

#         fits = _fit_models_s(Smax[fit_idx], y[fit_idx], sig[fit_idx])
#         if not fits: 
#             continue

#         xs = np.linspace(Smax[fit_idx].min(), Smax[fit_idx].max(), 600)
#         fig, ax = plt.subplots(figsize=(9.2, 4.3))
#         ax.errorbar(Smax[fit_idx]/1e6, y[fit_idx], yerr=sig[fit_idx],
#                     fmt='o', ms=3.5, alpha=0.8, label="CP data")

#         for name in ("CONST","LIN","SIN"):
#             if name in fits:
#                 bic = fits[name]["bic"]
#                 yhat = fits[name]["yhat_fn"](xs)
#                 ax.plot(xs/1e6, yhat, lw=2, label=f"{name} BIC={bic:.1f}")

#         # shade selection annulus
#         ax.axvspan(s0*(1-frac_width)/1e6, s0*(1+frac_width)/1e6, color='k', alpha=0.06, lw=0)

#         ax.set_xlabel(r"$s \equiv B/\lambda$  (M$\lambda$)")
#         ax.set_ylabel("Closure phase (deg)")
#         ax.grid(alpha=0.25)
#         ax.legend()
#         ttl = f"Triangle PAs≈{key} | selected by |u,v|≈{L0:.1f}±{100*frac_width:.0f}% | {fit_note} | N={fit_idx.sum()}"
#         ax.set_title(ttl)
#         outp = os.path.join(out_dir, f"{prefix}_s_annulus_{s0/1e6:.2f}_key_{hash(key)%10**6}.png")
#         fig.tight_layout(); fig.savefig(outp, dpi=180); plt.close(fig); print("[SAVE]", outp)

#     # Save summary CSV
#     csvp = os.path.join(out_dir, f"{prefix}_summary_L0_{L0:.1f}.csv")
#     df.to_csv(csvp, index=False); print("[SAVE]", csvp)
#     return df if return_dataframe else None

# --------------------- annuli helpers -------------------------

def choose_annuli_from_quantiles(geo: Geo, quantiles, use_s_annulus: bool):
    arr = geo.Smax if use_s_annulus else geo.Lmax
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise SystemExit("No valid samples to compute quantiles.")
    vals = [float(np.nanpercentile(arr, q)) for q in quantiles]
    return vals



# --- UD fit from OI_VIS2, then star-normalized spatial frequency x = pi B theta / lambda
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import j1

MAS2RAD = 4.848136811e-9
def _read_vis2_from_oifits(file_globs):
    """
    Read |B|, λ, V², σ(V²) for all unflagged OI_VIS2 rows, matching INSNAME to
    the correct OI_WAVELENGTH table. Returns flattened 1-D arrays (same length).
    """
    files = []
    for pat in file_globs:
        files += glob.glob(pat)

    Ls, lams, V2s, E2s = [], [], [], []

    for fn in files:
        with fits.open(fn) as hdul:
            # Build INSNAME->λ map (robust to column naming)
            wl_map = _collect_wavelength_sets(hdul)
            if not wl_map:
                continue  # no wavelength table in this file

            # Iterate all OI_VIS2 HDUs
            for vis2_hdu in (h for h in hdul if getattr(h, "name", "").upper() == "OI_VIS2"):
                ins = vis2_hdu.header.get("INSNAME", None)

                # Choose wavelength set: prefer matching INSNAME; else any one set
                if ins in wl_map:
                    wl = wl_map[ins]
                elif len(wl_map) == 1:
                    wl = next(iter(wl_map.values()))
                else:
                    # ambiguous; skip this HDU to avoid mismatched λ
                    continue

                t = vis2_hdu.data
                if t is None or len(t) == 0:
                    continue

                # Tolerant column access
                def col_any(cands, idx=0):
                    return _get_col_any(vis2_hdu, cands, fallback_index=idx)

                u   = np.asarray(t["UCOORD"], float)         # (nrow,)
                v   = np.asarray(t["VCOORD"], float)         # (nrow,)
                V2  = np.asarray(list(col_any(["VIS2DATA", "V2DATA"])), float)  # (nrow, nchan)
                E2  = np.asarray(list(col_any(["VIS2ERR",  "V2ERR"])),  float)  # (nrow, nchan)

                # FLAG may be absent; if so, treat as all False
                try:
                    FL = np.asarray(list(col_any(["FLAG"])), bool)              # (nrow, nchan)
                except Exception:
                    FL = np.zeros_like(V2, dtype=bool)

                # Basic shape checks
                nrow, nchan = V2.shape
                if wl.size != nchan:
                    # Per OIFITS, wl length must match channel count; if not, skip safely
                    continue

                # Broadcast geometry to channels: |B| per row per channel
                L = np.hypot(u[:, None], v[:, None])         # (nrow, nchan)
                lam = np.broadcast_to(wl, (nrow, nchan))     # (nrow, nchan)

                m = np.isfinite(V2) & np.isfinite(E2) & (~FL)
                if m.any():
                    Ls.append(L[m])
                    lams.append(lam[m])
                    V2s.append(V2[m])
                    E2s.append(E2[m])

    if not Ls:
        raise SystemExit("No OI_VIS2 usable samples found to fit θ⋆ (check flags/columns).")

    L   = np.concatenate(Ls)
    lam = np.concatenate(lams)
    V2  = np.concatenate(V2s)
    E2  = np.concatenate(E2s)
    return L, lam, V2, E2




###################
def _fit_pa_dipole(samples, geo, mask, use_Mlambda=True):
    """
    Fit CP(phi, s) = c + K * s * cos(phi - phi0) on the selected channels (mask).
    Returns dict with (c, K, phi0_deg, bic, n).
    K units: deg per Mλ if use_Mlambda else deg per (lambda^-1).
    """
    import numpy as np
    y   = samples.cp_deg[mask].astype(float)
    sig = samples.sig_deg[mask].astype(float)
    s   = geo.Smax[mask].astype(float)
    phi = np.radians(geo.PA_long[mask].astype(float))
    if use_Mlambda:
        s = s/1e6  # use Mλ so K is "deg per Mλ"

    m = np.isfinite(y) & np.isfinite(sig) & (sig > 0) & np.isfinite(s) & np.isfinite(phi)
    if m.sum() < 4:
        return None
    y, sig, s, phi = y[m], sig[m], s[m], phi[m]
    w = 1.0/np.maximum(sig, 1e-3)**2
    # Linear in [1, s*cos(phi), s*sin(phi)]
    X = np.vstack([np.ones_like(s), s*np.cos(phi), s*np.sin(phi)]).T
    Xw = X*np.sqrt(w[:,None]); yw = y*np.sqrt(w)
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    c, A, B = beta
    K = float(np.hypot(A, B))
    phi0 = float((np.degrees(np.arctan2(B, A)) % 180.0))
    # BIC vs this 3-parameter model
    yhat = X @ beta
    neg2loglike = np.sum(((y - yhat)/sig)**2 + np.log(2*np.pi*sig*sig))
    bic = 3*np.log(y.size) + neg2loglike
    return {"c": float(c), "K": K, "phi0_deg": phi0, "bic": float(bic), "n": int(y.size)}


# --------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="CP diagnostics from OIFITS (no CANDID)")
    ap.add_argument("--files", nargs="+", required=True, help="OIFITS glob(s)")
    ap.add_argument("--out_dir", type=str, default="./diag_out_oifits")
    ap.add_argument("--cp_floor_deg", type=float, default=None, help="σ floor (deg) added in quadrature")

    ap.add_argument("--annulus_quantiles", type=float, nargs="*", default=[65,80,90],
                    help="Quantiles to pick annulus centers (in s if --annulus_in_s, else in |B|)")
    ap.add_argument("--frac_width", type=float, default=0.10, help="±fractional width of annulus")

    ap.add_argument("--bins", type=int, default=12, help="PA bins for (1)")
    ap.add_argument("--nboot", type=int, default=800, help="Bootstrap reps for (1)")
    ap.add_argument("--tol_deg", type=float, default=6.0, help="PA rounding tolerance for triangle grouping")
    ap.add_argument("--min_channels", type=int, default=6, help="Min channels inside annulus to include a triangle")
    ap.add_argument("--plot_top", type=int, default=12, help="Max triangles to plot for (2)")
    ap.add_argument("--fit_all_channels", action="store_true", help="Fit CP(s) using ALL channels of a selected triangle (recommended).")
    ap.add_argument("--theta_ud_mas", type=float, default=3.2,
                    help="Uniform-disk diameter (mas). Enables star-normalized selection if combined with --star_annuli or --annulus_star_norm.")
    ap.add_argument("--star_annuli", type=float, nargs="*",
                    help="Annulus centers as multiples of the UD first null s_null=1.22/θ (e.g. 0.6 1.0 1.3). Implies selection in s=B/λ.")
    ap.add_argument("--annulus_star_norm", action="store_true",
                    help="Pick annulus centers by quantiles in s*=(B/λ)*θ (dimensionless). Requires --theta_ud_mas.")
    # (keep your existing)
    ap.add_argument("--annulus_in_s", action="store_true",
                    help="Select annuli by quantiles in s=B/λ (use when not using star-normalized options).")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)

    # Load
    samples = read_cp_from_oifits(args.files)
    if samples.cp_deg.size == 0:
        return

    if args.cp_floor_deg is not None:
        apply_cp_sigma_floor_inplace(samples, args.cp_floor_deg)
        print(f"[INFO] Applied CP σ-floor of {args.cp_floor_deg:.3f} deg (quadrature).")

    # Geometry
    geo = build_geometry(samples)
    lam_med = float(np.nanmedian(samples.lam))
    print(f"[INFO] λ_median = {lam_med:.3e} m")

    # ---------- INSERT THIS BLOCK FOR θUD + s* DIAGNOSTICS ----------

    mas2rad = np.deg2rad(1.0/3600_000.0)

    s_centers = None
    L_centers = None

    if args.theta_ud_mas and args.star_annuli:
        # star-normalized option
        theta_rad = args.theta_ud_mas * mas2rad  # diameter in radians
        s_null = 1.22 / theta_rad                # first UD null, in 1/rad (i.e., same units as s=B/λ)
        L_null = s_null * lam_med                # baseline at the null (meters)

        # centers in s from user multiples of s_null
        s_centers = [float(x) * s_null for x in args.star_annuli]
        L_centers = [s * lam_med for s in s_centers]

        print(f"[INFO] θ_UD={args.theta_ud_mas:.3f} mas → s_null={s_null/1e6:.2f} Mλ,  L_null={L_null:.2f} m")
        print("[INFO] Using star-normalized annuli at ξ×s_null:",
            ", ".join(f"ξ={x:.2f}→{sc/1e6:.2f} Mλ" for x, sc in zip(args.star_annuli, s_centers)),
            f"with ±{100*args.frac_width:.0f}% width")

        # Force selection in s (B/λ) for both CP-vs-PA and chromatic fits
        use_s_annulus = True

    else:
        # fall back to your existing behavior
        use_s_annulus = args.annulus_in_s
        centers = choose_annuli_from_quantiles(geo, args.annulus_quantiles, use_s_annulus)
        if use_s_annulus:
            s_centers = centers
            L_centers = [c * lam_med for c in s_centers]
            print("[INFO] Using annuli in s (Mλ):",
                ", ".join(f"{c/1e6:.2f}" for c in s_centers),
                f"with ±{100*args.frac_width:.0f}% width")
        else:
            L_centers = centers
            s_centers = [c / lam_med for c in L_centers]
            print("[INFO] Using annuli in |u,v|:",
                ", ".join(f"{c:.2f}" for c in L_centers),
                f"with ±{100*args.frac_width:.0f}% width")
    # ---- END NEW ----


    # Choose annuli
    # ---------- CHOOSE ANNULI (in |B|, in s=B/λ, or star-normalized) ----------
    lam_med = float(np.nanmedian(samples.lam))
    print(f"[INFO] λ_median = {lam_med:.3e} m")

    # UD diameter handling
    theta_mas = args.theta_ud_mas
    mas2rad = np.deg2rad(1.0/3600_000.0)
    theta_rad = theta_mas * mas2rad if theta_mas else None

    s_centers = None
    L_centers = None
    use_s_annulus = False  # whether CP-vs-PA will select in s instead of |B|

    if args.star_annuli and theta_rad:
        # explicit multiples of the UD first null
        s_null = 1.22 / theta_rad                  # 1/rad
        L_null = s_null * lam_med                  # meters
        s_centers = [float(x) * s_null for x in args.star_annuli]
        L_centers = [s * lam_med for s in s_centers]
        use_s_annulus = True
        print(f"[INFO] θ_UD={theta_mas:.3f} mas → s_null={s_null/1e6:.2f} Mλ, L_null={L_null:.2f} m")
        print("[INFO] Using star-normalized annuli at ξ×s_null:",
            ", ".join(f"ξ={x:.2f}→{sc/1e6:.2f} Mλ" for x, sc in zip(args.star_annuli, s_centers)),
            f"with ±{100*args.frac_width:.0f}% width")

    elif args.annulus_star_norm:
        # quantiles in s* = (B/λ)*θUD
        if not theta_rad:
            raise SystemExit("You set --annulus_star_norm but did not provide --theta_ud_mas.")
        sstar = geo.Smax * theta_rad                                  # dimensionless
        vals = [float(np.nanpercentile(sstar, q)) for q in args.annulus_quantiles]
        s_centers = [v / theta_rad for v in vals]                     # back to s (1/rad)
        L_centers = [s * lam_med for s in s_centers]
        use_s_annulus = True
        print("[INFO] Using annuli in s*=(B/λ)·θ (dimensionless):",
            ", ".join(f"{v:.2f}" for v in vals),
            " → s centers:",
            ", ".join(f"{sc/1e6:.2f} Mλ" for sc in s_centers),
            f"with ±{100*args.frac_width:.0f}% width")

    elif args.annulus_in_s:
        # plain quantiles in s=B/λ
        s_centers = choose_annuli_from_quantiles(geo, args.annulus_quantiles, use_s_annulus=True)
        L_centers = [s * lam_med for s in s_centers]
        use_s_annulus = True
        print("[INFO] Using annuli in s (Mλ):",
            ", ".join(f"{c/1e6:.2f}" for c in s_centers),
            f"with ±{100*args.frac_width:.0f}% width")

    else:
        # default: quantiles in |B|
        L_centers = choose_annuli_from_quantiles(geo, args.annulus_quantiles, use_s_annulus=False)
        s_centers = [L/lam_med for L in L_centers]
        print("[INFO] Using annuli in |u,v|:",
            ", ".join(f"{c:.2f}" for c in L_centers),
            f"with ±{100*args.frac_width:.0f}% width")
        
    # ---------- CHOOSE ANNULI (either in s* or in |u,v|/s=B/λ) ----------
    # if args.annulus_star_norm:
    #     if theta_mas is None:
    #         raise SystemExit("You set --annulus_star_norm but did not provide --theta_ud_mas.")
    #     arr_for_q = geo.Smax * theta_rad           # s*
    #     s_centers = [float(np.nanpercentile(arr_for_q, q)) for q in args.annulus_quantiles]
    #     L_centers = [c / theta_rad * lam_med for c in s_centers]  # for logging only
    #     print("[INFO] Using annuli in s* (dimensionless):",
    #           ", ".join(f"{c:.2f}" for c in s_centers),
    #           f"with ±{100*args.frac_width:.0f}% width")
    # else:
    #     # original behavior
    #     L_centers = choose_annuli_from_quantiles(geo, args.annulus_quantiles, use_s_annulus=False)
    #     s_centers = [c/lam_med for c in L_centers]
    #     print("[INFO] Using annuli in |u,v|:",
    #           ", ".join(f"{c:.2f}" for c in L_centers),
    #           f"with ±{100*args.frac_width:.0f}% width")
    # # centers = choose_annuli_from_quantiles(geo, args.annulus_quantiles, args.annulus_in_s)
    # # if args.annulus_in_s:
    # #     s_centers = centers
    # #     L_centers = [c*lam_med for c in s_centers]
    # #     print("[INFO] Using annuli in s (Mλ):", ", ".join(f"{c/1e6:.2f}" for c in s_centers),
    # #           f"with ±{100*args.frac_width:.0f}% width")
    # # else:
    # #     L_centers = centers
    # #     s_centers = [c/lam_med for c in L_centers]
    # #     print("[INFO] Using annuli in |u,v|:", ", ".join(f"{c:.2f}" for c in L_centers),
    # #           f"with ±{100*args.frac_width:.0f}% width")


    # ---- Run diagnostics per annulus ----
    summaries = []
    for L0, s0 in zip(L_centers, s_centers):
        print(f"\n=== Annulus center: |u,v|≈{L0:.2f}  &  s≈{s0/1e6:.2f} Mλ ±{100*args.frac_width:.0f}% ===")

        # (1) CP vs PA — selects points using whichever domain you chose:
        #     |B| if use_s_annulus==False, or s=B/λ if use_s_annulus==True.
        df_pa = cp_pa_circular_stats_annulus(
            samples, geo, out_dir,
            L0=L0, s0=s0, frac_width=args.frac_width,
            use_s_annulus=use_s_annulus,      # <-- this is the only new bit
            bins=args.bins, Nboot=args.nboot, tag="CP_circmean"
        )

        # (2) Chromatic model selection — always fits vs s=B/λ using s0
        df_bic = run_chromatic_selection(
            samples, geo, out_dir,
            L0=L0, s0=s0, frac_width=args.frac_width,
            tol_deg=args.tol_deg, min_channels=args.min_channels,
            plot_top=args.plot_top, fit_uses_all_channels=args.fit_all_channels,
            prefix="CP_chromatic", return_dataframe=True
        )

        # optional: collect a quick summary row
        digest = {"L0": L0, "s0_Mlambda": s0/1e6}
        if df_pa is not None:
            digest["median_|meanCP|_deg"] = float(np.nanmedian(np.abs(df_pa["mean_cp_deg"])))
            digest["min_Rayleigh_p"] = float(np.nanmin(df_pa["Rayleigh_p"]))
        if df_bic is not None and not df_bic.empty:
            vc = df_bic["best_model"].value_counts()
            for k, v in vc.items():
                digest[f"best_{k}"] = int(v)
            digest["median_dBIC_SIN_vs_CONST"] = float(np.nanmedian(df_bic["dBIC_SIN_vs_CONST"]))
        summaries.append(digest)
        

        # ----- photospheric dipole test inside this annulus -----
        # Select the same channels you use to "select" triangles: an s-annulus.
        band = (geo.Smax >= s0*(1-args.frac_width)) & (geo.Smax <= s0*(1+args.frac_width))
        dip = _fit_pa_dipole(samples, geo, band, use_Mlambda=True)
        if dip is not None:
            print(f"[DIPOLE] annulus s≈{s0/1e6:.2f} Mλ: "
                f"K={dip['K']:.3f} deg/Mλ, φ0={dip['phi0_deg']:.1f}°, "
                f"c={dip['c']:.3f} deg, N={dip['n']}, BIC={dip['bic']:.1f}")
        else:
            print("[DIPOLE] Not enough channels in this annulus for a stable fit.")

    # # Run diagnostics
    # summaries = []
    # for L0, s0 in zip(L_centers, s_centers):
    #     print(f"\n=== Annulus center: |u,v|≈{L0:.2f}  &  s≈{s0/1e6:.2f} Mλ ±{100*args.frac_width:.0f}% ===")

    #     df_pa = cp_pa_circular_stats_annulus(
    #         samples, geo, out_dir, L0, s0, args.frac_width,
    #         use_s_annulus=args.annulus_in_s, bins=args.bins, Nboot=args.nboot, tag="CP_circmean"
    #     )

    #     df_bic = run_chromatic_selection(
    #         samples, geo, out_dir, L0, s0, args.frac_width,
    #         tol_deg=args.tol_deg, min_channels=args.min_channels,
    #         plot_top=args.plot_top, fit_uses_all_channels=args.fit_all_channels,
    #         prefix="CP_chromatic", return_dataframe=True
    #     )

    #     digest = {"L0": L0, "s0_Mlambda": s0/1e6}
    #     if df_pa is not None:
    #         digest["median_|meanCP|_deg"] = float(np.nanmedian(np.abs(df_pa["mean_cp_deg"])))
    #         digest["min_Rayleigh_p"] = float(np.nanmin(df_pa["Rayleigh_p"]))
    #     if df_bic is not None and not df_bic.empty:
    #         vc = df_bic["best_model"].value_counts()
    #         for k,v in vc.items(): digest[f"best_{k}"] = int(v)
    #         digest["median_dBIC_SIN_vs_CONST"] = float(np.nanmedian(df_bic["dBIC_SIN_vs_CONST"]))
    #     summaries.append(digest)

    if summaries:
        df_sum = pd.DataFrame(summaries)
        csvp = os.path.join(out_dir, "CP_diagnostics_overview.csv")
        df_sum.to_csv(csvp, index=False); print("[SAVE]", csvp)

    print("\nDone. Outputs in:", out_dir)

if __name__ == "__main__":
    main()

"""
python cp_pa_annulus_diagnostics_2.py --files "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*fits" --out_dir ./diag_out_oifits --annulus_quantiles 65 80 90 --frac_width 0.25 --bins 12 --tol_deg 12 --min_channels 2 --plot_top 12
"""


"""
Nice—those numbers tell the story pretty cleanly:
	•	Inner annuli (≈48.7 and 69.0 Mλ):
K ≈ 0.03–0.04 deg/Mλ and |c| small (≈0–2°).
That’s a tiny, slowly varying CP consistent with a weak, photospheric dipole (spot/brightness gradient).
	•	Outer annuli (≈81–93 Mλ):
K and c explode and the BIC is gigantic. That’s exactly what we expect once you hit/clear the first null—CP wraps toward ±180° and the simple dipole model breaks (it’s not intended for the second lobe / circumstellar regime).

So: you’ve got quantitative separation already—small K inside, nonsense K outside.

To make this even more robust and avoid the “blow-ups” outside the UD, here are two small patches:

⸻


"""
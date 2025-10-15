#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANDID binary search with extras (patched)

Features:
- Global fitMap with bounded/coarse grid.
- Optional lock of diam* to UD from UD_fit.csv (prevents runaway diameters).
- Optional bootstrap uncertainties (fitBoot) with guard for formatting bug.
- Optional remove-and-refit (fitMap with removeCompanion).
- Optional detection-limit map (after removing the fitted companion).

Install CANDID from GitHub (not PyPI):
    python -m pip install "git+https://github.com/amerand/CANDID.git#egg=candid"
"""

import argparse, glob, json, os, re, fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import candid  # from GitHub repo

# ---------------- helpers ----------------

def pick_ud_at_wvl(ud_csv_path: str, wavemin_um: float, wavemax_um: float) -> float:
    """Return UD [mas] at midpoint wavelength from UD_fit.csv (index in meters; column 'ud_mean' in mas)."""
    ud = pd.read_csv(ud_csv_path, index_col=0)
    wvl0_um = 0.5 * (wavemin_um + wavemax_um)
    idx = np.argmin(abs(ud.index -  1e-6 * wvl0_um))#(ud.index - (1e-6 * wvl0_um)).abs().argmin()
    return float(ud['ud_mean'].iloc[idx])

def savefig(fig_num: int, path: str):
    try:
        plt.figure(fig_num); plt.tight_layout()
        plt.savefig(path, dpi=180)
        print(f"[SAVE] {path}")
    except Exception as e:
        print(f"[WARN] Could not save figure {fig_num}: {e}")


def np_json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    # Fallback: stringify anything else non-serializable (e.g. Path)
    return str(o)




#############
# 1) Plot Closure Phase vs PA
import numpy as np
import matplotlib.pyplot as plt

def _baseline_pa_deg(u, v):
    # Astronomical PA: angle East of North in [0, 180) for baselines
    # Using PA = atan2(u, v); fold 180° symmetry of baselines
    pa = np.degrees(np.arctan2(u, v)) % 180.0
    return pa

def extract_cp_vs_pa(o):
    """Return arrays of PA [deg], CP [deg], sigma_CP [deg] for all CP-like observables."""
    pa_all, cp_all, se_all = [], [], []
    for c in o._chi2Data:
        obs = c[0].split(';')[0]
        if obs in ('cp', 't3', 'icp', 'scp', 'ccp'):
            # c[1],c[2] = u1,v1 ; c[3],c[4] = u2,v2 ; third = -(u1+u2, v1+v2)
            u1, v1 = np.asarray(c[1]), np.asarray(c[2])
            u2, v2 = np.asarray(c[3]), np.asarray(c[4])
            u3, v3 = -(u1 + u2), -(v1 + v2)

            # Choose the *longest* triangle edge per sample to define the PA
            L1 = np.hypot(u1, v1)
            L2 = np.hypot(u2, v2)
            L3 = np.hypot(u3, v3)
            idx12 = L2 > L1
            umax = np.where(idx12, u2, u1)
            vmax = np.where(idx12, v2, v1)
            Lmax = np.where(idx12, L2, L1)
            use3 = L3 > Lmax
            umax = np.where(use3, u3, umax)
            vmax = np.where(use3, v3, vmax)

            pa = _baseline_pa_deg(umax, vmax).ravel()
            cp = np.asarray(c[-2]).ravel()            # radians
            se = np.asarray(c[-1]).ravel()            # radians

            pa_all.append(pa)
            cp_all.append(np.degrees(cp))
            se_all.append(np.degrees(se))

    if not pa_all:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(pa_all), np.concatenate(cp_all), np.concatenate(se_all))

def plot_cp_vs_pa(o, bins=36):
    pa, cp_deg, se_deg = extract_cp_vs_pa(o)
    if pa.size == 0:
        print("[CPvsPA] No CP-like data found.")
        return

    # binned median ± MAD/√N for readability
    edges = np.linspace(0, 180, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    med, lo, hi, n = [], [], [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (pa >= lo_e) & (pa < hi_e)
        if not np.any(m):
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); n.append(0)
        else:
            x = cp_deg[m]
            mmed = np.nanmedian(x)
            mad = 1.4826*np.nanmedian(np.abs(x - mmed))
            med.append(mmed); lo.append(mmed - mad/np.sqrt(np.sum(m))); hi.append(mmed + mad/np.sqrt(np.sum(m))); n.append(np.sum(m))

    plt.figure(figsize=(8,4.5))
    plt.scatter(pa, cp_deg, s=5, alpha=0.25, label="CP samples")
    plt.plot(centers, med, lw=2, label="binned median")
    plt.fill_between(centers, lo, hi, alpha=0.2, label="~1σ of median")
    plt.xlim(0,180); plt.xlabel("Baseline PA (deg, E of N)")
    plt.ylabel("Closure phase (deg)")
    plt.title("Closure phase vs baseline PA")
    plt.legend()
    plt.tight_layout()



#########
"""
2) Fit per PA bin (without rewriting CANDID)

Instead of physically removing data, we “ignore” points outside each PA bin by inflating their σ by a huge factor. That keeps array shapes intact and plays nicely with CANDID’s internals.
"""
from copy import deepcopy
import numpy as np

def pa_mask_arrays_for_tuple(c, pa_min, pa_max):
    """Return a mask (flattened shape-compatible with c[-2]/c[-1]) that is True for samples
    whose *longest* CP edge (or the V2 baseline) lies within [pa_min, pa_max) deg."""
    obs = c[0].split(';')[0]
    if obs == 'v2':
        u, v = np.asarray(c[1]), np.asarray(c[2])
        pa = _baseline_pa_deg(u, v).ravel()
        return (pa >= pa_min) & (pa < pa_max), pa.shape
    elif obs in ('cp', 't3', 'icp', 'scp', 'ccp'):
        u1, v1 = np.asarray(c[1]), np.asarray(c[2])
        u2, v2 = np.asarray(c[3]), np.asarray(c[4])
        u3, v3 = -(u1 + u2), -(v1 + v2)
        L1 = np.hypot(u1, v1)
        L2 = np.hypot(u2, v2)
        L3 = np.hypot(u3, v3)
        idx12 = L2 > L1
        umax = np.where(idx12, u2, u1)
        vmax = np.where(idx12, v2, v1)
        Lmax = np.where(idx12, L2, L1)
        use3 = L3 > Lmax
        umax = np.where(use3, u3, umax)
        vmax = np.where(use3, v3, vmax)
        pa = _baseline_pa_deg(umax, vmax).ravel()
        return (pa >= pa_min) & (pa < pa_max), pa.shape
    else:
        # Unknown observable: keep it by default
        shp = np.asarray(c[-2]).shape
        return np.ones(np.prod(shp), dtype=bool), shp
    

def count_by_obs(o, pa_lo, pa_hi):
    cnt = {"v2": 0, "cp": 0}
    for c in o._chi2Data:
        obs = c[0].split(';')[0]
        mask, _ = pa_mask_arrays_for_tuple(c, pa_lo, pa_hi)
        n = int(mask.sum())
        if obs == "v2":
            cnt["v2"] += n
        elif obs in ("cp", "t3", "icp", "scp", "ccp"):
            cnt["cp"] += n
    return cnt


def _robust_get_chi2r(bestFit: dict) -> float:
    for k in ("chi2r", "chi2nu", "chi2_red"):
        v = bestFit.get(k, np.nan)
        if np.isfinite(v):
            return float(v)
    if all(k in bestFit for k in ("chi2", "Ndof")) and bestFit["Ndof"]:
        return float(bestFit["chi2"]) / float(bestFit["Ndof"])
    return np.nan

def fit_in_pa_bins(o, fit_kwargs, pa_bin_deg=45.0, min_points=50):
    results = []
    bins = np.arange(0, 180 + 1e-6, pa_bin_deg)
    centers = 0.5*(bins[:-1] + bins[1:])

    # keep original data to restore each iteration
    orig_data = o._chi2Data

    for (pa_lo, pa_hi), center in zip(zip(bins[:-1], bins[1:]), centers):
        kept = 0
        new_data = []

        # build a NEW _chi2Data list with NaNs outside the bin
        for c in orig_data:
            mask, shp = pa_mask_arrays_for_tuple(c, pa_lo, pa_hi)
            dat = np.asarray(c[-2]).ravel().copy()
            err = np.asarray(c[-1]).ravel().copy()
            dat[~mask] = np.nan
            err[~mask] = np.nan
            kept += int(mask.sum())

            c_new = list(c)
            c_new[-2] = dat.reshape(shp)
            c_new[-1] = err.reshape(shp)
            new_data.append(c_new)

        if kept < min_points:
            results.append({"pa_bin_center_deg": float(center),
                            "error": f"insufficient samples (kept={kept})",
                            "sep_mas": np.nan, "pa_fit_deg": np.nan,
                            "f_percent": np.nan, "diam_star_mas": np.nan, "chi2r": np.nan})
            continue

        # swap in masked data
        o._chi2Data = new_data
        try:
            local = dict(fit_kwargs)
            local["fig"] = None
            if all(k in local for k in ("rmin","rmax","step")):
                span = float(local["rmax"]) - float(local["rmin"])
                if span <= 0 or (span / float(local["step"])) < 2:
                    local["step"] = max(span/10.0, 1e-3)

            # (optional) give CP more say in the bin fit
            # o.observables = ['cp','v2']  # or try ['cp'] only

            o.fitMap(**local)

            bf = o.bestFit.get("best", {})
            if {"x","y","f"}.issubset(bf):
                sep = float(np.hypot(bf["x"], bf["y"]))
                pa_fit = float(np.degrees(np.arctan2(bf["x"], bf["y"])) % 360.0)
                chi2r = _robust_get_chi2r(o.bestFit)
                results.append({"pa_bin_center_deg": float(center),
                                "sep_mas": sep, "pa_fit_deg": pa_fit,
                                "f_percent": float(bf["f"]),
                                "diam_star_mas": float(bf.get("diam*", np.nan)),
                                "chi2r": chi2r})
            else:
                results.append({"pa_bin_center_deg": float(center),
                                "sep_mas": np.nan, "pa_fit_deg": np.nan,
                                "f_percent": np.nan, "diam_star_mas": np.nan,
                                "chi2r": np.nan})
        except Exception as e:
            results.append({"pa_bin_center_deg": float(center), "error": str(e),
                            "sep_mas": np.nan, "pa_fit_deg": np.nan,
                            "f_percent": np.nan, "diam_star_mas": np.nan,
                            "chi2r": np.nan})
        finally:
            # always restore the original data for the next bin
            o._chi2Data = orig_data

    return results

def plot_pa_binned_results(results):
    import matplotlib.pyplot as plt
    import numpy as np
    cen = np.array([r["pa_bin_center_deg"] for r in results])
    f = np.array([r["f_percent"] for r in results], dtype=float)
    sep = np.array([r["sep_mas"] for r in results], dtype=float)
    chi2r = np.array([r["chi2r"] for r in results], dtype=float)

    fig, ax = plt.subplots(3,1, figsize=(8,7), sharex=True)
    ax[0].plot(cen, f, "-o", ms=4); ax[0].set_ylabel("f (% primary)")
    ax[1].plot(cen, sep, "-o", ms=4); ax[1].set_ylabel("sep (mas)")
    ax[2].plot(cen, chi2r, "-o", ms=4); ax[2].set_ylabel(r"$\chi^2_\nu$")
    ax[2].set_xlabel("PA bin center (deg, E of N)")
    for a in ax: a.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt

def _baseline_pa_deg(u, v):
    """Baseline PA (deg East of North), folded to [0, 180)."""
    return (np.degrees(np.arctan2(u, v)) % 180.0)

def extract_v2_vs_pa(o, Lmin=None, q_long=None):
    """
    Collect PA [deg], V2, sigma(V2), and baseline length L for all V2 points.
    Optionally filter to L >= Lmin, or to the top q_long% longest baselines.
    """
    pa_all, v2_all, se_all, L_all = [], [], [], []
    for c in o._chi2Data:
        obs = c[0].split(';')[0].lower()
        if obs in ('v2', 'vis2', 'v2amp'):
            u, v = np.asarray(c[1]), np.asarray(c[2])
            V2   = np.asarray(c[-2])
            sig  = np.asarray(c[-1])

            pa = _baseline_pa_deg(u, v).ravel()
            L  = np.hypot(u, v).ravel()
            V2 = V2.ravel()
            sig = sig.ravel()

            m = np.isfinite(pa) & np.isfinite(V2) & np.isfinite(sig) & np.isfinite(L)

            if Lmin is not None:
                m &= (L >= float(Lmin))

            if q_long is not None:
                # keep top q_long% by L (computed on the already-valid set)
                thr = np.nanpercentile(L[m], 100.0 - float(q_long))
                m &= (L >= thr)

            if np.any(m):
                pa_all.append(pa[m]); v2_all.append(V2[m]); se_all.append(sig[m]); L_all.append(L[m])

    if not pa_all:
        return (np.array([]),)*4
    return np.concatenate(pa_all), np.concatenate(v2_all), np.concatenate(se_all), np.concatenate(L_all)


def plot_L_vs_pa(o, bins=36):
    import numpy as np, matplotlib.pyplot as plt
    Ls = []
    PAs = []
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u, v = np.asarray(c[1]).ravel(), np.asarray(c[2]).ravel()
            Ls.append(np.hypot(u, v)); PAs.append((np.degrees(np.arctan2(u, v)) % 180.0))
    if not Ls:
        print("[LvsPA] No V² data."); return None
    L = np.concatenate(Ls); PA = np.concatenate(PAs)

    edges = np.linspace(0,180,bins+1); centers = 0.5*(edges[:-1]+edges[1:])
    medL = [np.nanmedian(L[(PA>=a0)&(PA<a1)]) if np.any((PA>=a0)&(PA<a1)) else np.nan
            for a0,a1 in zip(edges[:-1],edges[1:])]

    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.scatter(PA, L, s=6, alpha=0.25, label="baselines")
    ax.plot(centers, medL, lw=2, label="median |u,v| per bin")
    ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)")
    ax.set_ylabel("|u,v| (same units as CANDID)")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    return fig

def plot_v2_vs_pa_annulus(o, L0, frac_width=0.05, bins=24):
    import numpy as np, matplotlib.pyplot as plt
    PA_all=[]; V2_all=[]
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u, v = np.asarray(c[1]).ravel(), np.asarray(c[2]).ravel()
            V2   = np.asarray(c[-2]).ravel()
            L    = np.hypot(u, v)
            PA   = (np.degrees(np.arctan2(u, v)) % 180.0)
            band = (L>=L0*(1-frac_width)) & (L<=L0*(1+frac_width))
            PA_all.append(PA[band]); V2_all.append(V2[band])
    if not PA_all or np.sum([len(x) for x in PA_all])==0:
        print("[Annulus] No points in requested annulus."); return None
    PA = np.concatenate(PA_all); V2 = np.concatenate(V2_all)

    edges = np.linspace(0,180,bins+1); centers=0.5*(edges[:-1]+edges[1:])
    med=[]; lo=[]; hi=[]
    for a0,a1 in zip(edges[:-1],edges[1:]):
        m=(PA>=a0)&(PA<a1)
        if not np.any(m): med+= [np.nan]; lo+= [np.nan]; hi+= [np.nan]; continue
        x=V2[m]; mmed=np.nanmedian(x); mad=1.4826*np.nanmedian(np.abs(x-mmed)); s=mad/np.sqrt(np.sum(m))
        med.append(mmed); lo.append(mmed-s); hi.append(mmed+s)

    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.scatter(PA, V2, s=6, alpha=0.25, label=f"V² samples (|u,v|≈{L0}±{100*frac_width:.0f}%)")
    ax.plot(centers, med, lw=2, label="binned median")
    ax.fill_between(centers, lo, hi, alpha=0.2, label="~1σ of median")
    ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)"); ax.set_ylabel("V²")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    return fig

def plot_v2_residual_vs_pa(o, theta_mas, wav_um=1.65, bins=36, Lmin=None):
    """
    Divide measured V² by circular-UD model at the same |u,v|,
    then plot residual vs PA. theta_mas from your UD fit/CSV.
    wav_um only sets the spatial-frequency scale if your u,v are in meters/λ units;
    for CANDID’s internal scaling the ratio is what matters.
    """
    import numpy as np, matplotlib.pyplot as plt
    # UD visibility amplitude
    def V_ud(r):  # r = pi*theta*|u,v|/lambda  (scale absorbed by |u,v| units)
        from scipy.special import j1
        with np.errstate(divide='ignore', invalid='ignore'):
            x = r
            V = 2*j1(x)/x
            V[x==0] = 1.0
        return V

    theta_rad = theta_mas * (1e-3/206265.)  # mas -> rad
    PA_all=[]; R_all=[]
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u, v = np.asarray(c[1]).ravel(), np.asarray(c[2]).ravel()
            V2   = np.asarray(c[-2]).ravel()
            L    = np.hypot(u, v)
            if Lmin is not None:
                mL = L >= float(Lmin)
            else:
                mL = np.isfinite(L)
            PA   = (np.degrees(np.arctan2(u, v)) % 180.0)
            # model V² (scale constant cancels in practice)
            r = np.pi * theta_rad * L  # (λ scaling absorbed)
            V2_model = V_ud(r)**2
            m = mL & np.isfinite(V2_model) & np.isfinite(V2)
            PA_all.append(PA[m])
            R_all.append((V2[m]/V2_model[m]) - 1.0)

    if not PA_all:
        print("[UDresid] No points."); return None
    PA = np.concatenate(PA_all); R = np.concatenate(R_all)

    edges = np.linspace(0,180,bins+1); centers=0.5*(edges[:-1]+edges[1:])
    med=[]; lo=[]; hi=[]
    for a0,a1 in zip(edges[:-1],edges[1:]):
        m=(PA>=a0)&(PA<a1)
        if not np.any(m): med+= [np.nan]; lo+= [np.nan]; hi+= [np.nan]; continue
        x=R[m]; mmed=np.nanmedian(x); mad=1.4826*np.nanmedian(np.abs(x-mmed)); s=mad/np.sqrt(np.sum(m))
        med.append(mmed); lo.append(mmed-s); hi.append(mmed+s)

    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.axhline(0, ls='--', lw=1)
    ax.scatter(PA, R, s=6, alpha=0.25, label="(V² / V²_UD) − 1")
    ax.plot(centers, med, lw=2, label="binned median")
    ax.fill_between(centers, lo, hi, alpha=0.2, label="~1σ of median")
    ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)")
    ax.set_ylabel("residual V² / V²_UD − 1")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    return fig

def plot_v2_vs_pa(o, bins=36, Lmin=None, q_long=None, ax=None, title=None):
    """
    Scatter V2 vs PA with binned median ± ~1σ(MAD/√N).
    Use Lmin or q_long to restrict to longer baselines.
    """
    pa, V2, sig, L = extract_v2_vs_pa(o, Lmin=Lmin, q_long=q_long)
    if pa.size == 0:
        print("[V2vsPA] No V2 samples after filtering.")
        return None

    edges = np.linspace(0, 180, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    med, lo, hi = [], [], []

    for a0, a1 in zip(edges[:-1], edges[1:]):
        m = (pa >= a0) & (pa < a1)
        if not np.any(m):
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            x = V2[m]
            mmed = np.nanmedian(x)
            mad = 1.4826*np.nanmedian(np.abs(x - mmed))
            med.append(mmed)
            # uncertainty on the median for display
            s = mad/np.sqrt(np.sum(m))
            lo.append(mmed - s); hi.append(mmed + s)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    ax.scatter(pa, V2, s=6, alpha=0.25, label="V² samples")
    ax.plot(centers, med, lw=2, label="binned median")
    ax.fill_between(centers, lo, hi, alpha=0.2, label="~1σ of median")

    ax.set_xlim(0, 180)
    ax.set_xlabel("Baseline PA (deg, E of N)")
    ax.set_ylabel("V²")

    if title is None:
        if q_long is not None:
            title = f"V² vs PA (top {q_long:.0f}% longest baselines)"
        elif Lmin is not None:
            title = f"V² vs PA (L ≥ {Lmin:g})"
        else:
            title = "V² vs PA"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


from scipy.optimize import curve_fit

# ---- pull V², PA in a |u,v| annulus ----
def extract_v2_pa_annulus(o, L0, frac_width=0.10):
    PA_all, V2_all = [], []
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u = np.asarray(c[1]).ravel()
            v = np.asarray(c[2]).ravel()
            V2 = np.asarray(c[-2]).ravel()
            L  = np.hypot(u, v)
            PA = (np.degrees(np.arctan2(u, v)) % 180.0)
            m = (L >= L0*(1-frac_width)) & (L <= L0*(1+frac_width)) & np.isfinite(V2)
            if np.any(m):
                PA_all.append(PA[m])
                V2_all.append(V2[m])
    if not PA_all:
        return np.array([]), np.array([])
    return np.concatenate(PA_all), np.concatenate(V2_all)

# ---- cos(2(PA-PA0)) model and robust binning ----
def cos2_model(pa_deg, A, B, PA0_deg):
    # PA wrap-safe using radians internally
    return A + B * np.cos(2.0 * np.radians(pa_deg - PA0_deg))

def bin_pa(pa_deg, y, bins=18):
    edges = np.linspace(0,180,bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    med, sig = [], []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        m = (pa_deg >= a0) & (pa_deg < a1)
        if not np.any(m):
            med.append(np.nan); sig.append(np.nan)
        else:
            xm = np.nanmedian(y[m])
            mad = 1.4826*np.nanmedian(np.abs(y[m]-xm))
            med.append(xm)
            sig.append(mad/np.sqrt(np.sum(m)))  # ~uncertainty on median
    return centers, np.array(med), np.array(sig)

# ---- fit + plot wrapper ----
def fit_and_plot_v2_cos2(o, L0, frac_width=0.10, bins=18, outpath=None, title=None, do_bootstrap=True, Nboot=1000):
    pa, v2 = extract_v2_pa_annulus(o, L0, frac_width)
    if pa.size == 0:
        print("[cos2] No points in annulus."); return None, None

    cen, med, sig = bin_pa(pa, v2, bins=bins)
    m = np.isfinite(med)
    if np.sum(m) < 4:
        print("[cos2] Not enough binned points to fit."); return None, None

    # initial guesses: A ~ median, small B, PA0 ~ argmax
    A0 = np.nanmedian(med[m])
    B0 = 0.1 * A0
    PA0 = float(cen[m][np.nanargmax(med[m])])

    # weighted fit on binned medians
    sigma = np.where(np.isfinite(sig), sig, np.nanmedian(sig[m]))
    popt, pcov = curve_fit(cos2_model, cen[m], med[m], p0=[A0, B0, PA0], sigma=sigma[m], absolute_sigma=True, maxfev=10000)
    A, B, PA0_fit = popt
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([np.nan, np.nan, np.nan])

    # optional bootstrap of unbinned points for sanity
    boot_B = None
    if do_bootstrap:
        rng = np.random.default_rng(0)
        boot_B = np.empty(Nboot)
        for i in range(Nboot):
            idx = rng.integers(0, pa.size, pa.size)
            cen_b, med_b, sig_b = bin_pa(pa[idx], v2[idx], bins=bins)
            mb = np.isfinite(med_b)
            if np.sum(mb) < 4:
                boot_B[i] = np.nan
                continue
            sigma_b = np.where(np.isfinite(sig_b), sig_b, np.nanmedian(sig_b[mb]))
            try:
                p_b, _ = curve_fit(cos2_model, cen_b[mb], med_b[mb],
                                   p0=[A, max(B,1e-6), PA0_fit],
                                   sigma=sigma_b[mb], absolute_sigma=True, maxfev=10000)
                boot_B[i] = p_b[1]
            except Exception:
                boot_B[i] = np.nan

    # plot
    fig, ax = plt.subplots(figsize=(9,3.6))
    ax.scatter(pa, v2, s=8, alpha=0.25, label=f"V² samples (|u,v|≈{L0:.2f}±{100*frac_width:.0f}%)")
    ax.errorbar(cen[m], med[m], yerr=sigma[m], fmt='o', ms=4, capsize=2, label="binned median ±σ")
    pa_line = np.linspace(0,180,720)
    ax.plot(pa_line, cos2_model(pa_line, *popt), lw=2, label=f"fit: A={A:.3f}±{perr[0]:.3f},  B={B:.3f}±{perr[1]:.3f},  PA₀={PA0_fit:.1f}°±{perr[2]:.1f}°")
    ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)"); ax.set_ylabel("V²")
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=180)
        print(f"[SAVE] {outpath}")
    return (popt, perr), (boot_B if do_bootstrap else None)


## final plots 
import numpy as np, matplotlib.pyplot as plt, os
from scipy.optimize import curve_fit

def _baseline_pa_deg(u, v):
    # PA (deg East of North) folded to [0,180)
    return (np.degrees(np.arctan2(u, v)) % 180.0)

def _mad_sigma(x):
    m = np.nanmedian(x)
    mad = 1.4826*np.nanmedian(np.abs(x - m))
    return m, mad

def _bin_pa(pa_deg, y, bins=18):
    edges = np.linspace(0, 180, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    med, sig = [], []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        m = (pa_deg >= a0) & (pa_deg < a1)
        if not np.any(m):
            med.append(np.nan); sig.append(np.nan)
        else:
            xm, mad = _mad_sigma(y[m])
            med.append(xm)
            sig.append(mad/np.sqrt(np.sum(m)))  # ~uncertainty on median
    return centers, np.array(med), np.array(sig)

def _collect_L_v2(o):
    """All V2 baseline lengths L and (PA,V2)."""
    Ls, PAs, V2s = [], [], []
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u = np.asarray(c[1]).ravel(); v = np.asarray(c[2]).ravel()
            V2 = np.asarray(c[-2]).ravel()
            L  = np.hypot(u, v)
            pa = _baseline_pa_deg(u, v)
            m = np.isfinite(L) & np.isfinite(V2)
            Ls.append(L[m]); PAs.append(pa[m]); V2s.append(V2[m])
    if not Ls: 
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(Ls), np.concatenate(PAs), np.concatenate(V2s)

def _choose_annuli_from_quantiles(o, quantiles=(60, 75, 85, 92)):
    L_all, _, _ = _collect_L_v2(o)
    if not L_all.size:
        return []
    return [float(np.nanpercentile(L_all, q)) for q in quantiles]

def _cos2_model(pa_deg, A, B, PA0_deg):
    return A + B*np.cos(2.0*np.radians(pa_deg - PA0_deg))

def _extract_v2_pa_annulus(o, L0, frac_width=0.10):
    PA_all, V2_all = [], []
    for c in o._chi2Data:
        if c[0].split(';')[0].lower() in ('v2','vis2','v2amp'):
            u = np.asarray(c[1]).ravel(); v = np.asarray(c[2]).ravel()
            V2 = np.asarray(c[-2]).ravel()
            L  = np.hypot(u, v); PA = _baseline_pa_deg(u, v)
            m = (L >= L0*(1-frac_width)) & (L <= L0*(1+frac_width)) & np.isfinite(V2)
            if np.any(m):
                PA_all.append(PA[m]); V2_all.append(V2[m])
    if not PA_all: 
        return np.array([]), np.array([])
    return np.concatenate(PA_all), np.concatenate(V2_all)

def run_cos2_across_annuli(o, out_dir, L0_list=None, quantiles=(60,75,85,92),
                           frac_width=0.10, bins=12, tag="V2_PA_cos2"):
    os.makedirs(out_dir, exist_ok=True)
    if L0_list is None or len(L0_list) == 0:
        L0_list = _choose_annuli_from_quantiles(o, quantiles)

    rows = []
    for L0 in L0_list:
        pa, v2 = _extract_v2_pa_annulus(o, L0, frac_width)
        if pa.size == 0:
            print(f"[cos2] L0={L0:.2f}: no points"); 
            continue

        cen, med, sig = _bin_pa(pa, v2, bins=bins)
        m = np.isfinite(med)
        if np.sum(m) < 4:
            print(f"[cos2] L0={L0:.2f}: too few binned points")
            continue

        A0 = np.nanmedian(med[m]); B0 = 0.1*A0
        PA0 = float(cen[m][np.nanargmax(med[m])])
        sigma = np.where(np.isfinite(sig), sig, np.nanmedian(sig[m]))
        popt, pcov = curve_fit(_cos2_model, cen[m], med[m],
                               p0=[A0,B0,PA0], sigma=sigma[m], absolute_sigma=True, maxfev=10000)
        A,B,PA0_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov.size else (np.nan,np.nan,np.nan)
        frac = B/A if A else np.nan

        # plot
        fig, ax = plt.subplots(figsize=(9,3.6))
        ax.errorbar(cen[m], med[m], yerr=sigma[m], fmt='o', ms=4, capsize=2, label="binned median ±σ")
        pa_line = np.linspace(0,180,720)
        ax.plot(pa_line, _cos2_model(pa_line, *popt), lw=2,
                label=f"fit: A={A:.3f}±{perr[0]:.003f},  B={B:.3f}±{perr[1]:.003f},  PA₀={PA0_fit:.1f}°±{perr[2]:.1f}°")
        ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)"); ax.set_ylabel("V²")
        ax.set_title(f"V² vs PA in |u,v| annulus |u,v|≈{L0:.2f}±{100*frac_width:.0f}%")
        ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
        fpath = os.path.join(out_dir, f"{tag}_L0_{L0:.2f}.png")
        fig.savefig(fpath, dpi=180); plt.close(fig); print(f"[SAVE] {fpath}")

        rows.append({"L0":L0, "frac_width":frac_width, "bins":bins,
                     "A":A, "dA":perr[0], "B":B, "dB":perr[1],
                     "PA0_deg":PA0_fit, "dPA0_deg":perr[2], "B_over_A":frac})

    # CSV summary
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("L0")
        csv_path = os.path.join(out_dir, f"{tag}_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] {csv_path}")
    else:
        print("[cos2] No annuli produced a fit.")


### CP vs PA 
def _extract_cp_pa_annulus(o, L0, frac_width=0.10):
    """Return PA[deg], CP[deg] for CP-like observables whose *longest* edge
    has |u,v| within L0±(frac_width*L0)."""
    pa_list, cp_list = [], []
    for c in o._chi2Data:
        obs = c[0].split(';')[0].lower()
        if obs in ('cp','t3','icp','scp','ccp'):
            u1 = np.asarray(c[1]).ravel(); v1 = np.asarray(c[2]).ravel()
            u2 = np.asarray(c[3]).ravel(); v2 = np.asarray(c[4]).ravel()
            u3 = -(u1 + u2);              v3 = -(v1 + v2)

            L1 = np.hypot(u1, v1); L2 = np.hypot(u2, v2); L3 = np.hypot(u3, v3)
            # choose the longest edge to define (PA,L)
            idx12 = L2 > L1
            umax = np.where(idx12, u2, u1)
            vmax = np.where(idx12, v2, v1)
            Lmax = np.where(idx12, L2, L1)
            use3 = L3 > Lmax
            umax = np.where(use3, u3, umax)
            vmax = np.where(use3, v3, vmax)
            Lmax = np.where(use3, L3, Lmax)

            PA  = _baseline_pa_deg(umax, vmax)
            CP  = np.degrees(np.asarray(c[-2]).ravel())   # c[-2] is CP in radians
            m = (Lmax >= L0*(1-frac_width)) & (Lmax <= L0*(1+frac_width)) & np.isfinite(CP)
            if np.any(m):
                pa_list.append(PA[m]); cp_list.append(CP[m])
    if not pa_list:
        return np.array([]), np.array([])
    return np.concatenate(pa_list), np.concatenate(cp_list)

def plot_cp_vs_pa_annulus(o, out_dir, L0_list, frac_width=0.10, bins=18, tag="CP_PA_annulus"):
    os.makedirs(out_dir, exist_ok=True)
    for L0 in L0_list:
        pa, cp = _extract_cp_pa_annulus(o, L0, frac_width)
        if pa.size == 0:
            print(f"[CPvsPA] L0={L0:.2f}: no CP points"); 
            continue

        cen, med, sig = _bin_pa(pa, cp, bins=bins)
        fig, ax = plt.subplots(figsize=(9,3.6))
        ax.scatter(pa, cp, s=8, alpha=0.25, label="CP samples")
        m = np.isfinite(med)
        ax.errorbar(cen[m], med[m], yerr=sig[m], fmt='o', ms=4, capsize=2, label="binned median ±σ")
        ax.axhline(0, ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0,180); ax.set_xlabel("Baseline PA (deg, E of N)")
        ax.set_ylabel("Closure phase (deg)")
        ax.set_title(f"CP vs PA in |u,v| annulus |u,v|≈{L0:.2f}±{100*frac_width:.0f}%")
        ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
        fpath = os.path.join(out_dir, f"{tag}_L0_{L0:.2f}.png")
        fig.savefig(fpath, dpi=180); plt.close(fig); print(f"[SAVE] {fpath}")



## helper to extract info 
def _extract_chi2_info(bestfit: dict) -> dict:
    """Return a robust dict with chi2, Ndof, chi2r (chi2_nu)."""
    out = {"chi2": np.nan, "Ndof": np.nan, "chi2r": np.nan}
    if not isinstance(bestfit, dict):
        return out
    # try common keys used by CANDID across versions
    for k in ("chi2", "Chi2", "chisq"):
        if k in bestfit and np.isfinite(bestfit[k]):
            out["chi2"] = float(bestfit[k]); break
    for k in ("Ndof", "ndof", "NDOF"):
        if k in bestfit and np.isfinite(bestfit[k]):
            out["Ndof"] = float(bestfit[k]); break
    for k in ("chi2r", "chi2nu", "chi2_red", "reduced_chi2"):
        if k in bestfit and np.isfinite(bestfit[k]):
            out["chi2r"] = float(bestfit[k]); break
    # fill reduced chi2 if needed
    if (not np.isfinite(out["chi2r"])) and np.isfinite(out["chi2"]) and np.isfinite(out["Ndof"]) and out["Ndof"] > 0:
        out["chi2r"] = out["chi2"] / out["Ndof"]
    return out

def _count_points_by_obs(o) -> dict:
    """Count finite (meas & err) samples by observable in o._chi2Data."""
    counts = {}
    total = 0
    for c in getattr(o, "_chi2Data", []):
        kind = c[0].split(';')[0].lower()  # e.g. 'v2', 'cp', 't3'
        meas = np.asarray(c[-2]).ravel()
        errs = np.asarray(c[-1]).ravel()
        n = int(np.sum(np.isfinite(meas) & np.isfinite(errs)))
        counts[kind] = counts.get(kind, 0) + n
        total += n
    counts["all"] = total
    return counts



def apply_sigma_floors(o, cp_floor_deg=None, v2_floor_abs=None, v2_floor_frac=None):
    """
    Floors uncertainties in quadrature.
      cp_floor_deg   : add this many degrees (converted to rad) to CP σ
      v2_floor_abs   : add this absolute floor to V² σ (e.g. 0.01 for 1% in V² units)
      v2_floor_frac  : add this fractional floor to V² σ (e.g. 0.03 -> 3% of V² value)

    Only uncertainties are modified; the measured values stay untouched.
    """
    for c in o._chi2Data:
        kind = c[0].split(';')[0].lower()

        # CP-like: cp, t3, icp, scp, ccp
        if cp_floor_deg is not None and kind in ('cp','t3','icp','scp','ccp'):
            print(f"\n\napplying cp floor {cp_floor_deg}\n\n")
            err = np.asarray(c[-1], dtype=float)
            c[-1] = np.hypot(err, np.deg2rad(float(cp_floor_deg)))

        # V²-like: v2, vis2, v2amp
        if kind in ('v2','vis2','v2amp') and (v2_floor_abs is not None or v2_floor_frac is not None):
            print(f"\n\napplying V2 floor: abs :{v2_floor_abs}, frac:{v2_floor_frac}\n\n")
            meas = np.asarray(c[-2], dtype=float)   # V²
            err  = np.asarray(c[-1], dtype=float)   # σ(V²)
            a = float(v2_floor_abs) if v2_floor_abs is not None else 0.0
            f = float(v2_floor_frac) if v2_floor_frac is not None else 0.0
            # quadrature of absolute and fractional components
            extra2 = (a*a) + (f*np.abs(meas))**2
            c[-1] = np.sqrt(err*err + extra2)

# ---------------- main ----------------

def main():
    print('here')
    p = argparse.ArgumentParser(description="CANDID binary companion search with UD lock, bootstrap, remove/refit, and detection limits.")
    # paths & inputs
    p.add_argument("--paths_json", type=str, default="/home/rtc/Documents/long_secondary_periods/paths.json")
    p.add_argument("--comp_loc", type=str, default="ANU")
    p.add_argument("--ins", type=str, default="pionier")
    p.add_argument("--wavemin", type=float, default=None, help="min wavelength [um] (for UD lookup)")
    p.add_argument("--wavemax", type=float, default=None, help="max wavelength [um] (for UD lookup)")
    p.add_argument("--ud_csv", type=str, default=None, help="path to UD_fit.csv (to lock diam* and/or rmin>=UD/2)")
    p.add_argument("--outside_UD", action="store_true", help="enforce rmin >= UD/2 if UD is provided")
    # custom path (ignores --paths_json and --comp_loc)
    p.add_argument("--custom_dir", type=str, default=None, help="custom folder holding fits files to fit the candid models, if None then we use defaul paths definned by --paths_json and --comp_loc. Default: None")
    # candid controls
    p.add_argument("--ncores", type=int, default=1, help="processes for parallel fits (default: all)")
    p.add_argument("--step", type=float, default=12.0, help="global grid step [mas] (omit/negative -> let CANDID choose)")
    p.add_argument("--rmin", type=float, default=3.0, help="inner radius [mas]")
    p.add_argument("--rmax", type=float, default=60.0, help="outer radius [mas]")
    p.add_argument("--gravity_channel", type=str, default="SPECTRO_FT",
                   help="GRAVITY stream: SPECTRO_FT (FT) or SPECTRO_SC (science - this is very slow )")

    # extras
    p.add_argument("--bootstrap", action="store_true", help="run fitBoot() for uncertainties")
    p.add_argument("--bootstrap_N", type=int, default=None, help="fitBoot N (default: Ndata/2)")
    p.add_argument("--remove_and_refit", action="store_true", help="analytically remove best companion and re-run fitMap")
    p.add_argument("--do_limit", action="store_true", help="compute detection-limit map after removal")
    p.add_argument("--limit_step", type=float, default=None, help="step [mas] for detection-limit map")

    # outputs
    p.add_argument("--out_dir", type=str, default="./candid_out")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out_dir, f"{args.ins}"), exist_ok=True)

    # ---- resolve files (mirrors your PMOIRED logic) ----
    path_dict = json.load(open(args.paths_json))
    data_root = path_dict[args.comp_loc]["data"]

    gravity_bands = {
        'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],
        'Brg':[2.136, 2.196], 'NaI':[2.198, 2.218], 'NIII':[2.237, 2.261],
        'CO2-0':[2.2934, 2.298], 'CO3-1':[2.322,2.324], 'CO4-2':[2.3525,2.3555]
    }

    ins = args.ins
    wavemin, wavemax = args.wavemin, args.wavemax

    if ins == "pionier":
        if not args.custom_dir:
            files = glob.glob(os.path.join(data_root, "pionier/data/*.fits"))
        else:
            files = glob.glob(args.custom_dir + "*.fits")

        
        if wavemin is None or wavemax is None:
            wavemin, wavemax = 1.5, 1.8

    elif ins == "gravity":
        if not args.custom_dir:
            files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))
        else:
            files = glob.glob(args.custom_dir + "*.fits")
        if wavemin is None or wavemax is None:
            wavemin, wavemax = 2.100, 2.102

    elif fnmatch.fnmatch(ins, "gravity_line_*"):
        band = ins.split("gravity_line_")[-1]
        wavemin, wavemax = gravity_bands[band]
        files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))

    elif ins in ("matisse_LM", "matisse_L", "matisse_M"):
        #files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_chopped_L/*fits"))
        if wavemin is None or wavemax is None:
            if ins == "matisse_LM":   
                wavemin, wavemax = 3.1, 4.9
                if not args.custom_dir:
                    pth_tmp = os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_chopped_L/*.fits")
                else:
                    pth_tmp = args.custom_dir + "*.fits" #glob.glob(args.custom_dir + "*.fits")
                files = glob.glob(pth_tmp)
                if not files:
                    raise UserWarning(f'no files in directory {pth_tmp}')
    
            elif ins == "matisse_L":  
                wavemin, wavemax = 3.3, 3.6
                if not args.custom_dir:
                    pth_tmp = os.path.join(data_root, "matisse_wvl_filtered_L/*.fits")
                else:
                    pth_tmp = args.custom_dir + "*.fits" #glob.glob(args.custom_dir + "*.fits")
                files = glob.glob(pth_tmp )
                if not files:
                    raise UserWarning(f'no files in directory {pth_tmp}')
    
            else:                     
                wavemin, wavemax = 4.6, 4.9
                if not args.custom_dir:
                    pth_tmp = os.path.join(data_root, "matisse_wvl_filtered_M/*.fits")
                else:
                    pth_tmp = args.custom_dir + "*.fits" #glob.glob(args.custom_dir + "*.fits")
                files = glob.glob( pth_tmp  )
                if not files:
                    raise UserWarning(f'no files in directory {pth_tmp}')
                
    elif ins == "matisse_N":
        if not args.custom_dir:
            files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
        else:
            files = glob.glob(args.custom_dir + "*.fits")
        if wavemin is None or wavemax is None:
            raise SystemExit("For matisse_N please provide --wavemin and --wavemax [um].")

    elif ins in ("matisse_N_short", "matisse_N_mid", "matisse_N_long"):
        if not args.custom_dir:
            files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
        else:
            files = glob.glob(args.custom_dir + "*.fits")
        if wavemin is None or wavemax is None:
            if ins == "matisse_N_short": wavemin, wavemax = 8.0, 9.0
            elif ins == "matisse_N_mid": wavemin, wavemax = 9.0, 10.0
            else:                        wavemin, wavemax = 10.0, 13.0

    elif fnmatch.fnmatch(ins, "matisse_N_*um"):
        wvl_bin = 0.5
        m = re.match(r"^matisse_N_([\d.]+)um$", ins)
        wmin = round(float(m.group(1)), 1)
        if not args.custom_dir:
            files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
        else:
            files = glob.glob(args.custom_dir + "*.fits")
        if wavemin is None or wavemax is None:
            wavemin, wavemax = wmin, wmin + wvl_bin

    else:
        raise SystemExit("Unknown --ins option")

    if not files:
        raise SystemExit("No OIFITS files matched your selection.")

    # ---- CANDID config ----
    candid.CONFIG['Ncores'] = args.ncores           # None -> all
    candid.CONFIG['long exec warning'] = None       # both spellings for safety
    candid.CONFIG['longExecWarning']  = None

    
    # ---- Open & set observables/instrument ----
    if "gravity" not in ins.lower(): 
        o = candid.Open(files, largeCP=True)
    else: # we do FT data to be quicker 
        o = candid.Open(files, largeCP=True,  instruments = ['GRAVITY_SC_P1']) #,reducePoly=3) #,reducePoly=2)
        #candid.CONFIG['Nsmear'] = 1
    o.observables = ['v2','cp']                     # recommended for PIONIER/GRAVITY

    #print("Estimating covariance in data")
    #o.estimateCorrSpecChannels(verbose=True)


    #CP_FLOOR_DEG = 1  # try 0.3–1.0 deg
    #V2_FLOOR = 0.01  # try 0.3–1.0 deg
    #apply_sigma_floors(o, cp_floor_deg=CP_FLOOR_DEG, v2_floor_abs=0.01, v2_floor_frac=None)
    
    # CP triangles can be to constraining especially due to poor NS UV coverage , clip it! 
    # CP_FLOOR_DEG = 5  # try 0.3–1.0 deg
    # for c in o._chi2Data:
    #     if c[0].startswith('cp;'):
    #         c[-1] = np.hypot(c[-1], np.deg2rad(CP_FLOOR_DEG))

    # if "gravity" in ins.lower():
    #     o.instruments = [args.gravity_channel]      # 'SPECTRO_FT' or 'SPECTRO_SC'

    ########################################################################################
    #### PLOTTING V2 vs PA for different baselines 
    ########################################################################################
    ########################################################################################
    # 1) All baselines
    fig = plot_v2_vs_pa(o, bins=36)
    fig.savefig(os.path.join(args.out_dir, f"{args.ins}", "v2_vs_pa_all.png"), dpi=180)

    # 2) Only the longest 30% of baselines (unit-agnostic)
    fig = plot_v2_vs_pa(o, bins=36, q_long=30)
    fig.savefig(os.path.join(args.out_dir, f"{args.ins}", "v2_vs_pa_long30.png"), dpi=180)

    # 3) Absolute threshold if you know the units of u,v (same units in L)
    fig = plot_v2_vs_pa(o, bins=36, Lmin=80.0)  # example value
    fig.savefig(os.path.join(args.out_dir, f"{args.ins}", "v2_vs_pa_Lmin80.png"), dpi=180)

    fig = plot_L_vs_pa(o, bins=36)

    if fig is not None:
        fig.savefig(os.path.join(args.out_dir, f"{args.ins}",  "L_vs_PA.png"), dpi=180)
        plt.close(fig)

    # 1) V² vs PA: all baselines
    fig = plot_v2_vs_pa(o, bins=36)
    if fig is not None:
        fig.savefig(os.path.join(args.out_dir, f"{args.ins}",  "V2_vs_PA_all.png"), dpi=180)
        plt.close(fig)

    # 2) V² vs PA: top 30% longest baselines (unit-agnostic)
    fig = plot_v2_vs_pa(o, bins=36, q_long=30)
    if fig is not None:
        fig.savefig(os.path.join(args.out_dir, f"{args.ins}",  "V2_vs_PA_top30pct.png"), dpi=180)
        plt.close(fig)

    # 3) V² vs PA: absolute baseline threshold (set Lmin to taste)
    Lmin_thresh = 80.0  # <-- adjust if needed (same units as your u,v)
    fig = plot_v2_vs_pa(o, bins=36, Lmin=Lmin_thresh)
    if fig is not None:
        fig.savefig(os.path.join(args.out_dir, f"{args.ins}",  f"V2_vs_PA_Lmin{Lmin_thresh:g}.png"), dpi=180)
        plt.close(fig)

    # 4) V² vs PA in a narrow |u,v| annulus
    #    Pick a representative L0 automatically from your data (e.g., 80th percentile),
    #    then keep ±10% around it.
    _, _, _, L_all = extract_v2_vs_pa(o)   # returns PA, V2, σ, L
    if L_all.size:
        L0 = float(np.nanpercentile(L_all, 80))
        fig = plot_v2_vs_pa_annulus(o, L0=L0, frac_width=0.10, bins=24)
        if fig is not None:
            fig.savefig(os.path.join(args.out_dir, f"{args.ins}",  f"V2_vs_PA_annulus_L0_{L0:.1f}.png"), dpi=180)
            plt.close(fig)

    # pick L0 automatically from your dataset (e.g. 80th percentile)
    _, _, _, L_all = extract_v2_vs_pa(o)
    if L_all.size:
        L0 = float(np.nanpercentile(L_all, 80))  # same annulus center you used
        (popt, perr), boot_B = fit_and_plot_v2_cos2(
            o, L0=L0, frac_width=0.10, bins=12,
            outpath=os.path.join(args.out_dir, f"{args.ins}", f"V2_PA_cos2fit_L0_{L0:.1f}.png"),
            title="V² vs PA in |u,v| annulus with cos(2·) fit",
            do_bootstrap=True, Nboot=800
        )
        if popt is not None:
            A, B, PA0 = popt
            print(f"[cos2] A={A:.4f} ± {perr[0]:.4f},  B={B:.4f} ± {perr[1]:.4f},  PA0={PA0:.2f}° ± {perr[2]:.2f}°")
            if boot_B is not None and np.isfinite(boot_B).any():
                b_med = np.nanmedian(boot_B); b_lo = np.nanpercentile(boot_B,16); b_hi = np.nanpercentile(boot_B,84)
                print(f"[cos2] Bootstrap B: median={b_med:.4f},  16–84% = [{b_lo:.4f}, {b_hi:.4f}]")
    else:
        print("[cos2] Could not determine L0 (no V² points).")
        
    out_dir = os.path.join(args.out_dir, f"{args.ins}")
    run_cos2_across_annuli(o, out_dir,
                        L0_list=None,           # or e.g. [60.0, 80.0, 100.0]
                        quantiles=(60,75,85,92),
                        frac_width=0.10, bins=12)


    # Use the same L0_list you used for the V² cos2 fits
    L0_list = _choose_annuli_from_quantiles(o, quantiles=(60,75,85,92))
    plot_cp_vs_pa_annulus(o, out_dir=os.path.join(args.out_dir, f"{args.ins}"),
                        L0_list=L0_list, frac_width=0.10, bins=12)

    # # 5) Residual V² vs PA after dividing by a circular-UD model
    # #    Use diam* from bestFit if available, else fall back to UD_fit.csv.
    # theta_mas = np.nan
    # if getattr(o, "bestFit", None):
    #     theta_mas = float(o.bestFit.get("best", {}).get("diam*", np.nan))
    # if not np.isfinite(theta_mas) and args.ud_csv:
    #     try:
    #         theta_mas = float(pick_ud_at_wvl(args.ud_csv, wavemin, wavemax))
    #     except Exception as e:
    #         print(f"[UDresid] Could not read UD from CSV: {e}")

    # if np.isfinite(theta_mas):
    #     fig = plot_v2_residual_vs_pa(o, theta_mas=theta_mas, bins=36, Lmin=None)
    #     if fig is not None:
    #         fig.savefig(os.path.join(save_dir, f"V2_residual_vs_PA_UDtheta_{theta_mas:.3f}mas.png"), dpi=180)
    #         plt.close(fig)
    # else:
    #     print("[UDresid] No theta_mas available (neither bestFit['diam*'] nor ud_csv). Skipping residual plot.")
        
    ########################################################################################
    ########################################################################################


    # ---- Build global fit kwargs (avoid passing None) ----
    fit_kwargs = {"fig": 1}

    if args.step is not None and args.step > 0:
        fit_kwargs["step"] = float(args.step)

    # UD lock and rmin outside-UD
    addParam = {}
    doNotFit = []
    ud_locked = None
    if args.ud_csv:
        try:
            ud_locked = pick_ud_at_wvl(args.ud_csv, wavemin, wavemax)  # [mas]
            print(f"[INFO] UD from CSV at band midpoint: {ud_locked:.3f} mas")
            # sanity check: detect bad units/index
            if not (0.05 <= ud_locked <= 50.0):
                raise ValueError(f"UD={ud_locked:.2f} mas looks off; check CSV units/index.")
            addParam['diam*'] = float(ud_locked)
            #doNotFit.append('diam*')
            print(f"[INFO] Locking diam* = {ud_locked:.3f} mas for global grid.")
        except Exception as e:
            print(f"[WARN] UD lock skipped: {e}")

    rmin_eff = args.rmin
    if args.outside_UD and ud_locked is not None:
        rmin_eff = max(float(args.rmin), 0.5*ud_locked)
        print(f"[INFO] Enforcing outside-UD: rmin = max({args.rmin:.2f}, UD/2={0.5*ud_locked:.2f}) = {rmin_eff:.2f} mas")

    if rmin_eff is not None:      fit_kwargs["rmin"] = float(rmin_eff)
    if args.rmax is not None:     fit_kwargs["rmax"] = float(args.rmax)
    if addParam:                  fit_kwargs["addParam"] = addParam
    if doNotFit:                  fit_kwargs["doNotFit"] = doNotFit


    # ---- Global fitMap ----
    print("\n[STEP 1] Global fit-map kwargs:", {k:v for k,v in fit_kwargs.items() if k!='fig'})
    o.fitMap(**fit_kwargs)
    savefig(1, os.path.join(args.out_dir,f"{args.ins}", f"candid_fitmap_{ins}.png"))

    # getting some infos
    global_fit_quality = _extract_chi2_info(getattr(o, "bestFit", {}))
    data_counts = _count_points_by_obs(o)  # counts BEFORE any refit/masking

    if getattr(o, 'bestFit', None) is None:
        raise SystemExit("fitMap did not complete (bestFit is None). Try larger step and/or smaller rmax.")

    best  = o.bestFit.get("best", {})
    uncer = o.bestFit.get("uncer", {})

    best_global = o.bestFit.get("best", {}).copy()
    uncer_global = o.bestFit.get("uncer", {}).copy()

    print("\n[RESULT] Global best:")
    for k, unit in (("x","mas"),("y","mas"),("f","% primary"),("diam*","mas")):
        if k in best:
            e = uncer.get(k)
            line = f"  {k:6s} = {best[k]:.6g}"
            if e is not None: line += f" ± {e:.6g}"
            if unit:          line += f" [{unit}]"
            print(line)


    plot_cp_vs_pa(o, bins=36)
    plt.savefig(os.path.join(args.out_dir, f"{args.ins}", "cp_vs_pa.png"), dpi=180)

    
    pa_bin_deg = 40
    fit_kwargs_quick = dict(fit_kwargs)
    fit_kwargs_quick["step"] = max( fit_kwargs.get("step", 8.0), 8.0 )  # >=8 mas grid for speed

    pa_results = fit_in_pa_bins(o, fit_kwargs_quick, pa_bin_deg=pa_bin_deg) #,sigma_inflate=1e9)
    print( pa_results )
    fig = plot_pa_binned_results(pa_results)
    fig.savefig(os.path.join(args.out_dir, f"{args.ins}", f"fit_vs_PA_bins_{int(pa_bin_deg)}deg.png"), dpi=180)





    # ---- Optional bootstrap (lock diam* here as well to prevent drift) ----
    if args.bootstrap:
        boot_kwargs = {"fig": 2}
        #if addParam:  boot_kwargs["addParam"] = addParam
        if doNotFit:  boot_kwargs["doNotFit"] = doNotFit
        if args.bootstrap_N is not None:
            boot_kwargs["N"] = int(args.bootstrap_N)

        print("\n[STEP 2] fitBoot kwargs:", {k:v for k,v in boot_kwargs.items() if k!='fig'})
        try:
            o.fitBoot(**boot_kwargs)
        except ValueError as e:
            # Some CANDID versions have a harmless printf bug; continue
            print(f"[WARN] fitBoot raised ValueError while printing: {e}. Continuing.")
        savefig(2, os.path.join(args.out_dir,f"{args.ins}", f"candid_bootstrap_{ins}.png"))

    # ---- Optional remove-and-refit ----
    post_best = None
    post_fit_quality = None 
    if args.remove_and_refit:
        print("\n[STEP 3] Remove best companion analytically and re-run fitMap")
        # best = o.bestFit['best'] from the first fitMap
        ud_locked = best.get('diam*', None)   # or use a trusted UD value if you prefer

        # make sure you reuse *all* your global settings:
        refit_kwargs = dict(fit_kwargs)   # step, rmin, rmax, etc.
        refit_kwargs.update({
            "fig": 3,                       # or None if you don’t want the plot now
            "removeCompanion": best,        # analytical subtraction
        })

        # If the UD fit tended to diverge, lock it for the post-removal run:
        if ud_locked is not None and np.isfinite(ud_locked):
            refit_kwargs["addParam"] = {"diam*": float(ud_locked)}
            refit_kwargs["doNotFit"] = ["diam*"]

        # IMPORTANT: use refit_kwargs here (not fit_kwargs)
        o.fitMap(**refit_kwargs)
        savefig(refit_kwargs['fig'], os.path.join(args.out_dir, f"{args.ins}", f"candid_refit_removed_{ins}.png"))
        
        post_fit_quality = _extract_chi2_info(getattr(o, "bestFit", {}))
        # # reuse safe bounds
        # if rmin_eff is not None:      refit_kwargs["rmin"] = float(rmin_eff)
        # if "step" in fit_kwargs: refit_kwargs["step"] = fit_kwargs["step"]
        # if "rmin" in fit_kwargs: refit_kwargs["rmin"] = fit_kwargs["rmin"]
        # if "rmax" in fit_kwargs: refit_kwargs["rmax"] = fit_kwargs["rmax"]
        # addParam['diam*'] = float(ud_locked)
        # doNotFit.append('diam*')
        # DON'T lock diam* in the post-removal run unless you really want to
        #o.fitMap(**refit_kwargs)
        
        if getattr(o, 'bestFit', None) is None:
            print("[WARN] Post-removal fitMap did not complete; skipping post-best.")
        else:
            post_best = o.bestFit.get("best", {})
            print("\n[RESULT] Post-removal best:")
            for k, unit in (("x","mas"),("y","mas"),("f","% primary"),("diam*","mas")):
                if k in post_best:
                    print(f"  {k:6s} = {post_best[k]:.6g} [{unit}]")

    # ---- detection limits (after removal) ----
    if args.do_limit:
        print("\n[STEP 4] Detection-limit map (after analytic removal)")

        # Set this True for a quick preview (coarser/less accurate, faster)
        FAST_PREVIEW = False

        # Bandwidth-smearing sampling (smaller = faster)
        try:
            candid.CONFIG['Nsmear'] = 2 if FAST_PREVIEW else max(int(candid.CONFIG.get('Nsmear', 3)), 5)
            print(f"[DL] Using Nsmear={candid.CONFIG['Nsmear']}")
        except Exception:
            pass

        # Prefer the global-best (pre-refit) to define the removal
        rm_src = best_global if 'best_global' in locals() and best_global else (
            o.bestFit.get('best', {}) if getattr(o, 'bestFit', None) else {}
        )

        # Build removeCompanion dict (x,y,f[,diam*]) if possible
        rm = None
        have_xyz = all(k in rm_src and np.isfinite(rm_src[k]) for k in ('x', 'y', 'f'))
        if have_xyz:
            rm = {'x': float(rm_src['x']), 'y': float(rm_src['y']), 'f': float(rm_src['f'])}
            # Try to carry a sensible diameter for some CANDID builds
            diam_rm = None
            if 'diam*' in rm_src and np.isfinite(rm_src['diam*']):
                diam_rm = float(rm_src['diam*'])
            elif 'ud_locked' in locals() and ud_locked is not None:
                diam_rm = float(ud_locked)
            elif args.ud_csv and (args.wavemin is not None and args.wavemax is not None):
                try:
                    diam_rm = float(pick_ud_at_wvl(args.ud_csv, args.wavemin, args.wavemax))
                except Exception:
                    diam_rm = None
            if diam_rm is not None:
                rm['diam*'] = diam_rm
        else:
            print("[DL] No valid companion to remove; computing limits on original data.")

        # Reuse global bounds (fall back to safe defaults)
        dl_rmin = float(fit_kwargs.get('rmin', rmin_eff if rmin_eff is not None else 2.0))
        dl_rmax = float(fit_kwargs.get('rmax', args.rmax if args.rmax is not None else 40.0))
        if dl_rmax <= dl_rmin:
            dl_rmax = dl_rmin + 20.0

        # Choose step: CLI > smallestScale > default; coarser if FAST_PREVIEW
        s0 = getattr(o, "smallestScale", None)
        step0 = (float(args.limit_step) if args.limit_step is not None
                else (float(s0) if s0 is not None and np.isfinite(s0) else 2.0))
        if FAST_PREVIEW and step0 < 6.0:
            step0 = 6.0

        out_png = os.path.join(args.out_dir, f"{args.ins}", f"candid_detlimit_{ins}.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        max_tries, grow = 4, 1.7
        step = float(step0)
        last_err = None

        def _prep_dl_kwargs(step, rm):
            kw = {"fig": 4, "step": float(step), "rmin": float(dl_rmin), "rmax": float(dl_rmax)}
            if rm is not None:
                kw["removeCompanion"] = rm
            print("[DL] kwargs -> detectionLimit:",
                {k: kw[k] for k in ("step", "rmin", "rmax")},
                " removeCompanion=",
                {k: kw["removeCompanion"][k] for k in ("x", "y", "f")}
                if "removeCompanion" in kw else None,
                " diam*=", (kw["removeCompanion"].get("diam*") if "removeCompanion" in kw else None))
            return kw

        # Monkey-patch to prove what detectionLimit actually receives (remove later if noisy)
        _orig_dl = o.detectionLimit
        def _logged_dl(*a, **kw):
            print("[DL] detectionLimit received:",
                {k: kw.get(k) for k in ("step", "rmin", "rmax")},
                " removeCompanion keys=", list((kw.get("removeCompanion") or {}).keys()))
            return _orig_dl(*a, **kw)
        o.detectionLimit = _logged_dl

        for attempt in range(1, max_tries + 1):
            try:
                dl_kwargs = _prep_dl_kwargs(step, rm)

                # Seed internal bounds for older/quirky builds that ignore dl_kwargs on the first call
                if attempt == 1:
                    try:
                        _seed_add = {}
                        if rm and 'diam*' in rm:
                            _seed_add['diam*'] = float(rm['diam*'])
                        if _seed_add:
                            o.fitMap(fig=None, step=float(step), rmin=float(dl_rmin), rmax=float(dl_rmax),
                                    addParam=_seed_add, doNotFit=['diam*'])
                        else:
                            o.fitMap(fig=None, step=float(step), rmin=float(dl_rmin), rmax=float(dl_rmax))
                    except Exception as _e:
                        print(f"[DL] (non-fatal) seed fitMap failed: {_e}")

                print(f"[DL] Attempt {attempt}/{max_tries} with step={step:.3f} mas, "
                    f"rmin={dl_rmin:.2f}, rmax={dl_rmax:.2f}")
                #dl_kwargs['fig'] = 4
                o.detectionLimit(**dl_kwargs)

                #plt.figure(4)
                # try:
                #     plt.tight_layout()
                # except Exception:
                #     pass
                savefig(dl_kwargs['fig'], out_png)
                #plt.savefig(out_png, dpi=180)
                print(f"[SAVE] fig{dl_kwargs['fig']} as {out_png}")
                break

            except KeyError as e:
                # Some versions require 'diam*' in removeCompanion
                if str(e) == "'diam*'" and rm is not None and 'diam*' not in rm:
                    if 'ud_locked' in locals() and ud_locked is not None:
                        rm['diam*'] = float(ud_locked)
                        print("[DL] Injected diam* from UD lock and retrying…")
                        continue
                last_err = e
                print(f"[DL] detectionLimit failed (KeyError): {e} -> coarsen step…")
                step *= grow

            except Exception as e:
                last_err = e
                print(f"[DL] detectionLimit failed: {e} -> coarsen step…")
                step *= grow
        else:
            raise RuntimeError(f"detectionLimit failed after {max_tries} attempts; last error: {last_err}")

        detlim_meta = {
            "requested_step": float(step0),
            "final_step_used": float(step),
            "rmin": float(dl_rmin),
            "rmax": float(dl_rmax),
            "Nsmear": int(candid.CONFIG.get("Nsmear", -1)),
            "fast_preview": bool(FAST_PREVIEW),
            "removed_companion": rm,  # may include diam* if available
        }

    # ---- Save JSON summary ----

    f_snr = None
    if "f" in best and "f" in uncer and np.isfinite(uncer.get("f", np.nan)) and uncer["f"] > 0:
        f_snr = float(best["f"]) / float(uncer["f"])

    # record instruments/observables used
    obs_used = list(getattr(o, "observables", []))
    ins_used = list(getattr(o, "instruments", [])) if hasattr(o, "instruments") else None

    out_json = {
        "instrument": ins,
        "files": files,
        "wavemin_um": wavemin, "wavemax_um": wavemax,

        # what we asked CANDID to do globally
        "global_fit_kwargs": {k: v for k, v in fit_kwargs.items() if k != 'fig'},
        "observables": obs_used,
        "instruments": ins_used,
        "config": {
            "Ncores": candid.CONFIG.get("Ncores", None),
            "Nsmear": candid.CONFIG.get("Nsmear", None),
            "candid_version": getattr(candid, "__version__", None),
        },

        # best-fit (global) solution + uncertainties
        "global_best": best,
        "global_uncer": uncer,
        "global_fit_quality": global_fit_quality,   # <-- NEW
        "data_counts": data_counts,                 # <-- NEW
        "flux_ratio_snr": f_snr,                    # <-- NEW (percent / percent-err)

        # optional post-removal refit
        "post_remove_best": post_best,
        "post_remove_fit_quality": (post_fit_quality if 'post_fit_quality' in locals() else None),

        # detection-limit run summary (no map numbers, just settings used)
        "detlimit_meta": (detlim_meta if 'detlim_meta' in locals() else None),

        # other toggles the run used
        "extras": {
            "bootstrap": args.bootstrap, "bootstrap_N": args.bootstrap_N,
            "remove_and_refit": args.remove_and_refit,
            "do_limit": args.do_limit, "limit_step": args.limit_step
        }
    }
    # old one 
    # out_json = {
    #     "instrument": ins,
    #     "files": files,
    #     "wavemin_um": wavemin, "wavemax_um": wavemax,
    #     "global_fit_kwargs": {k: v for k, v in fit_kwargs.items() if k != 'fig'},
    #     "global_best": best, "global_uncer": uncer,
    #     "post_remove_best": post_best,
    #     "extras": {
    #         "bootstrap": args.bootstrap, "bootstrap_N": args.bootstrap_N,
    #         "remove_and_refit": args.remove_and_refit,
    #         "do_limit": args.do_limit, "limit_step": args.limit_step
    #     }
    # }
    out_js = os.path.join(args.out_dir,f"{args.ins}", f"candid_summary_{ins}.json")

    with open(os.path.join(args.out_dir,f"{args.ins}", f"candid_bestfit_{ins}.json"), "w") as f:
        json.dump(out_json, f, indent=2, default=np_json_default)
    # with open(out_js, "w") as f:
    #     json.dump(out_json, f, indent=2)
    print(f"\n[SAVE] {out_js}")

if __name__ == "__main__":
    main()





#python candid_fit.py --ins matisse_M --ud_csv /home/rtc/Documents/long_secondary_periods/data/UD_fit.csv --outside_UD --step 0.52 --rmin 2.1 --rmax 10 --ncores 1 --bootstrap --bootstrap_N 1000  --out_dir ./candid_out

# """
# python candid_fit.py \
#   --ins pionier \
#   --ud_csv /home/rtc/Documents/long_secondary_periods/data/UD_fit.csv \
#   --outside_UD \
#   --step 12 --rmin 3 --rmax 60 \
#   --ncores 8 \
#   --bootstrap --bootstrap_N 800 \
#   --remove_and_refit \
#   --do_limit --limit_step 2.0 \
#   --out_dir ./candid_out

#   python candid_fit.py   --ins pionier   --ud_csv /home/rtc/Documents/long_secondary_periods/data/UD_fit.csv   --outside_UD   --step 2 --rmin 2 --rmax 60   --ncores 
# 8   --bootstrap --bootstrap_N 800   --remove_and_refit   --do_limit --limit_step 2.0   --out_dir ./candid_out


# """

# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

# # """
# # CANDID binary search (coarse grid; optional UD lock)

# # - Loads OIFITS based on your paths.json & --ins
# # - Runs candid.fitMap() with a bounded/coarse grid
# # - Optionally locks diam* to UD from UD_fit.csv during the global grid
# # - Saves fit-map PNG and best-fit JSON
# # """

# # import argparse, glob, json, os, re, fnmatch
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import candid  # install from GitHub: pip install "git+https://github.com/amerand/CANDID.git#egg=candid"

# # def pick_ud_at_wvl(ud_csv_path: str, wavemin_um: float, wavemax_um: float) -> float:
# #     """Return UD [mas] at midpoint wavelength from your UD_fit.csv (index in meters; column 'ud_mean' in mas)."""
# #     ud = pd.read_csv(ud_csv_path, index_col=0)
# #     wvl0_um = 0.5 * (wavemin_um + wavemax_um)
# #     idx = (ud.index - (1e-6 * wvl0_um)).abs().argmin()
# #     return float(ud['ud_mean'].iloc[idx])

# # def main():
# #     p = argparse.ArgumentParser(description="Run a binary companion search with CANDID.")
# #     # inputs mirroring your PMOIRED script
# #     p.add_argument("--paths_json", type=str, default="/home/rtc/Documents/long_secondary_periods/paths.json")
# #     p.add_argument("--comp_loc", type=str, default="ANU")
# #     p.add_argument("--ins", type=str, default="pionier")
# #     p.add_argument("--wavemin", type=float, default=None, help="min wavelength [um] (only for UD lookup)")
# #     p.add_argument("--wavemax", type=float, default=None, help="max wavelength [um] (only for UD lookup)")
# #     p.add_argument("--ud_csv", type=str, default=None, help="path to UD_fit.csv to lock diam* during global grid")
# #     p.add_argument("--outside_UD", action="store_true", help="set rmin >= UD/2 if UD is provided")
# #     # CANDID controls
# #     p.add_argument("--ncores", type=int, default=None, help="processes for parallel fits (default: all)")
# #     p.add_argument("--step", type=float, default=12.0, help="grid step [mas] (coarse start)")
# #     p.add_argument("--rmin", type=float, default=3.0, help="inner radius [mas]")
# #     p.add_argument("--rmax", type=float, default=60.0, help="outer radius [mas]")
# #     p.add_argument("--gravity_channel", type=str, default="SPECTRO_FT", help="GRAVITY stream: SPECTRO_FT or SPECTRO_SC")
# #     # outputs
# #     p.add_argument("--out_dir", type=str, default="./candid_out")
# #     args = p.parse_args()

# #     os.makedirs(args.out_dir, exist_ok=True)

# #     # ---- resolve files from paths.json
# #     path_dict = json.load(open(args.paths_json))
# #     data_root = path_dict[args.comp_loc]["data"]

# #     gravity_bands = {
# #         'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],
# #         'Brg':[2.136, 2.196], 'NaI':[2.198, 2.218], 'NIII':[2.237, 2.261],
# #         'CO2-0':[2.2934, 2.298], 'CO3-1':[2.322,2.324], 'CO4-2':[2.3525,2.3555]
# #     }

# #     ins = args.ins
# #     wavemin, wavemax = args.wavemin, args.wavemax

# #     if ins == "pionier":
# #         files = glob.glob(os.path.join(data_root, "pionier/data/*.fits"))
# #         if wavemin is None or wavemax is None:
# #             wavemin, wavemax = 1.5, 1.8

# #     elif ins == "gravity":
# #         files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))
# #         if wavemin is None or wavemax is None:
# #             wavemin, wavemax = 2.100, 2.102

# #     elif fnmatch.fnmatch(ins, "gravity_line_*"):
# #         band = ins.split("gravity_line_")[-1]
# #         wavemin, wavemax = gravity_bands[band]
# #         files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))

# #     elif ins in ("matisse_LM", "matisse_L", "matisse_M"):
# #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_chopped_L/*fits"))
# #         if wavemin is None or wavemax is None:
# #             if ins == "matisse_LM":
# #                 wavemin, wavemax = 3.1, 4.9
# #             elif ins == "matisse_L":
# #                 wavemin, wavemax = 3.3, 3.6
# #             else:
# #                 wavemin, wavemax = 4.6, 4.9

# #     elif ins == "matisse_N":
# #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# #         if wavemin is None or wavemax is None:
# #             raise SystemExit("For matisse_N please provide --wavemin and --wavemax [um].")

# #     elif ins in ("matisse_N_short", "matisse_N_mid", "matisse_N_long"):
# #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# #         if wavemin is None or wavemax is None:
# #             if ins == "matisse_N_short":
# #                 wavemin, wavemax = 8.0, 9.0
# #             elif ins == "matisse_N_mid":
# #                 wavemin, wavemax = 9.0, 10.0
# #             else:
# #                 wavemin, wavemax = 10.0, 13.0

# #     elif fnmatch.fnmatch(ins, "matisse_N_*um"):
# #         wvl_bin = 0.5
# #         m = re.match(r"^matisse_N_([\d.]+)um$", ins)
# #         wmin = round(float(m.group(1)), 1)
# #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# #         if wavemin is None or wavemax is None:
# #             wavemin, wavemax = wmin, wmin + wvl_bin

# #     else:
# #         raise SystemExit("Unknown --ins option")

# #     if not files:
# #         raise SystemExit("No OIFITS files matched your selection.")

# #     # ---- CANDID config
# #     candid.CONFIG['Ncores'] = args.ncores        # None = all cores
# #     candid.CONFIG['long exec warning'] = None    # don't abort early

# #     # ---- Open & choose observables
# #     o = candid.Open(files)
# #     o.observables = ['v2','cp']                  # PIONIER best practice
# #     if "gravity" in ins:
# #         o.instruments = [args.gravity_channel]   # 'SPECTRO_FT' or 'SPECTRO_SC'

# #     # ---- Build kwargs for fitMap without passing None
# #     fit_kwargs = {"fig": 1}
# #     if args.step is not None:  fit_kwargs["step"] = float(args.step)
# #     # rmin possibly boosted by UD/2
# #     rmin = args.rmin
# #     if args.outside_UD and args.ud_csv:
# #         try:
# #             ud = pick_ud_at_wvl(args.ud_csv, wavemin, wavemax)
# #             rmin = max(rmin, 0.5*ud)
# #             print(f"[INFO] Enforcing outside-UD: rmin := max({args.rmin:.2f}, UD/2={0.5*ud:.2f}) -> {rmin:.2f} mas")
# #         except Exception as e:
# #             print(f"[WARN] Could not read UD from {args.ud_csv}: {e}")
# #     if rmin is not None:       fit_kwargs["rmin"] = float(rmin)
# #     if args.rmax is not None:  fit_kwargs["rmax"] = float(args.rmax)

# #     # Optionally lock diam* using UD from CSV
# #     addParam = {}
# #     doNotFit = []
# #     if args.ud_csv:
# #         try:
# #             ud = pick_ud_at_wvl(args.ud_csv, wavemin, wavemax)
# #             addParam['diam*'] = float(ud)
# #             doNotFit.append('diam*')
# #             print(f"[INFO] Locking diam* to UD={ud:.3f} mas for global grid.")
# #         except Exception as e:
# #             print(f"[WARN] UD lock skipped ({e})")

# #     # IMPORTANT: only pass these kwargs if they exist (avoid NoneType error)
# #     if addParam:
# #         fit_kwargs["addParam"] = addParam
# #     if doNotFit:
# #         fit_kwargs["doNotFit"] = doNotFit

# #     # ---- Run fitMap
# #     print("\n[STEP] Global fit-map with kwargs:", {k:v for k,v in fit_kwargs.items() if k!='addParam'})
# #     o.fitMap(**fit_kwargs)

# #     # ---- Save figure
# #     plt.figure(1); plt.tight_layout()
# #     out_png = os.path.join(args.out_dir, f"candid_fitmap_{ins}.png")
# #     try:
# #         plt.savefig(out_png, dpi=180)
# #         print(f"[SAVE] {out_png}")
# #     except Exception as e:
# #         print(f"[WARN] Could not save figure: {e}")

# #     # ---- Results
# #     if getattr(o, 'bestFit', None) is None:
# #         raise SystemExit("fitMap did not complete (bestFit is None). Try larger step / smaller rmax.")

# #     best = o.bestFit.get("best", {})
# #     uncer = o.bestFit.get("uncer", {})
# #     print("\nBEST BINARY FIT (CANDID):")
# #     for k, unit in (("x","mas"),("y","mas"),("f","% primary"),("diam*","mas")):
# #         if k in best:
# #             err = uncer.get(k, None)
# #             sval = f"{best[k]:.6g}"
# #             s = f"  {k:6s} = {sval}"
# #             if err is not None:
# #                 s += f" ± {err:.6g}"
# #             if unit:
# #                 s += f" [{unit}]"
# #             print(s)

# #     # ---- Save JSON
# #     out_json = {
# #         "instrument": ins,
# #         "files": files,
# #         "wavemin_um": wavemin, "wavemax_um": wavemax,
# #         "fit_kwargs": {k:v for k,v in fit_kwargs.items() if k!='fig'},
# #         "best": best,
# #         "uncer": uncer
# #     }
# #     out_js = os.path.join(args.out_dir, f"candid_bestfit_{ins}.json")
# #     with open(out_js, "w") as f:
# #         json.dump(out_json, f, indent=2)
# #     print(f"\n[SAVE] {out_js}")

# # if __name__ == "__main__":
# #     main()

# # """
# # python candid_fit.py \
# #   --ins pionier \
# #   --ud_csv /home/rtc/Documents/long_secondary_periods/data/UD_fit.csv \
# #   --outside_UD \
# #   --step 12 --rmin 2 --rmax 60 \
# #   --ncores 8 \
# #   --out_dir ./candid_out
# # """
# # # #!/usr/bin/env python3
# # # # -*- coding: utf-8 -*-

# # # """
# # # Binary fit with CANDID (Companion Analysis and Non-Detection in Interferometric Data)

# # # - Opens your OIFITS files (same selection logic as your PMOIRED script)
# # # - Runs a CANDID fit-map (grid of binary fits) to search for a companion
# # # - Saves a figure of the fit map
# # # - Optionally computes & saves a detection-limit map after removing the found companion

# # # Units (CANDID defaults):
# # # - positions (x,y): mas
# # # - flux ratio f: percent (%)

# # # Refs:
# # # - API usage: candid.Open(...), .fitMap(...), .detectionLimit(...); best fit in o.bestFit['best'].
# # # """

# # # import argparse
# # # import glob
# # # import json
# # # import fnmatch
# # # import re
# # # import os
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # import candid  # pip install from https://github.com/amerand/CANDID

# # # candid.CONFIG['long exec warning'] = None   # or a big number (seconds)
# # # candid.CONFIG['Ncores'] = 8                 # or your CPU count

# # # def pick_ud_at_wvl(ud_csv_path, wavemin_um, wavemax_um):
# # #     """Return UD diameter [mas] at the midpoint wavelength (microns) from your UD_fit.csv (index in meters)."""
# # #     ud_fits = pd.read_csv(ud_csv_path, index_col=0)
# # #     wvl0_um = 0.5*(wavemin_um + wavemax_um)
# # #     # ud_fits index is in meters; convert to meters:
# # #     idx = (ud_fits.index - (1e-6 * wvl0_um)).abs().argmin()
# # #     return float(ud_fits['ud_mean'].iloc[idx])

# # # def main():
# # #     p = argparse.ArgumentParser(description="Run a binary companion search with CANDID.")
# # #     # --- carry over your CLI
# # #     p.add_argument("--ins", type=str, default="pionier")
# # #     p.add_argument("--model", type=str, default="binary")   # ignored; we always run binary search here
# # #     p.add_argument("--wavemin", type=float, default=None, help="min wavelength [um] (used only to pick UD from CSV)")
# # #     p.add_argument("--wavemax", type=float, default=None, help="max wavelength [um] (used only to pick UD from CSV)")
# # #     p.add_argument("--binning", type=int, default=None)     # CANDID handles data internally; kept for parity

# # #     # --- CANDID controls
# # #     p.add_argument("--step", type=float, default=None, help="grid step [mas]; if not set, CANDID chooses")
# # #     p.add_argument("--rmin", type=float, default=None, help="inner search radius [mas]; default = CANDID's choice")
# # #     p.add_argument("--rmax", type=float, default=None, help="outer search radius [mas]; default = WL-smearing FoV")
# # #     p.add_argument("--outside_UD", action="store_true", help="enforce rmin >= UD/2 (requires UD CSV)")
# # #     p.add_argument("--ud_csv", type=str, default=None, help="path to UD_fit.csv to set UD constraint")
# # #     p.add_argument("--ncores", type=int, default=None, help="N cores for multiprocessing (default: all)")
# # #     p.add_argument("--gravity_channel", type=str, default="SPECTRO_FT",
# # #                    help="GRAVITY instrument stream to use: SPECTRO_FT or SPECTRO_SC")
# # #     p.add_argument("--do_limit", action="store_true", help="compute detection-limit map after removing companion")
# # #     p.add_argument("--limit_step", type=float, default=None, help="step [mas] for detection-limit map")

# # #     # paths.json gateway (your layout)
# # #     p.add_argument("--paths_json", type=str, default="/home/rtc/Documents/long_secondary_periods/paths.json")
# # #     p.add_argument("--comp_loc", type=str, default="ANU")

# # #     # outputs
# # #     p.add_argument("--out_dir", type=str, default="./candid_out")
# # #     args = p.parse_args()

# # #     os.makedirs(args.out_dir, exist_ok=True)

# # #     # ---------- resolve input files (same patterns you use) ----------
# # #     path_dict = json.load(open(args.paths_json))
# # #     data_root = path_dict[args.comp_loc]["data"]

# # #     gravity_bands = {
# # #         'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],
# # #         'Brg':[2.136, 2.196], 'NaI':[2.198, 2.218], 'NIII':[2.237, 2.261],
# # #         'CO2-0':[2.2934, 2.298],'CO3-1':[2.322,2.324],'CO4-2':[2.3525,2.3555]
# # #     }

# # #     ins = args.ins
# # #     wavemin = args.wavemin
# # #     wavemax = args.wavemax

# # #     if ins == "pionier":
# # #         files = glob.glob(os.path.join(data_root, "pionier/data/*.fits"))
# # #         if wavemin is None or wavemax is None:
# # #             wavemin, wavemax = 1.5, 1.8

# # #     elif ins == "gravity":
# # #         files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))
# # #         if wavemin is None or wavemax is None:
# # #             wavemin, wavemax = 2.100, 2.102  # narrow band as in your script

# # #     elif fnmatch.fnmatch(ins, "gravity_line_*"):
# # #         band_label = ins.split("gravity_line_")[-1]
# # #         wavemin, wavemax = gravity_bands[band_label]
# # #         files = glob.glob(os.path.join(data_root, "gravity/data/*.fits"))

# # #     elif ins in ("matisse_LM", "matisse_L", "matisse_M"):
# # #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_chopped_L/*fits"))
# # #         if wavemin is None or wavemax is None:
# # #             if ins == "matisse_LM":
# # #                 wavemin, wavemax = 3.1, 4.9
# # #             elif ins == "matisse_L":
# # #                 wavemin, wavemax = 3.3, 3.6
# # #             else:
# # #                 wavemin, wavemax = 4.6, 4.9

# # #     elif ins == "matisse_N":
# # #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# # #         if wavemin is None or wavemax is None:
# # #             raise SystemExit("For matisse_N please provide --wavemin and --wavemax (um).")

# # #     elif ins in ("matisse_N_short", "matisse_N_mid", "matisse_N_long"):
# # #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# # #         if wavemin is None or wavemax is None:
# # #             if ins == "matisse_N_short":
# # #                 wavemin, wavemax = 8.0, 9.0
# # #             elif ins == "matisse_N_mid":
# # #                 wavemin, wavemax = 9.0, 10.0
# # #             else:
# # #                 wavemin, wavemax = 10.0, 13.0

# # #     elif fnmatch.fnmatch(ins, "matisse_N_*um"):
# # #         wvl_bin = 0.5
# # #         match = re.match(r"^matisse_N_([\d.]+)um$", ins)
# # #         wmin = round(float(match.group(1)), 1)
# # #         files = glob.glob(os.path.join(data_root, "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits"))
# # #         if wavemin is None or wavemax is None:
# # #             wavemin, wavemax = wmin, wmin + wvl_bin

# # #     else:
# # #         raise SystemExit("Unknown --ins option")

# # #     if not files:
# # #         raise SystemExit("No OIFITS files matched your selection.")

# # #     # ---------- open in CANDID ----------
# # #     # Set cores BEFORE heavy calls
# # #     candid.CONFIG["Ncores"] = args.ncores  # None = all cores

# # #     o = candid.Open(files)  # multiple files OK if same instrument & single target per README
# # #     # For GRAVITY, restrict to meaningful observables/instrument stream (CP + V2; FT vs SC)
# # #     if "gravity" in ins:
# # #         o.observables = ['cp', 'v2']                 # closure phase + V2
# # #         o.instruments = [args.gravity_channel]       # 'SPECTRO_FT' (default) or 'SPECTRO_SC'

# # #     # (CANDID does not expose a documented per-wavelength channel slicer; use pre-filtered OIFITS if you need strict [wavemin,wavemax].)
# # #     o.observables = ['cp', 'v2'] 

# # #     # ---------- set search radii ----------
# # #     rmin = args.rmin
# # #     if args.outside_UD and args.ud_csv is not None:
# # #         ud_wvl_mas = pick_ud_at_wvl(args.ud_csv, wavemin, wavemax)  # UD diameter at band midpoint
# # #         rmin = max(rmin or 0.0, 0.5*ud_wvl_mas)  # enforce outside UD radius

# # #     # ---------- FIT MAP (binary search) ----------
# # #     fit_kwargs = {}
# # #     if args.step is not None: fit_kwargs["step"] = args.step
# # #     if rmin is not None:      fit_kwargs["rmin"] = rmin
# # #     if args.rmax is not None: fit_kwargs["rmax"] = args.rmax

# # #     # Let CANDID fit UD diameter and binary over a grid of starting positions
# # #     # (CANDID auto-chooses sane step & rmax if not provided)
# # #     o.fitMap(fig=1, **fit_kwargs)   # stores results in o.bestFit

# # #     # Save the figure
# # #     plt.figure(1)
# # #     plt.tight_layout()
# # #     plt.savefig(os.path.join(args.out_dir, f"candid_fitmap_{ins}.png"), dpi=180)

# # #     # Report best fit parameters
# # #     best = o.bestFit.get("best", {})
# # #     uncer = o.bestFit.get("uncer", {})
# # #     print("\nBEST BINARY FIT (CANDID):")
# # #     for k in ("x","y","f","diam*"):
# # #         if k in best:
# # #             sig = f" +- {uncer.get(k,'')}" if k in uncer else ""
# # #             print(f"  {k:6s} = {best[k]}{sig}")

# # #     # ---------- DETECTION LIMIT (optional) ----------
# # #     if args.do_limit:
# # #         # Remove detected companion analytically and compute detection limits
# # #         o.detectionLimit(fig=2,
# # #                          step=(args.limit_step if args.limit_step is not None else None),
# # #                          removeCompanion=best)  # pass dict with x,y,f,diam*

# # #         plt.figure(2)
# # #         plt.tight_layout()
# # #         plt.savefig(os.path.join(args.out_dir, f"candid_detlimit_{ins}.png"), dpi=180)

# # #     # ---------- Save best-fit JSON ----------
# # #     out_json = {
# # #         "instrument": ins,
# # #         "files": files,
# # #         "wavemin_um": wavemin, "wavemax_um": wavemax,
# # #         "fit_kwargs": fit_kwargs,
# # #         "best": best,
# # #         "uncer": uncer
# # #     }
# # #     with open(os.path.join(args.out_dir, f"candid_bestfit_{ins}.json"), "w") as f:
# # #         json.dump(out_json, f, indent=2)
# # #     print(f"\nSaved: {os.path.join(args.out_dir, f'candid_bestfit_{ins}.json')}")
# # #     print(f"Saved: {os.path.join(args.out_dir, f'candid_fitmap_{ins}.png')}")
# # #     if args.do_limit:
# # #         print(f"Saved: {os.path.join(args.out_dir, f'candid_detlimit_{ins}.png')}")

# # # if __name__ == "__main__":
# # #     main()




# # --- inside residualsOI(...) ---

# # old:
# # if 'PHI' in f:
# #     rf = lambda x: ((x + 180)%360 - 180)
# # else:
# #     rf = lambda x: x
# #
# # ...
# # tmp = rf(oi[ext[f]][k][f][mask] - m[ext[f]][k][f][mask])
# # if not ignoreErr is None:
# #     _i = i+np.arange(len(tmp))
# #     tmp /= (err[mask]*(1-ignoreErr[_i]) + ignoreErr[_i])
# #     i += len(tmp)
# # else:
# #     tmp /= err[mask]


# # # --- checking CP  ---

# # import numpy as np
# # import matplotlib.pyplot as plt

# # def _gather_cp_sigmas_rad(o, include_t3=True):
# #     """Return all CP-like sigma values (radians) flattened into one array."""
# #     kinds = {'cp', 'icp', 'scp', 'ccp'}
# #     if include_t3:
# #         kinds.add('t3')
# #     sigs = []
# #     for c in o._chi2Data:
# #         kind = c[0].split(';')[0]
# #         if kind in kinds:
# #             sigs.append(np.asarray(c[-1], dtype=float).ravel())
# #     if not sigs:
# #         return np.array([], dtype=float)
# #     arr = np.concatenate(sigs)
# #     return arr[np.isfinite(arr)]

# # # -------- collect "before" (in radians) --------
# # cp_sig_before_rad = _gather_cp_sigmas_rad(o, include_t3=False)  # set True to include T3
# # cp_sig_before_deg = np.rad2deg(cp_sig_before_rad)

# # # -------- apply your CP σ-floor (in quadrature) --------
# # CP_FLOOR_DEG = 0.6  # adjust 0.3–1.0 deg as needed
# # cp_floor_rad = np.deg2rad(CP_FLOOR_DEG)

# # for c in o._chi2Data:
# #     if c[0].split(';')[0] == 'cp':  # or use a set to include icp/scp/ccp/t3
# #         err = np.asarray(c[-1], dtype=float)
# #         c[-1] = np.hypot(err, cp_floor_rad)

# # # -------- collect "after" (in radians) --------
# # cp_sig_after_rad = _gather_cp_sigmas_rad(o, include_t3=False)
# # cp_sig_after_deg = np.rad2deg(cp_sig_after_rad)

# # # -------- quick stats --------
# # def _stats(x):
# #     return (np.nanmedian(x), np.nanpercentile(x, [16, 84]))
# # med_b, (p16_b, p84_b) = _stats(cp_sig_before_deg)
# # med_a, (p16_a, p84_a) = _stats(cp_sig_after_deg)
# # print(f"[CP σ] before: median {med_b:.3f}° (16–84%: {p16_b:.3f}–{p84_b:.3f})")
# # print(f"[CP σ]  after: median {med_a:.3f}° (16–84%: {p16_a:.3f}–{p84_a:.3f})")

# # # -------- plot histogram (degrees) --------
# # plt.figure(50, figsize=(6.5, 4.2))
# # finite_b = np.isfinite(cp_sig_before_deg)
# # finite_a = np.isfinite(cp_sig_after_deg)
# # # choose a common x-range up to the 99th percentile (after flooring)
# # xmax = np.nanpercentile(np.concatenate([cp_sig_before_deg[finite_b], cp_sig_after_deg[finite_a]]), 99) if finite_b.any() or finite_a.any() else 1.0
# # bins = np.linspace(0, xmax, 40)

# # plt.hist(cp_sig_before_deg[finite_b], bins=bins, histtype="step", linewidth=1.8, label="before floor")
# # plt.hist(cp_sig_after_deg[finite_a],  bins=bins, histtype="step", linewidth=1.8, label="after floor")

# # plt.axvline(CP_FLOOR_DEG, linestyle="--", linewidth=1.2, label=f"floor={CP_FLOOR_DEG:.2f}°")
# # plt.xlabel("Closure-phase σ (deg)")
# # plt.ylabel("Count")
# # plt.title("CP uncertainty distribution (before vs after σ-floor)")
# # plt.legend()
# # plt.tight_layout()

# # # Optional: save to your output folder
# # plt.savefig(os.path.join(args.out_dir, f"{args.ins}", "cp_sigma_hist.png"), dpi=180)
# # print("[SAVE]", os.path.join(args.out_dir, f"{args.ins}", "cp_sigma_hist.png"))



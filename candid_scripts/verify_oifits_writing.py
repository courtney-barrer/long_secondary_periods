#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
from astropy.io import fits

def _basename(path: str) -> str:
    return os.path.basename(path)

def _variant_path_for(orig_path: str, var_dir: str, suffix: str):
    """
    Map an original OIFITS path to a variant path in `var_dir`.

    We try both naming conventions:
      1) append suffix to the full basename:  base + suffix
         e.g.  file.fits + ".resid.fits"  → "file.fits.resid.fits"
      2) replace .fits extension:           root + suffix
         e.g.  file.fits  → "file" + ".resid.fits" → "file.resid.fits"
    """
    base = os.path.basename(orig_path)
    root, ext = os.path.splitext(base)

    candidates = [
        os.path.join(var_dir, base + suffix),   # append
        os.path.join(var_dir, root + suffix),   # replace .fits
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None

# def _variant_path_for(orig_path: str, var_dir: str, suffix: str):
#     """
#     Map an original OIFITS path to a variant path:
#       <var_dir>/<basename(orig)> + <suffix>
#     where <suffix> is something like ".resid.fits", ".model.fits", or ".data.fits".
#     """
#     base = _basename(orig_path)
#     cand = os.path.join(var_dir, base + suffix)
#     if os.path.exists(cand):
#         return cand
#     # Also try with no extra ".fits" if user passed suffix without the leading dot(s)
#     # but generally we want an exact suffix match to avoid false positives.
#     return None

def _read_vis2_table(hdul):
    try:
        t = hdul["OI_VIS2"].data
    except Exception:
        return None
    # Standard OIFITS keys
    have = t.columns.names
    need = ["VIS2DATA", "VIS2ERR", "FLAG"]
    for k in need:
        if k not in have:
            return None
    return t

def _read_t3_table(hdul):
    try:
        t = hdul["OI_T3"].data
    except Exception:
        return None
    have = t.columns.names
    need = ["T3PHI", "T3PHIERR", "FLAG"]
    for k in need:
        if k not in have:
            return None
    return t

def _compare_vis2(orig, var, tol_abs, tol_rel, max_print=5):
    """
    Compare VIS2DATA on unflagged channels. Returns (n_mismatch, n_checked).
    """
    n_bad = 0
    n_tot = 0
    printed = 0

    T0 = _read_vis2_table(orig)
    T1 = _read_vis2_table(var)
    if T0 is None or T1 is None:
        return (0, 0)  # no vis2 in one of the files → skip

    nrow = min(len(T0), len(T1))
    for i in range(nrow):
        v0 = np.array(T0["VIS2DATA"][i], float).ravel()
        e0 = np.array(T0["VIS2ERR"][i], float).ravel()
        f0 = np.array(T0["FLAG"][i], bool).ravel()

        v1 = np.array(T1["VIS2DATA"][i], float).ravel()
        e1 = np.array(T1["VIS2ERR"][i], float).ravel()
        f1 = np.array(T1["FLAG"][i], bool).ravel()

        nchan = min(v0.size, v1.size, f0.size, f1.size)
        if nchan == 0:
            continue

        v0 = v0[:nchan]; v1 = v1[:nchan]
        f  = ~(f0[:nchan] | f1[:nchan])  # only channels good in both
        if not np.any(f):
            continue

        dv  = np.abs(v1[f] - v0[f])
        # relative error denom
        denom = np.maximum(1e-30, np.abs(v0[f]))
        rel  = dv / denom

        bad = (dv > tol_abs) & (rel > tol_rel)
        nb  = int(np.sum(bad))
        n_bad += nb
        n_tot += int(np.sum(f))

        if nb and printed < max_print:
            idx = np.where(bad)[0][:max_print]
            for k in idx:
                print(f"    [V2 mismatch] row {i} ch {k}: orig={v0[f][k]:.6g} var={v1[f][k]:.6g} "
                      f"abs={dv[k]:.3g} rel={rel[k]:.3g}")
                printed += 1

    return (n_bad, n_tot)

def _wrap180(xdeg):
    y = (np.asarray(xdeg, float) + 180.0) % 360.0 - 180.0
    y[y <= -180.0] = 180.0
    return y

def _compare_cp(orig, var, tol_deg, max_print=5):
    """
    Compare T3PHI (deg) on unflagged channels, with wrap to [-180, 180].
    Returns (n_mismatch, n_checked).
    """
    n_bad = 0
    n_tot = 0
    printed = 0

    T0 = _read_t3_table(orig)
    T1 = _read_t3_table(var)
    if T0 is None or T1 is None:
        return (0, 0)

    nrow = min(len(T0), len(T1))
    for i in range(nrow):
        p0 = np.array(T0["T3PHI"][i], float).ravel()
        f0 = np.array(T0["FLAG"][i], bool).ravel()
        p1 = np.array(T1["T3PHI"][i], float).ravel()
        f1 = np.array(T1["FLAG"][i], bool).ravel()

        nchan = min(p0.size, p1.size, f0.size, f1.size)
        if nchan == 0:
            continue
        p0 = p0[:nchan]; p1 = p1[:nchan]
        f  = ~(f0[:nchan] | f1[:nchan])
        if not np.any(f):
            continue

        # wrap and diff on the circle
        a = _wrap180(p0[f]); b = _wrap180(p1[f])
        d = np.abs(_wrap180(b - a))
        bad = d > tol_deg
        nb  = int(np.sum(bad))
        n_bad += nb
        n_tot += int(np.sum(f))

        if nb and printed < max_print:
            idx = np.where(bad)[0][:max_print]
            for k in idx:
                print(f"    [CP mismatch] row {i} ch {k}: orig={a[k]:.3f}° var={b[k]:.3f}° |Δ|={d[k]:.3f}°")
                printed += 1

    return (n_bad, n_tot)

def verify(orig_glob, var_dir, variant_suffix, tol_v2_abs, tol_v2_rel, tol_cp_deg):
    files = sorted(glob.glob(orig_glob))
    if not files:
        print(f"No originals matched {orig_glob}")
        return

    print(f"[INFO] originals: {len(files)} files")
    print(f"[INFO] variant dir: {var_dir}")
    print(f"[INFO] required suffix for variant: '{variant_suffix}'")

    tot_v2_bad = tot_v2 = tot_cp_bad = tot_cp = 0
    matched = 0

    for f in files:
        fv = _variant_path_for(f, var_dir, variant_suffix)
        if fv is None:
            # report and continue
            print(f"  [SKIP] No match for '{_basename(f)}' with suffix '{variant_suffix}' in '{var_dir}'")
            continue

        with fits.open(f, mode="readonly") as h0, fits.open(fv, mode="readonly") as h1:
            matched += 1
            # quick per-file header sanity (same #channels in OI_WAVELENGTH):
            try:
                n0 = len(h0["OI_WAVELENGTH"].data["EFF_WAVE"])
                n1 = len(h1["OI_WAVELENGTH"].data["EFF_WAVE"])
                if n0 != n1:
                    print(f"  [WARN] nchan differ: {n0} (orig) vs {n1} (variant) for {_basename(f)}")
            except Exception:
                pass

            v2_bad, v2_tot = _compare_vis2(h0, h1, tol_v2_abs, tol_v2_rel)
            cp_bad, cp_tot = _compare_cp(h0, h1, tol_cp_deg)

            tot_v2_bad += v2_bad; tot_v2 += v2_tot
            tot_cp_bad += cp_bad;  tot_cp += cp_tot

            # Light per-file summary
            print(f"[FILE] {_basename(f)} → {_basename(fv)} | "
                  f"V2 mismatches: {v2_bad}/{v2_tot} | CP mismatches: {cp_bad}/{cp_tot}")

    print()
    print(f"[SUMMARY] matched files: {matched}")
    print(f"  total V2 mismatches beyond tol: {tot_v2_bad} / {tot_v2}")
    print(f"  total CP mismatches beyond tol: {tot_cp_bad} / {tot_cp}")

    if matched == 0:
        print("  RESULT: NO MATCHED FILES ⚠️ (check --var_dir and --variant_suffix)")
    elif tot_v2_bad == 0 and tot_cp_bad == 0:
        print("  RESULT: PASS ✅ (within tolerances)")
    else:
        print("  RESULT: FAIL ❌ (differences detected)")

def main():
    ap = argparse.ArgumentParser(description="Verify OIFITS variant files (data/model/resid) against originals.")
    ap.add_argument("--orig_glob", required=True, help="Glob for original OIFITS (e.g. /path/to/data/*.fits)")
    ap.add_argument("--var_dir", required=True, help="Directory containing variant files (e.g. .../resid)")
    ap.add_argument("--variant_suffix", default=".resid.fits",
                    help="Suffix appended to the original basename for the variant file "
                         "(e.g., .resid.fits, .model.fits, .data.fits)")
    ap.add_argument("--tol_v2_abs", type=float, default=1e-10, help="Absolute tol on V2")
    ap.add_argument("--tol_v2_rel", type=float, default=1e-7, help="Relative tol on V2")
    ap.add_argument("--tol_cp_deg", type=float, default=1e-6, help="Absolute tol on CP (deg) after wrapping")
    args = ap.parse_args()

    verify(args.orig_glob, args.var_dir, args.variant_suffix, args.tol_v2_abs, args.tol_v2_rel, args.tol_cp_deg)

if __name__ == "__main__":
    main()


# # python candid_scripts/verify_oifits_writing.py --orig_glob "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*.fits" --var_a_dir "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/data/" --variant_suffix ".resid.fits" --tol_v2_abs 1e-10 --tol_v2_rel 1e-7 --tol_cp_deg 1e-8

#python candid_scripts/verify_oifits_writing.py  --orig_glob "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*.fits" --var_dir "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/resid/" --variant_suffix ".resid.fits" --tol_v2_abs 1e-10 --tol_v2_rel 1e-7 --tol_cp_deg 1e-8



# #!/usr/bin/env python3
# """
# verify_oifits_variants.py

# Verify that rewritten OIFITS variants are consistent with the originals:
# - compare V^2 (VIS2DATA, VIS2ERR) in OI_VIS2 tables
# - compare CP (T3PHI, T3PHIERR) in OI_T3 tables
# - honor FLAG masks by default (can include flagged with --include_flagged)
# - robust matching of files by basename; optional fallback by WL channel count and MJD proximity
# """

# import os
# import sys
# import glob
# import math
# import argparse
# from collections import defaultdict
# import numpy as np
# from astropy.io import fits


# # --------------------------- helpers ---------------------------

# def _basename_noext(path):
#     b = os.path.basename(path)
#     # strip multiple suffixes e.g. ".data.fits"
#     if b.endswith(".fits"):
#         b = b[:-5]
#     return b

# def _ang_diff_deg(a, b):
#     """Smallest signed difference in degrees, result in [-180, 180]."""
#     d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
#     # put exactly -180 at +180 to avoid plotting artifacts
#     d[d <= -180.0] = 180.0
#     return d

# def _col(a, name_candidates):
#     """Return column with first matching name from list; None if none exists."""
#     for nm in name_candidates:
#         if nm in a.columns.names:
#             return a[nm]
#     return None

# def _get_mjd_span(hdul):
#     mjds = []
#     for name in ("OI_VIS2", "OI_T3", "OI_VIS"):
#         if name in hdul:
#             a = hdul[name].data
#             if "MJD" in a.columns.names:
#                 mj = np.asarray(a["MJD"]).ravel()
#                 if mj.size:
#                     mjds.append([np.nanmin(mj), np.nanmax(mj)])
#     if not mjds:
#         return (np.nan, np.nan)
#     lo = np.nanmin([x[0] for x in mjds])
#     hi = np.nanmax([x[1] for x in mjds])
#     return (lo, hi)

# def _get_nchan_wl(hdul):
#     if "OI_WAVELENGTH" not in hdul:
#         return None
#     t = hdul["OI_WAVELENGTH"].data
#     # PIONIER uses EFF_WAVE (um) & EFF_BAND; some other files might use EFF_WL
#     col = None
#     for c in ("EFF_WAVE", "EFF_WL"):
#         if c in t.columns.names:
#             col = t[c]
#             break
#     if col is None:
#         return None
#     arr = np.asarray(col)
#     return int(arr.size)

# def _match_variant_file(orig_path, variant_dir):
#     """
#     Try to find the variant file corresponding to orig_path in variant_dir.
#     Priority:
#         1) basename + *.fits (e.g., "foo*data*.fits", "foo*model*.fits", etc.)
#         2) any *.fits with same stem prefix
#         3) fallback: choose by closest MJD span and matching channel count
#     Returns path or None.
#     """
#     stem = _basename_noext(orig_path)
#     cand = sorted(glob.glob(os.path.join(variant_dir, stem + "*.fits")))
#     if len(cand) == 1:
#         return cand[0]
#     if len(cand) > 1:
#         # choose exact suffix flavors first
#         for suffix in (".data.fits", ".model.fits", ".resid.fits"):
#             exact = os.path.join(variant_dir, stem + suffix)
#             if exact in cand:
#                 return exact
#         # else keep all candidates for further filtering
#     else:
#         # 0 found with exact stem; try prefix match
#         cand = sorted(glob.glob(os.path.join(variant_dir, stem.split("_")[0] + "*.fits")))

#     if not cand:
#         return None

#     # Fallback: pick candidate with same #channels and closest MJD overlap
#     try:
#         with fits.open(orig_path, memmap=False) as hdul_o:
#             nchan_o = _get_nchan_wl(hdul_o)
#             mj_lo_o, mj_hi_o = _get_mjd_span(hdul_o)
#     except Exception:
#         return cand[0]  # last resort

#     best = None
#     best_score = None
#     for c in cand:
#         try:
#             with fits.open(c, memmap=False) as hdul_v:
#                 nchan_v = _get_nchan_wl(hdul_v)
#                 mj_lo_v, mj_hi_v = _get_mjd_span(hdul_v)
#             if (nchan_o is not None) and (nchan_v is not None) and (nchan_o != nchan_v):
#                 continue
#             # score = time mismatch (lower is better)
#             # allow NaN-safe comparison
#             dt = 0.0
#             if not (np.isnan(mj_lo_o) or np.isnan(mj_lo_v)):
#                 dt += abs(mj_lo_o - mj_lo_v)
#             if not (np.isnan(mj_hi_o) or np.isnan(mj_hi_v)):
#                 dt += abs(mj_hi_o - mj_hi_v)
#             score = dt
#             if (best is None) or (score < best_score):
#                 best = c
#                 best_score = score
#         except Exception:
#             continue

#     return best or (cand[0] if cand else None)


# def _compare_vis2(orig_hdul, var_hdul, tol_abs=1e-10, tol_rel=1e-7, include_flagged=False):
#     """Compare OI_VIS2: VIS2DATA / VIS2ERR / FLAG shapes and values."""
#     report = dict(kind="VIS2", n_rows=0, n_chan=0, n_bad=0, max_abs=0.0, max_rel=0.0)
#     if "OI_VIS2" not in orig_hdul or "OI_VIS2" not in var_hdul:
#         report["error"] = "OI_VIS2 missing in one or both files"
#         return report

#     to = orig_hdul["OI_VIS2"].data
#     tv = var_hdul["OI_VIS2"].data

#     # must have same number of rows
#     if len(to) != len(tv):
#         report["error"] = f"row count mismatch (orig {len(to)} vs var {len(tv)})"
#         return report

#     # column aliases
#     c_V2o  = _col(to, ["VIS2DATA", "V2DATA", "V2"])
#     c_EV2o = _col(to, ["VIS2ERR", "EVIS2", "EV2"])
#     c_FLo  = _col(to, ["FLAG"])
#     c_V2v  = _col(tv, ["VIS2DATA", "V2DATA", "V2"])
#     c_EV2v = _col(tv, ["VIS2ERR", "EVIS2", "EV2"])
#     c_FLv  = _col(tv, ["FLAG"])

#     if c_V2o is None or c_V2v is None:
#         report["error"] = "VIS2DATA column missing"
#         return report
#     # EV2 optional (we check when present)

#     n_bad = 0
#     n_rows = 0
#     all_abs = []
#     all_rel = []

#     for i in range(len(to)):
#         V2o = np.atleast_1d(np.asarray(c_V2o[i], float).ravel())
#         V2v = np.atleast_1d(np.asarray(c_V2v[i], float).ravel())
#         if V2o.size != V2v.size:
#             report["error"] = f"VIS2DATA length mismatch in row {i} ({V2o.size} vs {V2v.size})"
#             return report

#         if c_FLo is not None and c_FLv is not None and not include_flagged:
#             flo = np.asarray(c_FLo[i], bool).ravel()
#             flv = np.asarray(c_FLv[i], bool).ravel()
#             m = ~(flo | flv)
#         else:
#             m = slice(None)

#         diff = np.abs(V2o[m] - V2v[m])
#         all_abs.append(diff)
#         denom = np.maximum(1e-12, np.abs(V2o[m]))
#         all_rel.append(diff / denom)

#         n_bad += int(np.sum((diff > tol_abs) & (diff/denom > tol_rel)))
#         n_rows += 1

#         # EV2 if present
#         if (c_EV2o is not None) and (c_EV2v is not None):
#             Eo = np.atleast_1d(np.asarray(c_EV2o[i], float).ravel())
#             Ev = np.atleast_1d(np.asarray(c_EV2v[i], float).ravel())
#             if Eo.size == Ev.size:
#                 diffE = np.abs(Eo[m] - Ev[m])
#                 all_abs.append(diffE)  # track too
#             # else: ignore mismatch of errors silently

#     report["n_rows"] = n_rows
#     report["n_chan"] = int(sum(a.size for a in all_abs)) if all_abs else 0
#     report["n_bad"]  = int(n_bad)
#     if all_abs:
#         report["max_abs"] = float(np.max(np.concatenate(all_abs)))
#     if all_rel:
#         report["max_rel"] = float(np.max(np.concatenate(all_rel)))
#     return report


# def _compare_t3phi(orig_hdul, var_hdul, tol_deg=1e-8, include_flagged=False):
#     """Compare OI_T3: T3PHI / T3PHIERR / FLAG with modular angle diff."""
#     report = dict(kind="T3PHI", n_rows=0, n_chan=0, n_bad=0, max_abs_deg=0.0)
#     if "OI_T3" not in orig_hdul or "OI_T3" not in var_hdul:
#         report["error"] = "OI_T3 missing in one or both files"
#         return report

#     to = orig_hdul["OI_T3"].data
#     tv = var_hdul["OI_T3"].data

#     if len(to) != len(tv):
#         report["error"] = f"row count mismatch (orig {len(to)} vs var {len(tv)})"
#         return report

#     c_Po  = _col(to, ["T3PHI"])
#     c_EPo = _col(to, ["T3PHIERR", "ET3PHI"])
#     c_FLo = _col(to, ["FLAG"])

#     c_Pv  = _col(tv, ["T3PHI"])
#     c_EPv = _col(tv, ["T3PHIERR", "ET3PHI"])
#     c_FLv = _col(tv, ["FLAG"])

#     if c_Po is None or c_Pv is None:
#         report["error"] = "T3PHI column missing"
#         return report

#     n_bad = 0
#     n_rows = 0
#     all_abs = []

#     for i in range(len(to)):
#         Po = np.atleast_1d(np.asarray(c_Po[i], float).ravel())
#         Pv = np.atleast_1d(np.asarray(c_Pv[i], float).ravel())
#         if Po.size != Pv.size:
#             report["error"] = f"T3PHI length mismatch in row {i} ({Po.size} vs {Pv.size})"
#             return report

#         if c_FLo is not None and c_FLv is not None and not include_flagged:
#             flo = np.asarray(c_FLo[i], bool).ravel()
#             flv = np.asarray(c_FLv[i], bool).ravel()
#             m = ~(flo | flv)
#         else:
#             m = slice(None)

#         d = np.abs(_ang_diff_deg(Po[m], Pv[m]))
#         all_abs.append(d)
#         n_bad += int(np.sum(d > tol_deg))
#         n_rows += 1

#         # errors if present
#         if (c_EPo is not None) and (c_EPv is not None):
#             Eo = np.atleast_1d(np.asarray(c_EPo[i], float).ravel())
#             Ev = np.atleast_1d(np.asarray(c_EPv[i], float).ravel())
#             if Eo.size == Ev.size:
#                 all_abs.append(np.abs(Eo[m] - Ev[m]))

#     report["n_rows"] = n_rows
#     report["n_chan"] = int(sum(a.size for a in all_abs)) if all_abs else 0
#     report["n_bad"]  = int(n_bad)
#     report["max_abs_deg"] = float(np.max(np.concatenate(all_abs))) if all_abs else 0.0
#     return report


# def _summ_line(label, rep):
#     if "error" in rep:
#         return f"[{label}] {rep['kind']}: ERROR — {rep['error']}"
#     if rep["kind"] == "VIS2":
#         return (f"[{label}] VIS2: rows={rep['n_rows']} chans={rep['n_chan']} "
#                 f"n_bad={rep['n_bad']} max_abs={rep['max_abs']:.3e} max_rel={rep['max_rel']:.3e}")
#     else:
#         return (f"[{label}] T3PHI: rows={rep['n_rows']} chans={rep['n_chan']} "
#                 f"n_bad={rep['n_bad']} max_abs_deg={rep['max_abs_deg']:.6f}")


# # --------------------------- main ---------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Verify OIFITS variants against originals (V2 & CP consistency).")
#     ap.add_argument("--orig_glob", required=True,
#                     help='Glob or directory for original OIFITS (e.g. "/path/*.fits" or "/path")')
#     ap.add_argument("--var_a_dir", required=True,
#                     help="Directory containing first variant (e.g. .data.fits files)")
#     ap.add_argument("--var_b_dir", default=None,
#                     help="Optional directory containing second variant (e.g. .model.fits or .resid.fits)")
#     ap.add_argument("--tol_v2_abs", type=float, default=1e-10, help="Absolute tolerance for VIS2DATA")
#     ap.add_argument("--tol_v2_rel", type=float, default=1e-7, help="Relative tolerance for VIS2DATA")
#     ap.add_argument("--tol_cp_deg", type=float, default=1e-8, help="Absolute angular tolerance for T3PHI (deg)")
#     ap.add_argument("--include_flagged", action="store_true", help="Include flagged channels in comparisons")
#     args = ap.parse_args()

#     if os.path.isdir(args.orig_glob):
#         orig_list = sorted(glob.glob(os.path.join(args.orig_glob, "*.fits")))
#     else:
#         orig_list = sorted(glob.glob(args.orig_glob))

#     if not orig_list:
#         print("No original files found.")
#         sys.exit(2)

#     def _check_one_variant(vdir, label):
#         total_vis2_bad = 0
#         total_cp_bad = 0
#         n_files = 0

#         for f in orig_list:
#             var = _match_variant_file(f, vdir)
#             if var is None:
#                 print(f"[{label}] MISSING match for {os.path.basename(f)}")
#                 continue

#             try:
#                 with fits.open(f, memmap=False) as ho, fits.open(var, memmap=False) as hv:
#                     rep_v2 = _compare_vis2(ho, hv, tol_abs=args.tol_v2_abs, tol_rel=args.tol_v2_rel,
#                                            include_flagged=args.include_flagged)
#                     rep_cp = _compare_t3phi(ho, hv, tol_deg=args.tol_cp_deg,
#                                             include_flagged=args.include_flagged)
#                 print(f"FILE {os.path.basename(f)} ↔ {os.path.basename(var)}")
#                 print("   " + _summ_line(label, rep_v2))
#                 print("   " + _summ_line(label, rep_cp))
#                 if "error" not in rep_v2: total_vis2_bad += rep_v2["n_bad"]
#                 if "error" not in rep_cp: total_cp_bad += rep_cp["n_bad"]
#                 n_files += 1
#             except Exception as e:
#                 print(f"[{label}] ERROR opening/comparing {f}: {e}")
#                 continue

#         print(f"\n[{label}] SUMMARY over {n_files} matched file(s):")
#         print(f"   total V2 mismatches beyond tol: {total_vis2_bad}")
#         print(f"   total CP mismatches beyond tol: {total_cp_bad}")
#         ok = (total_vis2_bad == 0) and (total_cp_bad == 0)
#         print(f"   RESULT: {'PASS ✅' if ok else 'FAIL ❌'}\n")
#         return ok

#     ok_a = _check_one_variant(args.var_a_dir, "VAR_A")
#     ok_b = True
#     if args.var_b_dir:
#         ok_b = _check_one_variant(args.var_b_dir, "VAR_B")

#     sys.exit(0 if (ok_a and ok_b) else 1)


# if __name__ == "__main__":
#     main()


# # python candid_scripts/verify_oifits_writing.py --orig_glob "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*.fits" --var_a_dir "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/data/" --variant_suffix ".resid.fits" --tol_v2_abs 1e-10 --tol_v2_rel 1e-7 --tol_cp_deg 1e-8

# # python verify_oifits_writing.py \
# #   --orig_glob "/home/rtc/Documents/long_secondary_periods/data/pionier/data/*.fits" \
# #   --var_dir "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/resid" \
# #   --variant_suffix ".resid.fits" \
# #   --tol_v2_abs 1e-10 --tol_v2_rel 1e-7 --tol_cp_deg 1e-8
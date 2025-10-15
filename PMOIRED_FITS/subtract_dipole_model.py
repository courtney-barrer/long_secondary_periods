#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dipole model removal via complex-domain division (V, T3) on VLTI OIFITS
using PMOIRED sampling.

What it does
------------
1) Read observed data with PMOIRED (OI object).
2) Build a thermal-dipole brightness map (l=m=1), project to sky.
3) Sample the model on the **same uv coverage** (PMOIRED makeFakeVLTI) → 'oif'.
4) Build residuals by **division**:
   - V2_res = V2_data / V2_model  (ratio; use --v2_store diff for subtraction)
   - T3_res = T3_data / T3_model  (complex bispectrum division if T3AMP available),
     else CP_res = wrap(CP_data - CP_model).
5) Save CSV/NPZ and simple QA plots.

Notes
-----
- CP in PMOIRED is usually **radians**; set --cp_unit deg if yours are degrees.
- Near model nulls, division can blow up; we guard with eps_amp and you can mask later.

Requirements
------------
pip install pmoired astropy emcee corner

"""

import os, sys, glob, json, copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pmoired
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
import pandas as pd
# ----------------- utilities -----------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def sort_baseline_string(B_string):
    return ''.join(sorted([B_string[:2], B_string[2:]]))

def sort_triangle_string(B_string):
    return ''.join(sorted([B_string[:2], B_string[2:4], B_string[4:]]))

def enforce_ordered_baselines_keys(data, change_baseline_key_list):
    for i in range(len(data)):
        for k in change_baseline_key_list:
            if k == 'baselines':
                data[i][k] = [sort_baseline_string(b) for b in data[i][k]]
            else:
                for baseline_key in list(data[i][k].keys()):
                    new_key = sort_baseline_string(baseline_key)
                    if new_key != baseline_key:
                        data[i][k][new_key] = data[i][k].pop(baseline_key)

def enforce_ordered_triangle_keys(data, change_triangle_key_list):
    for i in range(len(data)):
        for k in change_triangle_key_list:
            if k == 'triangles':
                data[i][k] = [sort_triangle_string(t) for t in data[i][k]]
            else:
                for tri_key in list(data[i][k].keys()):
                    new_key = sort_triangle_string(tri_key)
                    if new_key != tri_key:
                        data[i][k][new_key] = data[i][k].pop(tri_key)

# ----------------- dipole & imaging -----------------

# Constants
h = 6.62607015e-34  # J s
c = 3.0e8           # m/s
k_B = 1.380649e-23  # J/K

def blackbody_intensity(T, wavelength_m):
    return (2*h*c**2 / wavelength_m**5) / (np.exp(h*c/(wavelength_m*k_B*T)) - 1)

def thermal_oscillation(theta, phi, t, T_eff, delta_T_eff, l, m, nu, psi_T):
    """Real Y_lm with cosine time dependence."""
    Y_lm = sph_harm(m, l, phi, theta)
    Ylmn = np.real(Y_lm)
    Ylmn /= np.max(np.abs(Ylmn)) if np.max(np.abs(Ylmn)) > 0 else 1.0
    return T_eff + delta_T_eff * Ylmn * np.cos(2*np.pi*nu*t + psi_T)

def rotate_to_observer_frame(theta, phi, theta_obs, phi_obs):
    # unit vector of observer direction
    x_obs = np.sin(theta_obs)*np.cos(phi_obs)
    y_obs = np.sin(theta_obs)*np.sin(phi_obs)
    z_obs = np.cos(theta_obs)
    v = np.array([x_obs, y_obs, z_obs])
    zhat = np.array([0,0,1.0])
    axis = np.cross(v, zhat)
    angle = np.arccos(np.clip(np.dot(v, zhat), -1, 1))
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0,0.0,0.0])
        angle = 0.0
    axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(angle*axis)

    # sphere points
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    xyz = np.stack([x,y,z], axis=-1).reshape(-1,3)
    xyzr = rot.apply(xyz).reshape(x.shape+(3,))
    xr, yr, zr = xyzr[...,0], xyzr[...,1], xyzr[...,2]

    rr = np.sqrt(xr**2+yr**2+zr**2) + 1e-15
    theta_r = np.arccos(np.clip(zr/rr, -1, 1))
    phi_r = np.arctan2(yr, xr)
    return theta_r, phi_r

def project_to_observer_plane(theta_r, phi_r, intensity, grid_size=512):
    x = np.sin(theta_r)*np.cos(phi_r)
    y = np.sin(theta_r)*np.sin(phi_r)
    z = np.cos(theta_r)
    vis = z > 0
    xv, yv, Iv = x[vis], y[vis], intensity[vis]

    xg = np.linspace(-1, 1, grid_size)
    yg = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(xg, yg)
    R = np.sqrt(X**2+Y**2)
    mask = R <= 1.0

    # nearest is robust; 'linear' also fine if you prefer
    from scipy.interpolate import griddata
    proj = np.zeros_like(X)
    if xv.size:
        proj[mask] = griddata(
            np.column_stack([xv, yv]), Iv, (X[mask], Y[mask]),
            method='linear', fill_value=0.0
        )
    return proj

def intensity_to_fits(image, dx_mas, dy_mas, name="dipole.fits", write_file=False, out_dir="."):
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    N = image.shape[0]
    hdu.header['CRPIX1'] = N/2
    hdu.header['CRPIX2'] = N/2
    hdu.header['CRVAL1'] = 0.0
    hdu.header['CRVAL2'] = 0.0
    hdu.header['CDELT1'] = float(dx_mas)
    hdu.header['CDELT2'] = float(dy_mas)
    hdu.header['CUNIT1'] = 'mas'
    hdu.header['CUNIT2'] = 'mas'
    hdu.header['HDUNAME'] = name
    hdul = fits.HDUList([hdu])
    if write_file:
        ensure_dir(out_dir)
        hdul.writeto(os.path.join(out_dir, name), overwrite=True)
    return hdul

def simulate_obs_from_image_reco_FAST(oi, image_hdul_or_path, img_pixscl_mas=None):
    """Sample a model image (fits path or HDUList) on oi's setup."""
    if isinstance(image_hdul_or_path, str):
        with fits.open(image_hdul_or_path) as d_model:
            img = np.array(d_model[0].data, dtype=float)
            units = d_model[0].header.get('CUNIT1','mas')
            pix = float(d_model[0].header.get('CDELT1', 1.0))
    else:
        d_model = image_hdul_or_path
        img = np.array(d_model[0].data, dtype=float)
        units = d_model[0].header.get('CUNIT1','mas')
        pix = float(d_model[0].header.get('CDELT1', 1.0))

    if img_pixscl_mas is None:
        img_pixscl_mas = pix
    if units.lower() == 'deg':
        img_pixscl_mas *= 3600.0*1e3
    elif units.lower() == 'mas':
        pass
    else:
        raise ValueError(f"Unsupported image CUNIT: {units}")

    oif = pmoired.OI()
    fake_blocks = []
    for a in oi.data:
        cube = {}
        N = img.shape[0]
        x = img_pixscl_mas * np.linspace(-N//2, N//2, N)
        y = img_pixscl_mas * np.linspace(-N//2, N//2, N)
        X, Y = np.meshgrid(x, y)
        cube['scale'] = img_pixscl_mas
        cube['X'], cube['Y'] = X, Y
        WL = a['WL']
        cube['image'] = np.array([img for _ in WL])
        cube['WL'] = WL

        # robust MJD
        if hasattr(a['MJD'], '__len__') and len(a['MJD']) > 0:
            mjd0 = [np.median(a['MJD'])]
        else:
            mjd0 = [a.get('MJD', 59000.0)]

        fake = pmoired.oifake.makeFakeVLTI(
            t=a['telescopes'],
            target=(a['header']['RA']*24/360, a['header']['DEC']),
            lst=[a['LST']],
            wl=WL,
            mjd0=mjd0,
            cube=cube
        )
        fake["filename"] = a.get("filename",None) # added so consistent sorting can be done between oi and oif
        if len(fake.get("MJD", [])) == 0:
            fake["MJD"] = mjd0
        fake_blocks.append(fake)

    oif.data = fake_blocks

    # enforce ordering consistency (PMOIRED minor quirk)
    change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
    change_triangle_key_list = ['triangles','OI_T3']
    enforce_ordered_baselines_keys(oif.data, change_baseline_key_list)
    enforce_ordered_triangle_keys(oif.data, change_triangle_key_list)
    return oif

# ----------------- complex-domain division -----------------


def subtract_by_division(oi_data, oi_model, *, copy_meta=True, verbose=True):
    """
    Build residual observables by dividing complex DATA by MODEL.
    - Triple products (OI_T3):  T3_res = T3_data / T3_model  →  residual CP = arg(T3_res) [deg],
      residual T3AMP = |T3_res|.
    - Complex visibilities (OI_VIS), if present: V_res = V_data / V_model  →  residual VISAMP & VISPHI.
    - For OI_VIS2 (squared vis), we *optionally* form a diagnostic ratio V2_res = V2_data / V2_model,
      but note this is not a complex subtraction.

    Parameters
    ----------
    oi_data : pmoired.OI
        Observed dataset (as returned by pmoired.OI(...)).
    oi_model : pmoired.OI
        Synthetic dataset (same epochs/triangles/baselines ordering as oi_data).
    copy_meta : bool
        If True, copy epoch-level metadata fields from data into residual.
    verbose : bool
        Print small warnings.

    Returns
    -------
    oi_res : pmoired.OI
        New OI object containing residual observables with same structure as input where possible.
    """

    def _degify(arr):
        """Ensure angles are in degrees (convert from rad if needed)."""
        if arr is None:
            return None
        arr = np.asarray(arr, float)
        # If max|arr| <= ~pi, assume radians and convert:
        return np.degrees(arr) if np.nanmax(np.abs(arr)) <= (np.pi * 1.05) else arr

    def _extract_t3_arrays(blk):
        """
        Return (T3AMP, ET3AMP, T3PHI_deg, ET3PHI_deg, S=Bmax/λ).
        Missing pieces come back as None.
        """
        A   = np.asarray(blk.get("T3AMP"), float)  if "T3AMP"  in blk else None
        Ae  = np.asarray(blk.get("ET3AMP"), float) if "ET3AMP" in blk else None
        phi = None
        for k in ("T3PHI", "t3phi", "CP"):  # CP not standard here, but allow
            if k in blk:
                phi = _degify(blk[k]); break
        phie = None
        for k in ("ET3PHI", "t3phierr", "CPERR"):
            if k in blk:
                phie = _degify(blk[k]); break
        S = None
        for k in ("Bmax/wl", "Bmax/wl_data", "Smax", "S"):
            if k in blk:
                S = np.asarray(blk[k], float); break
        flag = np.asarray(blk.get("FLAG"), bool) if "FLAG" in blk else None
        return A, Ae, phi, phie, S, flag

    def _extract_vis_arrays(blk):
        """
        Return (VISAMP, EVISAMP, VISPHI_deg, EVISPHI_deg) if present; else all None.
        """
        A   = np.asarray(blk.get("VISAMP"), float)  if "VISAMP"  in blk else None
        Ae  = np.asarray(blk.get("EVISAMP"), float) if "EVISAMP" in blk else None
        phi = _degify(blk["VISPHI"]) if "VISPHI" in blk else None
        phie= _degify(blk["EVISPHI"]) if "EVISPHI" in blk else None
        flag = np.asarray(blk.get("FLAG"), bool) if "FLAG" in blk else None
        return A, Ae, phi, phie, flag

    def _safe_div_complex(A_data, phi_data_deg, A_model, phi_model_deg):
        """Return complex (data/model), handling NaNs and zeros."""
        Td = A_data * np.exp(1j * np.deg2rad(phi_data_deg))
        Tm = A_model * np.exp(1j * np.deg2rad(phi_model_deg))
        with np.errstate(divide="ignore", invalid="ignore"):
            Tr = Td / Tm
        return Tr

    # --- Build output container
    oi_res = pmoired.OI()
    oi_res.data = []

    if len(oi_data.data) != len(oi_model.data):
        raise ValueError("DATA and MODEL do not have the same number of epochs.")

    for iepoch, (d, m) in enumerate(zip(oi_data.data, oi_model.data)):
        # Start residual epoch dict
        res = {}
        if copy_meta:
            # copy a few common meta fields if present
            for k in ("header", "MJD", "MJD2", "LST", "telescopes", "WL", "insname", "target"):
                if k in d:
                    res[k] = d[k]

        # ----- OI_T3 (closure quantities) -----
        res["OI_T3"] = {}
        if "OI_T3" in d and "OI_T3" in m:
            for tri in d["OI_T3"]:
                if tri not in m["OI_T3"]:
                    if verbose:
                        print(f"[WARN] epoch {iepoch}: triangle {tri} missing in MODEL; skipped.")
                    continue
                blk_d = d["OI_T3"][tri]
                blk_m = m["OI_T3"][tri]

                Ad, Aed, phid, ephid, Sd, flagd = _extract_t3_arrays(blk_d)
                Am, Aem, phim, ephim, Sm, flagm = _extract_t3_arrays(blk_m)

                if Ad is None or phid is None or Am is None or phim is None:
                    if verbose:
                        print(f"[WARN] epoch {iepoch}: no T3AMP/T3PHI for {tri}; skipped.")
                    continue

                # Shape & flag handling
                Ad   = np.asarray(Ad, float)
                Am   = np.asarray(Am, float)
                phid = np.asarray(phid, float)
                phim = np.asarray(phim, float)

                # Complex division
                T3r = _safe_div_complex(Ad, phid, Am, phim)
                cp_res_deg  = np.rad2deg(np.angle(T3r))
                t3amp_res   = np.abs(T3r)

                # Residual uncertainties: keep data errors (simple/optimistic)
                et3phi_res = ephid if ephid is not None else None
                et3amp_res = Aed  if Aed  is not None else None

                # Flags: union (be conservative)
                if flagd is not None or flagm is not None:
                    fd = flagd if flagd is not None else np.zeros_like(t3amp_res, dtype=bool)
                    fm = flagm if flagm is not None else np.zeros_like(t3amp_res, dtype=bool)
                    flag = (fd | fm)
                else:
                    flag = np.zeros_like(t3amp_res, dtype=bool)

                # Build residual triangle block
                rblk = {}
                # keep geometry/axes if present
                for k in ("B1","B2","B3","Bavg/wl","Bmax/wl","Bmin/wl","u1","v1","u2","v2","formula","MJD","MJD2"):
                    if k in blk_d:
                        rblk[k] = blk_d[k]
                # residuals
                rblk["T3AMP"]  = t3amp_res
                if et3amp_res is not None: rblk["ET3AMP"] = et3amp_res
                rblk["T3PHI"]  = cp_res_deg
                if et3phi_res is not None: rblk["ET3PHI"] = et3phi_res
                rblk["FLAG"]   = flag

                res["OI_T3"][tri] = rblk

        # ----- OI_VIS (complex vis) -----
        if "OI_VIS" in d and "OI_VIS" in m:
            res["OI_VIS"] = {}
            for bl in d["OI_VIS"]:
                if bl not in m["OI_VIS"]:
                    if verbose:
                        print(f"[WARN] epoch {iepoch}: baseline {bl} missing in MODEL(OI_VIS); skipped.")
                    continue

                blk_d = d["OI_VIS"][bl]
                blk_m = m["OI_VIS"][bl]
                VAd, VAed, VPd, EVPd, flagd = _extract_vis_arrays(blk_d)
                VAm, VAem, VPm, EVPm, flagm = _extract_vis_arrays(blk_m)
                if VAd is None or VPd is None or VAm is None or VPm is None:
                    # If either dataset lacks complex vis, skip this baseline
                    continue

                Vr = _safe_div_complex(VAd, VPd, VAm, VPm)
                vis_amp_res = np.abs(Vr)
                vis_phi_res = np.rad2deg(np.angle(Vr))

                if flagd is not None or flagm is not None:
                    fd = flagd if flagd is not None else np.zeros_like(vis_amp_res, dtype=bool)
                    fm = flagm if flagm is not None else np.zeros_like(vis_amp_res, dtype=bool)
                    flag = (fd | fm)
                else:
                    flag = np.zeros_like(vis_amp_res, dtype=bool)

                rblk = {}
                # keep geometry/axes if present
                for k in ("B/wl","u","v","MJD","MJD2"):
                    if k in blk_d:
                        rblk[k] = blk_d[k]
                rblk["VISAMP"]  = vis_amp_res
                if VAed is not None: rblk["EVISAMP"] = VAed
                rblk["VISPHI"]  = vis_phi_res
                if EVPd is not None: rblk["EVISPHI"] = EVPd
                rblk["FLAG"]    = flag

                res["OI_VIS"][bl] = rblk

        # ----- OI_VIS2 (diagnostic ratio) -----
        if "OI_VIS2" in d and "OI_VIS2" in m:
            res["OI_VIS2"] = {}
            for bl in d["OI_VIS2"]:
                if bl not in m["OI_VIS2"]:
                    if verbose:
                        print(f"[WARN] epoch {iepoch}: baseline {bl} missing in MODEL(OI_VIS2); skipped.")
                    continue
                blk_d = d["OI_VIS2"][bl]
                blk_m = m["OI_VIS2"][bl]
                V2d   = np.asarray(blk_d.get("V2"), float) if "V2" in blk_d else None
                EV2d  = np.asarray(blk_d.get("EV2"), float) if "EV2" in blk_d else None
                V2m   = np.asarray(blk_m.get("V2"), float) if "V2" in blk_m else None
                if V2d is None or V2m is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    V2r = V2d / V2m  # purely diagnostic (not complex)
                flagd = np.asarray(blk_d.get("FLAG"), bool) if "FLAG" in blk_d else np.zeros_like(V2r, bool)
                flagm = np.asarray(blk_m.get("FLAG"), bool) if "FLAG" in blk_m else np.zeros_like(V2r, bool)
                flag  = flagd | flagm

                rblk = {}
                for k in ("B/wl","u","v","MJD","MJD2"):
                    if k in blk_d: rblk[k] = blk_d[k]
                rblk["V2"]  = V2r
                if EV2d is not None: rblk["EV2"] = EV2d  # keep data uncertainty
                rblk["FLAG"] = flag
                res["OI_VIS2"][bl] = rblk

        # carry over baseline/triangle lists if present
        for key in ("baselines","triangles"):
            if key in d:
                res[key] = d[key]

        oi_res.data.append(res)

    return oi_res
# def subtract_by_division(oi, oif, *,
#                          cp_unit="rad",
#                          store_v2_as="ratio",   # "ratio" or "diff"
#                          eps_amp=1e-12):
#     """
#     Residuals by division: V2_res = V2_d / V2_m (or diff); T3_res = T3_d / T3_m (if T3AMP available)
#     Fallback for CP: CP_res = wrap(CP_d - CP_m).
#     """
#     to_rad = (np.pi/180.0) if cp_unit.lower().startswith("deg") else 1.0
#     oi_res = copy.deepcopy(oi)

#     for iblk in range(len(oi.data)):
#         d = oi.data[iblk]
#         m = oif.data[iblk]
#         r = oi_res.data[iblk]

#         # V²
#         if ("OI_VIS2" in d) and ("OI_VIS2" in m):
#             for bl in d["OI_VIS2"].keys():
#                 if bl not in m["OI_VIS2"]:
#                     continue
#                 v2d = np.asarray(d["OI_VIS2"][bl]["V2"], float)
#                 v2m = np.asarray(m["OI_VIS2"][bl]["V2"], float)
#                 if store_v2_as == "ratio":
#                     v2r = v2d / np.clip(v2m, eps_amp, None)
#                 else:
#                     v2r = v2d - v2m
#                 r["OI_VIS2"][bl]["V2"] = v2r

#         # T3 / CP
#         if ("OI_T3" in d) and ("OI_T3" in m):
#             for tri in d["OI_T3"].keys():
#                 if tri not in m["OI_T3"]:
#                     continue
#                 cpd = np.asarray(d["OI_T3"][tri]["CP"], float) * to_rad
#                 cpm = np.asarray(m["OI_T3"][tri]["CP"], float) * to_rad
#                 amp_d = np.asarray(d["OI_T3"][tri].get("T3AMP", []), float)
#                 amp_m = np.asarray(m["OI_T3"][tri].get("T3AMP", []), float)

#                 if amp_d.size and amp_m.size and amp_d.shape == cpd.shape and amp_m.shape == cpm.shape:
#                     T3d = amp_d * np.exp(1j*cpd)
#                     T3m = amp_m * np.exp(1j*cpm)
#                     T3r = T3d / np.clip(T3m, eps_amp, None)
#                     r["OI_T3"][tri]["CP"] = np.angle(T3r)
#                     r["OI_T3"][tri]["T3AMP"] = np.abs(T3r)
#                 else:
#                     r["OI_T3"][tri]["CP"] = _wrap_phase_rad(cpd - cpm)

#     return oi_res

# ----------------- model wrapper -----------------

def build_oif_from_params(oi, theta_o_deg, phi_o_deg, delta_T, theta_ud_mas,
                          wavelength_m=1.65e-6, grid_size=500,
                          T_eff=3000.0, nu=1/(757*24*3600), psi_T=0.0,
                          l=1, m=1, dx_mas=1.0, dy_mas=1.0):
    """
    Make dipole map → sample on oi → return oif.
    """
    # spherical grid (not too fine to keep speed)
    th = np.linspace(0, np.pi, 64)
    ph = np.linspace(0, 2*np.pi, 64)
    TH, PH = np.meshgrid(th, ph)

    Tloc = thermal_oscillation(TH, PH, 0.0, T_eff, delta_T, l, m, nu, psi_T)
    theta_o = np.deg2rad(theta_o_deg)
    phi_o   = np.deg2rad(phi_o_deg)
    THr, PHr = rotate_to_observer_frame(TH, PH, theta_o, phi_o)
    img = project_to_observer_plane(THr, PHr, blackbody_intensity(Tloc, wavelength_m), grid_size=grid_size)

    # build a FITS (in memory) with pixel scale set by UD (so diameter spans ~grid_size px)
    pix_mas = theta_ud_mas / grid_size
    hdul = intensity_to_fits(img, pix_mas, pix_mas, name="dipole_model.fits", write_file=False)

    oif = simulate_obs_from_image_reco_FAST(oi, hdul, img_pixscl_mas=pix_mas)
    return oif

# ----------------- saving diagnostics -----------------

def dump_residual_arrays(oi_res, out_dir, tag="residual"):
    """
    Flatten residual OI structure into tidy CSVs. Handles:
      • (1, N) row-vectors → coerced to 1-D (N,)
      • missing per-block wavelength -> fallback to epoch WL or NaN
      • missing S columns: B/wl -> Bavg/wl -> Bmax/wl
    Saves (if present):
      - {tag}_T3.csv   (CP/T3AMP residuals)
      - {tag}_VIS.csv  (VISAMP/VISPHI residuals)
      - {tag}_VIS2.csv (V2 diagnostic ratios)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- helpers ----------
    def _to_1d(x):
        """Return a 1-D float array; None -> empty 1-D."""
        if x is None:
            return np.empty(0, dtype=float)
        a = np.asarray(x, dtype=float)
        # squeeze then ravel: (1, N) → (N,), (N,) → (N,), scalars → (1,)
        a = np.squeeze(a).ravel()
        return a

    def _get1d(a, i):
        """Safe index into a 1-D array; NaN if out of range or empty."""
        a = _to_1d(a)
        if a.size == 0 or i >= a.size:
            return np.nan
        return float(a[i])

    def _pick_S(blk):
        """Pick best-available spatial freq vector from a block."""
        for key in ("B/wl", "Bavg/wl", "Bmax/wl"):
            if key in blk:
                return _to_1d(blk[key])
        return np.empty(0, dtype=float)

    # ---------------- T3 / CP ----------------
    rows_t3 = []
    for ie, ep in enumerate(oi_res.data):
        WL_ep = _to_1d(ep.get("WL", None))
        if "OI_T3" not in ep:
            continue
        for tri, blk in ep["OI_T3"].items():
            cp   = _to_1d(blk.get("T3PHI", None))   # deg
            ecp  = _to_1d(blk.get("ET3PHI", None))  # deg
            amp  = _to_1d(blk.get("T3AMP", None))
            eamp = _to_1d(blk.get("ET3AMP", None))
            S    = _pick_S(blk)

            n = max(cp.size, amp.size, S.size, WL_ep.size, 1)
            for i in range(n):
                rows_t3.append(dict(
                    epoch=ie,
                    triangle=tri,
                    chan=i,
                    CP_deg=_get1d(cp, i),
                    ET3PHI_deg=_get1d(ecp, i),
                    T3AMP=_get1d(amp, i),
                    ET3AMP=_get1d(eamp, i),
                    S=_get1d(S, i),
                    WL=_get1d(WL_ep, i)
                ))

    if rows_t3:
        df_t3 = pd.DataFrame(rows_t3)
        f = os.path.join(out_dir, f"{tag}_T3.csv")
        df_t3.to_csv(f, index=False)
        print("[SAVE]", f)

    # ---------------- VIS (complex) ----------------
    rows_vis = []
    for ie, ep in enumerate(oi_res.data):
        WL_ep = _to_1d(ep.get("WL", None))
        if "OI_VIS" not in ep:
            continue
        for bl, blk in ep["OI_VIS"].items():
            vamp  = _to_1d(blk.get("VISAMP", None))
            evamp = _to_1d(blk.get("EVISAMP", None))
            vphi  = _to_1d(blk.get("VISPHI", None))    # deg
            evphi = _to_1d(blk.get("EVISPHI", None))   # deg
            S     = _pick_S(blk)

            n = max(vamp.size, vphi.size, S.size, WL_ep.size, 1)
            for i in range(n):
                rows_vis.append(dict(
                    epoch=ie,
                    baseline=bl,
                    chan=i,
                    VISAMP=_get1d(vamp, i),
                    EVISAMP=_get1d(evamp, i),
                    VISPHI_deg=_get1d(vphi, i),
                    EVISPHI_deg=_get1d(evphi, i),
                    S=_get1d(S, i),
                    WL=_get1d(WL_ep, i)
                ))

    if rows_vis:
        df_vis = pd.DataFrame(rows_vis)
        f = os.path.join(out_dir, f"{tag}_VIS.csv")
        df_vis.to_csv(f, index=False)
        print("[SAVE]", f)

    # ---------------- VIS2 (diagnostic ratio) ----------------
    rows_v2 = []
    for ie, ep in enumerate(oi_res.data):
        WL_ep = _to_1d(ep.get("WL", None))
        if "OI_VIS2" not in ep:
            continue
        for bl, blk in ep["OI_VIS2"].items():
            V2  = _to_1d(blk.get("V2", None))
            EV2 = _to_1d(blk.get("EV2", None))
            S   = _pick_S(blk)

            n = max(V2.size, S.size, WL_ep.size, 1)
            for i in range(n):
                rows_v2.append(dict(
                    epoch=ie,
                    baseline=bl,
                    chan=i,
                    V2=_get1d(V2, i),
                    EV2=_get1d(EV2, i),
                    S=_get1d(S, i),
                    WL=_get1d(WL_ep, i)
                ))

    if rows_v2:
        df_v2 = pd.DataFrame(rows_v2)
        f = os.path.join(out_dir, f"{tag}_VIS2.csv")
        df_v2.to_csv(f, index=False)
        print("[SAVE]", f)

# def dump_residual_arrays(oi_res, out_dir, tag="residual"):
#     """
#     Flatten residual OI structure into tidy CSVs. Handles scalar vs vector
#     arrays safely and uses epoch-level WL when block-level WL is absent.
#     Saves three CSVs if the corresponding blocks are present:
#       - {tag}_T3.csv   (CP/T3AMP residuals)
#       - {tag}_VIS.csv  (VISAMP/VISPHI residuals)        [if OI_VIS present]
#       - {tag}_VIS2.csv (V2 diagnostic ratios)           [if OI_VIS2 present]
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     def _arr(x):
#         """np.atleast_1d on numbers or arrays; returns float array (or empty)."""
#         if x is None:
#             return np.array([], dtype=float)
#         a = np.asarray(x, dtype=float)
#         return a if a.ndim > 0 else np.atleast_1d(a)

#     def _get(a, i):
#         """Safe index: return a[i] if exists; else a[0] if nonempty; else NaN."""
#         a = _arr(a)
#         if a.size == 0:
#             return np.nan
#         return a[i] if i < a.size else a[0]

#     # ---------------- T3 / CP ----------------
#     rows_t3 = []
#     for ie, ep in enumerate(oi_res.data):
#         WL = _arr(ep.get("WL", None))  # epoch-level wavelengths (preferred)
#         if "OI_T3" not in ep:
#             continue
#         for tri, blk in ep["OI_T3"].items():
#             cp   = _arr(blk.get("T3PHI", None))  # deg
#             ecp  = _arr(blk.get("ET3PHI", None))
#             amp  = _arr(blk.get("T3AMP", None))
#             eamp = _arr(blk.get("ET3AMP", None))
#             smax = _arr(blk.get("Bmax/wl", None))  # often provided by PMOIRED

#             n = max(cp.size, amp.size, smax.size, 1)
#             for i in range(n):
#                 rows_t3.append(dict(
#                     epoch=ie,
#                     triangle=tri,
#                     chan=i,
#                     CP_deg=_get(cp, i),
#                     ET3PHI_deg=_get(ecp, i),
#                     T3AMP=_get(amp, i),
#                     ET3AMP=_get(eamp, i),
#                     Smax=_get(smax, i),
#                     WL=_get(WL, i)
#                 ))

#     if rows_t3:
#         df_t3 = pd.DataFrame(rows_t3)
#         df_t3.to_csv(os.path.join(out_dir, f"{tag}_T3.csv"), index=False)
#         print("[SAVE]", os.path.join(out_dir, f"{tag}_T3.csv"))

#     # ---------------- VIS (complex) ----------------
#     rows_vis = []
#     for ie, ep in enumerate(oi_res.data):
#         WL = _arr(ep.get("WL", None))
#         if "OI_VIS" not in ep:
#             continue
#         for bl, blk in ep["OI_VIS"].items():
#             vamp  = _arr(blk.get("VISAMP", None))
#             evamp = _arr(blk.get("EVISAMP", None))
#             vphi  = _arr(blk.get("VISPHI", None))   # deg
#             evphi = _arr(blk.get("EVISPHI", None))  # deg
#             bwl   = _arr(blk.get("B/wl", None))
#             n = max(vamp.size, vphi.size, bwl.size, 1)
#             for i in range(n):
#                 rows_vis.append(dict(
#                     epoch=ie,
#                     baseline=bl,
#                     chan=i,
#                     VISAMP=_get(vamp, i),
#                     EVISAMP=_get(evamp, i),
#                     VISPHI_deg=_get(vphi, i),
#                     EVISPHI_deg=_get(evphi, i),
#                     S=_get(bwl, i),
#                     WL=_get(WL, i)
#                 ))
#     if rows_vis:
#         df_vis = pd.DataFrame(rows_vis)
#         df_vis.to_csv(os.path.join(out_dir, f"{tag}_VIS.csv"), index=False)
#         print("[SAVE]", os.path.join(out_dir, f"{tag}_VIS.csv"))

#     # ---------------- VIS2 (diagnostic ratio) ----------------
#     rows_v2 = []
#     for ie, ep in enumerate(oi_res.data):
#         WL = _arr(ep.get("WL", None))
#         if "OI_VIS2" not in ep:
#             continue
#         for bl, blk in ep["OI_VIS2"].items():
#             V2  = _arr(blk.get("V2", None))
#             EV2 = _arr(blk.get("EV2", None))
#             bwl = _arr(blk.get("B/wl", None))
#             n = max(V2.size, bwl.size, 1)
#             for i in range(n):
#                 rows_v2.append(dict(
#                     epoch=ie,
#                     baseline=bl,
#                     chan=i,
#                     V2=_get(V2, i),
#                     EV2=_get(EV2, i),
#                     S=_get(bwl, i),
#                     WL=_get(WL, i)
#                 ))
#     if rows_v2:
#         df_v2 = pd.DataFrame(rows_v2)
#         df_v2.to_csv(os.path.join(out_dir, f"{tag}_VIS2.csv"), index=False)
#         print("[SAVE]", os.path.join(out_dir, f"{tag}_VIS2.csv"))

# # def dump_residual_arrays(oi_res, out_dir, tag="dipole_division"):
# #     ensure_dir(out_dir)

# #     # Flatten quick CSV/NPZ for V² & CP
# #     rows_v2 = []
# #     rows_cp = []
# #     for blk in oi_res.data:
# #         # V²
# #         if "OI_VIS2" in blk:
# #             for bl, v2d in blk["OI_VIS2"].items():
# #                 WL = np.asarray(v2d.get("WL", blk.get("WL", [])), float)
# #                 # Many PMOIRED blocks don't store WL per-baseline; try block level
# #                 if WL.size == 0 and "WL" in blk:
# #                     WL = np.asarray(blk["WL"], float)
# #                 if "V2" in v2d:
# #                     V2 = np.asarray(v2d["V2"], float)
# #                     for i in range(V2.size):
# #                         rows_v2.append(dict(baseline=bl, chan=i, V2=V2[i], WL=WL[i] if i<WL.size else np.nan))
# #         # CP
# #         if "OI_T3" in blk:
# #             for tri, t3d in blk["OI_T3"].items():
# #                 WL = np.asarray(t3d.get("WL", blk.get("WL", [])), float)
# #                 if WL.size == 0 and "WL" in blk:
# #                     WL = np.asarray(blk["WL"], float)
# #                 if "CP" in t3d:
# #                     CP = np.asarray(t3d["CP"], float)
# #                     T3A = np.asarray(t3d.get("T3AMP", []), float) if "T3AMP" in t3d else np.array([])
# #                     for i in range(CP.size):
# #                         rows_cp.append(dict(triangle=tri, chan=i, CP=CP[i], T3AMP=(T3A[i] if i<T3A.size else np.nan),
# #                                             WL=WL[i] if i<WL.size else np.nan))

# #     import pandas as pd
# #     if rows_v2:
# #         dfv = pd.DataFrame(rows_v2)
# #         dfv.to_csv(os.path.join(out_dir, f"{tag}_V2.csv"), index=False)
# #         np.savez(os.path.join(out_dir, f"{tag}_V2.npz"), **{k: dfv[k].values for k in dfv.columns})
# #         print("[SAVE]", os.path.join(out_dir, f"{tag}_V2.csv"))
# #     if rows_cp:
# #         dfc = pd.DataFrame(rows_cp)
# #         dfc.to_csv(os.path.join(out_dir, f"{tag}_CP.csv"), index=False)
# #         np.savez(os.path.join(out_dir, f"{tag}_CP.npz"), **{k: dfc[k].values for k in dfc.columns})
# #         print("[SAVE]", os.path.join(out_dir, f"{tag}_CP.csv"))

# import numpy as np, os, matplotlib.pyplot as plt

# def _wrap180(deg):
#     return (deg + 180.0) % 360.0 - 180.0

# def _to_deg(arr):
#     a = np.asarray(arr, float)
#     # Heuristic: small range ⇒ radians
#     if np.nanmax(np.abs(a)) <= 6.0:
#         a = np.degrees(a)
#     return a

# def _collect_cp_deg_vs_s(OBJ):
#     xs, ys = [], []
#     for blk in getattr(OBJ, "data", []):
#         if "OI_T3" not in blk:
#             continue

#         for tri, t3 in blk["OI_T3"].items():
#             # CP (deg)
#             if "T3PHI" in t3:
#                 cp_deg = _to_deg(t3["T3PHI"])   # already deg for PMOIRED, still robust
#             elif "CP" in t3:
#                 cp_deg = _to_deg(t3["CP"])      # radians → deg
#             else:
#                 continue

#             cp_deg = np.ravel(cp_deg)

#             # s = B/λ abscissa (prefer triangle's Bmax/wl)
#             if "Bmax/wl" in t3:
#                 s = np.ravel(np.asarray(t3["Bmax/wl"], float))
#             elif "Bavg/wl" in t3:
#                 s = np.ravel(np.asarray(t3["Bavg/wl"], float))
#             elif "Bmin/wl" in t3:
#                 s = np.ravel(np.asarray(t3["Bmin/wl"], float))
#             else:
#                 # last-ditch: try epoch-level WL to build a proxy, else index
#                 if "WL" in blk:
#                     lam = np.asarray(blk["WL"], float)
#                     s = np.ravel(np.ones_like(lam) / lam)  # very crude proxy
#                 else:
#                     s = np.arange(cp_deg.size, dtype=float)

#             # broadcast if needed
#             if s.size == 1 and cp_deg.size > 1:
#                 s = np.repeat(s, cp_deg.size)

#             n = min(s.size, cp_deg.size)
#             s = s[:n]; cp_deg = cp_deg[:n]

#             xs.append(s); ys.append(cp_deg)

#     if not xs:
#         return np.array([]), np.array([])
#     x = np.concatenate(xs)
#     y = np.concatenate(ys)
#     # sort by spatial frequency
#     idx = np.argsort(x)
#     return x[idx], _wrap180(y[idx])



def _wrap180(d): return (np.asarray(d,float)+180)%360-180
def _to_deg(a):  # tolerate rad input
    a = np.asarray(a, float)
    return np.degrees(a) if np.nanmax(np.abs(a))<=6.0 else a

def _infer_Mlambda(x):
    """PMOIRED uses B/λ with λ in μm => x already in Mλ; else detect & scale."""
    x = np.asarray(x, float).ravel()
    med = np.nanmedian(np.abs(x))
    if 0.2 <= med <= 300.0:        # looks like Mλ already (typical 30–150)
        return x
    if med > 1e5:                  # looks like rad^-1 (e.g. 6e7)
        return x/1e6               # convert to Mλ
    if med < 1.0:                  # odd (e.g. 4e-5) – probably over-scaled
        return x*1e6               # bring back to Mλ
    return x

def _collect_cp_deg_vs_Mlambda(OBJ, cp_err_max=None):
    xs, ys = [], []

    def to_deg_safe(a):
        a = np.asarray(a, float)
        return np.degrees(a) if np.nanmax(np.abs(a)) <= 6.0 else a

    def infer_Mlambda(x):
        x = np.asarray(x, float).ravel()
        med = np.nanmedian(np.abs(x))
        if 0.2 <= med <= 300:   # already Mλ
            return x
        if med > 1e5:           # rad^-1 → Mλ
            return x/1e6
        if med < 1.0:           # probably over-scaled (e.g. 4e-5)
            return x*1e6
        return x

    for blk in getattr(OBJ, "data", []):
        if "OI_T3" not in blk:
            continue
        for tri, t3 in blk["OI_T3"].items():
            # --- CP (deg) and its error (deg)
            if "T3PHI" in t3:
                cp  = to_deg_safe(t3["T3PHI"]).ravel()
                ecp = to_deg_safe(t3.get("ET3PHI", np.zeros_like(cp))).ravel()
            elif "CP" in t3:
                cp  = to_deg_safe(t3["CP"]).ravel()
                ecp = to_deg_safe(t3.get("ECP", np.zeros_like(cp))).ravel()
            else:
                continue

            # --- FLAG (True=bad)
            flg = np.asarray(t3.get("FLAG", np.zeros_like(cp, dtype=bool))).ravel()
            if flg.size != cp.size:
                flg = np.zeros_like(cp, dtype=bool)

            # --- abscissa B/λ
            if   "Bmax/wl" in t3: sraw = t3["Bmax/wl"]
            elif "Bavg/wl" in t3: sraw = t3["Bavg/wl"]
            elif "Bmin/wl" in t3: sraw = t3["Bmin/wl"]
            else:                 sraw = np.arange(cp.size)
            s = infer_Mlambda(sraw).ravel()

            # --- force common length
            L = min(cp.size, ecp.size, flg.size, s.size)
            if L == 0:
                continue
            cp  = cp [:L]
            ecp = ecp[:L]
            flg = flg[:L]
            s   = s  [:L]

            # --- optional error cut
            if cp_err_max is None:
                bad_err = np.zeros(L, dtype=bool)
            else:
                bad_err = ecp > float(cp_err_max)

            keep = ~(flg | bad_err)
            if np.any(keep):
                xs.append(s[keep])
                ys.append(((cp[keep] + 180) % 360) - 180)  # wrap to [-180,180)

    if not xs:
        return np.array([]), np.array([])
    x = np.concatenate(xs); y = np.concatenate(ys)
    order = np.argsort(x)
    return x[order], y[order]


def quick_plots(oi, oif, oi_res, out_dir, cp_err_max=None, v2_err_max=None, v2_ylim=None):
    """
    Make quick-look plots for CP and V^2 versus B/λ (in Mλ) for
    data / model / residual objects stored in PMOIRED-like dicts.

    Parameters
    ----------
    oi, oif, oi_res : PMOIRED OI-like objects
        Each must have a .data list with OI_T3 / OI_VIS2 blocks.
    out_dir : str
        Where PNGs are saved.
    cp_err_max : float or None
        If provided (deg), discard CP channels with ET3PHI > cp_err_max.
    v2_err_max : float or None
        If provided (absolute), discard V^2 channels with EV2 > v2_err_max.
    v2_ylim : tuple or None
        y-limits for V^2 plots, e.g. (0.0, 1.2). If None, auto-scale.

    Notes
    -----
    - CP uses per-triangle 'Bmax/wl' (preferred) in Mλ for the x-axis;
      falls back to 'Bavg/wl' then 'B/wl' if needed.
    - V^2 uses per-baseline 'B/wl' (preferred) in Mλ; falls back to
      'Bavg/wl' then 'Bmax/wl' if needed.
    - Channel FLAGs are honored (flagged -> dropped).
    - CP phases are wrapped to [-180, 180] deg for readability.
    """
    import numpy as np, os, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _wrap_deg(phi_deg):
        x = (np.asarray(phi_deg, float) + 180.0) % 360.0 - 180.0
        x[x <= -180.0] = 180.0
        return x

    # ---- helpers to pick a spatial frequency vector in Mλ ----
    def _pick_S_T3(blk):
        for key in ("Bmax/wl", "Bavg/wl", "B/wl"):
            if key in blk:
                return np.asarray(blk[key], float).ravel() / 1e6
        return np.array([], dtype=float)

    def _pick_S_VIS2(blk):
        for key in ("B/wl", "Bavg/wl", "Bmax/wl"):
            if key in blk:
                return np.asarray(blk[key], float).ravel() / 1e6
        return np.array([], dtype=float)

    # ---- collectors -------------------------------------------------
    def _collect_cp_deg_vs_Mlambda(OBJ):
        S, CP = [], []
        kept = 0
        for ep in getattr(OBJ, "data", []):
            if "OI_T3" not in ep:
                continue
            for tri, t3 in ep["OI_T3"].items():
                cp_deg  = np.asarray(t3.get("T3PHI", []), float).ravel()
                ep_deg  = np.asarray(t3.get("ET3PHI", []), float).ravel()
                flags   = np.asarray(t3.get("FLAG", np.zeros_like(cp_deg, dtype=bool)), bool).ravel()
                s_mlam  = _pick_S_T3(t3)
                if cp_deg.size == 0 or s_mlam.size == 0:
                    continue
                n = min(cp_deg.size, s_mlam.size, flags.size if flags.size else cp_deg.size,
                        ep_deg.size if ep_deg.size else cp_deg.size)
                cp_deg = cp_deg[:n]; s_mlam = s_mlam[:n]; flags = flags[:n]
                if ep_deg.size: ep_deg = ep_deg[:n]
                good = ~flags
                if cp_err_max is not None and ep_deg.size:
                    good &= (ep_deg <= float(cp_err_max))
                if not np.any(good):
                    continue
                CP.append(_wrap_deg(cp_deg[good]))
                S.append(s_mlam[good])
                kept += np.count_nonzero(good)
        if kept == 0:
            return np.array([]), np.array([])
        S = np.concatenate(S); CP = np.concatenate(CP)
        order = np.argsort(S)
        return S[order], CP[order]

    def _collect_v2_vs_Mlambda(OBJ):
        S, V2 = [], []
        kept = 0
        for ep in getattr(OBJ, "data", []):
            if "OI_VIS2" not in ep:
                continue
            for bl, v2 in ep["OI_VIS2"].items():
                v2_val = np.asarray(v2.get("V2", []), float).ravel()
                ev2    = np.asarray(v2.get("EV2", []), float).ravel()
                flags  = np.asarray(v2.get("FLAG", np.zeros_like(v2_val, dtype=bool)), bool).ravel()
                s_mlam = _pick_S_VIS2(v2)
                if v2_val.size == 0 or s_mlam.size == 0:
                    continue
                n = min(v2_val.size, s_mlam.size, flags.size if flags.size else v2_val.size,
                        ev2.size if ev2.size else v2_val.size)
                v2_val = v2_val[:n]; s_mlam = s_mlam[:n]; flags = flags[:n]
                if ev2.size: ev2 = ev2[:n]
                good = ~flags
                if v2_err_max is not None and ev2.size:
                    good &= (ev2 <= float(v2_err_max))
                if not np.any(good):
                    continue
                V2.append(v2_val[good]); S.append(s_mlam[good])
                kept += np.count_nonzero(good)
        if kept == 0:
            return np.array([]), np.array([])
        S = np.concatenate(S); V2 = np.concatenate(V2)
        order = np.argsort(S)
        return S[order], V2[order]

    # ---- plotting ---------------------------------------------------
    def _plot_cp(OBJ, label):
        s, cp = _collect_cp_deg_vs_Mlambda(OBJ)
        if s.size == 0:
            print(f"[quick_plots] No CP found for {label}.")
            return
        plt.figure(figsize=(9.2, 4.0))
        plt.plot(s, cp, ".", ms=3.0, alpha=0.7)
        plt.axhline(0, ls="--", lw=1, alpha=0.6)
        plt.xlabel(r"$B/\lambda$ (M$\lambda$)")
        plt.ylabel("CP (deg)")
        plt.title(f"Closure phase — {label}")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"qa_cp_{label}.png")
        plt.savefig(fn, dpi=160); plt.close()
        print("[SAVE]", fn)

    def _plot_v2(OBJ, label):
        s, v2 = _collect_v2_vs_Mlambda(OBJ)
        if s.size == 0:
            print(f"[quick_plots] No V2 found for {label}.")
            return
        plt.figure(figsize=(9.2, 4.0))
        plt.plot(s, v2, ".", ms=3.0, alpha=0.7)
        plt.xlabel(r"$B/\lambda$ (M$\lambda$)")
        plt.ylabel(r"$V^2$")
        if v2_ylim is not None:
            plt.ylim(*v2_ylim)
        plt.title(f"Squared visibility — {label}")
        plt.tight_layout()
        fn = os.path.join(out_dir, f"qa_v2_{label}.png")
        plt.savefig(fn, dpi=160); plt.close()
        print("[SAVE]", fn)

    print("QUICK PLOTS")
    for obj, lab in ((oi, "data"), (oif, "model"), (oi_res, "resid")):
        _plot_cp(obj, lab)
        _plot_v2(obj, lab)

# # Stable works but only CP plots below        
# def quick_plots(oi, oif, oi_res, out_dir, cp_err_max=None):
#     """
#     Make CP vs (B/λ) plots for data/model/residual with
#     - degrees (T3PHI) from PMOIRED
#     - wrap to [-180, 180] deg
#     - honor FLAG per channel
#     - optional error cut on ET3PHI (in deg)
#     - abscissa from each triangle's Bmax/wl (Mλ)
#     """
#     import numpy as np, os, matplotlib.pyplot as plt
#     os.makedirs(out_dir, exist_ok=True)

#     def _wrap_deg(phi_deg):
#         # wrap to [-180, 180]
#         x = (np.asarray(phi_deg, float) + 180.0) % 360.0 - 180.0
#         # put exactly -180 at +180 for nicer plotting
#         x[x <= -180.0] = 180.0
#         return x

#     def _collect_cp_deg_vs_Mlambda(OBJ):
#         S, CP = [], []
#         kept = 0
#         for blk in OBJ.data:
#             if "OI_T3" not in blk:
#                 continue
#             # per-triangle loop
#             for tri, t3 in blk["OI_T3"].items():
#                 # channel arrays (shape often (1, nchan)); ravel to 1-D
#                 cp_deg  = np.asarray(t3.get("T3PHI", []), float).ravel()
#                 ep_deg  = np.asarray(t3.get("ET3PHI", []), float).ravel()
#                 sMlam   = np.asarray(t3.get("Bmax/wl", []), float).ravel() / 1e6  # -> Mλ
#                 flags   = np.asarray(t3.get("FLAG", np.zeros_like(cp_deg, dtype=bool)), bool).ravel()

#                 if cp_deg.size == 0 or sMlam.size == 0:
#                     continue

#                 # align lengths (defensive; PMOIRED is usually consistent)
#                 n = min(cp_deg.size, sMlam.size, flags.size if flags.size else cp_deg.size,
#                         ep_deg.size if ep_deg.size else cp_deg.size)
#                 cp_deg = cp_deg[:n]; sMlam = sMlam[:n]; flags = flags[:n]
#                 if ep_deg.size: ep_deg = ep_deg[:n]

#                 # mask: good channels = not flagged and, if requested, error < cp_err_max
#                 good = ~flags
#                 if cp_err_max is not None and ep_deg.size:
#                     good &= (ep_deg <= float(cp_err_max))

#                 if not np.any(good):
#                     continue

#                 cp_w = _wrap_deg(cp_deg[good])
#                 s_w  = sMlam[good]
#                 S.append(s_w); CP.append(cp_w)
#                 kept += cp_w.size

#         if kept == 0:
#             return np.array([]), np.array([])
#         S = np.concatenate(S); CP = np.concatenate(CP)
#         # sort by spatial frequency for nicer plots
#         order = np.argsort(S)
#         return S[order], CP[order]

#     def _one(OBJ, label):
#         s, cp = _collect_cp_deg_vs_Mlambda(OBJ)
#         if s.size == 0:
#             print(f"[quick_plots] No CP found for {label}.")
#             return
#         plt.figure(figsize=(9.2, 4.0))
#         plt.plot(s, cp, ".", ms=3.0, alpha=0.7)
#         plt.axhline(0, ls="--", lw=1, alpha=0.6)
#         plt.xlabel(r"$B/\lambda$ (M$\lambda$)")
#         plt.ylabel("CP (deg)")
#         plt.title(f"Closure phase — {label}")
#         plt.tight_layout()
#         fn = os.path.join(out_dir, f"qa_cp_{label}.png")
#         plt.savefig(fn, dpi=160)
#         plt.close()
#         print("[SAVE]", fn)

#     print("QUICK PLOTS")
#     _one(oi, "data")
#     _one(oif, "model")
#     _one(oi_res, "resid")


# --- resolve input files from --dir and/or --files ---
def _gather_fits_from_path(p):
    exts = (".fits", ".fit", ".oifits", ".FITS", ".FIT", ".OIFITS")
    out = []
    if os.path.isdir(p):
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.endswith(exts):
                    out.append(os.path.join(root, fn))
    else:
        # treat as file or glob pattern
        out.extend(glob.glob(p))
    return out

def debug_dump_oi_t3_structure(oi, max_epochs=2, max_tris=5):
    print("[DEBUG] Inspecting OI_T3 structure...")
    for i, d in enumerate(oi.data[:max_epochs]):
        t3 = d.get("OI_T3", {})
        print(f"  epoch {i}: #triangles={len(t3)}")
        for tri, blk in list(t3.items())[:max_tris]:
            print(f"    triangle {tri}: keys={sorted(list(blk.keys()))}")
    print("[DEBUG] Done.")



def _extract_t3_arrays(blk):
    """
    Return (T3AMP, ET3AMP, T3PHI_deg, ET3PHI_deg, S) with graceful fallbacks.
    T3PHI is in degrees on output. S is Bmax/lambda (same shape as T3PHI).
    """
    # amplitudes
    A = np.asarray(blk.get("T3AMP"), float) if "T3AMP" in blk else None
    Ae = np.asarray(blk.get("ET3AMP"), float) if "ET3AMP" in blk else None

    # phases (deg); convert if they came in radians by mistake
    phi = None
    for k in ("T3PHI", "t3phi", "CP", "cp"):
        if k in blk:
            arr = np.asarray(blk[k], float)
            phi = np.degrees(arr) if np.nanmax(np.abs(arr)) <= np.pi*1.05 else arr
            break

    phie = None
    for k in ("ET3PHI", "t3phierr", "CPERR", "cperr"):
        if k in blk:
            arr = np.asarray(blk[k], float)
            phie = np.degrees(arr) if np.nanmax(np.abs(arr)) <= np.pi*1.05 else arr
            break

    # spatial frequency Bmax/λ
    S = None
    for k in ("Bmax/wl", "Bmax/wl_data", "Smax", "S"):
        if k in blk:
            S = np.asarray(blk[k], float)
            break

    return A, Ae, phi, phie, S


def _extract_vis_arrays(blk):
    """
    Return (VAMP, EVAMP, VPHI_deg, EVPHI_deg) if present; otherwise (None,...).
    """
    A  = np.asarray(blk.get("VISAMP"), float) if "VISAMP" in blk else None
    Ae = np.asarray(blk.get("EVISAMP"), float) if "EVISAMP" in blk else None

    phi = None
    if "VISPHI" in blk:
        arr = np.asarray(blk["VISPHI"], float)
        phi = np.degrees(arr) if np.nanmax(np.abs(arr)) <= np.pi*1.05 else arr
    phie = None
    if "EVISPHI" in blk:
        arr = np.asarray(blk["EVISPHI"], float)
        phie = np.degrees(arr) if np.nanmax(np.abs(arr)) <= np.pi*1.05 else arr

    return A, Ae, phi, phie

def peek_shapes(oi):
    for ie, ep in enumerate(oi.data):
        WL = np.asarray(ep.get("WL", []), float)
        print(f"Epoch {ie}: WL shape {WL.shape}")
        if "OI_T3" in ep:
            for tri, blk in ep["OI_T3"].items():
                a = np.asarray(blk.get("T3AMP", []))
                p = np.asarray(blk.get("T3PHI", []))
                s = np.asarray(blk.get("Bmax/wl", []))
                print(f"  T3 {tri}: T3AMP{a.shape} T3PHI{p.shape} Bmax/wl{s.shape}")
        if "OI_VIS2" in ep:
            for bl, blk in ep["OI_VIS2"].items():
                v = np.asarray(blk.get("V2", []))
                b = np.asarray(blk.get("B/wl", []))
                print(f"  V2 {bl}: V2{v.shape} B/wl{b.shape}")



# ------ for writing fits with residuals 
from astropy.io import fits

def _wrap_pi(x):
    """wrap to [-pi, pi)"""
    return (x + np.pi) % (2*np.pi) - np.pi

def _to1d(x):
    a = np.asarray(x, dtype=float)
    return np.squeeze(a).ravel()

def _station_maps_from_hdul(hdul):
    """Return {STA_INDEX:int -> STA_NAME:str} and reverse map, plus array name."""
    arr = None
    for h in hdul:
        if h.name == "OI_ARRAY":
            arr = h
            break
    if arr is None:
        raise RuntimeError("No OI_ARRAY HDU found")
    data = arr.data
    # OI-FITS v2: STA_INDEX (I), STA_NAME (A??), TEL_NAME, STA_X/Y/Z etc
    sta_idx = np.asarray(data["STA_INDEX"], int)
    try:
        sta_name = np.asarray(data["STA_NAME"]).astype(str)
    except Exception:
        # Some files store station labels in TEL_NAME
        sta_name = np.asarray(data["TEL_NAME"]).astype(str)
    idx2name = {int(i): str(n).strip() for i, n in zip(sta_idx, sta_name)}
    name2idx = {v: k for k, v in idx2name.items()}
    return idx2name, name2idx

def _row_baseline_key(row, idx2name):
    i1, i2 = int(row["STA_INDEX"][0]), int(row["STA_INDEX"][1])
    n1, n2 = idx2name[i1], idx2name[i2]
    # Sort so it matches pmoired's normalized naming convention
    return "".join(sorted([n1, n2]))

def _row_triangle_key(row, idx2name):
    i1, i2, i3 = (int(row["STA_INDEX"][0]), int(row["STA_INDEX"][1]), int(row["STA_INDEX"][2]))
    names = sorted([idx2name[i1], idx2name[i2], idx2name[i3]])
    return "".join(names)

def _build_epoch_lookup_from_oires(oi_res):
    """
    Build lookups:
      v2LUT[epoch][baseline]['V2'] -> (Nlam,)
      t3LUT[epoch][triangle]['T3PHI_deg'] -> (Nlam,)
      t3LUT[epoch][triangle]['T3AMP']     -> (Nlam,)  (if available)
    """
    v2LUT, t3LUT = [], []
    for ep in oi_res.data:
        v2_this = {}
        if "OI_VIS2" in ep:
            for bl, blk in ep["OI_VIS2"].items():
                v2_this[bl] = {
                    "V2": _to1d(blk.get("V2", []))
                }
        t3_this = {}
        if "OI_T3" in ep:
            for tri, blk in ep["OI_T3"].items():
                t3_this[tri] = {
                    "T3PHI_deg": _to1d(blk.get("T3PHI", [])),
                    "T3AMP":     _to1d(blk.get("T3AMP", [])) if "T3AMP" in blk else None
                }
        v2LUT.append(v2_this)
        t3LUT.append(t3_this)
    return v2LUT, t3LUT

def write_residual_oifits(obs_files, oi, oi_res, out_dir,
                          vis2_mode="ratio",       # 'ratio' or 'absolute'
                          t3_mode="phase_only",    # 'phase_only' or 'ratio'
                          copy_vis=True,
                          overwrite=True):
    """
    Create OI-FITS residual files, one per input OIFITS, by replacing observable columns.

    Parameters
    ----------
    obs_files : list[str]
        Original OIFITS filepaths used to build `oi`. Must be same ordering/length as `oi.data`.
    oi : pmoired.OI
        Observational dataset (used only to ensure epoch count/order).
    oi_res : pmoired.OI
        Residual dataset produced by your divide/subtract routine (same epoch order as `oi`).
    out_dir : str
        Destination directory for *_residual.fits files.
    vis2_mode : {'ratio','absolute'}
        If 'ratio' (recommended): write VIS2DATA_residual = your residual (data/model).
        If 'absolute': multiply original VIS2DATA by residual ratio and write absolute corrected V2.
    t3_mode : {'phase_only','ratio'}
        'phase_only' (recommended): write residual T3PHI (deg→rad), keep original T3AMP.
        'ratio': also write amplitude ratio to T3AMP (data/model).
    copy_vis : bool
        If True and OI_VIS exists, copy phases unchanged (or you can extend to correct them too).
    overwrite : bool
        Overwrite existing files.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(obs_files) != len(oi.data) or len(oi.data) != len(oi_res.data):
        print("[WARN] Epoch count mismatch between files/oi/oi_res; proceeding by index, but check alignment!")

    v2LUT, t3LUT = _build_epoch_lookup_from_oires(oi_res)

    for ie, in_f in enumerate(obs_files):
        with fits.open(in_f, mode="readonly") as hdul:
            idx2name, name2idx = _station_maps_from_hdul(hdul)

            # VIS2 block(s)
            for h in hdul:
                if h.name == "OI_VIS2":
                    data = h.data
                    for ir in range(len(data)):
                        row = data[ir]
                        key = _row_baseline_key(row, idx2name)  # e.g. 'A0B2'
                        resid_v2 = v2LUT[ie].get(key, {}).get("V2", None)
                        if resid_v2 is None or resid_v2.size == 0:
                            continue
                        nw = row["VIS2DATA"].shape[-1]
                        # Ensure same channel count; pad/truncate as needed
                        r = resid_v2
                        if r.size < nw:
                            r = np.pad(r, (0, nw - r.size), constant_values=np.nan)
                        elif r.size > nw:
                            r = r[:nw]

                        if vis2_mode == "ratio":
                            # Write the ratio directly; CANDID can handle V2≠[0..1] if you prefer absolute—choose!
                            row["VIS2DATA"] = r.astype(row["VIS2DATA"].dtype)
                        elif vis2_mode == "absolute":
                            row["VIS2DATA"] = (row["VIS2DATA"] * r).astype(row["VIS2DATA"].dtype)
                        else:
                            raise ValueError("vis2_mode must be 'ratio' or 'absolute'")

                        # Leave VIS2ERR unchanged (or you can propagate)
                        data[ir] = row

            # T3 block(s)
            for h in hdul:
                if h.name == "OI_T3":
                    data = h.data
                    for ir in range(len(data)):
                        row = data[ir]
                        key = _row_triangle_key(row, idx2name)  # e.g. 'A0B2C1'
                        resid = t3LUT[ie].get(key, None)
                        if resid is None:
                            continue
                        cp_deg = resid.get("T3PHI_deg", None)
                        if cp_deg is not None and cp_deg.size > 0:
                            nw = row["T3PHI"].shape[-1]
                            cp = cp_deg.copy()
                            if cp.size < nw:
                                cp = np.pad(cp, (0, nw - cp.size), constant_values=np.nan)
                            elif cp.size > nw:
                                cp = cp[:nw]
                            # OI-FITS expects radians:
                            row["T3PHI"] = _wrap_pi(np.deg2rad(cp)).astype(row["T3PHI"].dtype)

                        if t3_mode == "ratio":
                            amp = resid.get("T3AMP", None)
                            if amp is not None and amp.size > 0:
                                nw = row["T3AMP"].shape[-1]
                                a = amp.copy()
                                if a.size < nw:
                                    a = np.pad(a, (0, nw - a.size), constant_values=np.nan)
                                elif a.size > nw:
                                    a = a[:nw]
                                # If your residual T3AMP is a ratio, choose either to write the ratio or absolute:
                                row["T3AMP"] = a.astype(row["T3AMP"].dtype)
                        elif t3_mode == "phase_only":
                            # leave T3AMP unchanged
                            pass
                        else:
                            raise ValueError("t3_mode must be 'phase_only' or 'ratio'")

                        data[ir] = row

            # (Optional) VIS complex block — here we just copy through, or you can implement phase corrections
            if copy_vis:
                pass

            # Write new file
            base = os.path.basename(in_f)
            out_f = os.path.join(out_dir, base.replace(".fits", "_residual.fits"))
            if os.path.exists(out_f) and not overwrite:
                print("[SKIP] exists:", out_f)
            else:
                hdul.writeto(out_f, overwrite=True)
                print("[SAVE]", out_f)



##################################################
### re-writing original fits and the model fits (for verification with original fits )
def _name_from_sta_indices(hdul, idxs):
    """Return baseline or triangle key from integer STA_INDEX array in a FITS row."""
    arr = hdul["OI_ARRAY"].data
    # Prefer STA_NAME if present, else TEL_NAME
    if "STA_NAME" in arr.names:
        names = np.array(arr["STA_NAME"]).astype(str)
    elif "TEL_NAME" in arr.names:
        names = np.array(arr["TEL_NAME"]).astype(str)
    else:
        raise KeyError("Neither STA_NAME nor TEL_NAME in OI_ARRAY.")
    return "".join(sorted([names[i-1].strip() for i in np.atleast_1d(idxs)]))


def _find_best_pm_block_for_file(hdul, pm_obj_source):
    """
    Return the epoch index in the PMOIRED object whose WL count best matches the FITS file.
    """
    # FITS: nchan in OI_WAVELENGTH (EFF_WAVE or EFF_WL)
    wlcol, _ = _wl_cols(hdul)
    if wlcol is None:
        raise KeyError("OI_WAVELENGTH table missing in FITS.")
    nchan_f = len(hdul["OI_WAVELENGTH"].data[wlcol])

    # PMOIRED: WL per epoch is at ep["WL"]
    best_j, best_d = None, 1e9
    for j, ep in enumerate(getattr(pm_obj_source, "data", [])):
        nchan_pm = len(ep.get("WL", []))
        d = abs(nchan_pm - nchan_f)
        if d < best_d:
            best_d, best_j = d, j
    if best_j is None:
        raise RuntimeError("Could not match PMOIRED epoch by channel count.")
    return best_j

# def _find_best_pm_block_for_file(hdul, pm_obj_source):
#     """
#     Pick the PMOIRED epoch block that best corresponds to this FITS file.
#     Heuristic: match INSNAME and #channels; fall back to closest #channels.
#     """
#     import numpy as np

#     # Try to read INSNAME from a science table header to route wavelengths correctly
#     ins_from_file = None
#     for hdu in hdul:
#         if getattr(hdu, "name", "") in ("OI_VIS2", "OI_T3", "OI_VIS"):
#             ins_from_file = hdu.header.get("INSNAME", None)
#             if ins_from_file:
#                 break

#     # Robust wavelength read (meters), regardless of column naming
#     wl_file = _eff_wl_from_hdul(hdul, prefer_insname=ins_from_file)
#     nchan_f = int(np.asarray(wl_file).size)

#     # Walk PMOIRED epochs; each has a per-epoch WL vector (meters)
#     best_idx, best_score = None, +np.inf
#     for j, ep in enumerate(getattr(pm_obj_source, "data", [])):
#         wl_pm = np.asarray(ep.get("WL", []), float).ravel()
#         if wl_pm.size == 0:
#             continue
#         # Score by channel count difference; exact match wins
#         score = abs(wl_pm.size - nchan_f)
#         if score < best_score:
#             best_score, best_idx = score, j
#             if score == 0:
#                 break  # perfect match

#     if best_idx is None:
#         raise RuntimeError("No suitable PMOIRED epoch found to match file wavelengths.")

#     # Optional: sanity debug
#     # print(f"[DEBUG] Matched file INSNAME={ins_from_file} nchan={nchan_f} to epoch {best_idx} (nchan={len(pm_obj_source.data[best_idx].get('WL', []))})")
#     return best_idx

# def _find_best_pm_block_for_file(hdul, pm_obj):
#     """Choose the PMOIRED block in pm_obj.data that best matches this file."""
#     # Collect file-side station set & WL length & MJD
#     arr = hdul["OI_ARRAY"].data
#     if "STA_NAME" in arr.names:
#         sta_names = {s.strip() for s in arr["STA_NAME"].astype(str)}
#     else:
#         sta_names = {s.strip() for s in arr["TEL_NAME"].astype(str)}

#     # WL in file: assume single OI_WAVELENGTH table
#     nchan_f = len(hdul["OI_WAVELENGTH"].data["EFF_WL"])
#     mjd_f = None
#     if "OI_T3" in hdul:
#         tab = hdul["OI_T3"].data
#         if "MJD" in tab.names:
#             mjd_f = float(np.nanmedian(tab["MJD"]))
#     if mjd_f is None and "OI_VIS2" in hdul:
#         tab = hdul["OI_VIS2"].data
#         if "MJD" in tab.names:
#             mjd_f = float(np.nanmedian(tab["MJD"]))

#     # Score PMOIRED blocks
#     best = (-1, 1e9, None)   # (n_common_stations, |ΔMJD|, block_index)
#     for j, blk in enumerate(pm_obj.data):
#         # station set from PMOIRED block
#         if "baselines" in blk:
#             stas = set("".join(sorted(k)) for k in blk["baselines"])
#             # Expand to unique station letters
#             stas_flat = set("".join(sorted(list("".join(stas)))))
#         else:
#             stas_flat = set()
#         nchan_b = len(blk.get("WL", []))
#         if nchan_b != nchan_f:
#             continue
#         mjd_b = float(np.nanmedian(blk.get("MJD", [np.nan])))
#         n_common = len(stas_flat.intersection(sta_names)) if stas_flat else 0
#         d_mjd = abs((mjd_b - mjd_f)) if mjd_f is not None and np.isfinite(mjd_b) else 1e6
#         score = (n_common, -d_mjd)  # prefer more common stations; then closer MJD
#         if (score[0] > best[0]) or (score[0] == best[0] and -score[1] > -best[1]):
#             best = (score[0], -score[1], j)
#     return best[2]  # may be None


def _replace_vis2_columns(hdu_vis2, pm_epoch_vis2_dict, mode="data", time_tol_days=0.002):
    """
    Overwrite OI_VIS2 data in-place, matching each FITS row to the correct baseline
    by nearest (UCOORD,VCOORD) and (optionally) MJD proximity.
      mode:
        "data"  → write pm[V2], pm[EV2], pm[FLAG]
        "model" → write pm[V2], pm[EV2] (errors may be NaN)
        "resid" → write pm[V2] (ratio), EV2→NaN
      time_tol_days: if not None, require |ΔMJD| <= tol when MJD present on both sides.
    """
    t = hdu_vis2.data
    v2col, ev2col, ucol, vcol, mjdc, flagcol = _vis2_cols(hdu_vis2)
    if v2col is None or ucol is None or vcol is None:
        raise KeyError("VIS2 table missing required columns (VIS2DATA/V2, UCOORD, VCOORD).")

    nrows = len(t)
    for i in range(nrows):
        urow = float(np.asarray(t[ucol][i]).squeeze())
        vrow = float(np.asarray(t[vcol][i]).squeeze())
        mjdr = float(np.asarray(t[mjdc][i]).squeeze()) if mjdc else None

        key = _nearest_baseline_timeaware(pm_epoch_vis2_dict, urow, vrow, mjdr, time_tol_days)
        if key is None:
            continue

        pm = pm_epoch_vis2_dict[key]
        V2  = np.asarray(pm.get("V2",  []), float).ravel()
        EV2 = np.asarray(pm.get("EV2", []), float).ravel()
        FLG = np.asarray(pm.get("FLAG", np.zeros_like(V2, bool)), bool).ravel()

        nchan = len(t[v2col][i])

        def cut(a):
            if a.size == 0: return np.full(nchan, np.nan, float)
            if a.size < nchan:
                out = np.full(nchan, np.nan, float)
                out[:a.size] = a
                return out
            return a[:nchan]

        v2_out  = cut(V2)
        ev2_out = cut(EV2)
        flg_out = cut(FLG).astype(bool)

        if mode == "model":
            pass
        elif mode == "resid":
            ev2_out = np.full_like(v2_out, np.nan)
        elif mode == "data":
            pass
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        t[v2col][i] = v2_out
        if ev2col:  t[ev2col][i] = ev2_out
        if flagcol: t[flagcol][i] = flg_out

# def _replace_vis2_columns(hdu_vis2, pm_epoch_vis2_dict, mode="data"):
#     """
#     Overwrite the V2 part of an OI_VIS2 HDU using one PMOIRED epoch dict:
#       - hdu_vis2: astropy HDU for OI_VIS2 (one file)
#       - pm_epoch_vis2_dict: dict like ep["OI_VIS2"] (keys=baseline labels)
#       - mode: "data" (copy data), "model" (write model), or "resid" (write residuals/ratios)
#     We assume the order in pm_epoch_vis2_dict matches the row order well enough for
#     your PIONIER set; we also clip by channel length per row.
#     """
#     t = hdu_vis2.data
#     v2col, e2col, flagcol = _vis2_cols(hdu_vis2)
#     if v2col is None:
#         raise KeyError("No VIS2DATA/V2 column present in this OI_VIS2 HDU.")

#     # Gather pm arrays in a stable order:
#     pm_blocks = list(pm_epoch_vis2_dict.values())  # order assumed consistent
#     nrows = len(t)
#     nr = min(nrows, len(pm_blocks))
#     for i in range(nr):
#         pm = pm_blocks[i]
#         # PMOIRED names
#         v2_pm  = np.asarray(pm.get("V2", []),  float).ravel()
#         ev2_pm = np.asarray(pm.get("EV2", []), float).ravel()
#         fl_pm  = np.asarray(pm.get("FLAG", np.zeros_like(v2_pm, bool)), bool).ravel()

#         # What is the number of channels in the FITS row?
#         row = t[i]
#         v2_row = np.atleast_1d(row[v2col])
#         nchan = v2_row.size

#         # Clip PMOIRED arrays to this row shape
#         if v2_pm.size == 0:
#             continue
#         v2_pm  = v2_pm[:nchan]
#         ev2_pm = (ev2_pm[:nchan] if ev2_pm.size else np.full(nchan, np.nan))
#         fl_pm  = (fl_pm[:nchan]  if fl_pm.size  else np.zeros(nchan, bool))

#         # Decide what to write depending on label/mode
#         if mode == "data":
#             out_v2  = v2_pm
#             out_e2  = ev2_pm
#         elif mode == "model":
#             out_v2  = v2_pm          # we interpret these as model V²
#             out_e2  = np.full(nchan, np.nan)
#         elif mode == "resid":
#             # If you built residuals as ratios (data/model): write that
#             out_v2  = v2_pm
#             out_e2  = ev2_pm
#         else:
#             raise ValueError(f"Unknown mode '{mode}'")

#         # Write back
#         row[v2col] = out_v2
#         if e2col:
#             row[e2col] = out_e2
#         if flagcol:
#             row[flagcol] = fl_pm

#     # If PM fewer rows than FITS, we leave remaining rows untouched.

# def _replace_vis2_columns(hdul, blk_pm, mode="data"):
#     """
#     Replace VIS2/VIS2ERR in-place from PMOIRED block.
#     mode: "data", "model", "resid", "absolute" (explicit)
#     """
#     if "OI_VIS2" not in hdul: return
#     t = hdul["OI_VIS2"].data
#     # Walk rows; match baseline name and copy arrays
#     for i in range(len(t)):
#         if "STA_INDEX" in t.names:
#             bl = _name_from_sta_indices(hdul, [t["STA_INDEX"][i][0], t["STA_INDEX"][i][1]])
#         else:
#             # fallback: if baseline name already stored in table
#             bl = str(t["BASELINE"][i]).strip() if "BASELINE" in t.names else None
#         n = len(t["VIS2"][i])

#         # source arrays from PMOIRED block
#         src = blk_pm.get("OI_VIS2", {}).get(bl, {})
#         v2_data = np.asarray(src.get("V2", np.full(n, np.nan)), float).ravel()[:n]
#         ev2     = np.asarray(src.get("EV2", np.full(n, np.nan)), float).ravel()[:n]

#         if mode in ("data","absolute"):
#             new_v2 = v2_data
#         elif mode == "model":
#             # put model V2 (if unavailable, leave NaN)
#             new_v2 = v2_data
#         elif mode == "resid":
#             # residuals already computed in your oi_res object as absolute V2
#             new_v2 = v2_data
#         else:
#             new_v2 = v2_data

#         # write back (pad/truncate to length)
#         vv = np.array(t["VIS2"][i], dtype=float)
#         ee = np.array(t["VIS2ERR"][i], dtype=float)
#         vv[:len(new_v2)] = new_v2
#         if np.isfinite(ev2).any():
#             ee[:len(ev2)] = ev2
#         t["VIS2"][i] = vv
#         t["VIS2ERR"][i] = ee

def _replace_t3_columns(hdu_t3, pm_epoch_t3_dict, mode="data", time_tol_days=0.002):
    """
    Overwrite OI_T3 closure quantities in-place, matching each FITS row by nearest
    (U1COORD,V1COORD,U2COORD,V2COORD) and (optionally) MJD proximity.
    For 'resid', prefer 'CPdeg' from PM block when present (wrapped residual).
    """
    t = hdu_t3.data
    ccol, ecol, acol, aecol, u1c, v1c, u2c, v2c, mjdc, flagc = _t3_cols(hdu_t3)
    if (ccol is None and acol is None) or u1c is None or v1c is None or u2c is None or v2c is None:
        raise KeyError("OI_T3 table missing required columns.")

    nrows = len(t)
    for i in range(nrows):
        u1r = float(np.asarray(t[u1c][i]).squeeze())
        v1r = float(np.asarray(t[v1c][i]).squeeze())
        u2r = float(np.asarray(t[u2c][i]).squeeze())
        v2r = float(np.asarray(t[v2c][i]).squeeze())
        mjdr = float(np.asarray(t[mjdc][i]).squeeze()) if mjdc else None

        key = _nearest_triangle_timeaware(pm_epoch_t3_dict, u1r, v1r, u2r, v2r, mjdr, time_tol_days)
        if key is None:
            continue

        pm = pm_epoch_t3_dict[key]

        cp_deg   = np.asarray(pm.get("T3PHI",  []), float).ravel()
        ecp_deg  = np.asarray(pm.get("ET3PHI", []), float).ravel()
        cp_resid = np.asarray(pm.get("CPdeg",  []), float).ravel()  # residual CP (deg) if we computed it
        amp   = np.asarray(pm.get("T3AMP",  []), float).ravel()
        eamp  = np.asarray(pm.get("ET3AMP", []), float).ravel()
        flg   = np.asarray(pm.get("FLAG", np.zeros_like(cp_deg, bool)), bool).ravel()

        nchan = len(t[ccol][i]) if ccol else (len(t[acol][i]) if acol else 0)

        def cut(a):
            if a.size == 0: return np.full(nchan, np.nan, float)
            if a.size < nchan:
                out = np.full(nchan, np.nan, float)
                out[:a.size] = a
                return out
            return a[:nchan]

        if mode == "data":
            cp_out  = cut(cp_deg)
            ecp_out = cut(ecp_deg)
            amp_out = cut(amp)
            eamp_out= cut(eamp)
        elif mode == "model":
            cp_out  = cut(cp_deg)
            ecp_out = np.full(nchan, np.nan)
            amp_out = cut(amp)
            eamp_out= np.full(nchan, np.nan)
        elif mode == "resid":
            src_cp  = cp_resid if cp_resid.size else cp_deg
            cp_out  = cut(src_cp)
            ecp_out = np.full(nchan, np.nan)
            amp_out = cut(amp)           # if you stored a T3AMP ratio; otherwise harmless
            eamp_out= np.full(nchan, np.nan)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        if ccol:   t[ccol][i]   = cp_out
        if ecol:   t[ecol][i]   = ecp_out
        if acol:   t[acol][i]   = amp_out
        if aecol:  t[aecol][i]  = eamp_out
        if flagc:  t[flagc][i]  = cut(flg).astype(bool)

# def _replace_t3_columns(hdu_t3, pm_epoch_t3_dict, mode="data"):
#     """
#     Overwrite OI_T3 T3PHI/T3AMP columns from a PMOIRED epoch dict.
#     """
#     t = hdu_t3.data
#     ccol, ecol, acol, aecol, flagcol = _t3_cols(hdu_t3)
#     if ccol is None and acol is None:
#         raise KeyError("No T3 columns (T3PHI/T3AMP) found in this OI_T3 HDU.")

#     pm_blocks = list(pm_epoch_t3_dict.values())
#     nrows = len(t)
#     nr = min(nrows, len(pm_blocks))
#     for i in range(nr):
#         pm = pm_blocks[i]
#         cp   = np.asarray(pm.get("T3PHI", []),  float).ravel()
#         ecp  = np.asarray(pm.get("ET3PHI", []), float).ravel()
#         amp  = np.asarray(pm.get("T3AMP", []),  float).ravel()
#         eamp = np.asarray(pm.get("ET3AMP", []), float).ravel()
#         flg  = np.asarray(pm.get("FLAG",  np.zeros_like(cp, bool)), bool).ravel()

#         # Channel length from one available column in the FITS row
#         ref = None
#         for k in (ccol, acol):
#             if k:
#                 ref = np.atleast_1d(t[k][i])
#                 break
#         nchan = ref.size if ref is not None else cp.size

#         cp   = cp[:nchan]   if cp.size   else np.full(nchan, np.nan)
#         ecp  = ecp[:nchan]  if ecp.size  else np.full(nchan, np.nan)
#         amp  = amp[:nchan]  if amp.size  else np.full(nchan, np.nan)
#         eamp = eamp[:nchan] if eamp.size else np.full(nchan, np.nan)
#         flg  = flg[:nchan]  if flg.size  else np.zeros(nchan, bool)

#         if mode == "data":
#             out_cp, out_ecp = cp, ecp
#             out_amp, out_eamp = amp, eamp
#         elif mode == "model":
#             out_cp, out_ecp = cp, np.full(nchan, np.nan)
#             out_amp, out_eamp = amp, np.full(nchan, np.nan)
#         elif mode == "resid":
#             out_cp, out_ecp = cp, ecp
#             out_amp, out_eamp = amp, eamp
#         else:
#             raise ValueError(f"Unknown mode '{mode}'")

#         if ccol:   t[ccol][i]   = out_cp
#         if ecol:   t[ecol][i]   = out_ecp
#         if acol:   t[acol][i]   = out_amp
#         if aecol:  t[aecol][i]  = out_eamp
#         if flagcol:t[flagcol][i]= flg

# def _replace_t3_columns(hdul, blk_pm, mode="data"):
#     """
#     Replace T3PHI/T3AMP and errors in-place from PMOIRED block.
#     Assumes PMOIRED stores degrees for T3PHI/ET3PHI.
#     """
#     if "OI_T3" not in hdul: return
#     t = hdul["OI_T3"].data
#     for i in range(len(t)):
#         if "STA_INDEX" in t.names:
#             tri = _name_from_sta_indices(hdul, t["STA_INDEX"][i])
#         else:
#             tri = None
#         n = len(t["T3PHI"][i])

#         src = blk_pm.get("OI_T3", {}).get(tri, {})
#         phi = np.asarray(src.get("T3PHI", np.full(n, np.nan)), float).ravel()[:n]  # deg
#         ephi= np.asarray(src.get("ET3PHI", np.full(n, np.nan)), float).ravel()[:n]
#         amp = np.asarray(src.get("T3AMP", np.full(n, np.nan)), float).ravel()[:n]
#         eamp= np.asarray(src.get("ET3AMP", np.full(n, np.nan)), float).ravel()[:n]

#         # choose payload
#         if mode in ("data","absolute","model","resid"):
#             new_phi = phi   # already absolute phase in deg for data/model/resid
#             new_amp = amp
#         else:
#             new_phi = phi; new_amp = amp

#         # write back (preserve shape)
#         PHI = np.array(t["T3PHI"][i], dtype=float); PHI[:len(new_phi)] = new_phi
#         t["T3PHI"][i] = PHI
#         if "T3AMP" in t.names:
#             AMP = np.array(t["T3AMP"][i], dtype=float); AMP[:len(new_amp)] = new_amp
#             t["T3AMP"][i] = AMP
#         if "T3PHIERR" in t.names:
#             EPH = np.array(t["T3PHIERR"][i], dtype=float); EPH[:len(ephi)] = ephi
#             t["T3PHIERR"][i] = EPH
#         elif "ET3PHI" in t.names:  # PMOIRED naming sometimes mirrored in OIFITS
#             EPH = np.array(t["ET3PHI"][i], dtype=float); EPH[:len(ephi)] = ephi
#             t["ET3PHI"][i] = EPH
#         if "T3AMPERR" in t.names and eamp.size:
#             EAM = np.array(t["T3AMPERR"][i], dtype=float); EAM[:len(eamp)] = eamp
#             t["T3AMPERR"][i] = EAM


def write_oifits_variant(file_list, pm_obj_source, pm_obj_variant, label,  dest_dir, overwrite=False):
    """
    label ∈ {"data","model","resid"}
    - pm_obj_source: the PMOIRED object used to match epochs/chan counts
    - pm_obj_variant: the PMOIRED object whose arrays you want to write
    """
    import os
    from astropy.io import fits


    os.makedirs(dest_dir, exist_ok=True)

    subdir = ensure_dir(os.path.join(dest_dir, label))

    # TYPE GUARD: both must be PMOIRED-like objects with a .data list
    for name, obj in (("pm_obj_source", pm_obj_source), ("pm_obj_variant", pm_obj_variant)):
        if not hasattr(obj, "data") or not isinstance(getattr(obj, "data", None), list):
            raise TypeError(f"{name} must be a PMOIRED OI object with a '.data' list; "
                            f"got {type(obj).__name__}")
        
    # for f in file_list:
    #     with fits.open(f, mode="readonly") as hdul:
    
    for j, orig_path in enumerate(file_list):
        with fits.open(orig_path, memmap=False) as hdul:

            # pick epoch index that matches this file’s channel count
            j = _find_best_pm_block_for_file(hdul, pm_obj_source)
            ep = pm_obj_variant.data[j]  # PMOIRED epoch to take from

            # make a copy of the fits HDU list to modify
            hdunew = fits.HDUList([h.copy() for h in hdul])

            # VIS2 (present)
            if "OI_VIS2" in hdunew and "OI_VIS2" in ep:
                _replace_vis2_columns(hdunew["OI_VIS2"], ep["OI_VIS2"], mode=label)

            # VIS (may be missing in your PIONIER set)
            if "OI_VIS" in hdunew and "OI_VIS" in ep:
                # If you want to support VIS later, add a _replace_vis_columns
                pass  # or implement similarly with aliases

            # T3 (present)
            if "OI_T3" in hdunew and "OI_T3" in ep:
                _replace_t3_columns(hdunew["OI_T3"], ep["OI_T3"], mode=label)

            # # output path
            # base = os.path.basename(f)
            # outn = base.replace(".fits", f".{label}.fits")
            # outp = os.path.join(os.path.dirname(f), outn)
            # hdunew.writeto(outp, overwrite=overwrite)
            # print("[WRITE]", label, "→", outp)

            # build output path in the **dest_dir**
            base = os.path.basename(orig_path)
            stem = base[:-5] if base.lower().endswith(".fits") else base
            out_path = os.path.join( subdir  ,f"{stem}.{label}.fits")

            hdul.writeto(out_path, overwrite=overwrite)
            print(f"[WRITE] {label:5s} → {out_path}")

# def write_oifits_variant(file_list, pm_obj_source, out_dir, label="data", overwrite=True):
#     """
#     Rewrites each input OI-FITS into out_dir/{basename}.{label}.fits replacing:
#       - OI_VIS2 VIS2/VIS2ERR
#       - OI_T3   T3PHI/T3AMP (+ errors if present)
#     from the PMOIRED object 'pm_obj_source' (oi, oif, or oi_res).
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     for fin in file_list:
#         with fits.open(fin) as hdul:
#             j = _find_best_pm_block_for_file(hdul, pm_obj_source)
#             if j is None:
#                 print(f"[WARN] No matching PMOIRED block for {os.path.basename(fin)} — skipping.")
#                 continue
#             # deep copy to modify
#             hdun = fits.HDUList([h.copy() for h in hdul])

#             blk_pm = pm_obj_source.data[j]

#             # Replace columns in-place
#             _replace_vis2_columns(hdun, blk_pm, mode=label)
#             _replace_t3_columns(hdun, blk_pm, mode=label)

#             # Minimal provenance
#             if "PRIMARY" in hdun:
#                 hdr = hdun[0].header
#                 hdr["HIERARCH SUBPROC"] = (f"write_oifits_variant {label}", "filled VIS2/T3 from PMOIRED")

#             base = os.path.basename(fin)
#             fout = os.path.join(out_dir, base.replace(".fits", f".{label}.fits"))
#             hdun.writeto(fout, overwrite=overwrite)
#             print("[SAVE]", fout)

# def write_oifits_triplet(file_list, oi, oif, oi_res, out_dir, overwrite=True):
#     """
#     Write three parallel OI-FITS sets: .data.fits, .model.fits, .resid.fits
#     using identical mapping logic.
#     """
#     out_data  = os.path.join(out_dir, "oifits_data")
#     out_model = os.path.join(out_dir, "oifits_model")
#     out_resid = os.path.join(out_dir, "oifits_resid")
#     for d in (out_data, out_model, out_resid):
#         os.makedirs(d, exist_ok=True)


# def write_oifits_triplet(file_list, pm_obj_source, pm_obj_model, pm_obj_resid,
#                          out_dir, overwrite=False):

# def write_oifits_triplet(file_list,
#                          pm_ref,       # PMOIRED OI used to match epochs/channels (usually the original data 'oi')
#                          pm_data,      # PMOIRED OI whose arrays you want in the ".data.fits" files (often same as pm_ref)
#                          pm_model,     # PMOIRED OI → arrays for ".model.fits"
#                          pm_resid,     # PMOIRED OI → arrays for ".resid.fits"
#                          out_dir, 
#                          overwrite=False):
#     os.makedirs(out_dir, exist_ok=True)
#     # data (sanity copy)
#     write_oifits_variant(file_list, pm_data, label="data",
#                          dest_dir=out_dir, overwrite=overwrite)
#     # model
#     write_oifits_variant(file_list, pm_model,  label="model",
#                          dest_dir=out_dir, overwrite=overwrite)
#     # residual
#     write_oifits_variant(file_list, pm_resid,  label="resid",
#                          dest_dir=out_dir, overwrite=overwrite)
    
    
def write_oifits_triplet(file_list,
                         pm_ref,       # PMOIRED OI used to match epochs/channels (usually the original data 'oi')
                         pm_data,      # PMOIRED OI whose arrays you want in the ".data.fits" files (often same as pm_ref)
                         pm_model,     # PMOIRED OI → arrays for ".model.fits"
                         pm_resid,     # PMOIRED OI → arrays for ".resid.fits"
                         out_dir,
                         overwrite=False):
    """
    Write three OIFITS variants next to the originals:
      • *.data.fits  — arrays from pm_data
      • *.model.fits — arrays from pm_model
      • *.resid.fits — arrays from pm_resid
    All files are matched to the original FITS’ channel counts using `pm_ref`.
    """

    write_oifits_variant(file_list, pm_ref, pm_data,  label="data",  dest_dir=out_dir, overwrite=overwrite)
    write_oifits_variant(file_list, pm_ref, pm_model, label="model", dest_dir=out_dir, overwrite=overwrite)
    write_oifits_variant(file_list, pm_ref, pm_resid, label="resid", dest_dir=out_dir,overwrite=overwrite)

    # print("[WRITE] re-written data OIFITS (sanity copy)")
    # write_oifits_variant(file_list, oi,     out_data,  label="data",  overwrite=overwrite)

    # print("[WRITE] model OIFITS")
    # write_oifits_variant(file_list, oif,    out_model, label="model", overwrite=overwrite)

    # print("[WRITE] residual OIFITS")
    # write_oifits_variant(file_list, oi_res, out_resid, label="resid", overwrite=overwrite)


def _eff_wl_from_hdul(hdul, prefer_insname=None):
    """
    Return the 1-D array of effective wavelengths (meters) from an OIFITS HDUList.
    Respects INSNAME routing when multiple OI_WAVELENGTH tables exist.
    """
    import numpy as np
    candidates = []
    # First pass: gather OI_WAVELENGTH HDUs that match INSNAME (if provided)
    for hdu in hdul:
        if getattr(hdu, "name", "") != "OI_WAVELENGTH":
            continue
        if prefer_insname and hdu.header.get("INSNAME", "") != prefer_insname:
            continue
        cols = list(hdu.data.columns.names)
        # Try common column name variants (OIFITS v2 is EFF_WAVE)
        for col in ("EFF_WAVE", "EFF_WL", "WAVELENGTH", "LAMBDA", "EFFLAMBDA", "WAVE"):
            if col in cols:
                wl = np.asarray(hdu.data[col], float).ravel()
                if wl.size:
                    candidates.append((wl, hdu.header.get("INSNAME","")))
                break

    # If nothing matched INSNAME, allow any OI_WAVELENGTH
    if not candidates:
        for hdu in hdul:
            if getattr(hdu, "name", "") != "OI_WAVELENGTH":
                continue
            cols = list(hdu.data.columns.names)
            for col in ("EFF_WAVE", "EFF_WL", "WAVELENGTH", "LAMBDA", "EFFLAMBDA", "WAVE"):
                if col in cols:
                    wl = np.asarray(hdu.data[col], float).ravel()
                    if wl.size:
                        candidates.append((wl, hdu.header.get("INSNAME","")))
                    break

    if not candidates:
        # Helpful error with column names we actually found
        found = []
        for hdu in hdul:
            if getattr(hdu, "name", "") == "OI_WAVELENGTH":
                found.extend(hdu.data.columns.names)
        raise KeyError(f"No wavelength column found in OI_WAVELENGTH. Columns present: {sorted(set(found))}")

    # Prefer the one with matching INSNAME if present
    if prefer_insname:
        for wl, ins in candidates:
            if ins == prefer_insname:
                return wl
    # Else return the first candidate
    return candidates[0][0]

# diagnostics 
# --- DIAG 1: what columns exist in your FITS tables? ---
from astropy.io import fits
import glob, os

def inspect_oifits_columns(file_glob):
    files = sorted(sum([glob.glob(g) for g in ([file_glob] if isinstance(file_glob,str) else file_glob)], []))
    for f in files[:3]:   # show a few
        with fits.open(f) as hdul:
            print("\nFILE:", os.path.basename(f))
            for name in ("OI_WAVELENGTH","OI_VIS2","OI_VIS","OI_T3"):
                if name in hdul:
                    cols = list(hdul[name].columns.names)
                    print(f"  {name} cols:", cols)
                else:
                    print(f"  {name}: MISSING")

# Example:
# inspect_oifits_columns("/path/to/*.fits")
# --- DIAG 2: what keys are inside your PMOIRED objects? ---
def inspect_pmoi_blocks(oi):
    for ie, ep in enumerate(getattr(oi, "data", [])):
        print(f"\nEpoch {ie}:")
        for blk_name in ("OI_VIS2","OI_VIS","OI_T3"):
            blk = ep.get(blk_name, {})
            print(" ", blk_name, ": ", len(blk), "entries")
            # print one example dict’s keys
            for k,(key,sub) in enumerate(blk.items()):
                print("   ", key, "→", sorted(sub.keys()))
                break
        print("  WL keys:", "WL" in ep, "shape", (None if "WL" not in ep else (len(ep["WL"]),)))


def _choose_col(cols, options):
    """Return the first existing column name from `options`."""
    for name in options:
        if name in cols:
            return name
    return None

def _wl_cols(hdul):
    """Column names for OI_WAVELENGTH table."""
    if "OI_WAVELENGTH" not in hdul:
        return None, None
    cols = hdul["OI_WAVELENGTH"].columns.names
    wl  = _choose_col(cols, ["EFF_WAVE","EFF_WL"])
    bw  = _choose_col(cols, ["EFF_BAND","EFF_BW"])
    return wl, bw

def _vis2_cols(hdu):
    """Column names inside OI_VIS2 HDU (one table)."""
    cols = hdu.columns.names
    v2   = _choose_col(cols, ["VIS2DATA","V2"])
    ev2  = _choose_col(cols, ["VIS2ERR","EVIS2","EV2"])
    flag = "FLAG" if "FLAG" in cols else None
    return v2, ev2, flag

def _t3_cols(hdu):
    """Column names inside OI_T3 HDU (one table)."""
    cols = hdu.columns.names
    cphi = _choose_col(cols, ["T3PHI","T3PHASE","CP"])
    ecph = _choose_col(cols, ["T3PHIERR","ET3PHI","CPERR"])
    tamp = _choose_col(cols, ["T3AMP","|T3|","T3ABS"])
    etam = _choose_col(cols, ["T3AMPERR","ET3AMP"])
    flag = "FLAG" if "FLAG" in cols else None
    return cphi, ecph, tamp, etam, flag



########### AFTER REALISING WROTE ORIGINAL FILES 

def _col(hdu, *names):
    """Return the first column name in this HDU that exists, else None."""
    cols = [c.name for c in hdu.columns]
    for n in names:
        if n in cols:
            return n
    return None

def _vis2_cols(hdu_vis2):
    # data + err + uv + time + flags
    v2   = _col(hdu_vis2, "VIS2DATA", "V2")
    ev2  = _col(hdu_vis2, "VIS2ERR",  "EV2")
    u    = _col(hdu_vis2, "UCOORD",   "U")
    v    = _col(hdu_vis2, "VCOORD",   "V")
    mjdc = _col(hdu_vis2, "MJD")
    flag = _col(hdu_vis2, "FLAG")
    return v2, ev2, u, v, mjdc, flag

def _t3_cols(hdu_t3):
    # cp + ecp + amp + eamp + uv1/uv2 + time + flags
    cphi  = _col(hdu_t3, "T3PHI",     "CP")
    ecphi = _col(hdu_t3, "T3PHIERR",  "ET3PHI")
    amp   = _col(hdu_t3, "T3AMP")
    eamp  = _col(hdu_t3, "T3AMPERR",  "ET3AMP")
    u1    = _col(hdu_t3, "U1COORD",   "U1")
    v1    = _col(hdu_t3, "V1COORD",   "V1")
    u2    = _col(hdu_t3, "U2COORD",   "U2")
    v2    = _col(hdu_t3, "V2COORD",   "V2")
    mjdc  = _col(hdu_t3, "MJD")
    flag  = _col(hdu_t3, "FLAG")
    return cphi, ecphi, amp, eamp, u1, v1, u2, v2, mjdc, flag

def _median_uv_from_pm(pm_blk):
    uu = np.asarray(pm_blk.get("u", []), float).ravel()
    vv = np.asarray(pm_blk.get("v", []), float).ravel()
    if uu.size == 0 or vv.size == 0:
        return None
    return (float(np.nanmedian(uu)), float(np.nanmedian(vv)))

def _median_uvs_from_pm_tri(pm_blk):
    u1 = np.asarray(pm_blk.get("u1", []), float).ravel()
    v1 = np.asarray(pm_blk.get("v1", []), float).ravel()
    u2 = np.asarray(pm_blk.get("u2", []), float).ravel()
    v2 = np.asarray(pm_blk.get("v2", []), float).ravel()
    if min(u1.size, v1.size, u2.size, v2.size) == 0:
        return None
    return (float(np.nanmedian(u1)), float(np.nanmedian(v1)),
            float(np.nanmedian(u2)), float(np.nanmedian(v2)))

def _median_mjd_from_pm(pm_blk):
    # PMOIRED blocks usually carry 'MJD' (and sometimes 'MJD2'); use either.
    for k in ("MJD", "MJD2"):
        arr = pm_blk.get(k, None)
        if arr is None:
            continue
        arr = np.asarray(arr, float).ravel()
        if arr.size:
            return float(np.nanmedian(arr))
    return None

def _nearest_baseline_timeaware(pm_vis2_dict, urow, vrow, mjd_row=None, time_tol_days=None):
    """
    Pick the PMOIRED baseline block whose median (u,v) is closest to (urow,vrow),
    but if time_tol_days is given and both sides have MJD, require |ΔMJD| <= time_tol_days.
    Fallback: if no time-close candidate, use uv-only nearest.
    """
    # First pass: restrict by time if possible
    cand = []
    if mjd_row is not None and time_tol_days is not None:
        for k, blk in pm_vis2_dict.items():
            mjd_blk = _median_mjd_from_pm(blk)
            if mjd_blk is None:
                continue
            if abs(mjd_blk - mjd_row) <= time_tol_days:
                cand.append(k)
    # If no time-close candidates, consider all
    if not cand:
        cand = list(pm_vis2_dict.keys())

    best_key, best_d2 = None, np.inf
    for k in cand:
        uv = _median_uv_from_pm(pm_vis2_dict[k])
        if uv is None:
            continue
        d2 = (uv[0]-urow)**2 + (uv[1]-vrow)**2
        if d2 < best_d2:
            best_key, best_d2 = k, d2
    return best_key

def _nearest_triangle_timeaware(pm_t3_dict, u1r, v1r, u2r, v2r, mjd_row=None, time_tol_days=None):
    """
    Like above, but for triangles: compare (u1,v1,u2,v2), allow 1↔2 swap, with optional time filter.
    """
    cand = []
    if mjd_row is not None and time_tol_days is not None:
        for k, blk in pm_t3_dict.items():
            mjd_blk = _median_mjd_from_pm(blk)
            if mjd_blk is None:
                continue
            if abs(mjd_blk - mjd_row) <= time_tol_days:
                cand.append(k)
    if not cand:
        cand = list(pm_t3_dict.keys())

    best_key, best_d2 = None, np.inf
    for k in cand:
        uv = _median_uvs_from_pm_tri(pm_t3_dict[k])
        if uv is None:
            continue
        # two permutations
        d2a = (uv[0]-u1r)**2 + (uv[1]-v1r)**2 + (uv[2]-u2r)**2 + (uv[3]-v2r)**2
        d2b = (uv[2]-u1r)**2 + (uv[3]-v1r)**2 + (uv[0]-u2r)**2 + (uv[1]-v2r)**2
        d2 = d2a if d2a < d2b else d2b
        if d2 < best_d2:
            best_key, best_d2 = k, d2
    return best_key
# ----------------- main -----------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dipole-model residuals by division (V² & CP/T3) using PMOIRED sampling")
    ap.add_argument(
    "--dir", nargs="+", default=None,
    help="One or more directories to search recursively for FITS files."
    )
    ap.add_argument("--files", nargs="+", default=None, help="OIFITS glob(s)")
    ap.add_argument("--ins", choices=["pionier","gravity"], required=True)
    ap.add_argument("--out_dir", default="./dipole_residuals")

    # dipole params (optional if you load JSON)
    ap.add_argument("--theta_o_deg", type=float, default=170.0, help="inclination-like angle (deg)")
    ap.add_argument("--phi_o_deg",   type=float, default=168.0, help="position angle (deg)")
    ap.add_argument("--delta_T",     type=float, default=212.0, help="temperature contrast (K)")
    ap.add_argument("--theta_ud_mas",type=float, default=3.28,  help="UD diameter (mas) for pixel scale")

    ap.add_argument("--load_mcmc_json", type=str, default=None, help="Load MCMC JSON (your format) to get median params")

    # options
    ap.add_argument("--grid_size", type=int, default=500)
    ap.add_argument("--cp_unit", choices=["rad","deg"], default="rad")
    ap.add_argument("--v2_store", choices=["ratio","diff"], default="ratio",
                    help="Store V2 residual as ratio (default) or difference")
    
    # ap.add_argument(
    # "--oifits_out_dir", type=str, default="./written_oifits",
    # help="Directory to write the OIFITS triplet (data/model/resid). Will be created."
    # )
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)

    #oifits_out_dir = ensure_dir(args.out_dir)

    # build file list 
    file_list = []
    if args.dir:
        for d in args.dir:
            file_list += _gather_fits_from_path(d)
    if args.files:
        for pat in args.files:
            file_list += _gather_fits_from_path(pat)

    file_list = sorted(set(file_list))
    if not file_list:
        raise SystemExit("No FITS files found. Provide --dir and/or --files.")

    # # 1) read OIFITS in PMOIRED
    # file_list = []
    # for pat in args.files:
    #     file_list += glob.glob(pat)
    # file_list = sorted(set(file_list))
    # if not file_list:
    #     raise SystemExit("No files matched.")

    if args.ins == "pionier":
        oi = pmoired.OI(file_list, binning=None, insname=None)
        lam_nom = 1.65e-6
    else:
        oi = pmoired.OI(file_list, binning=400, insname='GRAVITY_SC_P1')
        lam_nom = 2.2e-6


    
    # sort blocks by MJD for stability
    oi.data = sorted(oi.data, key=lambda x: x['MJD'][0] if hasattr(x['MJD'], '__len__') else x['MJD'])
    # check debug- delete once stable 
    #debug_dump_oi_t3_structure(oi, max_epochs=1, max_tris=8)
    #peek_shapes(oi)

    inspect_oifits_columns(file_list)
    inspect_pmoi_blocks(oi)

    # enforce ordering quirks
    change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
    change_triangle_key_list = ['triangles','OI_T3']
    enforce_ordered_baselines_keys(oi.data, change_baseline_key_list)
    enforce_ordered_triangle_keys(oi.data, change_triangle_key_list)

    # 2) get dipole params (CLI or JSON posteriors)
    theta_o_deg = args.theta_o_deg
    phi_o_deg   = args.phi_o_deg
    delta_T     = args.delta_T
    theta_ud_mas= args.theta_ud_mas

    if args.load_mcmc_json and os.path.isfile(args.load_mcmc_json):
        with open(args.load_mcmc_json, 'r') as f:
            d = json.load(f)
        # Expect your keys like ['$\\theta_o$', '$\\phi_o$', '$\\Delta T$', '$\\theta_{UD}$']
        # Use medians
        try:
            th = np.median(np.array(d.get('$\\theta_o$', d.get('theta_o', []))).ravel())
            ph = np.median(np.array(d.get('$\\phi_o$', d.get('phi_o', []))).ravel())
            dT = np.median(np.array(d.get('$\\Delta T$', d.get('delta_T', []))).ravel())
            ud = np.median(np.array(d.get('$\\theta_{UD}$', d.get('theta_ud', []))).ravel())
            # convert if your JSON stored radians for the two angles:
            if np.abs(th) <= 2*np.pi and np.abs(ph) <= 2*np.pi:
                th = np.degrees(th); ph = np.degrees(ph)
            theta_o_deg, phi_o_deg, delta_T, theta_ud_mas = float(th), float(ph), float(dT), float(ud)
            print(f"[INFO] Loaded medians from JSON: theta_o={theta_o_deg:.1f} deg, phi_o={phi_o_deg:.1f} deg, ΔT={delta_T:.1f} K, θ_UD={theta_ud_mas:.3f} mas")
        except Exception as e:
            print("[WARN] Could not parse JSON posteriors, using CLI values. Err:", e)

    # 3) build model on same sampling
    oif = build_oif_from_params(
        oi, theta_o_deg, phi_o_deg, delta_T, theta_ud_mas,
        wavelength_m=lam_nom, grid_size=args.grid_size,
        T_eff=3000.0, nu=1/(757*24*3600), psi_T=0.0, l=1, m=1,
        dx_mas=theta_ud_mas/args.grid_size, dy_mas=theta_ud_mas/args.grid_size
    )

    # 4) divide to get residuals
    # oi_res = subtract_by_division(
    #     oi, oif, cp_unit=args.cp_unit, store_v2_as=args.v2_store, eps_amp=1e-12
    # )

    oi_res = subtract_by_division(oi, oif, verbose=True)

    # 5) save dumps & QA plots 
    # newer that writes data, model and residuals fits files 
    write_oifits_triplet(
        file_list=file_list,
        pm_ref=oi,            # reference for matching epochs/channels
        pm_data=oi,           # write a clean "data" copy in our standardized format
        pm_model=oif,         # model
        pm_resid=oi_res,      # residuals
        out_dir= out_dir,
        overwrite=True
    )
    # old way 
    ## dump_residual_arrays(oi_res, out_dir, tag="dipole_division")

    # write_oifits_triplet(
    #     file_list,   # the original list of .fits you loaded
    #     oi,          # PMOIRED data (original)
    #     oif,         # PMOIRED synthetic model
    #     oi_res,      # PMOIRED residuals
    #     out_dir,     # e.g., "./out_for_candid"
    #     overwrite=True
    # )
    
    quick_plots(oi, oif, oi_res, out_dir)#, ins=args.ins)

    # OLD RESIDUALS 
    # write_residual_oifits(
    #     obs_files=file_list, 
    #     oi = oi, 
    #     oi_res = oi_res, 
    #     out_dir = out_dir,
    #     vis2_mode="ratio",        # or "absolute"
    #     t3_mode="phase_only",     # or "ratio"
    #     copy_vis=True,
    #     overwrite=True
    #     )
    
    # #  pickle the residual PMOIRED object for later reuse
    # try:
    #     import pickle
    #     with open(os.path.join(out_dir, "oi_residual_pmoired.pkl"), "wb") as f:
    #         pickle.dump(oi_res, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print("[SAVE]", os.path.join(out_dir, "oi_residual_pmoired.pkl"))
    # except Exception as e:
    #     print("[WARN] Could not pickle residual OI object:", e)

    # print("\nDone. Residuals written to:", out_dir)

if __name__ == "__main__":
    main()
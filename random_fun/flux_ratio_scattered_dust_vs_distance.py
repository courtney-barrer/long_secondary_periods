#!/usr/bin/env python3
"""
Scattered flux from a dusty companion: F_scat/F_* vs separation (mas),
and a 2D heatmap vs separation (mas) and dust mass, for a fixed clump size.

Assumptions:
- Single scattering, optically thin (sanity contour for tau_ext=0.3 provided).
- Compact clump at separation R (R >> clump radius a).
- Henyey–Greenstein phase function with asymmetry g.
- Mass scattering coefficient kappa_sca(λ) (cm^2 g^-1).
- A fixed distance d = 505 pc is used to convert AU <-> mas.

Key relation:
  F_scat/F_* = (kappa_sca * M_d / R^2) * p_HG(theta; g)

Where p_HG(θ; g) is normalized so ∫ p dΩ = 1 (units sr^-1).

You can change the band/dust preset and geometry in the USER PARAMETERS block.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------- Physical constants -----------------
AU_IN_CM = 1.495978707e13
PC_IN_CM = 3.08567758149e18
M_H = 1.6735575e-24     # gram
MU_GAS = 1.4            # mean molecular weight per H (includes He)

# ----------------- Distance and angle conversions -----------------
D_PC = 505.0  # <-- set distance here
def au_to_mas(R_AU: np.ndarray | float, d_pc: float = D_PC) -> np.ndarray | float:
    """
    Small-angle approx: theta[arcsec] = R[AU] / d[pc]; theta[mas] = 1000 * R/d.
    """
    return 1000.0 * (np.asarray(R_AU) / d_pc)

def mas_to_au(theta_mas: np.ndarray | float, d_pc: float = D_PC) -> np.ndarray | float:
    return np.asarray(theta_mas) * d_pc / 1000.0

# ----------------- Phase function -----------------
def hg_phase(theta_rad: float, g: float) -> float:
    """
    Henyey–Greenstein phase function p_HG(θ; g), normalized so ∫ p dΩ = 1.
    Returns p in sr^-1.
    """
    ct = math.cos(theta_rad)
    denom = (1 + g**2 - 2*g*ct)**1.5
    return (1 - g**2) / (4 * math.pi * denom)

# ----------------- Dust presets (order-of-mag; replace with WD01/Draine tables for precision) -----------------
@dataclass
class DustPreset:
    name: str
    lambda_um: float
    kappa_sca_cmg: float  # cm^2 g^-1
    g: float              # HG asymmetry
    albedo: float         # single-scattering albedo (for tau check)

PRESETS = {
    "V": DustPreset("V", 0.55, 4e3, 0.60, 0.60),
    "J": DustPreset("J", 1.25, 1.5e3, 0.60, 0.60),
    "H": DustPreset("H", 1.65, 1.0e3, 0.60, 0.60),
    "K": DustPreset("K", 2.20, 6e2, 0.55, 0.50),
}

# ----------------- Core relations -----------------
def fscat_fraction_from_mass(Md_g, R_cm, theta_deg, kappa_sca_cmg, g):
    """
    Scattered flux fraction F_scat / F_* for a compact clump with dust mass Md.
    Accepts Nd arrays or scalars for R_cm.
    """
    p = hg_phase(math.radians(theta_deg), g)
    return (kappa_sca_cmg * Md_g / (np.asarray(R_cm) ** 2)) * p

def mass_from_density_sphere(rho_d_gcm3, a_cm):
    """Uniform-density sphere mass from dust density rho_d and radius a."""
    return (4.0 / 3.0) * math.pi * a_cm**3 * rho_d_gcm3

def tau_ext_from_mass(Md_g, a_cm, kappa_sca_cmg, albedo):
    """
    Approximate extinction optical depth through the clump along a diameter (length 2a):
      tau_ext = kappa_ext * rho_d * (2a), with rho_d = Md / Volume and kappa_ext = kappa_sca / albedo.
    Good for sanity checking the single-scattering assumption (tau_ext <~ 0.3).
    """
    volume = (4.0 / 3.0) * math.pi * a_cm**3
    rho_d = Md_g / volume
    kappa_ext = kappa_sca_cmg / max(albedo, 1e-8)
    return kappa_ext * rho_d * (2.0 * a_cm)

# ----------------- USER PARAMETERS -----------------
# Band/dust model and geometry
dust = PRESETS["H"]     # choose: "V","J","H","K"
theta_deg = 89.0 #60.0        # scattering angle (star–dust–observer)
a_AU = 2 #0.2              # clump radius in AU (used for density->mass and tau checks)
a_cm = a_AU * AU_IN_CM

# Line-plot settings (fixed density -> fixed mass via size a)
rho_d_fixed = 1e-14     # g cm^-3  (example optically-thin density for a=0.2 AU)
Md_fixed = mass_from_density_sphere(rho_d_fixed, a_cm)

# Separation grids
R_AU_line = np.logspace(0, 1.7, 220)      # 1 to ~50 AU
R_cm_line = R_AU_line * AU_IN_CM
sep_mas_line = au_to_mas(R_AU_line, D_PC)

# Heatmap grids (mass vs separation)
R_AU_grid = np.logspace(0, 1.7, 240)      # 1 to ~50 AU
R_cm_grid = R_AU_grid * AU_IN_CM
sep_mas_grid = au_to_mas(R_AU_grid, D_PC)

Md_grid = np.logspace(20, 26, 240)        # 1e20 .. 1e26 g
# (If you prefer a density grid, convert to Md using a_cm and rho_d.)

# ----------------- Compute -----------------
# 1) Line: F_scat/F_* vs separation (mas)
F_line = fscat_fraction_from_mass(Md_fixed, R_cm_line, theta_deg, dust.kappa_sca_cmg, dust.g)

# 2) Heat map: log10(F_scat/F_*) over (Md, R)
p_theta = hg_phase(math.radians(theta_deg), dust.g)
F_grid = (dust.kappa_sca_cmg * p_theta) * (Md_grid[:, None] / (R_cm_grid[None, :]**2))
logF = np.log10(F_grid)

# 3) Single-scattering sanity: tau_ext from Md (independent of R), draw tau_ext=0.3 as a horizontal contour line
tau_ext_vec = tau_ext_from_mass(Md_grid, a_cm, dust.kappa_sca_cmg, dust.albedo)

# ----------------- Plot 1: line plot (separation in mas) -----------------
plt.figure()
plt.loglog(sep_mas_line, F_line)
plt.xlabel("Separation [mas]  (d = 505 pc)")
plt.ylabel("Scattered flux fraction  $F_{\\rm scat}/F_*$")
plt.title(
    f"{dust.name}-band: $F_{{\\rm scat}}/F_*$ vs separation "
    f"(ρ_d={rho_d_fixed:.0e} g cm$^{{-3}}$, a={a_AU:g} AU, θ={theta_deg:.1f}°)"
)
plt.grid(True, which='both', ls=':')

# ----------------- Plot 2: heat map (separation in mas vs dust mass) -----------------
plt.figure()
extent = [sep_mas_grid.min(), sep_mas_grid.max(), Md_grid.min(), Md_grid.max()]
plt.imshow(logF, extent=extent, aspect='auto', origin='lower')
cbar = plt.colorbar()
cbar.set_label(r"$\log_{10}(F_{\rm scat}/F_*)$")

# 1% contour (F = 0.01 → log10 F = -2)
SEP_MAS_MESH, MD_MESH = np.meshgrid(sep_mas_grid, Md_grid)
CS1 = plt.contour(SEP_MAS_MESH, MD_MESH, logF, levels=[-2.0], colors='k')
plt.clabel(CS1, inline=True, fmt=lambda _: "1%")

# tau_ext = 0.3 line (horizontal; independent of separation for fixed a and dust)
# Find Md where tau_ext crosses 0.3 (it might be a range; plot as a dashed line)
# For robustness, we draw a horizontal line at Md where tau_ext≈0.3
# Interpolate in log-space for clarity
import numpy as _np
idx = _np.argmin(_np.abs(tau_ext_vec - 0.3))
Md_tau03 = Md_grid[idx]
plt.axhline(Md_tau03, linestyle='--', linewidth=1.2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Separation [mas]  (d = 505 pc)")
plt.ylabel("Dust mass $M_d$ [g]")
plt.title(
    f"{dust.name}-band: $F_{{\\rm scat}}/F_*$ heat map (a={a_AU:g} AU, θ={theta_deg:.1f}°)\n"
    r"Contours: 1% (solid),  $\tau_{\rm ext}=0.3$ (dashed)"
)

# ----------------- Save figures -----------------

#plt.savefig("fscat_vs_sep_mas_line.png", dpi=220, bbox_inches='tight')
#plt.savefig("fscat_heatmap_mass_vs_sepmas.png", dpi=220, bbox_inches='tight')
plt.show()
print("Saved: fscat_vs_sep_mas_line.png, fscat_heatmap_mass_vs_sepmas.png")

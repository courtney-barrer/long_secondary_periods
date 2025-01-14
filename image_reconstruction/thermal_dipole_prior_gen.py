import numpy as np
from astropy.io import fits
import os 
import argparse
import matplotlib.pyplot as plt
import glob 
import pmoired
import json
import sys
sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

### cannot be (easily) parametrically generated from pmoired. So we do it custom here


# Default values
default_save_fits_path = '/home/rtc/Documents/long_secondary_periods/PMOIRED_FITS/best_models/' #'/home/rtc/Documents/long_secondary_periods/image_reco/thermal_dipole_prior/'
default_wavelength = 1.6e-6  # m
default_Npixels = 10
default_UD = 3.3  # mas
default_psi_T = 0
default_delta_T = 212
default_T_eff = 3000
default_incl = 170
default_projang = 172
default_l = 1
default_m = 1
default_plot = True

# Calculate default pixel size
default_pixelsize = round(
    default_wavelength / (4 * 120) * 1e3 * 3600 * 180 / np.pi, 2
)

# Argument parser setup
parser = argparse.ArgumentParser(description="Thermal dipole prior simulation.")

parser.add_argument("--save_fits_path", type=str, default=default_save_fits_path,
                    help="Path to save FITS files.")
parser.add_argument("--wavelength", type=float, default=default_wavelength,
                    help="Wavelength in meters.")
parser.add_argument("--Npixels", type=int, default=default_Npixels,
                    help="Number of pixels.")
parser.add_argument("--UD", type=float, default=default_UD,
                    help="Uniform disk diameter in mas.")
parser.add_argument("--psi_T", type=float, default=default_psi_T,
                    help="Phase offset of the temperature variation.")
parser.add_argument("--delta_T", type=float, default=default_delta_T,
                    help="Amplitude of temperature variation.")
parser.add_argument("--T_eff", type=float, default=default_T_eff,
                    help="Effective temperature of the star.")
parser.add_argument("--incl", type=float, default=default_incl,
                    help="Inclination angle in degrees.")
parser.add_argument("--projang", type=float, default=default_projang,
                    help="Projection angle in degrees.")
parser.add_argument("--l", type=int, default=default_l,
                    help="Spherical harmonic degree.")
parser.add_argument("--m", type=int, default=default_m,
                    help="Spherical harmonic order.")
parser.add_argument("--plot", type=lambda x: x.lower() == 'true', default=default_plot,
                    help="Whether to plot results (True/False).")

# Parse arguments
args = parser.parse_args()

# Variables
save_fits_path = args.save_fits_path
wavelength = args.wavelength
Npixels = args.Npixels
UD = args.UD
psi_T = args.psi_T
delta_T = args.delta_T
T_eff = args.T_eff
incl = args.incl
projang = args.projang
l = args.l
m = args.m
plot = args.plot

# Recompute pixel size if wavelength changes
pixelsize = round(wavelength / (4 * 120) * 1e3 * 3600 * 180 / np.pi, 2)



# # command line arguments
# save_fits_path = f'/home/rtc/Documents/long_secondary_periods/image_reco/thermal_dipole_prior/'

# wavelength = 1.6e-6 #m
# #grid_size = 60
# pixelsize = round(wavelength / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2)
# Npixels = 10
# UD = 3.3 #mas
# psi_T = 0 
# delta_T = 212
# T_eff = 3000
# incl = 170
# projang = 172
# l = 1
# m = 1           # Spherical harmonic degree and order
# plot = True


if not os.path.exists(save_fits_path):
    os.makedirs(save_fits_path)
    
# derived parameters
grid_size = int( UD / pixelsize ) 
dx = pixelsize
dy = pixelsize 
pad_factor = Npixels / grid_size 

# we redifine variables for clarity (legacy)
theta_o = np.deg2rad( incl )
phi_o = np.deg2rad( projang  )

nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)

# Stellar surface grid
theta = np.linspace(0, np.pi, 200)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 200)  # Full longitude
theta, phi = np.meshgrid(theta, phi)

# Observer position
#theta_obs = theta_o # 45 degrees
#phi_obs = phi_o   # 60 degrees
header_dict = {"wvl":wavelength, "psi_T":psi_T, "nu":nu, "l":l, "m":m, "delta_T_eff":delta_T, "T_eff":T_eff, 'phi_obs':phi_o, 'theta_obs':theta_o}

# Calculate local effective temperature
T_eff_local = plot_util.thermal_oscillation(theta, phi, 0, T_eff, delta_T, l, m, nu, psi_T)

# Rotate to observer frame
theta_rot, phi_rot = plot_util.rotate_to_observer_frame(theta, phi, theta_o, phi_o)

# Project onto observer plane
projected_intensity = plot_util.project_to_observer_plane(theta_rot, phi_rot, plot_util.blackbody_intensity(T_eff_local, wavelength), grid_size =grid_size )

projected_intensity = np.pad( projected_intensity,  int(projected_intensity.shape[0] * pad_factor), mode='constant', constant_values=0)

name = 'thermal_dipole_prior.fits'
plot_util.intensity_2_fits(projected_intensity, dx=dx, dy=dy, name=name, data_path = save_fits_path, header_dict=header_dict)

if plot: 
    fig , axes = plt.subplots() 
    ax2 = axes

    origin = 'lower'
    extent = dx * grid_size * pad_factor / 2 * np.array([1, -1, -1, 1])

    # Plot the heatmap
    heatmap = ax2.imshow(
        projected_intensity / np.max( projected_intensity), origin=origin, cmap="inferno",
        extent=extent, vmin=0, vmax=1
    )


    # Add axis labels and title
    ax2.set_title("Projected Intensity", fontsize=15)
    ax2.set_ylabel('$\Delta$ DEC -> N [mas]', fontsize=15)
    ax2.set_xlabel('$\Delta$ RA <- E [mas]', fontsize=15)
    ax2.tick_params(labelsize=15)

    # Adjust colorbar with padding
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    cbar = plt.colorbar(heatmap, cax=cax)
    cbar.set_label("Normalized Intensity", fontsize=15)
    cbar.ax.tick_params(labelsize=15)  # Adjust colorbar tick label size

    plt.show()
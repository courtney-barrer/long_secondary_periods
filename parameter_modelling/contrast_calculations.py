import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 6.626e-34  # Planck's constant (J·s)
c = 3.0e8      # Speed of light (m/s)
k_B = 1.38e-23 # Boltzmann's constant (J/K)

# Planck function to calculate intensity
def planck_intensity(wavelength, T):
    """Calculate Planck intensity for a given wavelength and temperature."""
    return (2 * h * c**2) / (wavelength**5 * (np.exp(h * c / (wavelength * k_B * T)) - 1))

# Function to calculate contrast as a function of wavelength
def calculate_contrast(wavelengths, T_star, T_companion):
    """Calculate the contrast as a function of wavelength."""
    contrasts = []
    for wavelength in wavelengths:
        I_star = planck_intensity(wavelength, T_star)
        I_companion = planck_intensity(wavelength, T_companion)
        contrast = I_companion / I_star
        contrasts.append(contrast)
    return contrasts

# Extended wavelength range (e.g., 1 to 13 microns)
wavelengths = np.linspace(1e-6, 13e-6, 1000)  # Wavelengths in meters

# Temperatures
T_primary = 3000  # Primary star temperature (K)
companion_temperatures = [500, 750, 1000, 1250, 1500]  # Companion temperatures (K)

# Plot contrast vs. wavelength for various companion temperatures
plt.figure(figsize=(10, 7))
for T_companion in companion_temperatures:
    contrasts = calculate_contrast(wavelengths, T_primary, T_companion)
    plt.plot(wavelengths * 1e6, contrasts, label=f"T_companion = {T_companion} K")
plt.xlabel("Wavelength (microns)", fontsize=14)
plt.ylabel("Contrast", fontsize=14)
plt.title("Contrast vs Wavelength (1-13 microns)", fontsize=16)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.show()

# --------------------------------------------------------------------
# Contrast vs Radius Model (Assuming Power-law T(r) = T_inner * (r/r_inner)^-q)
# --------------------------------------------------------------------

distance = 540 #parsec
au2mas = 1# np.rad2deg( 4.848e-6 / distance * 1e3 * 3600 )  

# Parameters for dust temperature model
T_inner = 1500  # Inner dust sublimation temperature (K)
r_inner = au2mas  * 3 * (T_primary / T_inner)**2  # Dust sublimation radius (AU)
# Reference: Typical values for oxygen-rich AGB stars (Bladh et al. 2013)
# r_inner depends on luminosity and temperature, typically 3-5 AU for AGB stars.

# Radial range
radii = au2mas * np.linspace(r_inner, 50, 500)  # Radii in AU (3 AU to 50 AU)
q = 0.5  # Power-law exponent for temperature profile (reasonable for O-rich AGB)

# Calculate dust temperature profile
T_dust = T_inner * (radii / r_inner)**-q

# Calculate contrast vs. radius at a fixed wavelength (e.g., 10 microns)
wavelength_fixed = 10e-6  # 10 microns
contrasts_radius = [planck_intensity(wavelength_fixed, T) / planck_intensity(wavelength_fixed, T_primary) for T in T_dust]

# Plot contrast vs. radius
plt.figure(figsize=(10, 7))
plt.plot(radii, contrasts_radius, label=f"q = {q}, λ = {wavelength_fixed * 1e6} µm")
plt.axvline(r_inner, color='r', linestyle='--', label=f"Inner Dust Radius = {r_inner:.2f} AU")
plt.xlabel("Radius (AU)", fontsize=14)
plt.ylabel("Contrast", fontsize=14)
plt.title("Contrast vs Radius for Dust around O-rich AGB Star", fontsize=16)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.show()

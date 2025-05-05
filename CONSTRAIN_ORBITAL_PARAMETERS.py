

import numpy as np
import matplotlib.pyplot as plt
import corner


# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
AU = 1.496e11  # Astronomical unit in meters
M_sun = 1.989e30  # Solar mass, kg
day = 86400  # Seconds in a day

# Orbital period in seconds
P = 757 * day  # Increase period to 1000 days

# Number of samples for Monte Carlo simulation
n_samples = 100000

# Primary star mass distribution (normal, constrained to >1 M_sun)
mean_primary_mass = 3  # Solar masses
std_primary_mass = 1  # Solar masses
primary_mass_samples = np.random.uniform(1, 5, n_samples) #np.random.normal(mean_primary_mass, std_primary_mass, n_samples)
primary_mass_samples = primary_mass_samples[primary_mass_samples > 1]  # Constraint > 1 M_sun

# Ensure the number of samples matches
n_primary_samples = len(primary_mass_samples)

# Companion mass distribution (log-normal, mean=0.01 M_sun)
mean_log_companion_mass = np.log(0.01 * M_sun)  # Log mean in kg
std_log_companion_mass = 5  # Reduced log std deviation
companion_mass_samples = np.random.lognormal(mean_log_companion_mass, std_log_companion_mass, n_primary_samples)

#plt.figure(); plt.hist( companion_mass_samples /  M_sun ) ;plt.xscale('log');plt.show()

# Calculate semi-major axis (a) in meters
a = ((G * (primary_mass_samples * M_sun + companion_mass_samples) * (P ** 2)) / (4 * np.pi ** 2)) ** (1 / 3)

# Radial velocity amplitude for the primary star (m/s)
v_r = (2 * np.pi * a / P) * (companion_mass_samples / (primary_mass_samples * M_sun + companion_mass_samples))

# Sample inclinations uniformly in cos(i) space (0 to 90 degrees)
cos_inclination_samples = np.random.uniform(0, 1, n_primary_samples)
inclination_samples = np.arccos(cos_inclination_samples)  # Inclination in radians

# Update radial velocity amplitude calculation to include inclination
v_r_inclined = v_r * np.sin(inclination_samples)

# Inspect radial velocity distribution
plt.figure(figsize=(10, 6))
plt.hist(v_r_inclined, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Radial Velocity Amplitude (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Radial Velocity Amplitudes', fontsize=16)
plt.grid(True)
plt.show()

# Filter results where v_r < 4 m/s
filtered_indices = v_r_inclined < 6e3 #6e3
filtered_companion_masses = companion_mass_samples[filtered_indices] / M_sun  # Convert to M_sun

# Plotting results
plt.figure(figsize=(10, 6))
plt.hist(filtered_companion_masses, bins=np.logspace(-6,-1,20), alpha=0.7, color='blue', edgecolor='black')
plt.xscale('log')
plt.xlabel('Companion Mass ($M_\\odot$)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Companion Mass Distribution for $v_r < 4 \, \\mathrm{m/s}$', fontsize=16)
#plt.grid(True)
plt.axvline( 3e-6 , color='green', label=r"$M_{Earth}$")
plt.axvline( 0.000954 , color='pink', label=r"$M_{Jupyter}$")
plt.axvline( 0.013 , color='brown', ls='--',label=r"lower $M_{brown\ dwarf}$")
plt.axvline( 0.08 , color='brown', ls='-',label=r"upper $M_{brown\ dwarf}$")
plt.legend()
plt.show()


# Statistics of filtered companion masses
if len(filtered_companion_masses) > 0:
    mean_mass = np.mean(filtered_companion_masses)
    median_mass = np.median(filtered_companion_masses)
    std_mass = np.std(filtered_companion_masses)

    print(f"Mean Companion Mass: {mean_mass:.5f} M_sun")
    print(f"Median Companion Mass: {median_mass:.5f} M_sun")
    print(f"Standard Deviation: {std_mass:.5f} M_sun")
else:
    print("No companion masses satisfy the radial velocity constraint.")




#### ORBITAL SEPERATION 
# Constants

# Calculate semi-major axis (a) in meters for the filtered masses
filtered_primary_masses = primary_mass_samples[filtered_indices]  # Filter primary masses
a_filtered = ((G * (filtered_primary_masses * M_sun + filtered_companion_masses * M_sun) * (P ** 2)) / 
              (4 * np.pi ** 2)) ** (1 / 3)  # Semi-major axis in meters

# Convert semi-major axis to AU
a_filtered_au = a_filtered / AU

# Plot orbital separation distribution
plt.figure(figsize=(10, 6))
plt.hist(a_filtered_au, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Orbital Separation (AU)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Orbital Separation Distribution for $v_r < 4 \, \\mathrm{m/s}$', fontsize=16)
plt.grid(True)
plt.show()

# Statistics of orbital separation
if len(a_filtered_au) > 0:
    mean_separation = np.mean(a_filtered_au)
    median_separation = np.median(a_filtered_au)
    std_separation = np.std(a_filtered_au)

    print(f"Mean Orbital Separation: {mean_separation:.5f} AU")
    print(f"Median Orbital Separation: {median_separation:.5f} AU")
    print(f"Standard Deviation: {std_separation:.5f} AU")
else:
    print("No orbital separations could be computed.")





# Combine the sampled orbital separations and companion masses
data = np.vstack((np.log10(filtered_companion_masses), a_filtered_au)).T

# Define labels for the corner plot
labels = [r"log$_{10}$(Companion Mass [$M_\odot$])", r"Orbital Separation (AU)"]
#labels = [r"Companion Mass [$M_\odot$]", r"Orbital Separation (AU)"]

fig = plt.figure(figsize=(10, 10))
ranges = [(-6, 1), (1.5, 4)]
# Create the corner plot
corner.corner(
    data,
    labels=labels,
    show_titles=True,
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],
    hist_kwargs={"density": True},
    label_kwargs={"fontsize": 14},
    title_fmt=".3f",
    range=ranges,
    fig=fig,
)


# Add vertical lines to the companion mass histogram
axes = fig.axes
mass_axis = axes[0]  # First axis corresponds to the companion mass
#mass_axis.set_xscale('log')
mass_axis.axvline(np.log10(3e-6), color='green', linestyle='-', label=r"$M_{Earth}$")
mass_axis.axvline(np.log10(0.000954), color='pink', linestyle='-', label=r"$M_{Jupiter}$")
mass_axis.axvline(np.log10(0.013), color='brown', linestyle='--', label=r"lower $M_{BD}$")
mass_axis.axvline(np.log10(0.08), color='brown', linestyle='-', label=r"upper $M_{BD}$")

# Add legend
mass_axis.legend(fontsize=10)

# Adjust plot aesthetics
#plt.suptitle("Companion Mass and Orbital Separation Distribution", fontsize=16)
plt.tight_layout()
#plt.savefig("MC_rtpav_orbital_parameters_constraints.png",dpi=300)
plt.show()






############### REPEAT AFTER PEER REVIEW 
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11         # Gravitational constant [m^3 kg^-1 s^-2]
Msun = 1.989e30         # Solar mass [kg]
AU = 1.496e11           # Astronomical unit [m]
day = 86400             # Seconds in a day

# Fixed orbital period
P_days = 757            # Period in days
P = P_days * day        # Period in seconds

# Number of simulated systems
N = 10000

# Sample parameters from uniform distributions
M1 = np.random.uniform(1.0, 5.0, N)     # Primary mass in M_sun
M2 = np.random.uniform(0.0, 1.0, N)     # Companion mass in M_sun
i_deg = np.random.uniform(0.0, 90.0, N)   # Inclination in degrees
i_rad = np.deg2rad(i_deg)                 # Convert inclination to radians

# Convert masses to kg
M1_kg = M1 * Msun
M2_kg = M2 * Msun

# Calculate orbital separation (semi-major axis) using Kepler's Third Law:
# a = ((G * (M1 + M2) * P^2) / (4*pi^2))^(1/3)
a = ((G * (M1_kg + M2_kg) * P**2) / (4 * np.pi**2))**(1/3)  # in meters
a_AU = a / AU  # Convert separation to astronomical units

# Calculate the radial velocity amplitude for the primary star
# a1 = a * M2 / (M1 + M2)
# K1 = (2*pi*a1*sin(i)) / P
a1 = a * (M2_kg / (M1_kg + M2_kg))
K1 = (2 * np.pi * a1 * np.sin(i_rad)) / P   # in m/s
K1_km_s = K1 / 1e3  # Convert to km/s

# Plot histogram of orbital separations in AU
plt.figure(figsize=(8, 5))
plt.hist(a_AU, bins=50, edgecolor='black')
plt.xlabel('Orbital Separation (AU)')
plt.ylabel('Number of Systems')
plt.title(f'Orbital Separation Distribution for P = {P_days} Days')
plt.grid(True)
plt.show()

# Plot histogram of radial velocity amplitudes in km/s
plt.figure(figsize=(8, 5))
plt.hist(K1_km_s, bins=50, color='orange', edgecolor='black')
plt.xlabel('Radial Velocity Amplitude K1 (km/s)')
plt.ylabel('Number of Systems')
plt.title(f'Primary Radial Velocity Distribution for P = {P_days} Days')
plt.grid(True)
plt.show()

# Print basic statistics
print("Mean orbital separation (AU):", np.mean(a_AU))
print("Median orbital separation (AU):", np.median(a_AU))
print("Mean K1 (km/s):", np.mean(K1_km_s))
print("Median K1 (km/s):", np.median(K1_km_s))









##############################################################
### with ellipticity 

# Sample orbital eccentricities and arguments of periastron
ecc_samples = np.random.uniform(0, 0.9, n_primary_samples)
omega_samples = np.random.uniform(0, 2*np.pi, n_primary_samples)

# Sample inclination
cos_incl = np.random.uniform(0, 1, n_primary_samples)
incl_samples = np.arccos(cos_incl)

# Calculate semi-amplitude K1 in m/s
term1 = (2 * np.pi * G / P) ** (1/3)
mass_term = companion_mass_samples * np.sin(incl_samples) / (primary_mass_samples * M_sun + companion_mass_samples) ** (2/3)
K1 = term1 * mass_term / np.sqrt(1 - ecc_samples**2)  # Now includes eccentricity

# Apply radial velocity constraint
K1_constraint = 6e3  # 6 km/s in m/s
filtered = K1 < K1_constraint




############ with ellipcitiy 

import numpy as np
import matplotlib.pyplot as plt
import corner

# Constants
G = 6.67430e-11       # m^3 kg^-1 s^-2
AU = 1.496e11         # m
M_sun = 1.989e30      # kg
day = 86400           # s

# Orbital parameters
P = 757 * day         # Orbital period in seconds

# Sampling
n_samples = 100_000

# Primary mass (uniform 1–5 M_sun)
primary_mass_samples = np.random.uniform(1, 5, n_samples)

# Companion mass (log-normal centered on 0.01 M_sun, log-space std dev = 5)
mean_log_mc = np.log(0.01 * M_sun)
std_log_mc = 5
companion_mass_samples = np.random.lognormal(mean_log_mc, std_log_mc, n_samples)

# Inclination (uniform in cos(i))
cos_i = np.random.uniform(0, 1, n_samples)
i_samples = np.arccos(cos_i)

# Eccentricity (uniform or beta for more realism)
ecc_samples = np.random.beta(0.867, 3.03, n_samples)  # Kipping 2013 distribution

# Semi-amplitude K1 (m/s), includes eccentricity
term1 = (2 * np.pi * G / P) ** (1/3)
total_mass = primary_mass_samples * M_sun + companion_mass_samples
mass_term = companion_mass_samples * np.sin(i_samples) / total_mass**(2/3)
K1 = term1 * mass_term / np.sqrt(1 - ecc_samples**2)

# Apply constraint: K1 < 6 km/s
mask = K1 < 6e3
filtered_Mp = primary_mass_samples[mask]
filtered_Mc = companion_mass_samples[mask]
filtered_ecc = ecc_samples[mask]

# Compute semi-major axis (m) and convert to AU
a_filtered = ((G * (filtered_Mp * M_sun + filtered_Mc) * (P ** 2)) / (4 * np.pi ** 2)) ** (1 / 3)
a_filtered_AU = a_filtered / AU

# Companion mass in solar units
filtered_Mc_solar = filtered_Mc / M_sun

# --- Plots ---

# 1. Histogram: Companion Mass
plt.figure()
plt.hist(filtered_Mc_solar, bins=np.logspace(-6, -1, 40), color='blue', edgecolor='k', alpha=0.7)
plt.xscale('log')
plt.xlabel('Companion Mass ($M_\\odot$)')
plt.ylabel('Count')
plt.title('Companion Mass Distribution ($K_1 < 6$ km/s)')
plt.axvline(3e-6, color='green', label=r"$M_\oplus$")
plt.axvline(0.000954, color='pink', label=r"$M_{Jupiter}$")
plt.axvline(0.013, color='brown', ls='--', label=r"lower $M_{BD}$")
plt.axvline(0.08, color='brown', ls='-', label=r"upper $M_{BD}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Histogram: Orbital Separation
plt.figure()
plt.hist(a_filtered_AU, bins=50, color='orange', edgecolor='k', alpha=0.7)
plt.xlabel('Orbital Separation (AU)')
plt.ylabel('Count')
plt.title('Orbital Separation Distribution ($K_1 < 6$ km/s)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Histogram: Eccentricity
plt.figure()
plt.hist(filtered_ecc, bins=40, color='purple', edgecolor='k', alpha=0.7)
plt.xlabel('Eccentricity')
plt.ylabel('Count')
plt.title('Eccentricity Distribution ($K_1 < 6$ km/s)')
plt.tight_layout()
plt.grid(True)
plt.show()

# 4. Corner plot: log10(Mc), a [AU], e
corner_data = np.vstack([
    np.log10(filtered_Mc_solar),
    a_filtered_AU,
    filtered_ecc
]).T

fig = corner.corner(
    corner_data,
    labels=[
        r"log$_{10}$(Companion Mass [$M_\odot$])",
        r"Orbital Separation [AU]",
        r"Eccentricity"
    ],
    show_titles=True,
    title_fmt=".3f",
    quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 14},
    title_kwargs={"fontsize": 12},
    hist_kwargs={"density": True}
)

# Add vertical lines to mass histogram
axes = fig.axes
axes[0].axvline(np.log10(3e-6), color='green', linestyle='-', label=r"$M_\oplus$")
axes[0].axvline(np.log10(0.000954), color='pink', linestyle='-', label=r"$M_{Jupiter}$")
axes[0].axvline(np.log10(0.013), color='brown', linestyle='--', label=r"lower $M_{BD}$")
axes[0].axvline(np.log10(0.08), color='brown', linestyle='-', label=r"upper $M_{BD}$")
axes[0].legend(fontsize=10)
plt.tight_layout()
plt.show()

# --- Summary statistics ---
if len(filtered_Mc_solar) > 0:
    print(f"Number of systems satisfying K1 < 6 km/s: {len(filtered_Mc_solar)}")
    print(f"Mean Companion Mass: {np.mean(filtered_Mc_solar):.5f} M_sun")
    print(f"Median Companion Mass: {np.median(filtered_Mc_solar):.5f} M_sun")
    print(f"Mean Orbital Separation: {np.mean(a_filtered_AU):.3f} AU")
    print(f"Median Orbital Separation: {np.median(a_filtered_AU):.3f} AU")
    print(f"Mean Eccentricity: {np.mean(filtered_ecc):.3f}")
else:
    print("No systems satisfy the RV constraint.")








def compute_semi_major_axis(K1, M_p, M_c, i, e):
    """
    Compute orbital semi-major axis from RV semi-amplitude including eccentricity.

    Parameters:
    - K1 : float or np.array, radial velocity semi-amplitude (m/s)
    - M_p : float or np.array, mass of primary star (kg)
    - M_c : float or np.array, mass of companion (kg)
    - i : float or np.array, orbital inclination (radians)
    - e : float or np.array, orbital eccentricity

    Returns:
    - a : float or np.array, semi-major axis (meters)
    """
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    numerator = G * M_c**2 * np.sin(i)**2
    denominator = K1**2 * (1 - e**2) * (M_p + M_c)
    a = numerator / denominator
    return a




# Constants
G = 6.67430e-11  # m^3/kg/s^2
M_sun = 1.989e30
AU = 1.496e11

# Inputs
K1 = 3e3  # m/s
M_p = 5 * M_sun  # kg
M_c = 1e0 * M_sun  # kg
i = np.radians(90)  # radians
e = np.linspace(0,0.9,20) #0. #0.3

plt.figure()
plt.plot( e, compute_semi_major_axis(K1, M_p, M_c, i, e)/ AU, label='Mc=M_sun')
plt.plot( e, compute_semi_major_axis(K1, M_p, 0.5 * M_c, i, e)/ AU, label='Mc=M_sun')
plt.plot( e, compute_semi_major_axis(K1, M_p, 0.1 * M_c, i, e)/ AU, label='Mc=M_sun')
plt.yscale('log')
plt.show() 




# Equation
numerator = G * M_c**2 * np.sin(i)**2
denominator = K1**2 * (1 - e**2) * (M_p + M_c)
a_meters = numerator / denominator
a_AU = a_meters / AU

print(f"Semi-major axis a = {a_meters:.3e} m = {a_AU:.3f} AU")



import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatLogSlider, FloatSlider, fixed

# --- Function ---
def compute_semi_major_axis(K1, M_p, M_c, i, e):
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    numerator = G * M_c**2 * np.sin(i)**2
    denominator = K1**2 * (1 - e**2) * (M_p + M_c)
    a = numerator / denominator
    return a

# --- Plotting Function ---
def plot_a(K1_kms, M_p_solar, M_c_solar, i_deg, e):
    # Convert to SI
    M_sun = 1.989e30
    K1 = K1_kms * 1e3
    M_p = M_p_solar * M_sun
    M_c = M_c_solar * M_sun
    i = np.radians(i_deg)
    
    a_m = compute_semi_major_axis(K1, M_p, M_c, i, e)
    a_AU = a_m / 1.496e11  # Convert to AU
    
    print(f"Semi-major axis: {a_AU:.3f} AU")
    
    # Optional: draw a bar plot or static point
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.axvline(a_AU, color='blue')
    ax.set_xlim(0, 10)
    ax.set_xlabel("Semi-major axis (AU)")
    ax.set_yticks([])
    ax.set_title(f"a = {a_AU:.3f} AU")
    plt.grid(True)
    plt.show()

# --- Widgets ---
interact(
    plot_a,
    K1_kms=FloatSlider(value=3, min=0.5, max=6.0, step=0.1, description="K₁ [km/s]"),
    M_p_solar=FloatSlider(value=2.5, min=1, max=5, step=0.1, description="Mₚ [M☉]"),
    M_c_solar=FloatLogSlider(value=0.01, base=10, min=-5, max=-0.5, step=0.01, description="M_c [M☉]"),
    i_deg=FloatSlider(value=60, min=0, max=90, step=1, description="i [deg]"),
    e=FloatSlider(value=0.3, min=0.0, max=0.9, step=0.01, description="e"),
);



"""
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

# Get coordinates from SIMBAD
result = Simbad.query_object("RT Pav")
ra, dec = result["RA"][0], result["DEC"][0]

# Convert to degrees
from astropy.coordinates import SkyCoord
import astropy.units as u
coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

# Gaia cone search
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
job = Gaia.cone_search_async(coord, radius=u.Quantity(1, u.arcsec))
gaia_results = job.get_results()
print(gaia_results)

"""
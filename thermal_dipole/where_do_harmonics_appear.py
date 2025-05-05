

import json
import numpy as np
import corner
import matplotlib.pyplot as plt 
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata 
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import json
from matplotlib.animation import FuncAnimation
import pandas as pd 
import matplotlib.patches as mpatches

def blackbody_intensity(T, wavelength):
    return (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * k_B * T)) - 1)

def thermal_oscillation(theta, phi, t, T_eff, delta_T_eff, l, m, nu, psi_T):
    """
    Calculate the local effective temperature of a star with a thermal oscillation mode.
    """
    Y_lm = sph_harm(m, l, phi, theta)
    Y_lm_normalized = np.real(Y_lm) / np.max(np.real(Y_lm))
    time_dependent_term = np.cos(2 * np.pi * nu * t + psi_T)
    return T_eff + delta_T_eff * Y_lm_normalized * time_dependent_term

def rotate_to_observer_frame(theta, phi, theta_obs, phi_obs):
    """
    Rotate the stellar coordinates such that the observer's position aligns with the new z-axis.
    
    Parameters:
        theta, phi: Spherical coordinates of the stellar surface.
        theta_obs, phi_obs: Observer's position in spherical coordinates.
    
    Returns:
        theta_rot, phi_rot: Rotated spherical coordinates.
    """
    # Convert observer position to Cartesian coordinates
    x_obs = np.sin(theta_obs) * np.cos(phi_obs)
    y_obs = np.sin(theta_obs) * np.sin(phi_obs)
    z_obs = np.cos(theta_obs)

    # Define rotation: observer's position -> z-axis
    observer_direction = np.array([x_obs, y_obs, z_obs])
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(observer_direction, z_axis)
    rotation_angle = np.arccos(np.dot(observer_direction, z_axis))
    if np.linalg.norm(rotation_axis) > 1e-10:
        rotation_axis /= np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([1, 0, 0])  # Arbitrary axis when already aligned

    # Rotation matrix
    rotation = R.from_rotvec(rotation_angle * rotation_axis)

    # Convert stellar surface to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords = np.stack((x, y, z), axis=-1)

    # Rotate coordinates
    rotated_coords = rotation.apply(coords.reshape(-1, 3)).reshape(coords.shape)

    # Convert back to spherical coordinates
    x_rot, y_rot, z_rot = rotated_coords[..., 0], rotated_coords[..., 1], rotated_coords[..., 2]
    r_rot = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
    theta_rot = np.arccos(np.clip(z_rot / r_rot, -1, 1))
    phi_rot = np.arctan2(y_rot, x_rot)

    return theta_rot, phi_rot

def project_to_observer_plane(theta_rot, phi_rot, intensity, grid_size=500):
    """
    Project the rotated stellar surface onto the observer's 2D image plane.
    """
    # Convert spherical to Cartesian
    x = np.sin(theta_rot) * np.cos(phi_rot)
    y = np.sin(theta_rot) * np.sin(phi_rot)
    z = np.cos(theta_rot)
    
    # Only keep the visible hemisphere (z > 0)
    visible = z > 0
    x_visible = x[visible]
    y_visible = y[visible]
    intensity_visible = intensity[visible]

    # Create the observer plane grid
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    x_plane, y_plane = np.meshgrid(x_grid, y_grid)
    
    # Mask points outside the unit circle
    r_plane = np.sqrt(x_plane**2 + y_plane**2)
    mask = r_plane <= 1

    # Interpolate intensity from spherical to plane
    points = np.vstack((x_visible, y_visible)).T
    projected_intensity = np.zeros_like(x_plane)
    projected_intensity[mask] = griddata(
        points, intensity_visible, (x_plane[mask], y_plane[mask]), method='linear', fill_value=0
    )
    
    return projected_intensity

def wienslaw(T):
    # T in Kelvin
    lambda_peak = 2898 / T * 1e-6 # m
    return lambda_peak

def limb_darkening(theta_rot, intensity, model="linear", **kwargs):
    """
    Apply limb darkening to the stellar surface intensity.
    
    Parameters:
        theta_rot: Array of colatitudes in the observer's frame (radians).
        intensity: Array of intensities before limb darkening.
        model: Limb-darkening model to use. Options: "linear", "quadratic", "powerlaw".
        kwargs: Additional parameters for the specific model:
            - Linear: u (coefficient).
            - Quadratic: u1, u2 (coefficients).
            - Power-law: u (coefficient).
        
    Returns:
        intensity_ld: Intensity with limb darkening applied.
    """
    # Calculate mu = cos(theta) for the observer's frame
    mu = np.cos(theta_rot)
    
    # Ensure mu is non-negative (visible hemisphere)
    mu = np.maximum(mu, 0)
    
    if model == "linear":
        # Linear limb darkening: I(mu) = I_0 * (1 - u + u * mu)
        u = kwargs.get("u", 0.5)  # Default u = 0.5
        limb_darkening_factor = (1 - u + u * mu)
    
    elif model == "quadratic":
        # Quadratic limb darkening: I(mu) = I_0 * (1 - u1 * (1 - mu) - u2 * (1 - mu)^2)
        u1 = kwargs.get("u1", 0.5)  # Default u1 = 0.5
        u2 = kwargs.get("u2", 0.5)  # Default u2 = 0.5
        limb_darkening_factor = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2
    
    elif model == "powerlaw":
        # Power-law limb darkening: I(mu) = I_0 * mu^u
        u = kwargs.get("u", 0.5)  # Default power-law exponent u = 0.5
        limb_darkening_factor = mu**u
    
    else:
        raise ValueError(f"Unknown limb-darkening model: {model}")
    
    # Apply the limb-darkening factor to the intensity
    intensity_ld = intensity * limb_darkening_factor
    
    # Ensure no negative intensity due to limb darkening
    intensity_ld = np.maximum(intensity_ld, 0)
    
    return intensity_ld



# Function to classify the presence of a second harmonic
def classify_harmonic_presence(flux, tgrid, threshold=0.05, method='lomb'):
    """
    Classify whether a second harmonic is present in the light curve.
    INPUT tgrid MUST BE NORMALIZED TO PRIMARY PERIOD (1 = PERIOD)

    Parameters:
        flux (array): Light curve flux values.
        tgrid (array): Time grid corresponding to the flux.
        threshold (float): Threshold ratio for the harmonic detection (default 5%).
    
    Returns:
        bool: True if second harmonic is present, False otherwise.
        dict: PSD analysis results for debugging (optional).
    """

    if method=='psd':
        # Compute the PSD using the Welch method
        freqs, psd = welch(flux, fs=1/np.mean(np.diff(tgrid)), nperseg=len(tgrid)//4)
    elif method=='lomb':
        frequency_resolution=10
        # Define the frequency grid
        min_freq = 1 / (tgrid[-1] - tgrid[0])  # Minimum frequency (inverse of total duration)
        max_freq = 1 / (2 * np.mean(np.diff(tgrid)))  # Nyquist frequency
        freqs = np.linspace(min_freq, max_freq, len(tgrid) * frequency_resolution)

        # Compute the Lomb-Scargle periodogram
        ls = LombScargle(tgrid, flux)
        psd = ls.power(freqs)

    # Find the fundamental frequency (largest PSD peak)
    fundamental_idx = np.argmin(np.abs(freqs - 1))  # Index of the fundamental frequency
    fundamental_freq = freqs[fundamental_idx]
    fundamental_amplitude = psd[fundamental_idx]

    # Find the second harmonic (2x fundamental frequency)
    harmonic_freq = 2 * fundamental_freq
    harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))  # Closest frequency bin
    harmonic_amplitude = psd[harmonic_idx]

    # Classify based on the amplitude ratio
    harmonic_present = harmonic_amplitude >= threshold * fundamental_amplitude

    # Optional: Return results for debugging or further analysis
    analysis_results = {
        "fundamental_freq": fundamental_freq,
        "fundamental_amplitude": fundamental_amplitude,
        "harmonic_freq": harmonic_freq,
        "harmonic_amplitude": harmonic_amplitude,
        "psd": psd,
        "freqs": freqs,
    }

    
    return harmonic_present, analysis_results


def classify_harmonic_presence_correlation(flux, tgrid, threshold=0.5, fundamental_freq=1):
    """
    Classify whether a second harmonic is present using correlation with synthetic sinusoids.

    Parameters:
        flux (array): Light curve flux values.
        tgrid (array): Time grid corresponding to the flux.
        threshold (float): Threshold for the correlation ratio to detect the harmonic.
        fundamental_freq (float): Fundamental frequency for the sinusoid (default: 1).

    Returns:
        bool: True if second harmonic is present, False otherwise.
        dict: Correlation results, including values for fundamental and harmonic.
    """
    # Normalize the flux to [0, 1]
    flux_normalized = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

    # Generate synthetic sinusoids
    synthetic_fundamental = 0.5 * (1 + np.sin(2 * np.pi * fundamental_freq * tgrid))  # Normalize to [0, 1]
    synthetic_harmonic = 0.5 * (1 + np.sin(2 * np.pi * 2 * fundamental_freq * tgrid))  # Second harmonic

    # Compute correlations
    correlation_fundamental = np.corrcoef(flux_normalized, synthetic_fundamental)[0, 1]
    correlation_harmonic = np.corrcoef(flux_normalized, synthetic_harmonic)[0, 1]

    # Classify based on correlation ratio
    harmonic_present = correlation_harmonic >= threshold * correlation_fundamental

    # if 1: #check
    #     plt.figure()
    #     plt.plot(tgrid,flux_normalized)
    #     plt.plot(tgrid,synthetic_fundamental)
    #     plt.show()

    # Return results
    analysis_results = {
        "correlation_fundamental": correlation_fundamental,
        "correlation_harmonic": correlation_harmonic,
        "harmonic_present": harmonic_present,
    }
    return harmonic_present, analysis_results



def classify_harmonic_presence_peaks(flux, tgrid, threshold=0.05, plot=False):
    """
    Classify whether a second harmonic is present by analyzing peak intervals.

    Parameters:
        flux (array): Light curve flux values.
        tgrid (array): Time grid corresponding to the flux.
        threshold (float): Minimum threshold for peak prominence (default: 0.05).
        plot (bool): If True, plot the light curve with detected peaks.

    Returns:
        bool: True if second harmonic is present, False otherwise.
        dict: Analysis results, including average time between peaks.
    """
    # Find peaks in the flux time series
    peaks, properties = find_peaks(flux)

    # Filter peaks by flux value
    min_flux = threshold * np.max(flux)
    peaks = peaks[flux[peaks] >= min_flux]

    # Check if there are enough peaks for analysis
    if len(peaks) < 2:
        return False, {"average_delta_t": None, "harmonic_present": False, "peaks": peaks}

    # Calculate time intervals between consecutive peaks
    delta_t = np.diff(tgrid[peaks])

    # Compute the average time between peaks
    average_delta_t = np.mean(delta_t)

    # Classify the harmonic presence
    harmonic_present = np.isclose(average_delta_t, 0.5, atol=0.1)

    # Optional: Plot the light curve with detected peaks
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(tgrid, flux, label="Flux Time Series", alpha=0.8)
        plt.plot(tgrid[peaks], np.array(flux[peaks]), "ro", label="Detected Peaks")
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.yscale('log')
        plt.title("Peak Detection in Light Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Return results
    analysis_results = {
        "peaks": peaks,
        "delta_t": delta_t,
        "average_delta_t": average_delta_t,
        "harmonic_present": harmonic_present,
    }
    return harmonic_present, analysis_results


# Constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 3.0e8           # Speed of light (m/s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)
# Constants for Wien's law
wien_constant = 2898e-6  # Wien's displacement constant in meters·Kelvin


# Main Parameters
T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 #/ (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)
u=0.5 #limd darkenning
pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
# Stellar surface grid
theta = np.linspace(0, np.pi, 50)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 50)  # Full longitude
theta, phi = np.meshgrid(theta, phi)

# some parameters that reproduce it 
best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 312.46496105857034, 3.3016914552936942

tgrid = np.linspace( -4/nu, 4/nu , 200)
wavelength_aaso = 806e-9 #I-band #551e-9


# some parameters that reproduce it 
best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 312.46496105857034, 3.3016914552936942
best_theta_o = 0.08726646259971647
flux=[]
projected_intensities = [] 
for t in tgrid:

   #  Compute local effective temperature
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff, best_delta_T, l=1, m=1, nu=nu, psi_T=psi_T)
    
    # Calculate intensity using blackbody radiation
    intensity = blackbody_intensity(T_eff_local, wavelength_aaso)
    
    # Rotate to observer's frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)
    
    # Apply limb darkening with the current coefficient
    intensity_ld = limb_darkening(theta_rot, intensity, model="powerlaw", u=u) #limb_darkening(theta_rot, intensity, model="linear", u=u)
    
    # Step 5: Project the intensity (with limb darkening applied) onto the observer's plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, intensity_ld, grid_size=grid_size)
    
    # Append results
    projected_intensities.append(projected_intensity)
    flux.append(np.sum(projected_intensity))


freqs, psd = welch(flux, fs=1/np.mean(np.diff(tgrid)), nperseg=len(tgrid)//4)
fig,ax=plt.subplots(2,1)
ax[0].plot(tgrid, flux)
ax[1].semilogy(freqs, psd)
plt.show()


res, res_dict=classify_harmonic_presence(flux, tgrid, threshold=0.05)






############################################ 
# grid search ffixed effective temperature 

# Define grids for inclination and delta_T_eff
wavelength = 806e-9
delta_T_eff_grid = np.linspace(0, 500, 5)  # Dipole amplitude grid (K)
theta_obs_grid = np.linspace(0, np.pi/4, 5)  # Inclination grid (colatitude in radians)
phi_obs = 0  # Fix azimuthal angle for simplicity (you can vary it if needed)
num_samples = len(delta_T_eff_grid)*len(theta_obs_grid)
# Data storage
results = []

# Grid-based loop
cnt=0
for delta_T_eff in delta_T_eff_grid:
    for theta_obs in theta_obs_grid:
        print(f"delta_T_eff: {delta_T_eff}, theta_obs: {theta_obs}")
        print(f"iteration {cnt}/{num_samples}")
        # Fixed parameters
        T_eff = 3000  # Fixed effective temperature (K)

        # Generate light curve
        flux = []
        for t in tgrid:
            T_eff_local = thermal_oscillation(theta, phi, t, T_eff, delta_T_eff, l=1, m=1, nu=nu, psi_T=psi_T)
            intensity = blackbody_intensity(T_eff_local, wavelength)
            theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_obs, phi_obs)

            # Apply limb darkening
            intensity_ld = limb_darkening(theta_rot, intensity, model="powerlaw", u=0.5)

            # Project onto the observer's plane
            projected_intensity = project_to_observer_plane(theta_rot, phi_rot, intensity_ld, grid_size=grid_size)
            flux.append(np.sum(projected_intensity))

        # Classify harmonic presence
        res, res_dict = classify_harmonic_presence(flux, tgrid, threshold=0.05)

        # Store results
        results.append({
            "T_eff": T_eff,
            "delta_T_eff": delta_T_eff,
            "theta_obs": theta_obs,
            "harmonic_present": res,
            "fundamental_freq": res_dict["fundamental_freq"],
            "harmonic_freq": res_dict["harmonic_freq"],
            "harmonic_amplitude_ratio": res_dict["harmonic_amplitude"] / res_dict["fundamental_amplitude"]
        })

        cnt+=1

# Convert results to a DataFrame
df = pd.DataFrame(results)


# Save to a file for further analysis
df.to_csv(f"grid_dT-incl_harmonic_classification_Teff-{T_eff}K.csv", index=False)

# Summary statistics
print(df["harmonic_present"].value_counts(normalize=True))

# Visualization
plt.figure(figsize=(8, 6))
harmonic_matrix = df.pivot_table(index="theta_obs", columns="delta_T_eff", values="harmonic_present")
plt.imshow(harmonic_matrix, extent=(delta_T_eff_grid[0], delta_T_eff_grid[-1],
                                    theta_obs_grid[0], theta_obs_grid[-1]),
           aspect="auto", origin="lower", cmap="coolwarm")
plt.colorbar(label="Harmonic Presence (0 = False, 1 = True)")
plt.xlabel("Dipole Amplitude ($\delta T_{\mathrm{eff}}$)")
plt.ylabel("Inclination (radians)")
plt.title("Harmonic Presence as a Function of Dipole Amplitude and Inclination")
plt.show()



###########################################
#  grid is defined directly in terms of the ratio \lambda_{\text{peak}} / \lambda_{\text{obs}}

# Fixed parameters
wavelength_obs = 806e-9  # Observed wavelength (I-band) in meters
delta_T_eff_fixed = 300  # Fixed dipole amplitude in Kelvin
theta_obs_grid = np.linspace(np.deg2rad(1), np.deg2rad(40), 20)  # Inclination grid (colatitude in radians)
ratio_grid = np.logspace(-1, 1.2, 20)  # Grid for ratio lambda_peak / wavelength_obs
num_samples = len(ratio_grid)*len(theta_obs_grid)

nu = 1  # Frequency (Hz)
psi_T = 0  # Phase offset (rad)
grid_size = 500
theta = np.linspace(0, np.pi, 40)  # Stellar colatitude
phi = np.linspace(0, 2 * np.pi, 40)  # Stellar longitude
theta, phi = np.meshgrid(theta, phi)
tgrid = np.linspace(-4 / nu, 4 / nu, 100)  # Time grid

finer_factor = 4 
# Create a finer time grid
tgrid_fine = np.linspace(tgrid[0], tgrid[-1], len(tgrid) * finer_factor)

legend_patches = []

interpolate_ts= True
plt.figure(figsize=(8, 6))
method = 'peaks'

for hatchss, lss,delta_T_eff_fixed in zip(['.', '/','\\'],[':','--','-'],[100, 300, 500]):
    # Data storage
    results = []
    cnt=0
    # Grid search loop
    for ratio_lambda in ratio_grid:
        # Calculate effective temperature based on ratio
        lambda_peak = ratio_lambda * wavelength_obs
        T_eff = wien_constant / lambda_peak

        for theta_obs in theta_obs_grid:
            print(f"Ratio: {ratio_lambda}, T_eff: {T_eff}, theta_obs: {theta_obs}")
            print(f"iteration {cnt}/{num_samples}")
            # Generate light curve
            flux = []
            for t in tgrid:
                T_eff_local = thermal_oscillation(theta, phi, t, T_eff, delta_T_eff_fixed, l=1, m=1, nu=nu, psi_T=psi_T)
                intensity = blackbody_intensity(T_eff_local, wavelength_obs)
                theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_obs, 0)  # Fixed azimuth
                intensity_ld = limb_darkening(theta_rot, intensity, model="powerlaw", u=0.5)
                projected_intensity = project_to_observer_plane(theta_rot, phi_rot, intensity_ld, grid_size=grid_size)
                flux.append(np.sum(projected_intensity))


            # Interpolate the flux onto the finer time grid using spline
            if interpolate_ts:
                interpolator = interp1d(tgrid, flux, kind="cubic")

                # Interpolate onto a finer grid
                flux_fine = interpolator(tgrid_fine)
                
                #flux_extended = np.concatenate([flux_fine[::-1], flux_fine, flux_fine[::-1]])
                #tgrid_extended = np.

                # Classify harmonic presence
                if method=='psd':
                    res, res_dict = classify_harmonic_presence(flux_fine, tgrid_fine, threshold=0.05, method=method)
                elif method=='peaks':
                    res, res_dict = classify_harmonic_presence_peaks(flux_fine, tgrid_fine, threshold=0.05, plot=False)
                if method=='lomb':
                    res, res_dict = classify_harmonic_presence(flux_fine, tgrid_fine, threshold=0.05, method=method)
            else:
                if method=='psd':
                    res, res_dict = classify_harmonic_presence(flux, tgrid, threshold=0.05, method=method)
                elif method=='peaks':
                    res, res_dict = classify_harmonic_presence_peaks(flux, tgrid, threshold=0.05, plot=False)
                if method=='lomb':
                    res, res_dict = classify_harmonic_presence(flux, tgrid, threshold=0.05, method=method)
            # # if ratio_lambda > 2 :
            # plt.figure()
            # plt.plot( tgrid_fine, flux_fine)
            # plt.ylabel('flux')
            # plt.xlabel('time')
            # plt.title(f'Teff={T_eff}, ratio={ratio_lambda}, incl={theta_obs*180/3.14}')
            # plt.show()
            # Store results

            if method=='psd':
                results.append({
                    "ratio_lambda": ratio_lambda,
                    "T_eff": T_eff,
                    "theta_obs": theta_obs,
                    "harmonic_present": res,
                    "fundamental_freq": res_dict["fundamental_freq"],
                    "harmonic_freq": res_dict["harmonic_freq"],
                    "harmonic_amplitude_ratio": res_dict["harmonic_amplitude"] / res_dict["fundamental_amplitude"]
                })
            elif method == 'peaks':
                results.append({
                    "ratio_lambda": ratio_lambda,
                    "T_eff": T_eff,
                    "theta_obs": theta_obs,
                    "harmonic_present": res,
                })
            elif method == 'lomb':
                results.append({
                    "ratio_lambda": ratio_lambda,
                    "T_eff": T_eff,
                    "theta_obs": theta_obs,
                    "harmonic_present": res,
                    "fundamental_freq": res_dict["fundamental_freq"],
                    "harmonic_freq": res_dict["harmonic_freq"],
                    "harmonic_amplitude_ratio": res_dict["harmonic_amplitude"] / res_dict["fundamental_amplitude"]
                })
            cnt+=1

            
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save results for further analysis
    df.to_csv(f"grid_wein_ratio-incl_harmonic_classification_{round(delta_T_eff_fixed)}K.csv", index=False)

    # # Visualization: Contour plot for harmonic presence
    # plt.figure(figsize=(8, 6))
    # harmonic_matrix = df.pivot_table(index="theta_obs", columns="ratio_lambda", values="harmonic_present")
    # plt.imshow(harmonic_matrix, extent=(ratio_grid[0], ratio_grid[-1],
    #                                     theta_obs_grid[0], theta_obs_grid[-1]),
    #            aspect="auto", origin="lower", cmap="coolwarm")
    # plt.colorbar(label="Harmonic Presence (0 = False, 1 = True)")
    # plt.xlabel(r"$\lambda_{\mathrm{BB,peak}} / \lambda_{\mathrm{obs}}$", fontsize=12)
    # plt.ylabel("Inclination (radians)", fontsize=12)
    # #plt.title("Harmonic Presence as a Function of $\lambda_{\mathrm{peak}} / \lambda_{\mathrm{obs}}$ and Inclination", fontsize=14)
    # plt.tight_layout()
    # plt.show()
    # Visualization: Contour plot for harmonic presence boundary



    # Create a pivot table from the DataFrame
    harmonic_matrix = df.pivot_table(index="theta_obs", columns="ratio_lambda", values="harmonic_present")

    # Convert theta_obs to degrees for better visualization
    theta_obs_deg = np.rad2deg(theta_obs_grid)

    # Create 2D grid for ratio_lambda and theta_obs_deg
    X, Y = np.meshgrid(ratio_grid, theta_obs_deg)

    # Plot the contour for the harmonic presence boundary (0.5 threshold)
    cs = plt.contour(X, Y, harmonic_matrix.values, levels=[0.5],  linewidths=2, linestyles=lss)#, label=r'$\delta T='+f'{round(delta_T_eff_fixed)}K')
    plt.contourf(X, Y, harmonic_matrix.values, levels=[0.5, 1.0], hatches=hatchss, colors='none', alpha=0)
    #artists, labels = cs.legend_elements()
    legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='black', hatch=hatchss,
               label=r'$\delta T = ' + f'{delta_T_eff_fixed}K$') )
    #plt.legend(artists, labels, handleheight=2)
# Labels and Title
fs=15
plt.legend(handles=legend_patches, fontsize=13, loc='upper left')
plt.xlabel(r"$\lambda_{\mathrm{BB,peak}} / \lambda_{\mathrm{obs}}$", fontsize=15)
plt.ylabel("Dipole inclination [degrees]", fontsize=fs)
plt.gca().tick_params(labelsize=fs)
plt.xscale('log')
#plt.title("Harmonic Presence Boundary", fontsize=14)

# Improve layout
plt.tight_layout()
plt.savefig('harmonic_appearence_boundary.png', bbox_inches='tight',dpi=300)

# Show the plot
plt.show()


## to look at the actual ratio
# Assuming `df` contains your simulation results with columns:
# "ratio_lambda", "theta_obs", "harmonic_amplitude_ratio"

# Pivot the data to create a matrix for the heatmap
harmonic_ratio_matrix = df.pivot_table(index="theta_obs", columns="ratio_lambda", values="harmonic_amplitude_ratio")

# Convert theta_obs to degrees for better visualization
theta_obs_deg = np.rad2deg(theta_obs_grid)

# Create 2D grid for ratio_lambda and theta_obs_deg
X, Y = np.meshgrid(df["ratio_lambda"].unique(), theta_obs_deg)

# Plot the heatmap
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, np.log10(harmonic_ratio_matrix.values), levels=50, cmap="viridis")
cbar = plt.colorbar(contour)
cbar.set_label("Harmonic-to-Fundamental Ratio", fontsize=12)

# Set log scale for the x-axis
plt.xscale("log")

# Labels and Title
plt.xlabel(r"$\lambda_{\mathrm{BB,peak}} / \lambda_{\mathrm{obs}}$ (Log Scale)", fontsize=12)
plt.ylabel("Inclination (Degrees)", fontsize=12)
plt.title("Harmonic-to-Fundamental Ratio Across Parameter Space", fontsize=14)

# Improve layout
plt.tight_layout()
plt.show()




### check particular ts 


results = []
# Grid search loop
theta_obs = np.deg2rad(40)
for ratio_lambda in [0.5, 3]:
    # Calculate effective temperature based on ratio
    lambda_peak = ratio_lambda * wavelength_obs
    T_eff = wien_constant / lambda_peak

    #for theta_obs in theta_obs_grid:
    print(f"Ratio: {ratio_lambda}, T_eff: {T_eff}, theta_obs: {theta_obs}")
    print(f"iteration {cnt}/{num_samples}")
    # Generate light curve
    flux = []
    for t in tgrid:
        T_eff_local = thermal_oscillation(theta, phi, t, T_eff, delta_T_eff_fixed, l=1, m=1, nu=nu, psi_T=psi_T)
        intensity = blackbody_intensity(T_eff_local, wavelength_obs)
        theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_obs, 0)  # Fixed azimuth
        intensity_ld = limb_darkening(theta_rot, intensity, model="powerlaw", u=0.5)
        projected_intensity = project_to_observer_plane(theta_rot, phi_rot, intensity_ld, grid_size=grid_size)
        flux.append(np.sum(projected_intensity))

    plt.plot(tgrid, np.array(flux)/np.max(flux), label=f'{ratio_lambda}')
plt.legend()
plt.show()


freqs, psd = welch(flux, fs=1/np.mean(np.diff(tgrid)), nperseg=len(tgrid)//4)
fig,ax=plt.subplots(2,1)
ax[0].plot(tgrid, flux)
ax[1].semilogy(freqs, psd)
plt.show()

###########################################
#Monte Carlo

# Simulation parameters
T_eff_range = [2000, 5000]  # Effective temperature range (K)
delta_T_eff_range = [0, 500]  # Dipole amplitude range (K)
num_samples = 1000  # Number of Monte Carlo simulations
wavelength = 806e-9  # Fixed observed wavelength (I-band)
nu = 1  # Frequency (Hz)
psi_T = 0  # Phase offset (rad)
grid_size = 500
theta = np.linspace(0, np.pi, 20)  # Stellar colatitude
phi = np.linspace(0, 2 * np.pi, 20)  # Stellar longitude
theta, phi = np.meshgrid(theta, phi)
tgrid = np.linspace(-4 / nu, 4 / nu, 100)  # Time grid

# Function to sample random inclinations (uniform over 3D sphere)
def random_inclination():
    z = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.arccos(z)
    return theta, phi

# Data storage
results = []

# Monte Carlo loop
for i in range(num_samples):
    print(f"iteration {i}/{num_samples}")
    # Sample parameters
    T_eff = 3000 #np.random.uniform(*T_eff_range)
    delta_T_eff = np.random.uniform(*delta_T_eff_range)
    theta_obs, phi_obs = random_inclination()
    
    # Generate light curve
    flux = []
    for t in tgrid:
        T_eff_local = thermal_oscillation(theta, phi, t, T_eff, delta_T_eff, l=l, m=m, nu=nu, psi_T=psi_T)
        intensity = blackbody_intensity(T_eff_local, wavelength_aaso)
        theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_obs, phi_obs)
        # Apply limb darkening with the current coefficient
        intensity_ld = limb_darkening(theta_rot, intensity, model="powerlaw", u=u) #limb_darkening(theta_rot, intensity, model="linear", u=u)
        
        # Step 5: Project the intensity (with limb darkening applied) onto the observer's plane
        projected_intensity = project_to_observer_plane(theta_rot, phi_rot, intensity_ld, grid_size=grid_size)
        
        flux.append(np.sum(projected_intensity))
    
    # Classify harmonic presence
    res, res_dict = classify_harmonic_presence(flux, tgrid, threshold=0.05)
    
    # Store results
    results.append({
        "T_eff": T_eff,
        "delta_T_eff": delta_T_eff,
        "theta_obs": theta_obs,
        "phi_obs": phi_obs,
        "harmonic_present": res,
        "fundamental_freq": res_dict["fundamental_freq"],
        "harmonic_freq": res_dict["harmonic_freq"],
        "harmonic_amplitude_ratio": res_dict["harmonic_amplitude"] / res_dict["fundamental_amplitude"]
    })

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save to a file for further analysis
#df.to_csv("monte_carlo_harmonic_results.csv", index=False)

# Summary statistics
print(df["harmonic_present"].value_counts(normalize=True))

# Optional: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(df["T_eff"], df["delta_T_eff"], c=df["harmonic_present"], cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Harmonic Presence (0 = False, 1 = True)")
plt.xlabel("Effective Temperature (K)")
plt.ylabel("Dipole Amplitude (K)")
plt.title("Monte Carlo Simulation: Harmonic Presence")
plt.show()


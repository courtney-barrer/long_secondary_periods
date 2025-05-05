

import json
import numpy as np
import corner
import matplotlib.pyplot as plt 
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata 
import json
from matplotlib.animation import FuncAnimation

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


# Constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 3.0e8           # Speed of light (m/s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)



########################################
#### Looking at light curves and "secondary eclipses" reproduced by thermal modes 
#########################################

# Main Parameters
T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)

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

tgrid = np.linspace( -2/nu, 2/nu , 100)
wavelength_aaso = 551e-9
flux_dict = {}
theta0_grid = np.deg2rad( np.array([0, 5, 20, 50,  80, 90, 100]) ) # np.linspace( np.deg2rad(0 ),np.deg2rad( 110 ) , 8 )
plt.figure()
for best_theta_o in theta0_grid: 
    flux=[]
    projected_intensities = []  
    for t in tgrid[:len(tgrid)//2]:

        T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

        # Rotate to observer frame
        theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

        # Project onto observer plane
        projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

        # Append results
        projected_intensities.append(projected_intensity)
        flux.append(np.sum(projected_intensity))

    plt.plot( (tgrid[:len(tgrid)//2]-np.min(tgrid[:len(tgrid)//2])) / (757 * 3600 * 24), (flux-np.min(flux))/np.max(flux-np.min(flux)) , label  = r'$i$='+f'{round(np.rad2deg(best_theta_o))} deg')
    flux_dict[best_theta_o]  = flux 

plt.legend() 
plt.xlabel('LSP Phase',fontsize=15)
plt.ylabel( "Normalized Flux",fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.figure(); plt.plot( flux ); plt.show()
plt.tight_layout()
plt.savefig("light_curve_secondary_eclipse_with_dipole.png",dpi=300)
plt.show()


########################################
#### function of observed wavelength  

# Main Parameters
T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)

pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
# Stellar surface grid
theta = np.linspace(0, np.pi, 50)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 50)  # Full longitude
theta, phi = np.meshgrid(theta, phi)

best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 312.46496105857034, 3.3016914552936942
best_theta_o = np.deg2rad( 5 )

tgrid = np.linspace( -2/nu, 2/nu , 100)
#wavelength_aaso = 551e-9
flux_dict = {}
wave_grid = np.linspace( 300e-9, 2000e-9, 10)
plt.figure()
for wavelength_aaso in wave_grid: 
    flux=[]
    projected_intensities = []  
    for t in tgrid[:len(tgrid)//2]:

        T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

        # Rotate to observer frame
        theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

        # Project onto observer plane
        projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

        # Append results
        projected_intensities.append(projected_intensity)
        flux.append(np.sum(projected_intensity))

    plt.plot( (tgrid[:len(tgrid)//2]-np.min(tgrid[:len(tgrid)//2])) / (757 * 3600 * 24), (flux-np.min(flux))/np.max(flux-np.min(flux)) , label  = r'$\lambda$='+f'{round(wavelength_aaso*1e9)}nm')
    flux_dict[best_theta_o]  = flux 

plt.legend() 
plt.xlabel('LSP Phase',fontsize=15)
plt.ylabel( "Normalized Flux",fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.figure(); plt.plot( flux ); plt.show()
plt.tight_layout()


########################################
########################################
#### for give incl. is there a contour effective temperature and observed wavelength (Wein displacement law)
# where we maintain these secondary "eclipses"?

# Main Parameters
#T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)

pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
# Stellar surface grid
theta = np.linspace(0, np.pi, 60)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 60)  # Full longitude
theta, phi = np.meshgrid(theta, phi)

best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 200, 3.3016914552936942
best_theta_o = np.deg2rad( 5 )

tgrid = np.linspace( -2/nu, 2/nu , 100)
#wavelength_aaso = 551e-9
flux_dict = {}
T_grid = [400, 500,  1000, 2000, 3000] #np.logspace( 2.2, 3.5, 10)



plt.figure()

for T_eff in T_grid: 
    wavelength_aaso = 0.55 * wienslaw(T_eff)
    flux=[]
    projected_intensities = []  
    for t in tgrid[:len(tgrid)//2]:

        T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

        # Rotate to observer frame
        theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

        # Project onto observer plane
        projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

        # Append results
        projected_intensities.append(projected_intensity)
        flux.append(np.sum(projected_intensity))

    plt.plot( (tgrid[:len(tgrid)//2]-np.min(tgrid[:len(tgrid)//2])) / (757 * 3600 * 24), (flux-np.min(flux))/np.max(flux-np.min(flux)) , \
             label  = r"$T_{eff}$="+f'{round(T_eff)}K, ' + r"$\delta T$="+f'{round(best_delta_T)}K, ' +r'$\lambda$='+f"{round(wavelength_aaso*1e6,1)}um, i = 5deg")
    flux_dict[best_theta_o]  = flux 

### reference LSP (no secondary eclipse )
#T_eff = 3000
flux=[]
projected_intensities = []  

wavelength_aaso = 0.55 * wienslaw(T_eff)
for t in tgrid[:len(tgrid)//2]:

    T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, np.pi/4, best_phi_o)

    # Project onto observer plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

    # Append results
    projected_intensities.append(projected_intensity)
    flux.append(np.sum(projected_intensity))

plt.plot( (tgrid[:len(tgrid)//2]-np.min(tgrid[:len(tgrid)//2])) / (757 * 3600 * 24), (flux-np.min(flux))/np.max(flux-np.min(flux)) , \
            label  = r"$T_{eff}$="+f'{round(T_eff)}K, ' + r"$\delta T$="+f'{round(best_delta_T)}K, ' +r'$\lambda$='+f"{round(wavelength_aaso*1e6,1)}um, i={45}deg")


plt.legend() 
plt.xlabel('LSP Phase',fontsize=15)
plt.ylabel( "Normalized Flux",fontsize=15)
plt.gca().tick_params(labelsize=15)
#plt.figure(); plt.plot( flux ); plt.show()
plt.tight_layout()
plt.savefig( "lightcurves_of_secondary_minima_dipole_mode.png", dpi=300)

plt.show()




#########################
# Movie time

l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 # 0.7 * np.pi * 2 # Phase offset (rad)

pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
# Stellar surface grid
theta = np.linspace(0, np.pi, 60)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 60)  # Full longitude
theta, phi = np.meshgrid(theta, phi)

best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 200, 3.3016914552936942
best_theta_o = np.deg2rad( 10 )

tgrid = np.linspace( -2/nu, 2/nu , 100)

#T_eff = 3000
flux=[]
projected_intensities = []  
T_eff = 1000
wavelength_aaso = 0.55 * wienslaw(T_eff)
for t in tgrid:

    T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

    # Project onto observer plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

    # Append results
    projected_intensities.append(projected_intensity)
    flux.append(np.sum(projected_intensity))




# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

flux = np.array( flux ) / np.max( flux)
times = ( tgrid - tgrid[0] ) / ( 24 * 3600)
# Subplot 1: Magnitude vs time
ax1 = axes[0]
line, = ax1.plot([], [])
ax1.set_xlim(np.min(times), np.max(times))
ax1.set_ylim(0, 1)
ax1.set_xlabel("Time")
ax1.set_ylabel(r"$\delta$ Flux")
#ax1.legend()



# Subplot 2: Heatmap of projected intensity
ax2 = axes[1]

origin = 'lower'
extent =  np.array([1, -1, -1, 1])

# Plot the heatmap
max_int = np.max(projected_intensities)
heatmap = ax2.imshow(
    projected_intensities[0]/max_int, origin=origin, cmap="inferno",
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

# heatmap = ax2.imshow(
#     projected_intensities[0], origin="lower", cmap="inferno",
#     extent=[-1, 1, -1, 1], vmin=np.min(projected_intensities), vmax=np.max(projected_intensities)
# )
# ax2.set_title("Projected Intensity")
# ax2.set_xlabel("Observer Plane X")
# ax2.set_ylabel("Observer Plane Y")
# plt.colorbar(heatmap, ax=ax2, label="Intensity")

# Update function for animation
def update(frame):
    # Update line plot
    line.set_data( times[:frame], np.array( flux[:frame]) / np.max( flux )  )
    #line.set_data(1/(24*60*60) * tgrid[:frame+1], magnitude[:frame+1])
    
    # Update heatmap
    heatmap.set_data(projected_intensities[frame]/max_int)
    return line, heatmap

# Create animation
ani = FuncAnimation(fig, update, frames=len(projected_intensities)-1, interval=10, blit=True)

# Save or display the animation
ani.save("light_curve_thermal_dipole_secondary_minima.mp4", fps=10, dpi=150)
#plt.show()


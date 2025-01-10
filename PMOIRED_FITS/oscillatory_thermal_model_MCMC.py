###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
############################# #####################
############################# #####################
#########################          ################
############################# #####################
############################# #####################
############################# #####################
############################# #####################
############################# #####################
###################################################
#%% MCMC fitting

import numpy as np
import emcee 
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.interpolate import griddata 
from astropy.io import fits
import os 
import glob 
import pmoired
import corner
import json
import sys
import argparse
sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

# Constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 3.0e8           # Speed of light (m/s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)



"""
input parameters to fit: theta_o, phi_o, delta_T
fixed input: theta, phi, T_eff, t, nu, psi_T, l, m, wavelength, grid_size, obs_files
output: reduced chi^2 

def thermal_dipole_fit( fit_param, fixed_param)

    theta_o, phi_o, delta_T  = fit_param

    theta, phi, T_eff, t, nu, psi_T, l, m, wavelength, grid_size,save_fits_path, img_pixscl, binning, insname, obs_files = fixed_param

    # Calculate local effective temperature
    T_eff_local = thermal_oscillation(theta, phi, 0, T_eff, delta_T, l, m, nu, psi_T)

    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_o, phi_o)

    # Project onto observer plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength), grid_size =grid_size )

    # pad 
    projected_intensity = np.pad( projected_intensity, projected_intensity.shape[0] * pad_factor, mode='constant', constant_values=0)
    
    # name an save to fits
    name = f'theta-{theta_o}_phi-{phi_o}_T-{delta_T}.fits'
    intensity_2_fits(projected_intensity, dx=dx, dy=dy, name=name, data_path = save_fits_path, header_dict=header_dict)

    # format the observed data  (oi) and generate synthetic observed data (oif) from the observed conditions and the image file
    oi, oif = simulate_obs_from_image_reco( obs_files, save_fits_path+name, img_pixscl=img_pixscl, binning=binning, insname = insname)

    
    comp_dict_v2=plot_util.compare_models(oi, oif , measure='V2', kwargs=kwargs)
    comp_dict_cp=plot_util.compare_models(oi, oif , measure='CP', kwargs=kwargs)

    v2_chi2 = np.mean(flatten( get_all_values( comp_dict_v2['chi2'] ) ) ) )
    cp_chi2 = np.mean(flatten( get_all_values( comp_dict_cp['chi2'] ) ) ))
    chi2 = (v2_chi2 + cp_chi2) / 2

    return chi2
"""



def blackbody_intensity(T, wavelength):
    """
    Calculate black body intensity using Planck's law.
    """
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


def intensity_2_fits(projected_intensity, dx, dy, name, data_path, header_dict={}, write_file=True):

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print( 'Path created: ', data_path )
    
    p = fits.PrimaryHDU( projected_intensity )

    # set headers 
    p.header.set('CRPIX1', projected_intensity.shape[0] / 2 ) # Reference pixel 

    p.header.set('CRPIX2', projected_intensity.shape[0] / 2 ) # Reference pixel 

    p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel 

    p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel  

    #p.header.set('CDELT1', dx * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT1', dx ) # Coord. incr. per pixel 

    #p.header.set('CDELT2', dy * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT2', dy  ) # Coord. incr. per pixel 

    p.header.set('CUNIT1', 'mas'  ) # Physical units for CDELT1 and CRVAL1 

    p.header.set('CUNIT2', 'mas'  ) # Physical units for CDELT1 and CRVAL1 
    
    p.header.set('HDUNAME', name ) # Physical units for CDELT1 and CRVAL1 

    if header_dict:
        for k, v in header_dict.items():
            p.header.set(k, v)

    h = fits.HDUList([])
    h.append( p ) 

    if write_file:
        h.writeto( data_path + name, overwrite=True )

    return h 



def sort_baseline_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:] ] ))
    return(n_key)

def sort_triangle_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:4],B_string[4:]  ] ))
    return(n_key)


def enforce_ordered_baselines_keys(data, change_baseline_key_list):
    """_summary_

    Args:
        data (_type_): _description_
        change_baseline_key_list (_type_): list of keys that have baselines in them
    """
    for i in range(len(data)):
        #enforce correct ordering of baseline keys (sometimes we get 'C0D1' and the other is 'D1C0')
        for k in change_baseline_key_list:
            if (k == 'baselines') :
                tmp = [sort_baseline_string(baseline_key ) for baseline_key in data[i][k]]
                data[i][k] = tmp
            else:
                for baseline_key in data[i][k].copy().keys():
                    new_key = sort_baseline_string(baseline_key )
                    data[i][k][new_key] = data[i][k].pop(baseline_key)
           
           
             
def enforce_ordered_triangle_keys(data, change_triangle_key_list):
    """_summary_

    Args:
        data (_type_): _description_
        change_baseline_key_list (_type_): list of keys that have baselines in them
    """
    for i in range(len(data)):
        #enforce correct ordering of baseline keys (sometimes we get 'C0D1' and the other is 'D1C0')
        for k in change_triangle_key_list:
            if (k == 'triangles'):
                tmp = [sort_triangle_string(baseline_key ) for baseline_key in data[i][k]]
                data[i][k] = tmp     
            else:
                for traingle_key in data[i][k].copy().keys():
                    new_key = sort_triangle_string(traingle_key )
                    data[i][k][new_key] = data[i][k].pop( traingle_key )
         
def simulate_obs_from_image_reco_FAST( oi, image_file , img_pixscl = None):
    # FAST because we don't re-read in obs_files to generate oi each time 

    if type(image_file)==str: #then passing a fits file name

        with fits.open(image_file) as d_model:
            img = d_model[0].data

            #assert (abs( float( d_model[0].header['CUNIT2'] ) ) - abs( float(  d_model[0].header['CUNIT1']) ) ) /  float( d_model[0].header['CDELT1'] ) < 0.001
            # we assert the image has to be square..
            #assert abs(float( d_model[0].header['CDELT2'])) == abs(float(d_model[0].header['CDELT1']))
            
            img_units = d_model[0].header['CUNIT1']

    else: # its a open fits file
        d_model = image_file
        img = d_model[0].data
        img_units = d_model[0].header['CUNIT1']
        
    if img_pixscl is None:
        img_pixscl = d_model[0].header['CDELT1']     

    if img_units == 'deg':
        img_pixscl *= 3600*1e3 # convert to mas
    if img_units == 'mas':
        pass 
    else:  
        raise TypeError('Units not recognized')

    #oi = pmoired.OI(obs_files, binning = binning, insname = insname)

    oif = pmoired.OI()

    fake_obs_list = []
    for a in oi.data: 
        
        cube = {}
        cube['scale'] = img_pixscl # mas / pixel
        x = img_pixscl * np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0])  # mas
        y = img_pixscl * np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0])  # mas
        cube['X'] , cube['Y'] =  np.meshgrid(x, y)
        cube['image'] = np.array([ img  for _ in a['WL']] )
        cube['WL'] = a['WL']
        
        if hasattr(a['MJD'], '__len__'):
            mjd = np.median( a['MJD'] ) 
            #print( "here   ---\n\n")
        else:
            print("weird that a[MJD] has no length")

        fake_obs_list.append( 
            pmoired.oifake.makeFakeVLTI(\
                t= a['telescopes'],\
                target = ( a['header']['RA']* 24 / 360 , a['header']['DEC'] ),\
                lst = [a['LST']], \
                wl = a['WL'], \
                mjd0 = [mjd], #a[ 'MJD'], #[mjd], #a[ 'MJD'],\
                cube = cube ) 
        )
        # if WARNING:  1 out of 1 LST are not observable! then doesn't append mjd.. hack fix:
        if  len( fake_obs_list[-1]["MJD"] )==0:
            fake_obs_list[-1]["MJD"] = [mjd]
        try:
            del mjd
        except:
            print("no mjd to delete")

    # makefake does some operation on MJD so still doesn't match.  
    oif.data = fake_obs_list

    # sort data by MJD - issues with Gravity when x['MJD'] is a list... TO DO 
    
    # DO ALL THIS SORTING ON oi BEFORE INPUTING 
    # oi.data = sorted(oi.data, key=lambda x: x['MJD'][0]) # have to take first one because sometimes a list 
    
    # oif.data = sorted(oif.data, key=lambda x: x['MJD'][0])
    
    # ## SILLY BUG IN PMOIRED WHERE BASELINE/TRIANGLE KEYS ARE INCONSISTENT (e.g. 'D0C1' then 'C1D0')
    # ## we fix this here by ordering all relevant keys
    change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
    change_triangle_key_list = ['triangles','OI_T3']
    # enforce_ordered_baselines_keys(oi.data, change_baseline_key_list)
    enforce_ordered_baselines_keys(oif.data, change_baseline_key_list)
    # enforce_ordered_triangle_keys(oi.data, change_triangle_key_list)
    enforce_ordered_triangle_keys(oif.data, change_triangle_key_list)

    return( oif )



def flatten(xss):
    return [x for xs in xss for x in xs]


def get_all_values(nested_dict):
    """
    Recursively extract all root values from a nested dictionary.

    Parameters:
        nested_dict (dict): The input dictionary, which may have arbitrary nesting.

    Returns:
        list: A flattened list of all values in the nested dictionary.
    """
    values = []
    
    def extract_values(d):
        for key, value in d.items():
            if isinstance(value, dict):  # If value is a dictionary, recurse
                extract_values(value)
            else:  # Otherwise, add the value to the list
                values.append(value)
    
    extract_values(nested_dict)
    return values


def thermal_dipole_fit(fit_param,  **kwargs):
    """
    Calculate the reduced chi-square for fitting the thermal dipole model, 
    with instrument-specific parameters derived inside the function.

    Parameters:
        fit_param (list): [theta_o, phi_o, delta_T] - Parameters to fit.
        kwargs: Fixed parameters and configurations (see below).

    Fixed Parameters (kwargs):
        ins (str): Instrument name ('pionier', 'gravity', etc.).
        theta, phi (np.ndarray): Stellar grid coordinates. (Default: 200x200 grid)
        T_eff (float): Effective temperature (K). (Default: 3000 K)
        t (float): Time (s). (Default: 0)
        nu (float): Frequency (Hz). (Default: 1 / (757 * 24 * 60 * 60))
        psi_T (float): Phase offset (radians). (Default: 0.7 * 2π)
        l, m (int): Spherical harmonic degree and order. (Default: 1, 1)
        grid_size (int): Grid size for projection. (Default: 500)
        pad_factor (float): Padding factor for intensity images. (Default: 2)
        dx, dy (float): Pixel scale in mas. (Default: 1 mas each)
        save_fits_path (str): Path to save FITS files. (Default: './')
        save_fits (bool): Whether to save the FITS file. (Default: True)
        header_dict (dict): Header metadata for the FITS file. (Default: Empty dict)

    Returns:
        float: Reduced chi-square of the fit.
    """

    fit_ud = kwargs.get('fit_ud', True)

    theta = kwargs.get('theta', np.linspace(0, np.pi, 50))
    phi = kwargs.get('phi', np.linspace(0, 2 * np.pi, 50))
    theta, phi = np.meshgrid(theta, phi)

    ins = kwargs.get('ins', 'pionier')
    T_eff = kwargs.get('T_eff', 3000)
    t = kwargs.get('t', 0)  # Default time is 0
    nu = kwargs.get('nu', 1 / (757 * 24 * 60 * 60))
    psi_T = kwargs.get('psi_T', 0.7 * np.pi * 2)
    l = kwargs.get('l', 1)
    m = kwargs.get('m', 1)
    grid_size = kwargs.get('grid_size', 500)
    pad_factor = kwargs.get('pad_factor', 2)
    dx = kwargs.get('dx', 1)  # Default pixel scale in mas
    dy = kwargs.get('dy', 1)
    save_fits = kwargs.get('save_fits', True)
    header_dict = kwargs.get('header_dict', {})


    # Unpack fit parameters
    if fit_ud:
        theta_o, phi_o, delta_T, UD_diam  = fit_param
    else: 
        theta_o, phi_o, delta_T = fit_param
    # Extract fixed parameters or assign defaults
    oi = kwargs.get('oi', None)
    assert oi is not None

    if not fit_ud:
        UD_diam = kwargs.get( 'UD_diam', 3)


    # to save fits files on parameteric model images 
    save_fits_path = f'/home/rtc/Downloads/fine_thermal_model_grid_{ins}/'

    # Derive instrument-specific parameters
    if ins == 'pionier':
        #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
        wavelength = 1600e-9  # Observation wavelength (m)
        #binning = None
        #UD_diam = 3.3  # Uniform disk diameter in mas (prior)
        #insname = None
    elif ins == 'gravity':
        #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
        wavelength = 2200e-9
        #binning = 400
        #UD_diam = 3.5  # Uniform disk diameter in mas (prior)
        #insname = 'GRAVITY_SC_P1'
    else:
        raise ValueError(f"Unknown instrument: {ins}")

    # 1. Calculate local effective temperature
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff, delta_T, l, m, nu, psi_T)

    # 2. Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_o, phi_o)

    # 3. Project onto observer plane
    projected_intensity = project_to_observer_plane(
        theta_rot, phi_rot,
        blackbody_intensity(T_eff_local, wavelength),
        grid_size=grid_size
    )

    # 4. Pad intensity image
    # if pad_factor > 1.0:
    #     pad_size = int(projected_intensity.shape[0] * (pad_factor - 1) // 2)
    #     projected_intensity = np.pad(
    #         projected_intensity, pad_size,
    #         mode='constant', constant_values=0
    #     )

    # 5. Save FITS file (optional)
    #if save_fits:
    # don't save the file and just pass the fits directly to avoid re-reading it in! 
    name = f'theta-{theta_o:.2f}_phi-{phi_o:.2f}_T-{delta_T:.2f}.fits'
    h = intensity_2_fits(
        projected_intensity,
        dx=dx,
        dy=dy,
        name=name,
        data_path=save_fits_path,
        header_dict=header_dict,
        write_file=False
    )

    # 6. Simulate observed data
    synthetic_file = f"{save_fits_path}{name}"

    # oif = simulate_obs_from_image_reco_FAST(
    #     oi, synthetic_file, img_pixscl = UD_diam / grid_size
    # )
    # dont read the file in an just pass the fits (h) directly!!!! 
    oif = simulate_obs_from_image_reco_FAST(
        oi, h, img_pixscl = UD_diam / grid_size
    )
    # 7. Compare synthetic and observed data
    kwargs = {'v2_err_min':0.001,'cp_err_min':0.1} 
    comp_dict_v2 = plot_util.compare_models(oi, oif, measure='V2' ,**kwargs)
    comp_dict_cp = plot_util.compare_models(oi, oif, measure='CP',**kwargs)

    # 8. Calculate reduced chi-square
    v2_chi2 = np.mean(flatten(get_all_values(comp_dict_v2['chi2'])))
    cp_chi2 = np.mean(flatten(get_all_values(comp_dict_cp['chi2'])))
    chi2_reduced = (v2_chi2 + cp_chi2) / 2

    return chi2_reduced

# Log-probability function
def log_probability(params, **kwargs):
    """
    Log-probability function for MCMC sampling.
    
    Parameters:
        params (list): [theta_o, phi_o, delta_T]
        kwargs: Additional fixed parameters for `thermal_dipole_fit`.
    
    Returns:
        float: Log-probability.
    """
    fit_ud = kwargs.get('fit_ud', True)

    if fit_ud:
        theta_o, phi_o, delta_T, UD_diam = params
        
    else:
        theta_o, phi_o, delta_T = params
    
    # Enforce parameter bounds (prior) #(-np.pi <= theta_o <= np.pi) 
    if not (0 <= theta_o <= 2*np.pi) or not (0 <= phi_o <= 2 * np.pi) or not (0 <= delta_T <= 500):
        return -np.inf  # Log-probability is -inf for invalid parameters
    

    # Compute reduced chi-square
    if fit_ud:
        chi_squared_reduced = thermal_dipole_fit([theta_o, phi_o, delta_T,UD_diam], **kwargs)
    else:
        chi_squared_reduced = thermal_dipole_fit([theta_o, phi_o, delta_T], **kwargs)
    # Convert to log-probability
    log_prob = -0.5 * chi_squared_reduced
    return log_prob



# Initialize the MCMC sampler
def run_mcmc(log_prob_func, nwalkers, ndim, nsteps, **kwargs):
    """
    Run the MCMC algorithm to fit parameters.
    
    Parameters:
        log_prob_func (function): The log-probability function.
        nwalkers (int): Number of walkers.
        ndim (int): Number of parameters.
        nsteps (int): Number of steps for the MCMC run.
        kwargs: Additional arguments for the log-probability function.
    
    Returns:
        sampler: The MCMC sampler object.
    """

    fit_ud = kwargs.get('fit_ud', True)

    if fit_ud:
        # Initial guesses for walkers
        initial_pos = [
            [
                np.random.uniform(0, 2*np.pi/2),#np.random.uniform(-np.pi/2, np.pi/2),  # Random θ_o between 0 and π
                np.random.uniform(0, 2 * np.pi),  # Random φ_o between 0 and 2π
                np.random.uniform(0, 500), # Random ΔT between 0 and 500
                3.5 + np.random.randn()/2 # normal for UD diameter
            ]
            for _ in range(nwalkers)
        ]
    else:
        # Initial guesses for walkers
        initial_pos = [
            [
                np.random.uniform(0, np.pi),  # Random θ_o between 0 and π
                np.random.uniform(0, 2 * np.pi),  # Random φ_o between 0 and 2π
                np.random.uniform(0, 500)  # Random ΔT between 0 and 500
            ]
            for _ in range(nwalkers)
        ]
    
    # Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, kwargs=kwargs)
    
    # Run MCMC
    print(f"Running MCMC with {nwalkers} walkers, {ndim} dimensions, and {nsteps} steps...")
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    
    return sampler


parser = argparse.ArgumentParser(description="Run MCMC thermal dipole fitting.")
parser.add_argument("--ins", type=str, required=True, choices=["pionier", "gravity"],
                    help="Instrument name (e.g., 'pionier' or 'gravity').")
parser.add_argument("--fit_ud", action="store_true",
                    help="Flag to include uniform disk diameter fitting.")


args = parser.parse_args()

ins = args.ins #'pionier' #'gravity'
fit_ud = args.fit_ud 

print( f"\n\n{ins} fitting UD: {fit_ud}")
# paths to data depending on what computer im running things on 
path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU'  # computer location

# To include fitting uniform disk diameter 
fit_ud = True # False 

if ins == 'pionier':
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
    wavelength = 1600e-9  # Observation wavelength (m)
    binning = None
    UD_diam = 3.3  # Uniform disk diameter in mas (prior)
    insname = None
elif ins == 'gravity':
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
    wavelength = 2200e-9
    binning = 400
    UD_diam = 3.55  # Uniform disk diameter in mas (prior)
    insname = 'GRAVITY_SC_P1'
else:
    raise ValueError(f"Unknown instrument: {ins}")

# format data in pmoired style
oi = pmoired.OI(obs_files, binning = binning, insname = insname)

# sort baseline labels etc to avoid errors

# sort data by MJD - issues with Gravity when x['MJD'] is a list... TO DO 

# DO ALL THIS SORTING ON oi BEFORE INPUTING 
oi.data = sorted(oi.data, key=lambda x: x['MJD'][0]) # have to take first one because sometimes a list 
# ## SILLY BUG IN PMOIRED WHERE BASELINE/TRIANGLE KEYS ARE INCONSISTENT (e.g. 'D0C1' then 'C1D0')
# ## we fix this here by ordering all relevant keys
change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
change_triangle_key_list = ['triangles','OI_T3']
enforce_ordered_baselines_keys(oi.data, change_baseline_key_list)
enforce_ordered_triangle_keys(oi.data, change_triangle_key_list)



default_kwargs = {
    # Observational data
    'oi': oi,  # Preformatted observational data (required)
    
    # if fixed an not fitted (defines pixel scale)
    'fit_ud':fit_ud,
    'UD_diam': UD_diam,
    # Stellar grid coordinates
    'theta': np.linspace(0, np.pi, 50), #200 # Full colatitude
    'phi': np.linspace(0, 2 * np.pi, 50), #200 # Full longitude
    
    # Stellar parameters
    'T_eff': 3000,  # Average effective temperature (K)
    't': 0,  # Time (s)
    'nu': 1 / (757 * 24 * 60 * 60),  # Frequency (Hz), 1 cycle per 757 days
    'psi_T': 0.7 * np.pi * 2,  # Phase offset (radians)
    'l': 1,  # Spherical harmonic degree
    'm': 1,  # Spherical harmonic order

    # Observation parameters
    'grid_size': 500,  # Grid size for the projection (pixels per stellar diameter)
    'pad_factor': 2,  # Padding factor for projected intensity

    # Pixel scale
    'dx': 1,  # Pixel scale in mas
    'dy': 1,  # Pixel scale in mas

    # FITS output
    'save_fits': True,  # Save the FITS file
    'header_dict': {},  # Metadata for the FITS file

    # Synthetic observation parameters
    'kwargs': {},  # Additional arguments for plotting utilities
}

# testing 
#theta_o, phi_o, delta_T = 0, 0 , 200 
theta_o, phi_o, delta_T, UD_diam = np.deg2rad(170),  np.deg2rad(168),  212 , 3.28
thermal_dipole_fit(fit_param=[theta_o, phi_o, delta_T, UD_diam ],  **default_kwargs)

# MCMC parameters
nwalkers = 32  # Number of walkers
if fit_ud:
    ndim = 4 # Number of parameters: [theta_o, phi_o, delta_T]
else:
    ndim = 3 
nsteps = 500  # Number of steps for the MCMC run

if fit_ud:
    labels = [r"$\theta_o$", r"$\phi_o$", r"$\Delta T$", r"$\theta_{UD}$"]
else:
    labels = [r"$\theta_o$", r"$\phi_o$", r"$\Delta T$"]

# Run the MCMC algorithm
sampler = run_mcmc(log_probability, nwalkers, ndim, nsteps, **default_kwargs)

# extract raw samples and save for future use
raw_samples = sampler.get_chain()
sample_dict = {k:v for k,v in zip(labels,raw_samples.T.tolist())}
import json
with open(f'MCMC_SMALLER_GRID_{ins}_thermal_dipole_nwalkers_{nwalkers}_fitud-{fit_ud}_nsteps-{nsteps}.json', 'w') as f:
    json.dump(sample_dict, f)

# Extract and filter the samples
samples = sampler.get_chain(discard=100, thin=10, flat=True)  # Discard burn-in, thin samples

# # Plot the results
# fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(sampler.get_chain()[:, :, i], alpha=0.5)
#     ax.set_ylabel(labels[i])
#     ax.set_xlabel("Step number")
# plt.tight_layout()
# plt.show()
# plt.savefig(f'delme_{ins}_sample_series.png')

# # Corner plot of the posterior distributions
# try:
#     import corner
#     fig = corner.corner(samples, labels=labels)
#     plt.show()
# except ImportError:
#     print("Install 'corner' for posterior plots: pip install corner")
# plt.savefig(f'delme_{ins}_mcmc_corner.png')

# # Best-fit parameters
# best_fit_params = np.mean(samples, axis=0)
# print(f"Best-fit parameters: θ_o={best_fit_params[0]:.3f}, φ_o={best_fit_params[1]:.3f}, ΔT={best_fit_params[2]:.3f}")





# file_path = "/Users/bencb/Documents/long_secondary_periods/MCMC_pionier_thermal_dipole_nwalkers_32_fitud-True_nsteps-500.json"
# with open(file_path, 'r') as json_file:
#     data = json.load(json_file)

# #In [3]: data.keys()
# #Out[3]: dict_keys(['$\\theta_o$', '$\\phi_o$', '$\\Delta T$', '$\\theta_{UD}$'])

data = sample_dict

# Assuming the keys represent parameter names and values are lists of samples
parameters = list(data.keys())

samples = np.array([np.array( data[key] ).reshape(-1) for key in parameters]).T  # Transpose for corner plot

samples[:,0] = 180 / np.pi * np.mod(samples[:,0], 2* np.pi )
samples[:,1] *= 180 / np.pi 

# Create the corner plot
fig = corner.corner(
    samples,
    labels=parameters,
    quantiles=[0.15, 0.5, 0.85],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 14},
    hist_kwargs={"color": "blue"},
    plot_contours=True,
    contour_kwargs={"colors": "red"},
    quantile_kwargs={"linestyles": "--", "colors": "black"}
)
# Customize axes labels and tick labels
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)  # Set tick label font size
    if ax.xaxis.get_label():
        ax.xaxis.get_label().set_fontsize(14)  # Set x-axis label font size
    if ax.yaxis.get_label():
        ax.yaxis.get_label().set_fontsize(14)  # Set y-axis label font size


plt.savefig( "CORNER"+file_path.split('/')[-1].split('.json')[0] , dpi = 300)
plt.show()

"""
## #ANALYSIS

# RT PAV light curve 
import json
import numpy as np
import corner
import matplotlib.pyplot as plt 

# Constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 3.0e8           # Speed of light (m/s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)


#file = "MCMC_SMALLER_GRID_pionier_thermal_dipole_nwalkers_32_fitud-True_nsteps-500.json" #
file = "MCMC_pionier_thermal_dipole_nwalkers_32_fitud-True_nsteps-500.json"
file_path = f"/Users/bencb/Documents/long_secondary_periods/{file}"
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

#In [3]: data.keys()
#Out[3]: dict_keys(['$\\theta_o$', '$\\phi_o$', '$\\Delta T$', '$\\theta_{UD}$'])


# Assuming the keys represent parameter names and values are lists of samples
parameters = list(data.keys())

samples = np.array([np.array( data[key] ).reshape(-1) for key in parameters]).T  # Transpose for corner plot

samples[:,0] = 180 / np.pi * np.mod(samples[:,0], 2* np.pi )
# flip degeneracy around 0  and 180 degrees 
#samples[:,1] = 180 / np.pi * np.mod( samples[:,1], np.pi)  #* np.mod(samples[:,0], np.pi )
#samples[:,1][ samples[:,1] <= 90] = samples[:,1][ samples[:,1] <= 90] + 180
samples[:,1] *= 180 / np.pi

labels = ["incl.", "proj. angle", "$\\Delta T$", "$\\theta_{UD}$"]

# Create the corner plot
fig = corner.corner(
    samples,
    labels=labels, #parameters,
    quantiles=[0.15, 0.5, 0.85],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    hist_kwargs={"color": "blue"},
    plot_contours=True,
    contour_kwargs={"colors": "red"},
    quantile_kwargs={"linestyles": "--", "colors": "black"}
)
# Customize axes labels and tick labels
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)  # Set tick label font size
    if ax.xaxis.get_label():
        ax.xaxis.get_label().set_fontsize(14)  # Set x-axis label font size
    if ax.yaxis.get_label():
        ax.yaxis.get_label().set_fontsize(14)  # Set y-axis label font size


plt.savefig( "CORNER"+file_path.split('/')[-1].split('.json')[0] , dpi = 300)
plt.show()



# RT PAV light curve 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.time import Time
from scipy.interpolate import interp1d 

data_path = '/Users/bencb/Documents/long_secondary_periods/lightcurves/LSP_light_curves_data/'
os.chdir(data_path )

#files = os.listdir()

#stars = [xx.split('_aavso')[0] for xx in files]


#plsp = {'RZ_Ari' : 479, 'S_lep' : 880, 'GO_vel' : 501, 'RT_Cnc' : 542}


#%% plot of light curve extended to p109 

light_curve = pd.read_csv('RT_pav_asassn.csv')

light_curve.columns = ['JD', 'camera', 'filter', 'Magnitude', 'mag err', 'flux (mJy)', 'flux err']


fff = 15 
t = np.sort(light_curve['JD']) 
mag = [x for _, x in sorted(zip(light_curve['JD'], light_curve['Magnitude']))]

plt.figure(figsize=(8,5))

RTpav_interp_fn = interp1d(t,mag)
tn = np.linspace(np.min(t),np.max(t),1000)
magn = RTpav_interp_fn(tn)

tn2 = np.linspace(np.min(t),(1 + 7e-4)* np.max(t), 1000)
plt.plot(t,mag,'x', label = 'ASAS data')
plt.plot(tn2, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoidal fit')

plt.axvspan(Time('2022-03-01').jd, Time('2022-09-30').jd, alpha=0.3, color='red',label='P109')

plt.legend(loc='lower left',fontsize=13)

plt.axvline(Time('2022-03-01').jd)
plt.axvline(Time('2022-09-30').jd)

plt.gca().invert_yaxis()
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('time (JD)',fontsize=fff)
plt.ylabel('mag (V)',fontsize=fff)





#################
### looking at secondary eclipse with thermal mode - seems to come on at inclination angles near 180 degrees 

#best_delta_T = 200
#best_theta_o, best_phi_o = np.deg2rad( 160), np.deg2rad( 160 )

# get coordinates right!!! 

"""
theta_0 (i.e. inclination i) is 0 when dipole is orthogonal to line of sight and 90 when aligned with line of sight
phi_0 is rotation around line of sight 

"""

best_theta_o, best_phi_o = 0, 0
l=1
m=1
T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

# Rotate to observer frame
theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

# Project onto observer plane
projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

origin = 'lower' 
extent = [1, -1, -1, 1]

plt.figure();
plt.ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
plt.xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
plt.imshow( projected_intensity  , extent=extent, origin=origin)
plt.show()


from matplotlib.widgets import Slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
im = ax.imshow(projected_intensity, origin='lower', extent=[-1, 1, -1, 1])
plt.colorbar(im, ax=ax, label="Projected Intensity")

# Slider setup
ax_theta = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_theta = Slider(ax_theta, 'Theta_o', 0, np.pi, valinit=best_theta_o)
slider_phi = Slider(ax_phi, 'Phi_o', 0, 2 * np.pi, valinit=best_phi_o)

# Update function
def update(val):
    theta_o = slider_theta.val
    phi_o = slider_phi.val
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff, best_delta_T, l, m, nu, psi_T)
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_o, phi_o)
    projected_intensity = project_to_observer_plane(
        theta_rot.flatten(),
        phi_rot.flatten(),
        blackbody_intensity(T_eff_local.flatten(), wavelength_aaso),
        grid_size
    )
    im.set_data(projected_intensity)
    fig.canvas.draw_idle()

# Connect sliders to the update function
slider_theta.on_changed(update)
slider_phi.on_changed(update)

plt.show()


#### Looking at light curves and "secondary eclipses"
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

best_phi_o, best_delta_T, best_ud = 3.3161255787892263, 312.46496105857034, 3.3016914552936942

tgrid = tn2 * 24 * 60 * 60 #np.linspace( -2/nu, 2/nu , 100)
wavelength_aaso = 551e-9
flux_dict = {}
theta0_grid = np.deg2rad( np.array([0, 5, 10, 20, 50,  80, 90, 100]) ) # np.linspace( np.deg2rad(0 ),np.deg2rad( 110 ) , 8 )
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



######################################################
#### COMPARING VLTI FIT WITH LIGHT CURVE AND MAKE MOVIE

from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata 
from matplotlib.animation import FuncAnimation
# Get image 

cnt = 0 
# Main Parameters
T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0 #0.7 * np.pi * 2 # Phase offset (rad), for VLTI fit we use 0.7 phase at t=0

pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
# Stellar surface grid
theta = np.linspace(0, np.pi, 50)  # Full colatitude
phi = np.linspace(0, 2 * np.pi, 50)  # Full longitude
theta, phi = np.meshgrid(theta, phi)


samples = np.array([np.array( data[key] ).reshape(-1) for key in parameters]).T  # Transpose for corner plot
best_theta_o, best_phi_o, best_delta_T, best_ud = np.median( samples, axis = 0 )


######################

#best_delta_T = 200
#best_theta_o, best_phi_o = np.deg2rad( 160), np.deg2rad( 160 )

psi_T = 0.5 * np.pi * 2 
tgrid = tn2 * 24 * 60 * 60 #np.linspace( -2/nu, 2/nu , 100)
wavelength_aaso = 551e-9
flux=[]
projected_intensities = []
for t in tgrid:
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T)

    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi,  best_theta_o, best_phi_o)

    # Project onto observer plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

    # Append results
    projected_intensities.append(projected_intensity)
    flux.append(np.sum(projected_intensity))
#plt.figure(); plt.plot( flux ); plt.show()



# Compute magnitude (logarithmic flux)
magnitude = -2.5 * np.log10(flux / np.min(flux))
mag_offset = 9.0 # 8.45
# Create the figure and subplots



plt.figure() 
plt.plot(np.sort(light_curve['JD']) , mag,'x', label = 'ASAS data')
#plt.plot(tn2, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoidal fit')
plt.plot(tn2  , mag_offset  + magnitude , color='k', label="thermal dipole model fit to Pionier P109 data")
#plt.plot(tn2  , mag_offset  +  np.array(flux)/np.max(flux) , color='k', label="thermal dipole model fit to Pionier P109 data")

plt.axvspan(Time('2022-03-01').jd, Time('2022-09-30').jd, alpha=0.3, color='red',label='P109')

plt.legend(loc='lower left',fontsize=13)

plt.axvline(Time('2022-03-01').jd)
plt.axvline(Time('2022-09-30').jd)

plt.gca().invert_yaxis()
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('time (JD)',fontsize=fff)
plt.ylabel('mag (V)',fontsize=fff)
plt.tight_layout()
plt.savefig('light_curve_with_pionier_bestfit_thermal_dipole.png')
plt.show() 



# Subplot 2: Heatmap of projected intensity
# Create the figure and subplots
t = 0
T_eff_local = thermal_oscillation(theta, phi, t, T_eff,  best_delta_T, l, m, nu, psi_T=1.2*np.pi*2) # 0.7+0.5 (for 180 degree phase shift required to sync light curve)

# Rotate to observer frame
theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

# Project onto observer plane
projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )




fig, ax = plt.subplots(1, 1, figsize=(6, 6))

origin = 'lower'
extent = best_ud / 2 * np.array([1, -1, -1, 1])

# Plot the heatmap
heatmap = ax.imshow(
    projected_intensity/np.max(projected_intensity), origin=origin, cmap="inferno",
    extent=extent, vmin=0, vmax=1
)

# Add axis labels and title
#ax.set_title("Projected Intensity", fontsize=15)
ax.set_ylabel('$\Delta$ DEC -> N [mas]', fontsize=15)
ax.set_xlabel('$\Delta$ RA <- E [mas]', fontsize=15)
ax.tick_params(labelsize=15)

# Adjust colorbar with padding
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
cbar = plt.colorbar(heatmap, cax=cax)
cbar.set_label("Normalized Intensity", fontsize=15)
cbar.ax.tick_params(labelsize=15)  # Adjust colorbar tick label size

# Show the plot
plt.tight_layout()
plt.savefig( "best_fit_dipole_intensity.png")
plt.show()

#########################
# Movie 

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Magnitude vs time
ax1 = axes[0]
# line, = ax1.plot([], [], label="Magnitude (V)")
# ax1.set_xlim(1/(24*60*60) * tgrid[0], 1/(24*60*60) * tgrid[-1])
# ax1.set_ylim(np.min(magnitude) - 0.1, np.max(magnitude) + 0.1)
# ax1.set_xlabel("Time")
# ax1.set_ylabel(r"$\delta$ Magnitude (V)")
# ax1.legend()



ax1.plot(np.sort(light_curve['JD']) , mag,'x', label = 'ASAS data')
#plt.plot(tn2, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoidal fit')
line, =ax1.plot([],[], color='k', label="thermal dipole model fit to Pionier P109 data")

ax1.axvspan(Time('2022-03-01').jd, Time('2022-09-30').jd, alpha=0.3, color='red',label='P109')

ax1.legend(loc='lower left',fontsize=10)

ax1.axvline(Time('2022-03-01').jd)
ax1.axvline(Time('2022-09-30').jd)

ax1.invert_yaxis()
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('time (JD)',fontsize=fff)
ax1.set_ylabel('mag (V)',fontsize=fff)


# Subplot 2: Heatmap of projected intensity
ax2 = axes[1]

origin = 'lower'
extent = best_ud / 2 * np.array([1, -1, -1, 1])

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
    line.set_data(tn2[:frame+1] , mag_offset +  magnitude[:frame+1])
    #line.set_data(1/(24*60*60) * tgrid[:frame+1], magnitude[:frame+1])
    
    # Update heatmap
    heatmap.set_data(projected_intensities[frame]/max_int)
    return line, heatmap

# Create animation
ani = FuncAnimation(fig, update, frames=len(tgrid), interval=100, blit=True)

# Save or display the animation
ani.save("light_curve_thermal_dipole.mp4", fps=50, dpi=150)
plt.show()


"""
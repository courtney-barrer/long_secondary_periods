import numpy as np
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.interpolate import griddata 
from astropy.io import fits
import os 
import glob 
import pmoired
import json
import sys
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


def intensity_2_fits(projected_intensity, dx, dy, name, data_path, header_dict={}):

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

    h.writeto( data_path + name, overwrite=True )





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
             
                      
def simulate_obs_from_image_reco( obs_files, image_file , img_pixscl = None, binning=None, insname = None):
    
    # change wvl_band_dict[feature] to wvl_lims
    d_model = fits.open( image_file )
    
    img = d_model[0].data

    #assert (abs( float( d_model[0].header['CUNIT2'] ) ) - abs( float(  d_model[0].header['CUNIT1']) ) ) /  float( d_model[0].header['CDELT1'] ) < 0.001
    # we assert the image has to be square..
    #assert abs(float( d_model[0].header['CDELT2'])) == abs(float(d_model[0].header['CDELT1']))
    
    img_units = d_model[0].header['CUNIT1']
    
    if img_pixscl is None:
        img_pixscl = d_model[0].header['CDELT1']     

    if img_units == 'deg':
        img_pixscl *= 3600*1e3 # convert to mas
    if img_units == 'mas':
        pass 
    else:  
        raise TypeError('Units not recognized')

    oi = pmoired.OI(obs_files, binning = binning, insname = insname)

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
    
    oi.data = sorted(oi.data, key=lambda x: x['MJD'][0]) # have to take first one because sometimes a list 
    
    oif.data = sorted(oif.data, key=lambda x: x['MJD'][0])
    
    ## SILLY BUG IN PMOIRED WHERE BASELINE/TRIANGLE KEYS ARE INCONSISTENT (e.g. 'D0C1' then 'C1D0')
    ## we fix this here by ordering all relevant keys
    change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
    change_triangle_key_list = ['triangles','OI_T3']
    enforce_ordered_baselines_keys(oi.data, change_baseline_key_list)
    enforce_ordered_baselines_keys(oif.data, change_baseline_key_list)
    enforce_ordered_triangle_keys(oi.data, change_triangle_key_list)
    enforce_ordered_triangle_keys(oif.data, change_triangle_key_list)

    return( oi, oif )


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


path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))

comp_loc = 'ANU' # computer location

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = path_dict[comp_loc]['data'] 

ins = 'gravity' # 'pionier'

if ins == 'pionier':
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
    wavelength = 1600e-9  # Observation wavelength (m)
    binning = None
    UD_diam = 3.3 #mas
    insname=None

elif ins == 'gravity':
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
    wavelength = 2200e-9 
    binning = 400
    UD_diam = 3.5 #mas
    insname='GRAVITY_SC_P1'

# # Main Parameters
# T_eff = 3000          # Average effective temperature (K)
# delta_T_eff = 100     # Amplitude of temperature variation (K)
# l, m = 1, 1           # Spherical harmonic degree and order
# nu = 1e-6             # Frequency (Hz)
# psi_T = 0             # Phase offset (rad)
# #wavelength = 1.6e-6   # Observation wavelength (500 nm)

# # Stellar surface grid
# theta = np.linspace(0, np.pi, 200)  # Full colatitude
# phi = np.linspace(0, 2 * np.pi, 200)  # Full longitude
# theta, phi = np.meshgrid(theta, phi)

# # Observer position
# theta_obs = 2 * np.pi / 4  # 45 degrees
# phi_obs = np.pi / 3    # 60 degrees

# # Calculate local effective temperature
# T_eff_local = thermal_oscillation(theta, phi, 0, T_eff, delta_T_eff, l, m, nu, psi_T)

# # Rotate to observer frame
# theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_obs, phi_obs)

# # Project onto observer plane
# projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength),grid_size=500)

# # Visualization
# plt.figure(figsize=(8, 8))
# extent = [-1, 1, -1, 1]
# plt.imshow(projected_intensity, extent=extent, origin='lower', cmap='inferno')
# plt.colorbar(label='Intensity (W/m^2/sr/m)')
# plt.title('Projected Stellar Intensity from Observer Position')
# plt.xlabel('Observer Plane X')
# plt.ylabel('Observer Plane Y')
# plt.gca().set_aspect('equal')
# plt.show()

# course grid 
theta_grid = np.linspace(0, np.pi, 7) 
phi_grid = np.linspace(0, np.pi, 7)  
delta_T_grid = np.linspace(0,400, 10)  

# fine grid 
# best from course : 0 2.356194490192345 200.0
# theta_grid = np.linspace(-np.pi/10, np.pi/10, 5) 
# phi_grid = np.linspace(2, 2.6, 5)  
# delta_T_grid = np.linspace(150, 400, 10)  

N = len( theta_grid ) * len( phi_grid ) * len( delta_T_grid )

save_fits_path = f'/home/rtc/Downloads/fine_thermal_model_grid_{ins}/'
theta = 0 
cnt = 0 
# Main Parameters
T_eff = 3000          # Average effective temperature (K)
l, m = 1, 1           # Spherical harmonic degree and order
nu = 1 / (757*24*60*60) #1e-6             # Frequency (Hz)
psi_T = 0.7 * np.pi * 2 # Phase offset (rad)

pad_factor = 2
grid_size = 500
dx =  1  # mas <---------
dy = 1  # mas
for theta_o in theta_grid:
    for phi_o in phi_grid:
        for delta_T in delta_T_grid:# Amplitude of temperature variation (K)

            print( f'{100 * cnt/N}%')

            name = f'theta-{theta_o}_phi-{phi_o}_T-{delta_T}.fits'

            # to write to fits headers 
            header_dict = {"wvl":wavelength, "psi_T":psi_T, "nu":nu, "l":l, "m":m, "delta_T_eff":delta_T, "T_eff":T_eff, 'phi_obs':phi_o, 'theta_obs':theta_o}

            # Stellar surface grid
            theta = np.linspace(0, np.pi, 200)  # Full colatitude
            phi = np.linspace(0, 2 * np.pi, 200)  # Full longitude
            theta, phi = np.meshgrid(theta, phi)

            # Observer position
            #theta_obs = theta_o # 45 degrees
            #phi_obs = phi_o   # 60 degrees

            # Calculate local effective temperature
            T_eff_local = thermal_oscillation(theta, phi, 0, T_eff, delta_T, l, m, nu, psi_T)

            # Rotate to observer frame
            theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, theta_o, phi_o)

            # Project onto observer plane
            projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength), grid_size =grid_size )

            projected_intensity = np.pad( projected_intensity, projected_intensity.shape[0] * pad_factor, mode='constant', constant_values=0)
            intensity_2_fits(projected_intensity, dx=dx, dy=dy, name=name, data_path = save_fits_path, header_dict=header_dict)

            cnt += 1 






# model_files = glob.glob( data_path + '*.fits')

# fig, ax = plt.subplots( 10,10 )

# for f,axx in zip( model_files, ax.reshape(-1)):
#     with fits.open( f ) as a:
#         axx.imshow( a[0].data )
#         axx.set_title( f"{round( float( a[0].header['PHI_OBS']) ,2 )} - {round( float( a[0].header['THETA_OBS']) ,2 )}" )

# plt.axis('off')


model_files = glob.glob( save_fits_path + '*.fits')

#f= model_files[0]


img_pixscl = UD_diam / grid_size

kwargs = {
    'wvl_lims': [-np.inf, np.inf],
    'cp_err_min': 0.2,
    'cp_err_max': 30,
    'v2_err_min': 0.01,
    'v2_err_max': 0.5,
    'cp_min': -np.inf,
    'cp_max': np.inf,
    'v2_min': 0,
    'v2_max': np.inf
}

phi_o = []
theta_o = []
delta_T = []
v2_chi2 = []
cp_chi2 = []

for i,f in enumerate( model_files ):
    print( f"\n\n\n{100* i / len(model_files)}%")
    with fits.open(f) as d:
        phi_o.append(float(d[0].header['PHI_OBS']))
        theta_o.append(float(d[0].header['HIERARCH theta_obs']))
        delta_T.append(float(d[0].header['HIERARCH delta_T_eff']))

    oi, oif = simulate_obs_from_image_reco( obs_files, f, img_pixscl=img_pixscl, binning=binning, insname = insname)

    #plot_util.plotV2CP( oi ,wvl_band_dict={"i":[1.5e-6, 1.7e-6]}, feature="i", CP_ylim = 180,  logV2 = True, savefig_folder='',savefig_name='delme')
    comp_dict_v2=plot_util.compare_models(oi, oif , measure='V2', kwargs=kwargs)
    comp_dict_cp=plot_util.compare_models(oi, oif , measure='CP', kwargs=kwargs)

    v2_chi2.append( np.mean(flatten( get_all_values( comp_dict_v2['chi2'] ) ) ) )
    cp_chi2.append( np.mean(flatten( get_all_values( comp_dict_cp['chi2'] ) ) ))


best_i_v2=np.argmin( v2_chi2 )
best_i_cp=np.argmin( cp_chi2 )

best_i=np.argmin(np.mean( [v2_chi2,cp_chi2 ], axis=0))

print( f'chi2 = {(v2_chi2[best_i]+cp_chi2[best_i])/2}')
print( f'chi2 cp = {(v2_chi2[best_i_cp]+cp_chi2[best_i_cp])/2}')
print( f'chi2 v2 = {(v2_chi2[best_i_v2]+cp_chi2[best_i_v2])/2}')


chi2_T = (np.array( v2_chi2 ) + np.array( cp_chi2) )/2


#################
# get best parameters ! 

import pandas as pd
df = pd.DataFrame( {"f":model_files,"chi2":chi2_T, "phi_o":phi_o, "theta_o":theta_o, "deltaT":delta_T})
df.to_csv(save_fits_path+f"{ins}_results.csv")

# print(phi_o[best_i], theta_o[best_i], delta_T[best_i])

# print(phi_o[best_i_v2], theta_o[best_i_v2], delta_T[best_i_v2])

# print(phi_o[best_i_cp], theta_o[best_i_cp], delta_T[best_i_cp])


import corner 
corner.corner( df[ ['phi_o', 'theta_o', 'deltaT','chi2']],
               labels=['phi_o', 'theta_o', 'deltaT','chi2'],
               show_titles= True
               )
plt.savefig('delme.png')


bi = best_i #_cp
best_phi_o=phi_o[bi]
best_theta_o=theta_o[bi]
best_delta_T=delta_T[bi]

f = model_files[bi]


print( f'chi2 = {(v2_chi2[bi]+cp_chi2[bi])/2}')
print(f, best_phi_o, best_theta_o, best_delta_T)





#################
# plot grid search results 
plt.figure()
plt.semilogy( delta_T, v2_chi2, '.', label='v2' )
plt.semilogy( delta_T, cp_chi2, '.', label='cp' )
plt.xlabel(r"$\delta T$")
plt.ylabel(r"$\chi_\nu^2$")
plt.legend()
plt.savefig('delme.png')

plt.figure()
plt.semilogy( theta_o, v2_chi2, '.', label='v2' )
plt.semilogy( theta_o, cp_chi2, '.', label='cp' )
plt.ylabel(r"$\chi_\nu^2$")
plt.xlabel(r"P.A")
plt.legend()
plt.savefig('delme.png')

plt.figure()
plt.semilogy( phi_o, v2_chi2, '.', label='v2' )
plt.semilogy( phi_o, cp_chi2, '.', label='cp' )
plt.ylabel(r"$\chi_\nu^2$")
plt.xlabel("i")
plt.legend()
plt.savefig('delme.png')


print()
d =fits.open( f )
plt.figure()
with fits.open( f ) as d:
    plt.imshow( d[0].data/np.max(d[0].data) )
plt.colorbar(label='normalized intensity')
plt.title(r"best fit: $\delta T$"+f"={round(delta_T[bi])}K, i={round(np.rad2deg(phi_o[bi]))}deg,P.A={round(np.rad2deg(theta_o[bi]))}deg")
plt.savefig('delme.png')

#################
# plot V2 and CP 
plt.close('all')

oi, oif = simulate_obs_from_image_reco( obs_files, f, img_pixscl=img_pixscl , binning = binning)

kwargs =  {
    'wvl_lims':[-np.inf, np.inf],\
    'model_col': 'orange',\
    'obs_col':'grey',\
    'fsize':18,\
    'logV2':True,\
    'ylim':[0,1],
    'CPylim':180
    } # 'CP_ylim':180,

plot_util.compare_CP_obs_vs_image_reco( oi, oif , return_data = False, savefig='delme.png' ,kwargs=kwargs)
plot_util.compare_V2_obs_vs_image_reco( oi, oif , return_data = False, savefig='delme.png'  ,kwargs=kwargs)


############ exploring different UD diameters 
#plot_util.

UD_diam_grid = np.linspace( 3, 5, 30)
fbest = model_files[bi]

v2_chi2=[]
cp_chi2=[]
for UD_diam in UD_diam_grid:
    print("/n/n/n/n", UD_diam)
    img_pixscl = UD_diam / grid_size
    oi, oif = simulate_obs_from_image_reco( obs_files, fbest, img_pixscl=img_pixscl, binning=binning, insname=insname)

    #plot_util.plotV2CP( oi ,wvl_band_dict={"i":[1.5e-6, 1.7e-6]}, feature="i", CP_ylim = 180,  logV2 = True, savefig_folder='',savefig_name='delme')
    comp_dict_v2=plot_util.compare_models(oi,oif , measure='V2')
    comp_dict_cp=plot_util.compare_models(oi,oif , measure='CP')

    v2_chi2.append( np.mean(flatten( get_all_values( comp_dict_v2['chi2'] ) ) ) )
    cp_chi2.append( np.mean(flatten( get_all_values( comp_dict_cp['chi2'] ) ) ))


#plot_util.compare_V2_obs_vs_image_reco( oi, oif , return_data = False,  savefig='delme.png')#, **kwargs )

#plot_util.compare_CP_obs_vs_image_reco( oi, oif , return_data = False,  savefig='delme.png')#, **kwargs )

plt.figure()
plt.semilogy( UD_diam_grid, v2_chi2, label='v2' )
plt.semilogy( UD_diam_grid, cp_chi2, label='cp' )
plt.legend()
plt.xlabel(r"$\theta$ [mas]")
plt.ylabel(r"$\chi_\nu^2$")
plt.savefig('delme.png')

#print( np.mean( v2_chi2 ) )

#print( np.mean( cp_chi2 ) )


#########################
## can we reproduce light curve amplitude ?

tgrid = np.linspace( -2/nu, 2/nu , 100)
wavelength_aaso = 551e-9
flux=[]
for t in tgrid:
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff, 300, l, m, nu, psi_T)

    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)

    # Project onto observer plane
    projected_intensity = project_to_observer_plane(theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size =grid_size )

    flux.append( np.sum (projected_intensity))

plt.figure()
offset  = 8.5
plt.plot( tgrid / 24 / 3600, offset + 2.5 *np.log10(flux/np.min(flux) ))
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel(r"time [days]")
plt.ylabel(r"mag")
plt.savefig('delme.png')






##### movie 

# Prepare flux storage
flux = []

# Precompute data for animation
projected_intensities = []
for t in tgrid:
    # Compute local temperature
    T_eff_local = thermal_oscillation(theta, phi, t, T_eff, best_delta_T, l, m, nu, psi_T)
    
    # Rotate to observer frame
    theta_rot, phi_rot = rotate_to_observer_frame(theta, phi, best_theta_o, best_phi_o)
    
    # Project onto observer plane
    projected_intensity = project_to_observer_plane(
        theta_rot, phi_rot, blackbody_intensity(T_eff_local, wavelength_aaso), grid_size=grid_size
    )
    
    # Append results
    projected_intensities.append(projected_intensity)
    flux.append(np.sum(projected_intensity))

# Compute magnitude (logarithmic flux)
magnitude = 2.5 * np.log10(flux / np.min(flux))

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1: Magnitude vs time
ax1 = axes[0]
line, = ax1.plot([], [], label="Magnitude (V)")
ax1.set_xlim(tgrid[0], tgrid[-1])
ax1.set_ylim(np.min(magnitude) - 0.1, np.max(magnitude) + 0.1)
ax1.set_xlabel("Time")
ax1.set_ylabel(r"$\delta$ Magnitude (V)")
ax1.legend()

# Subplot 2: Heatmap of projected intensity
ax2 = axes[1]
heatmap = ax2.imshow(
    projected_intensities[0], origin="lower", cmap="inferno",
    extent=[-1, 1, -1, 1], vmin=np.min(projected_intensities), vmax=np.max(projected_intensities)
)
ax2.set_title("Projected Intensity")
ax2.set_xlabel("Observer Plane X")
ax2.set_ylabel("Observer Plane Y")
plt.colorbar(heatmap, ax=ax2, label="Intensity")

# Update function for animation
def update(frame):
    # Update line plot
    line.set_data(tgrid[:frame+1], magnitude[:frame+1])
    
    # Update heatmap
    heatmap.set_data(projected_intensities[frame])
    return line, heatmap

# Create animation
ani = FuncAnimation(fig, update, frames=len(tgrid), interval=100, blit=True)

# Save or display the animation
ani.save("light_curve_animation.mp4", fps=10, dpi=150)
plt.show()






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

                
def simulate_obs_from_image_reco_FAST( oi, image_file , img_pixscl = None):
    # FAST because we don't re-read in obs_files to generate oi each time 

    # change wvl_band_dict[feature] to wvl_lims
    with fits.open(image_file) as d_model:
        img = d_model[0].data

        #assert (abs( float( d_model[0].header['CUNIT2'] ) ) - abs( float(  d_model[0].header['CUNIT1']) ) ) /  float( d_model[0].header['CDELT1'] ) < 0.001
        # we assert the image has to be square..
        #assert abs(float( d_model[0].header['CDELT2'])) == abs(float(d_model[0].header['CDELT1']))
        
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
    theta = kwargs.get('theta', np.linspace(0, np.pi, 200))
    phi = kwargs.get('phi', np.linspace(0, 2 * np.pi, 200))
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
    if save_fits:
        name = f'theta-{theta_o:.2f}_phi-{phi_o:.2f}_T-{delta_T:.2f}.fits'
        intensity_2_fits(
            projected_intensity,
            dx=dx,
            dy=dy,
            name=name,
            data_path=save_fits_path,
            header_dict=header_dict
        )

    # 6. Simulate observed data
    synthetic_file = f"{save_fits_path}{name}"

    oif = simulate_obs_from_image_reco_FAST(
        oi, synthetic_file, img_pixscl = UD_diam / grid_size
    )

    # 7. Compare synthetic and observed data
    comp_dict_v2 = plot_util.compare_models(oi, oif, measure='V2')
    comp_dict_cp = plot_util.compare_models(oi, oif, measure='CP')

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
    
    # Enforce parameter bounds (prior)
    if not (-np.pi <= theta_o <= np.pi) or not (0 <= phi_o <= 2 * np.pi) or not (0 <= delta_T <= 500):
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
                np.random.uniform(-np.pi/2, np.pi/2),  # Random θ_o between 0 and π
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



ins = 'gravity'

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
    UD_diam = 3.5  # Uniform disk diameter in mas (prior)
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
    'theta': np.linspace(0, np.pi, 200),  # Full colatitude
    'phi': np.linspace(0, 2 * np.pi, 200),  # Full longitude
    
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
#thermal_dipole_fit(fit_param=[theta_o, phi_o, delta_T],  **default_kwargs)

# MCMC parameters
nwalkers = 32  # Number of walkers
if fit_ud:
    ndim = 4 # Number of parameters: [theta_o, phi_o, delta_T]
else:
    ndim = 3 
nsteps = 500  # Number of steps for the MCMC run

# Run the MCMC algorithm
sampler = run_mcmc(log_probability, nwalkers, ndim, nsteps, **default_kwargs)

# Extract the samples
samples = sampler.get_chain(discard=100, thin=10, flat=True)  # Discard burn-in, thin samples

# Plot the results
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
if fit_ud:
    labels = [r"$\theta_o$", r"$\phi_o$", r"$\Delta T$", r"$\theta_{UD}$"]
else:
    labels = [r"$\theta_o$", r"$\phi_o$", r"$\Delta T$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(sampler.get_chain()[:, :, i], alpha=0.5)
    ax.set_ylabel(labels[i])
    ax.set_xlabel("Step number")
plt.tight_layout()
plt.show()

# Corner plot of the posterior distributions
try:
    import corner
    fig = corner.corner(samples, labels=labels)
    plt.show()
except ImportError:
    print("Install 'corner' for posterior plots: pip install corner")

# Best-fit parameters
best_fit_params = np.mean(samples, axis=0)
print(f"Best-fit parameters: θ_o={best_fit_params[0]:.3f}, φ_o={best_fit_params[1]:.3f}, ΔT={best_fit_params[2]:.3f}")
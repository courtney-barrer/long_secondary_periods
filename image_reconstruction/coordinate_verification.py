# 1. fit parametric model 
# 2. create image 
# 3. generate synthetic data from image given observations 
# 4. image reconstruction 

## key here is we take one file from a long baseline
# copy it and replace square visibilities with pmoired fake data
# generated from the original data coordinates - but with a parametric model
# we then input this fake file to image reconstruction MiRA algorithm 


import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits 
import sys
import os 
import glob 
import json
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fpdf import FPDF
from PIL import Image
import argparse
import pmoired 

sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MAKE SURE THIS IS IN PATH FIRST
# start new terminal
#export PATH="$HOME/easy-yorick/bin/:$PATH"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



"""
In [9]: d_fits.info()
Filename: /home/rtc/Documents/long_secondary_periods/data/pionier/data/PIONI.2022-05-04T09-26-46.303_oidataCalibrated.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU     668   ()      
  1  OI_TARGET     1 BinTableHDU     58   1R x 17C   [1I, 8A, 1D, 1D, 1E, 1D, 1D, 1D, 7A, 7A, 1D, 1D, 1D, 1D, 1E, 1E, 1A]   
  2  OI_WAVELENGTH    1 BinTableHDU     19   6R x 2C   [1E, 1E]   
  3  OI_ARRAY      1 BinTableHDU     30   4R x 5C   [3A, 2A, 1I, 1E, 3D]   
  4  OI_VIS2       1 BinTableHDU     44   6R x 10C   [1I, 1D, 1D, 1D, 6D, 6D, 1D, 1D, 2I, 6L]   
  5  OI_T3         1 BinTableHDU     58   4R x 14C   [1I, 1D, 1D, 1D, 6D, 6D, 6D, 6D, 1D, 1D, 1D, 1D, 3I, 6L]   


OI_TARGET - copy 
OI_WAVELENGTH - copy
OI_ARRAY - copy , check d_fits['OI_ARRAY'].data['STA_NAME']
OI_VIS2 - replace 'VIS2DATA' , 'UCOORD  ', 'VCOORD  ' , 'STA_INDEX' 
d_fits['OI_VIS2'].data['VCOORD' ].shape = (6,) 
d_fits['OI_VIS2'].data['VIS2DATA' ].shape = (6, 6) # work out if rows or columns are baselines/wavelength 
d_fits['OI_VIS2'].data['FLAG    ' ].shape = (6, 6)

in the oi file

oi.data[0]['OI_VIS2'].keys() =  dict_keys(['D0J3', 'K0J3', 'K0D0', 'G2D0', 'G2J3', 'K0G2'])
oi.data[0]['OI_VIS2']['K0J3']['V2'].shape = (1, 6)

so just work out if rows 

"""


plt.ion()

def pmoiredModel_2_fits( oi, imFov = 200 , imPix= 2, name='untitled'):
    """
    save fits files with heade standard required by OImaging so pmoired images can be uploaded as priors

    Parameters
    ----------
    bestmodel : TYPE
    DESCRIPTION. e.g. input oi.bestfit['best']

    Returns
    -------
    fits file

    """
    _ = pmoired.oimodels.computeLambdaParams( oi.bestfit['best'] ) 

    oi.computeModelImages( imFov=imFov , imPix=imPix)

    im = oi.images['cube'] 

    dx = np.mean( np.diff( oi.images['X'], axis = 1) ) #mas 

    dy = np.mean( np.diff( oi.images['Y'], axis = 0) ) #mas 

    p = fits.PrimaryHDU( im[0] )

    # set headers 
    p.header.set('CRPIX1', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

    p.header.set('CRPIX2', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

    p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel 

    p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel  

    #p.header.set('CDELT1', dx * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT1', dx ) # Coord. incr. per pixel 

    #p.header.set('CDELT2', dy * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT2', dy  ) # Coord. incr. per pixel 

    p.header.set('CUNIT1', 'mas'  ) # Physical units for CDELT1 and CRVAL1 

    p.header.set('CUNIT2', 'mas'  ) # Physical units for CDELT1 and CRVAL1 
    
    p.header.set('HDUNAME', name ) # Physical units for CDELT1 and CRVAL1 

    h = fits.HDUList([])
    h.append( p ) 

    """
    #example 
    oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

    h = pmoiredModel_2_fits( oi, imFov = 200 , name='bens_test')

    h.writeto( '/Users/bencb/Downloads/hello_rtpav.fits' ) 

    """

    return( h )


def fit_a_prior( prior_type, obs_files, fov, pixelsize, save_path,label="some_prior" ,**kwargs):
    
    wavemin = kwargs.get("wavemin", -np.inf ) 
    wavemax = kwargs.get("wavemax", np.inf ) 
    binning= kwargs.get("binning", 1)
    max_rel_V2_error = kwargs.get("max_rel_V2_error", 1)
    max_rel_CP_error = kwargs.get("max_rel_CP_error", 10)
    
    if prior_type == "UD":
        param_grid = kwargs.get( "param_grid", np.logspace( 0,2.5,50) )
        
        oi = pmoired.OI(obs_files, binning = binning)
        bestchi2 = np.inf

        oi.setupFit({'obs':['V2', 'T3PHI'],
                    'min relative error':{'V2':0.0},
                    'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                    'wl ranges':[[wavemin, wavemax]]})

        for udtmp in param_grid:
            ud_model = {'*,ud':udtmp}

            oi.doFit(ud_model)
            if oi.bestfit['chi2'] < bestchi2:
                bestchi2 = oi.bestfit['chi2']
                bestud = oi.bestfit['best']['*,ud']

            ud_model = {'*,ud':bestud}
            oi.doFit(ud_model)
    else:
        print( "prior type does not exist.")
        
        
    
    best_fits = pmoiredModel_2_fits( oi, imFov = fov, imPix = pixelsize, name=f"{prior_type}_prior")
    # write the fits     
    best_fits.writeto( save_path + label + '.fits' , overwrite = True)
    
    dx = best_fits[0].header['CDELT1'] #mas * 3600 * 1e3
    x = np.linspace( -best_fits[0].data.shape[0]//2 * dx , best_fits[0].data.shape[0]//2 * dx,  best_fits[0].data.shape[0])

    dy = best_fits[0].header['CDELT2'] #mas * 3600 * 1e3
    y = np.linspace( -best_fits[0].data.shape[1]//2 * dy , best_fits[0].data.shape[1]//2 * dy,  best_fits[0].data.shape[1])

    origin = 'lower'
    
    # save the image 
    plt.figure()
    plt.imshow( best_fits[0].data , origin = origin, extent = [np.max(x), np.min(x), np.min(y), np.max(y) ] )
    plt.colorbar()
    plt.title("IMAGE RECONSTRUCTION PRIOR\n" + save_path.split('/')[-1] )
    plt.savefig( save_path + label + '.png' ) 
    plt.close()
    
    return( best_fits )




"""
if pixelscale and/or fov is not specified it defaults to 
pixelsize = lambda/4B_max (in milliarc sec) for the central wavelength 
of the given instrument (default instrument is Pionier), 
and fov = 60 * pixelsize.


found issues with gravity and pmoired that some observations are “not observable” (because we observed a little below 20 deg)
- changed this default limit in pmoired fakeoi to 15 degrees in the following functions“

def projBaseline(T1, T2, target, lst, ip1=1, ip2=3, DL1=None, DL2=None,
min_altitude=15, max_OPD=100, max_airmass=None)

def nTelescopes(T, target, lst, ip=None, DLconstraints=None,
min_altitude=15, max_OPD=100, max_vcmPressure=None,
max_airmass=None, flexible=True, plot=False, STS=True)

"""
experiment_lab = 'reco_parametric_models'

parser = argparse.ArgumentParser(description="Script to run image reconstruction with customizable parameters.")
    
# Add arguments
parser.add_argument("--ins", type=str, default="pionier",
                    help="Instrument name (default: pionier)")
parser.add_argument("--prior", type=str, default="Dirac", 
                    help="prior to use in image reconstruction. Can either be Dirac, Random, UD,.."), # or a path to a fits file to use as prior. You can use the following to build this fits file from a parametric model : /home/rtc/Documents/long_secondary_periods/image_reconstruction/creating_parameteric_model_priors.py")
parser.add_argument("--I_really_want_to_use_this_prior", type=str, default=None,
                    help="override any default prior and use the prior specified by this path (must be an appropiate fits file)")
parser.add_argument("--savefig", type=str, default=None,
                    help="Base directory to save plots (default: None which goes to f'/home/rtc/Documents/long_secondary_periods/tests/{ins}/{experiment_lab}/')")
parser.add_argument("--regul", type=str, default="hyperbolic",
                    help="Regularization method (default: hyperbolic)")
parser.add_argument("--pixelsize", type=float, default=None,
                    help="Pixel size in milliarcseconds (default: None in which case we set it based on the instrument at lambda/4Bmax)")
parser.add_argument("--fov", type=float, default=None,
                    help="Field of view in milliarcseconds (default: 35 mas)")
parser.add_argument("--use_vis2", type=str, default="all",
                    help="Use vis2 data ('all', 'phi', or 'none'; default: all)")
parser.add_argument("--use_t3", type=str, default="phi",
                    help="Use T3 data ('phi', 'none', or 'all'; default: phi)")
parser.add_argument("--mu", type=float, default=3e2,
                    help="Relative strength of the prior (default: 300)")
parser.add_argument("--tau", type=float, default=1e-5,
                    help="Edge threshold for regularization (default: 1e-5)")
parser.add_argument("--eta", type=float, default=1,
                    help="Scale for local gradient estimation (default: 1)")
parser.add_argument("--ftol", type=float, default=0,
                    help="Function tolerance (default: 0)")
parser.add_argument("--gtol", type=float, default=0,
                    help="Gradient tolerance (default: 0)")
parser.add_argument("--wavemin", type=float, default=None,
                    help="minimum wavelength in microns")
parser.add_argument("--wavemax", type=float, default=None,
                    help="maximum wavelength in microns")
parser.add_argument("--plot_logV2", action='store_true',
                    help="plot V^2 in log scale")
parser.add_argument("--plot_image_logscale", action='store_true',
                    help="plot image reco in log scale")
# Parse arguments and run the script
args = parser.parse_args()

path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU' # computer location

print( f'\n\n=====\nplot_logV2={args.plot_logV2}\n\n')

# Parameters
ins = args.ins
prior = args.prior
regul = args.regul
pixelsize = args.pixelsize
fov = args.fov
use_vis2 = args.use_vis2
use_t3 = args.use_t3
mu = args.mu
tau = args.tau
eta = args.eta
ftol = args.ftol
gtol = args.gtol
wavemin = args.wavemin #um
wavemax = args.wavemax #um

#input_files = args.input_files


if args.savefig is None:
    savefig = f'/home/rtc/Documents/long_secondary_periods/tests/{ins}/{experiment_lab}/'
    if not os.path.exists(savefig):
        os.makedirs(savefig)
        
else:
    savefig = args.savefig
    if not os.path.exists(savefig):
        os.makedirs(savefig)
        
        
        
# map instrument to particular data files and parameters if no user specification
if ins == 'pionier':
    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_PIONIER_MERGED_2022-04-29.fits'
    #glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
    #
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
    
    if args.pixelsize is None:
        pixelsize = round(1.6e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2)# lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize


    if (wavemin is None) or (wavemax is None):
        wavemin = 1.5
        wavemax = 1.8  
        
elif ins == 'gravity':
    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_pav_GRAVITY_SC_P1_MERGED_2022-06-24.fits'
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
    
    if args.pixelsize is None:
        pixelsize = round(2.2e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2)# lambda/4Bmax in mas
 
    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 2.100#2.05
        wavemax = 2.102#2.40  
               
elif ins == 'matisse_LM':
    ## best not use combine L and M and use 'matisse_L' or 'matisse_M' 
    print("Using CHOPPED MATISSE_LM data (not full merged data)")
    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_LM_chopped_2022-04-08.fits' #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits') #)
    #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    #
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 3.4e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi, 2) # lambda/4Bmax in mas
   
    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 3.1 
        wavemax = 4.9       
             
elif ins == 'matisse_L':
    print("Using CHOPPED MATISSE_LM data (not full merged data)")
    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_LM_chopped_2022-04-08.fits' #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits') #)
    #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    #
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 3.4e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi, 2) # lambda/4Bmax in mas
   
    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 3.3 #2.8
        wavemax = 3.6 #3.8   
        
elif ins == 'matisse_M':
    print("Using CHOPPED MATISSE_LM data (not full merged data)")
    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_LM_chopped_2022-04-08.fits' #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits') #)
    #'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    #
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 3.4e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi, 2) # lambda/4Bmax in mas
   
    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 4.6 
        wavemax = 4.9    

elif ins == 'matisse_N':
    
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # to compare model plots with the observed data
    #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 10e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 11.0#7.5
        wavemax = 12.0 #13.0

elif ins == 'matisse_N_8um':
    
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # to compare model plots with the observed data
    #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 8.0#7.5
        wavemax = 9.0 #13.0

elif ins == 'matisse_N_9um':
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # # to compare model plots with the observed data
    # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 9.0#7.5
        wavemax = 10.0 #13.0
    
elif ins == 'matisse_N_10um':
    
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # # to compare model plots with the observed data
    # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 10e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 10.0#7.5
        wavemax = 11.0 #13.0

elif ins == 'matisse_N_11um':
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # # to compare model plots with the observed data
    # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 11e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 11.0 
        wavemax = 12.0 

elif ins == 'matisse_N_12um':
    # using flipped phases and CP phases ( )
    """
    modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
    in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
    """
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    # ORIGINAL (non phase flipped)
    # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # #path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits'
    # # to compare model plots with the observed data
    # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( 12e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = 12.0 
        wavemax = 13.0 


# file ID given the input parameters
if args.I_really_want_to_use_this_prior is None:
    fid = f'{ins}_prior-{prior}_regul-{regul}_pixelscale-{pixelsize}_fov-{fov}_wavemin-{wavemin}_wavemax-{wavemax}_mu-{mu}_tau-{tau}_eta-{eta}_usev2-{use_vis2}_uset3-{use_t3}'
else: 
    fid = f'{ins}_prior-specificFile_regul-{regul}_pixelscale-{pixelsize}_fov-{fov}_wavemin-{wavemin}_wavemax-{wavemax}_mu-{mu}_tau-{tau}_eta-{eta}_usev2-{use_vis2}_uset3-{use_t3}'
## Priors
prior_save_path = savefig + "priors/"
if not os.path.exists( prior_save_path ):
    os.makedirs( prior_save_path )
    
prior_kwargs = {"wavemin" : wavemin, "wavemax": wavemax, "binning":1, "max_rel_V2_error":1,"max_rel_CP_error": 10, "param_grid": np.logspace( 0,2.5,50) } 

label = f"prior_{fid}"
# create (grid search to fit parameteric model to data) prior from parameteric model and save it to the save path
#parametric_image_fit = fit_a_prior( prior_type = 'UD', obs_files = obs_files, fov = fov, pixelsize = pixelsize, label = label, save_path= prior_save_path, **prior_kwargs)

oi = pmoired.OI()
model =  {'*,ud':9.0, '*,x':5 , '*,y':5, '*,incl':70, '*,projang':-45}

#oi.showModel(model, WL=np.linspace( 1,1.5), imFov=fov, showSED=False)
#plt.close()
#plt.savefig('delme2.png')
#oi.bestfit['best'] = model

d_model = plot_util.create_parametric_prior(pmoired_model = model, fov =  fov, pixelsize= pixelsize) #pmoiredModel_2_fits( oi, imFov =  fov, imPix= pixelsize, name='untitled')

prior_file =  'prior_test.fits'
d_model.writeto( prior_file, overwrite=True)
#real_data = fits.open(  path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_PIONIER_MERGED_2022-04-29.fits' )

# parametric_image_file_path = prior_save_path + label + '.fits'

# image_file = parametric_image_file_path 
# #image_file = '/home/rtc/Documents/long_secondary_periods/tests/pionier/reco_parametric_models/priors/prior_pionier_prior-Dirac_regul-hyperbolic_pixelscale-0.69_fov-41.4_wavemin-1.5_wavemax-1.8_mu-300.0_tau-1e-05_eta-1_usev2-all_uset3-phi.fits'

# #oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, parametric_image_file_path )



if 1:
     # change wvl_band_dict[feature] to wvl_lims
    #d_model = fits.open( image_file )
    
    img = d_model[0].data

    #assert (abs( float( d_model[0].header['CUNIT2'] ) ) - abs( float(  d_model[0].header['CUNIT1']) ) ) /  float( d_model[0].header['CDELT1'] ) < 0.001
    # we assert the image has to be square..
    #assert abs(float( d_model[0].header['CDELT2'])) == abs(float(d_model[0].header['CDELT1']))

    img_units = d_model[0].header['CUNIT1']

    img_pixscl = d_model[0].header['CDELT1']     
    if img_units == 'deg':
        img_pixscl *= 3600*1e3 # convert to mas
    if img_units == 'mas':
        pass 
    else:  
        raise TypeError('Units not recognized')



    oi = pmoired.OI(obs_files, dMJD=1e9)

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
            
        fake_obs_list.append( \
            pmoired.oifake.makeFakeVLTI(\
                t= a['telescopes'],\
                target = ( a['header']['RA']* 24 / 360 , a['header']['DEC'] ),\
                lst = [a['LST']], \
                wl = a['WL'], \
                mjd0 = a[ 'MJD'], #[mjd], #a[ 'MJD'],\
                cube = cube ) 
        )

    # makefake does some operation on MJD so still doesn't match.  
    oif.data = fake_obs_list




i =  2 # just look at one file on a large baseline 
data = fits.open( obs_files[i] )
vcoord = [] 
ucoord = []
v2 = []
v2err = []

for b in oif.data[i]['OI_VIS2']: 
    v2.append( oif.data[i]['OI_VIS2'][b]['V2'][0].tolist() )
    v2err.append(  oif.data[i]['OI_VIS2'][b]['EV2'][0].tolist() ) 
    ucoord.append(  oif.data[i]['OI_VIS2'][b]['u'][0]  )
    vcoord.append(  oif.data[i]['OI_VIS2'][b]['v'][0]  )

data['OI_VIS2'].data['UCOORD'] = np.array( ucoord )
data['OI_VIS2'].data['VCOORD'] = np.array( vcoord )
data['OI_VIS2'].data['VIS2DATA'] = np.array( v2 )
data['OI_VIS2'].data['VIS2ERR'] = np.array( v2err )


input_file = 'test123.fits'
data.writeto( input_file , overwrite = True ) 

output_imreco_file = 'test_reco_para123.fits'
#write_oifits(oif.data, output_filename = input_file)

# send the ccommand to the terminal
# reconstruct the image with only squared visibilities on the copied fake file


#prior_file = "Dirac"
#mu = 10000
tau = 1
input_str = f"ymira -initial={prior_file} -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} -gtol={gtol} -ftol={ftol} -flux=1 -min=0 -save_initial -bootstrap=1 -save_dirty_map -save_dirty_beam -use_vis=none -use_vis2={use_vis2} -overwrite -use_t3=none {input_file} {output_imreco_file}"
#f"ymira -initial=Dirac -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} {input_files} {output_imreco_file}"
#
os.system( input_str )

reco_fits = fits.open( output_imreco_file )


## comparing the prior fits to the reconstruction 
fig,ax = plt.subplots(1,3)
ax[0].imshow( d_model[0].data )
ax[0].set_title( 'real data')
ax[1].imshow( reco_fits['IMAGE-OI INITIAL IMAGE'].data ) # d_model[0].data )
ax[1].set_title( 'reconstruction\nprior')
ax[2].imshow( reco_fits[0].data )
ax[2].set_title( 'reconstruction')
plt.savefig('delme.png')

# now look at the PMOIRED coordinates of the prior !!!
# oi.showModel(model, WL=np.linspace( 1,1.5), imFov=10, showSED=False)
# #plt.close()
# plt.savefig('delme2.png')


# using function 
plot_util.plot_image_reconstruction( output_imreco_file  , single_plot = False , verbose=True, plot_logscale=False, savefig='delme3.png', prior = "UD")









# #plt.figure() ;plt.imshow( reco_fits[0].data ); plt.savefig('delme.png')



# #plot_util.plot_image_reconstruction( output_imreco_file , single_plot = False , verbose=True, plot_logscale=False, savefig='delme.png', prior = "Dirac")









# def explore_pmoired_data(data):
#     """
#     Explore the nested structure of the pmoired fake VLTI observations.
    
#     Args:
#         data (list or dict): The top-level data structure of the pmoired observations.
#     """
#     def recursive_explore(obj, prefix=""):
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 if isinstance(value, (dict, list)):
#                     print(f"{prefix}{key}: {type(value).__name__}")
#                     recursive_explore(value, prefix=prefix + "    ")
#                 else:
#                     print(f"{prefix}{key}: {type(value).__name__}")
#         elif isinstance(obj, list):
#             print(f"{prefix}List with {len(obj)} elements.")
#             for i, item in enumerate(obj[:5]):  # Limit to first 5 for large lists
#                 print(f"{prefix}[{i}]: {type(item).__name__}")
#                 recursive_explore(item, prefix=prefix + "    ")
#         else:
#             print(f"{prefix}{obj}: {type(obj).__name__}")

#     # Start exploration
#     recursive_explore(data)







# def create_oi_target(fake_data):
#     """
#     Create an OI_TARGET HDU compliant with the OIFITS format expected by MiRA.

#     Args:
#         fake_data (list): List of nested dictionaries representing observations.

#     Returns:
#         fits.BinTableHDU: A BinTableHDU for OI_TARGET.
#     """
#     # Extract target data (assuming one target for simplicity)
#     target_id = [1]  # Example: Single target ID
#     target_name = ['Target']  # Replace with actual target name
#     ra = [float(fake_data[0]['targname'].split()[0])]  # RA in degrees
#     dec = [float(fake_data[0]['targname'].split()[1])]  # Dec in degrees
#     equinox = [2000.0]  # Standard equinox
#     ra_err = [0.0]  # Placeholder for RA error
#     dec_err = [0.0]  # Placeholder for Dec error
#     sysvel = [0.0]  # Placeholder for system velocity
#     velt_type = ['']  # Placeholder for velocity type
#     velt_def = ['']  # Placeholder for velocity definition
#     pm_ra = [0.0]  # Placeholder for proper motion in RA
#     pm_dec = [0.0]  # Placeholder for proper motion in Dec
#     pm_ra_err = [0.0]  # Placeholder for RA proper motion error
#     pm_dec_err = [0.0]  # Placeholder for Dec proper motion error
#     parallax = [0.0]  # Placeholder for parallax
#     parallax_err = [0.0]  # Placeholder for parallax error
#     spec_type = ['']  # Placeholder for spectral type

#     # Define columns for OI_TARGET table
#     columns = [
#         fits.Column(name='TARGET_ID', format='1I', array=target_id),
#         fits.Column(name='TARGET', format='8A', array=target_name),
#         fits.Column(name='RAEP0', format='1D', unit='deg', array=ra),
#         fits.Column(name='DECEP0', format='1D', unit='deg', array=dec),
#         fits.Column(name='EQUINOX', format='1E', unit='year', array=equinox),
#         fits.Column(name='RA_ERR', format='1D', unit='deg', array=ra_err),
#         fits.Column(name='DEC_ERR', format='1D', unit='deg', array=dec_err),
#         fits.Column(name='SYSVEL', format='1D', unit='m/s', array=sysvel),
#         fits.Column(name='VELTYP', format='7A', array=velt_type),
#         fits.Column(name='VELDEF', format='7A', array=velt_def),
#         fits.Column(name='PMRA', format='1D', unit='deg/yr', array=pm_ra),
#         fits.Column(name='PMDEC', format='1D', unit='deg/yr', array=pm_dec),
#         fits.Column(name='PMRA_ERR', format='1D', unit='deg/yr', array=pm_ra_err),
#         fits.Column(name='PMDEC_ERR', format='1D', unit='deg/yr', array=pm_dec_err),
#         fits.Column(name='PARALLAX', format='1E', unit='deg', array=parallax),
#         fits.Column(name='PARA_ERR', format='1E', unit='deg', array=parallax_err),
#         fits.Column(name='SPECTYP', format='1A', array=spec_type),
#     ]

#     # Create and return the OI_TARGET HDU
#     return fits.BinTableHDU.from_columns(columns, name='OI_TARGET')




# def write_oifits(fake_data, output_filename):
#     """
#     Convert a pmoired fake VLTI observations object to an OIFITS-compliant file,
#     handling multidimensional fields for visibility data.

#     Args:
#         fake_data (list): List of nested dictionaries representing observations.
#         output_filename (str): Name of the output OIFITS file.
#     """
#     # Primary HDU
#     primary_hdu = fits.PrimaryHDU()

#     # Store representative MJD in the header
#     primary_hdu.header['MJD'] = list( fake_data[0]['OI_VIS2'].keys( ))[0]  # Example: First MJD
#     primary_hdu.header['COMMENT'] = "Representative MJD stored in the header."
    
#     # Target HDU
#     # target_rows = []
#     # for data in fake_data:
#     #     target_name = data['targname']
#     #     ra, dec = map(float, target_name.split())  # Assuming targname is 'RA Dec'
#     #     target_rows.append((target_name, ra, dec))
#     # target_table = fits.BinTableHDU.from_columns([
#     #     fits.Column(name='TARGET', format='20A', array=[row[0] for row in target_rows]),
#     #     fits.Column(name='RA', format='D', array=[row[1] for row in target_rows]),
#     #     fits.Column(name='DEC', format='D', array=[row[2] for row in target_rows])
#     # ], name='OI_TARGET')
#     target_hdu = create_oi_target(fake_data)

#     # Telescope and Array Configuration
#     telescope_names = fake_data[0]['telescopes']
#     array_table = fits.BinTableHDU.from_columns([
#         fits.Column(name='TEL_NAME', format='16A', array=telescope_names),
#         fits.Column(name='TEL_X', format='D', array=np.zeros(len(telescope_names))),  # Placeholder
#         fits.Column(name='TEL_Y', format='D', array=np.zeros(len(telescope_names))),  # Placeholder
#         fits.Column(name='TEL_Z', format='D', array=np.zeros(len(telescope_names)))   # Placeholder
#     ], name='OI_ARRAY')
    
#     # Observables: VIS2 with multidimensional fields
#     vis2_tables = []
#     for key, obs_data in fake_data[0]['OI_VIS2'].items():
#         n_rows = obs_data['V2'].shape[0]  # Number of rows
#         n_cols = obs_data['V2'].shape[1]  # Number of columns (e.g., 6)

#         # Broadcast u and v to match the number of rows
#         u_broadcasted = np.tile(obs_data['u'], (n_rows, 1)).flatten()
#         v_broadcasted = np.tile(obs_data['v'], (n_rows, 1)).flatten()

#         vis2_tables.append(fits.BinTableHDU.from_columns([
#             fits.Column(name='UCOORD', format='1D', array=u_broadcasted),
#             fits.Column(name='VCOORD', format='1D', array=v_broadcasted),
#             fits.Column(name='VIS2DATA', format=f'{n_cols}D', array=obs_data['V2']),
#             fits.Column(name='VIS2ERR', format=f'{n_cols}D', array=obs_data['EV2']),
#             fits.Column(name='FLAG', format=f'{n_cols}L', array=obs_data['FLAG'])
#         ], name=f'OI_VIS2_{key}'))

#     # Combine all HDUs
#     hdus = [primary_hdu, target_hdu, array_table] + vis2_tables
#     hdul = fits.HDUList(hdus)
    
#     # Write to file
#     hdul.writeto(output_filename, overwrite=True)
#     print(f"Successfully wrote OIFITS file to {output_filename}")


























# #%%%% 


# def write_oif_to_oifits(oif, filename):
#     """
#     Convert a list of VLTI-like observations (oif.data) to an OIFITS-compliant FITS file.

#     Parameters:
#         oif (object): Object containing observation data in oif.data.
#         filename (str): Path to the output FITS file.
#     """
#     from astropy.io import fits
#     import numpy as np

#     hdus = fits.HDUList()
#     primary_hdu = fits.PrimaryHDU()
#     primary_hdu.header['ORIGIN'] = 'Synthetic Data'
#     primary_hdu.header['DATE'] = '2024-12-12'
#     hdus.append(primary_hdu)

#     # Process each observation in oif.data
#     for obs in oif.data:
#         # OI_TARGET
#         oi_target_data = [
#             (1, obs['targname'], 0.0, 0.0, 2000.0, 0.0, 0.0, 'none', 0.0, 0.0)
#         ]
#         oi_target_cols = fits.ColDefs([
#             fits.Column(name='TARGET_ID', format='I', array=[x[0] for x in oi_target_data]),
#             fits.Column(name='TARGET', format='32A', array=[x[1] for x in oi_target_data]),
#             fits.Column(name='RAEP0', format='D', array=[x[2] for x in oi_target_data]),
#             fits.Column(name='DECEP0', format='D', array=[x[3] for x in oi_target_data]),
#             fits.Column(name='EQUINOX', format='E', array=[x[4] for x in oi_target_data]),
#             fits.Column(name='RA_ERR', format='E', array=[x[5] for x in oi_target_data]),
#             fits.Column(name='DEC_ERR', format='E', array=[x[6] for x in oi_target_data]),
#             fits.Column(name='SYSTEM', format='16A', array=[x[7] for x in oi_target_data]),
#             fits.Column(name='PMRA', format='E', array=[x[8] for x in oi_target_data]),
#             fits.Column(name='PMDEC', format='E', array=[x[9] for x in oi_target_data])
#         ])

#         hdus.append(fits.BinTableHDU.from_columns(oi_target_cols, name='OI_TARGET'))

#         # OI_WAVELENGTH
#         wavelengths = obs['WL']
#         oi_wavelength_data = [
#             (wl, 0.0) for wl in wavelengths
#         ]
#         oi_wavelength_cols = fits.ColDefs([
#             fits.Column(name='EFF_WAVE', format='E', array=[x[0] for x in oi_wavelength_data]),
#             fits.Column(name='EFF_BAND', format='E', array=[x[1] for x in oi_wavelength_data])
#         ])
#         hdus.append(fits.BinTableHDU.from_columns(oi_wavelength_cols, name='OI_WAVELENGTH'))

#         # # OI_T3 (Closure Phases)
#         # t3_data = []
#         # for triangle, t3 in obs['OI_T3'].items():
#         #     # for i, mjd in enumerate(obs['MJD']):
#         #     #     t3_data.append((
#         #     #         triangle, mjd, t3['T3AMP'][i], t3['ET3AMP'][i], t3['T3PHI'][i], t3['ET3PHI'][i],
#         #     #         t3['u1'][i], t3['v1'][i], t3['u2'][i], t3['v2'][i]
#         #     #     ))

#         #     for mjd in t3['MJD']:
#         #         for i in range(len(t3['T3AMP'])):
#         #             print(f"i: {i}, mjd: {mjd}, len(T3AMP): {len(t3['T3AMP'])}")
#         #             t3_data.append((
#         #                 triangle, mjd, t3['T3AMP'][i], t3['ET3AMP'][i], t3['T3PHI'][i], t3['ET3PHI'][i],
#         #                 t3['u1'][i], t3['v1'][i], t3['u2'][i], t3['v2'][i]
#         #             ))
#         # oi_t3_cols = fits.ColDefs([
#         #     fits.Column(name='TRIANGLE', format='16A', array=[x[0] for x in t3_data]),
#         #     fits.Column(name='MJD', format='D', array=[x[1] for x in t3_data]),
#         #     fits.Column(name='T3AMP', format='E', array=[x[2] for x in t3_data]),
#         #     fits.Column(name='T3AMPERR', format='E', array=[x[3] for x in t3_data]),
#         #     fits.Column(name='T3PHI', format='E', array=[x[4] for x in t3_data]),
#         #     fits.Column(name='T3PHIERR', format='E', array=[x[5] for x in t3_data]),
#         #     fits.Column(name='U1COORD', format='E', array=[x[6] for x in t3_data]),
#         #     fits.Column(name='V1COORD', format='E', array=[x[7] for x in t3_data]),
#         #     fits.Column(name='U2COORD', format='E', array=[x[8] for x in t3_data]),
#         #     fits.Column(name='V2COORD', format='E', array=[x[9] for x in t3_data])
#         # ])
#         # hdus.append(fits.BinTableHDU.from_columns(oi_t3_cols, name='OI_T3'))

#         # # Add other tables as needed (OI_VIS2, OI_FLUX, etc.)
#         # # The pattern is the same: extract the relevant data and create columns.

#     # Write to file
#     hdus.writeto(filename, overwrite=True)


# def write_oif_to_oifits(oif, filename):
#     """
#     Convert a list of VLTI-like observations (oif.data) to an OIFITS-compliant FITS file.

#     Parameters:
#         oif (object): Object containing observation data in oif.data.
#         filename (str): Path to the output FITS file.
#     """
#     from astropy.io import fits
#     import numpy as np

#     # Initialize HDU list with a primary HDU
#     hdus = fits.HDUList()
#     primary_hdu = fits.PrimaryHDU()
#     primary_hdu.header['ORIGIN'] = 'Synthetic Data'
#     primary_hdu.header['DATE'] = '2024-12-12'
#     primary_hdu.header['CONTENT'] = 'OIFITS'
#     primary_hdu.header['REVISION'] = '1'  # OI-FITS revision
#     hdus.append(primary_hdu)

#     # Process each observation in oif.data
#     for obs in oif.data:
#         # OI_TARGET
#         oi_target_data = [
#             (1, obs['targname'], 0.0, 0.0, 2000.0, 0.0, 0.0, 'none', 0.0, 0.0)
#         ]
#         oi_target_cols = fits.ColDefs([
#             fits.Column(name='TARGET_ID', format='I', array=[x[0] for x in oi_target_data]),
#             fits.Column(name='TARGET', format='32A', array=[x[1] for x in oi_target_data]),
#             fits.Column(name='RAEP0', format='D', array=[x[2] for x in oi_target_data]),
#             fits.Column(name='DECEP0', format='D', array=[x[3] for x in oi_target_data]),
#             fits.Column(name='EQUINOX', format='E', array=[x[4] for x in oi_target_data]),
#             fits.Column(name='RA_ERR', format='E', array=[x[5] for x in oi_target_data]),
#             fits.Column(name='DEC_ERR', format='E', array=[x[6] for x in oi_target_data]),
#             fits.Column(name='SYSTEM', format='16A', array=[x[7] for x in oi_target_data]),
#             fits.Column(name='PMRA', format='E', array=[x[8] for x in oi_target_data]),
#             fits.Column(name='PMDEC', format='E', array=[x[9] for x in oi_target_data])
#         ])
#         hdus.append(fits.BinTableHDU.from_columns(oi_target_cols, name='OI_TARGET'))

#         # OI_WAVELENGTH
#         wavelengths = obs['WL']
#         oi_wavelength_data = [
#             (wl, 0.0) for wl in wavelengths
#         ]
#         oi_wavelength_cols = fits.ColDefs([
#             fits.Column(name='EFF_WAVE', format='D', array=[x[0] for x in oi_wavelength_data]),
#             fits.Column(name='EFF_BAND', format='D', array=[x[1] for x in oi_wavelength_data])
#         ])
#         hdus.append(fits.BinTableHDU.from_columns(oi_wavelength_cols, name='OI_WAVELENGTH'))

#         # Placeholder for OI_VIS2 (Squared Visibility)
#         if 'OI_VIS2' in obs:
#             vis2_data = obs['OI_VIS2']
#             oi_vis2_cols = fits.ColDefs([
#                 fits.Column(name='TARGET_ID', format='I', array=vis2_data['TARGET_ID']),
#                 fits.Column(name='MJD', format='D', array=vis2_data['MJD']),
#                 fits.Column(name='VIS2DATA', format='E', array=vis2_data['V2']),
#                 fits.Column(name='VIS2ERR', format='E', array=vis2_data['EV2'])
#             ])
#             hdus.append(fits.BinTableHDU.from_columns(oi_vis2_cols, name='OI_VIS2'))
#         else:
#             hdus.append(fits.BinTableHDU.from_columns(fits.ColDefs([]), name='OI_VIS2'))

#         # Placeholder for OI_T3 (Closure Phases)
#         if 'OI_T3' in obs:
#             t3_data = obs['OI_T3']
#             oi_t3_cols = fits.ColDefs([
#                 fits.Column(name='TRIANGLE', format='16A', array=t3_data.get('TRIANGLE', [])),
#                 fits.Column(name='MJD', format='D', array=t3_data.get('MJD', [])),
#                 fits.Column(name='T3AMP', format='E', array=t3_data.get('T3AMP', [])),
#                 fits.Column(name='T3AMPERR', format='E', array=t3_data.get('ET3AMP', [])),
#                 fits.Column(name='T3PHI', format='E', array=t3_data.get('T3PHI', [])),
#                 fits.Column(name='T3PHIERR', format='E', array=t3_data.get('ET3PHI', [])),
#             ])
#             hdus.append(fits.BinTableHDU.from_columns(oi_t3_cols, name='OI_T3'))
#         else:
#             hdus.append(fits.BinTableHDU.from_columns(fits.ColDefs([]), name='OI_T3'))

#     # Ensure required extensions are included
#     if not any(hdu.name == 'OI_TARGET' for hdu in hdus):
#         hdus.append(fits.BinTableHDU.from_columns(fits.ColDefs([]), name='OI_TARGET'))

#     if not any(hdu.name == 'OI_WAVELENGTH' for hdu in hdus):
#         hdus.append(fits.BinTableHDU.from_columns(fits.ColDefs([]), name='OI_WAVELENGTH'))

#     # Write to file
#     hdus.writeto(filename, overwrite=True)
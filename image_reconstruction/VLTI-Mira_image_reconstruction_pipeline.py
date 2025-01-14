
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
import re
import fnmatch
sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MAKE SURE THIS IS IN PATH FIRST
# start new terminal
#export PATH="$HOME/easy-yorick/bin/:$PATH"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


plt.ion()

gravity_bands = {'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO2-0':[2.2934, 2.298],\
                                   'CO3-1':[2.322,2.324],'CO4-2':[2.3525,2.3555]}

# def pmoiredModel_2_fits( oi, imFov = 200 , imPix= 2, name='untitled'):
#     """
#     save fits files with heade standard required by OImaging so pmoired images can be uploaded as priors

#     Parameters
#     ----------
#     bestmodel : TYPE
#     DESCRIPTION. e.g. input oi.bestfit['best']

#     Returns
#     -------
#     fits file

#     """
#     _ = pmoired.oimodels.computeLambdaParams( oi.bestfit['best'] ) 

#     oi.computeModelImages( imFov=imFov , imPix=imPix)

#     im = oi.images['cube'] 

#     dx = np.mean( np.diff( oi.images['X'], axis = 1) ) #mas 

#     dy = np.mean( np.diff( oi.images['Y'], axis = 0) ) #mas 

#     p = fits.PrimaryHDU( im[0] )

#     # set headers 
#     p.header.set('CRPIX1', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

#     p.header.set('CRPIX2', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

#     p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel 

#     p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel  

#     #p.header.set('CDELT1', dx * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
#     p.header.set('CDELT1', dx ) # Coord. incr. per pixel 

#     #p.header.set('CDELT2', dy * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
#     p.header.set('CDELT2', dy  ) # Coord. incr. per pixel 

#     p.header.set('CUNIT1', 'mas'  ) # Physical units for CDELT1 and CRVAL1 

#     p.header.set('CUNIT2', 'mas'  ) # Physical units for CDELT1 and CRVAL1 
    
#     p.header.set('HDUNAME', name ) # Physical units for CDELT1 and CRVAL1 

#     h = fits.HDUList([])
#     h.append( p ) 

#     """
#     #example 
#     oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#     h = pmoiredModel_2_fits( oi, imFov = 200 , name='bens_test')

#     h.writeto( '/Users/bencb/Downloads/hello_rtpav.fits' ) 

#     """

#     return( h )



# def create_parametric_prior(pmoired_model ,fov, pixelsize, save_path=None, label="some_prior" ):
#     """
#     pmoired_model is dictionary holding parameteric model parameter in pmoired format 

#     e.g: 
#     UD:
#     pmoired_model = {'*,ud':9.0}

#     offset ellipse:
#     pmoired_model = {'*,ud':9.0, '*,x':5 , '*,y':5, '*,incl':70, '*,projang':-45}
#     """

#     oi = pmoired.OI()
#     # wavelength doesn't really matter here since we just want the image
#     oi.showModel(pmoired_model, WL=np.linspace( 1, 2, 10), imFov=fov, showSED=False)
#     plt.close()
#     #plt.savefig('delme2.png')
#     oi.bestfit = {}
#     oi.bestfit['best'] = pmoired_model

#     prior_fits = pmoiredModel_2_fits( oi, imFov =  fov, imPix= pixelsize, name='untitled')

#     #prior_file =  'prior_test.fits'
#     if save_path is not None:
#         prior_fits.writeto( save_path + label + '.fits' , overwrite = True)
        
#     return prior_fits 


# def fit_a_prior( prior_type, obs_files, fov, pixelsize, save_path, label="some_prior" ,**kwargs):
    
#     wavemin = kwargs.get("wavemin", -np.inf ) 
#     wavemax = kwargs.get("wavemax", np.inf ) 
#     binning= kwargs.get("binning", 1)
#     max_rel_V2_error = kwargs.get("max_rel_V2_error", 1)
#     max_rel_CP_error = kwargs.get("max_rel_CP_error", 10)
    
#     if prior_type == "UD":
#         param_grid = kwargs.get( "param_grid", np.logspace( 0,2.5,50) )
        
#         oi = pmoired.OI(obs_files, binning = binning)
#         bestchi2 = np.inf

#         oi.setupFit({'obs':['V2', 'T3PHI'],
#                     'min relative error':{'V2':0.0},
#                     'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
#                     'wl ranges':[[wavemin, wavemax]]})

#         for udtmp in param_grid:
#             ud_model = {'*,ud':udtmp}

#             oi.doFit(ud_model)
#             if oi.bestfit['chi2'] < bestchi2:
#                 bestchi2 = oi.bestfit['chi2']
#                 bestud = oi.bestfit['best']['*,ud']

#             ud_model = {'*,ud':bestud}
#             oi.doFit(ud_model)
#     else:
#         print( "prior type does not exist.")
        
    
#     best_fits = pmoiredModel_2_fits( oi, imFov = fov, imPix = pixelsize, name=f"{prior_type}_prior")
#     # write the fits     
#     best_fits.writeto( save_path + label + '.fits' , overwrite = True)
    
#     dx = best_fits[0].header['CDELT1'] #mas * 3600 * 1e3
#     x = np.linspace( -best_fits[0].data.shape[0]//2 * dx , best_fits[0].data.shape[0]//2 * dx,  best_fits[0].data.shape[0])

#     dy = best_fits[0].header['CDELT2'] #mas * 3600 * 1e3
#     y = np.linspace( -best_fits[0].data.shape[1]//2 * dy , best_fits[0].data.shape[1]//2 * dy,  best_fits[0].data.shape[1])

#     origin = 'lower'
    
#     # save the image 
#     plt.figure()
#     plt.imshow( best_fits[0].data , origin = origin, extent = [np.max(x), np.min(x), np.min(y), np.max(y) ] )
#     plt.colorbar()
#     plt.title("IMAGE RECONSTRUCTION PRIOR\n" + save_path.split('/')[-1] )
#     plt.savefig( save_path + label + '.png' ) 
#     plt.close()
    
#     return( best_fits )



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
experiment_lab = 'dirac_prior_mu_explore'

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


#The uniform disk fits within these absorption regions give diameters of 3.90$\pm$0.05mas, 4.06 $\pm$0.07mas, 3.94 $\pm$ at wavelengths 2.294, 2.322, 2.353$\mu$m, corresponding to CO(2-1) , CO(3-1), CO(4-2) band heads respectively. 
elif fnmatch.fnmatch(ins, 'gravity_line_*'):

    #extract the band 
    band_label = ins.split('gravity_line_')[-1]
    # get the wavelength limits 
    wavemin = gravity_bands[band_label][0]
    wavemax = gravity_bands[band_label][1]

    input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_pav_GRAVITY_SC_P1_MERGED_2022-06-24.fits'
    # to compare model plots with the observed data
    obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
    
    if args.pixelsize is None:
        pixelsize = round(2.2e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2)# lambda/4Bmax in mas
 
    if args.fov is None:
        fov = 60 * pixelsize

               
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


elif fnmatch.fnmatch(ins, 'matisse_N_*um'): 
    # using 0.5um binning in N-band for reconstruction
    wvl_bin = 0.5 #um 
    pattern = r"^matisse_N_([\d.]+)um$"
    match = re.match( pattern, ins)
    wmin = round(float(match.group(1)) , 1) 
    
    obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
    input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
    # ORIGINAL (non phase flipped)
    #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
    # to compare model plots with the observed data
    #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
    if args.pixelsize is None:
        pixelsize = round( wmin*1e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

    if args.fov is None:
        fov = 60 * pixelsize

    if (wavemin is None) or (wavemax is None):
        wavemin = wmin 
        wavemax = wmin + wvl_bin

else: 
    raise UserWarning('input instrument (ins) is not a valid options' )

# elif ins == 'matisse_N_8um':
    
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # to compare model plots with the observed data
#     #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 8.0#7.5
#         wavemax = 9.5 #13.0

# elif ins == 'matisse_N_8.5um':
    
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # to compare model plots with the observed data
#     #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 8.5#7.5
#         wavemax = 9.0 #13.0

# elif ins == 'matisse_N_9um':
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # # to compare model plots with the observed data
#     # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 9.0#7.5
#         wavemax = 9.5 #13.0
    
# elif ins == 'matisse_N_9.5um':
    
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # to compare model plots with the observed data
#     #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 9.5#7.5
#         wavemax = 10.0 #13.0

# elif ins == 'matisse_N_10um':
    
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # # to compare model plots with the observed data
#     # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 10e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 10.0#7.5
#         wavemax = 10.5 #13.0

# elif ins == 'matisse_N_10.5um':
    
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     #input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # to compare model plots with the observed data
#     #obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 9e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 10.5#7.5
#         wavemax = 11.0 #13.0

# elif ins == 'matisse_N_11um':
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
    
#     # ORIGINAL (non phase flipped)
#     # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # # to compare model plots with the observed data
#     # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 11e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 11.0 
#         wavemax = 12.0 

# elif ins == 'matisse_N_12um':
#     # using flipped phases and CP phases ( )
#     """
#     modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
#     in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
#     """
#     obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")
#     input_files = path_dict[comp_loc]['data'] + "merged_files/modifiedCP_RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits"
#     # ORIGINAL (non phase flipped)
#     # input_files = path_dict[comp_loc]['data'] + 'merged_files/RT_Pav_MATISSE_N-band_MERGED_2022-04-24.fits'
#     # #path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits'
#     # # to compare model plots with the observed data
#     # obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_merged_N/*fits')
    
#     if args.pixelsize is None:
#         pixelsize = round( 12e-6 / (4 * 120) * 1e3 * 3600 * 180/np.pi , 2) # lambda/4Bmax in mas

#     if args.fov is None:
#         fov = 60 * pixelsize

#     if (wavemin is None) or (wavemax is None):
#         wavemin = 12.0 
#         wavemax = 13.0 


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
if (prior != 'Dirac') & (prior != 'Random') & (args.I_really_want_to_use_this_prior is None):
    label = f"prior_{fid}"
    # create (grid search to fit parameteric model to data) prior from parameteric model and save it to the save path
    aprior = plot_util.fit_a_prior( prior_type = prior, obs_files = obs_files, fov = fov, pixelsize = pixelsize, label = label, save_path= prior_save_path, **prior_kwargs)

    prior_file_path = prior_save_path + label + '.fits'
    
elif args.I_really_want_to_use_this_prior is not None:

    prior_file_path = args.I_really_want_to_use_this_prior # specific file path to use as prior (no fitting)
    
    if prior_file_path.split('.')[-1] == "json": 
        # output model parameters from pmoired
        # we need to read it in and create a fits file from it 
        # to input into MiRA using the specified FOV and pixelscale

        print('\nREADING IN PMOIRED MODEL\n')
        
        label = f"prior_{fid}"

        with open(prior_file_path, 'r') as file:
            pmoired_model = json.load(file)

        aprior = plot_util.create_parametric_prior(pmoired_model=pmoired_model ,fov=args.fov, pixelsize=args.pixelsize, save_path=prior_save_path, label=label)

        prior_file_path = prior_save_path + label + '.fits'

        print( f'using {prior_file_path}')

    elif prior_file_path.split('.')[-1] == "fits":
        # fits file built from plot_util.pmoiredModel_2_fits() function
        
        print( f'using {prior_file_path}')

    else:
        raise UserWarning('prior_file_path needs to be json from pmoired model or fits file')
        

else:
    ### HERE WE DON"T USE MIRA INBUILT DEFAULTs  - INSTEAD WE CREATE IT FOR 
    label = f"prior_{fid}"
    aprior = plot_util.create_parametric_prior(pmoired_model =  {'*,ud':0.1} ,fov = fov, pixelsize=pixelsize, label = label, save_path= prior_save_path )
    prior_file_path = prior_save_path + label + '.fits'
    #prior_file_path = prior # otherwise its a inbuilt MIRA prior keyword

# output file name
output_imreco_file = savefig + f'imageReco_{fid}.fits'

# The only thing that matters for the prior is prior_file_path" to a compatible fits file holding the prior! 
# send the command to the terminal
input_str = f"ymira -initial={prior_file_path} -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} -gtol={gtol} -ftol={ftol} -flux=1 -min=0 -bootstrap=1 -save_initial -save_dirty_map -save_dirty_beam -use_vis=none -use_vis2={use_vis2} -overwrite -use_t3={use_t3} {input_files} {output_imreco_file}"
#f"ymira -initial=Dirac -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} {input_files} {output_imreco_file}"
#
os.system(input_str)

# read in the image reconstruction data
plot_util.plot_image_reconstruction( output_imreco_file, single_plot = False , verbose=True, plot_logscale = args.plot_image_logscale, savefig=savefig + f'image_reco_w_dirtybeam_{fid}.png', prior = prior  )

# compare the observed data to the synthetic image reconstruction data
oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, output_imreco_file )


# only use logV for Pionier 
kwargs =  {
    'wvl_lims':[wavemin, wavemax],\
    'model_col': 'orange',\
    'obs_col':'grey',\
    'fsize':18,\
    'logV2':args.plot_logV2,\
    'ylim':[0,1],
    'CPylim':180
    } # 'CP_ylim':180,

v2dict = plot_util.compare_V2_obs_vs_image_reco(oi, oif, return_data=True, savefig=savefig+f'v2_obs_vs_imreco_{fid}.png', **kwargs)
cpdict = plot_util.compare_CP_obs_vs_image_reco(oi, oif, return_data=True, savefig=savefig+f'cp_obs_vs_imreco_{fid}.png', **kwargs)

plt.close('all')


## Need to add own calculation of chi2 based on image reconstruction 
kwargs = {'v2_err_min':0.01,'cp_err_min':0.1} 
comp_dict_v2 = plot_util.compare_models( oi, oif ,measure = 'V2', **kwargs)
comp_dict_cp = plot_util.compare_models( oi, oif ,measure = 'CP', **kwargs)


# Calculate reduced chi-square

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

v2_chi2 = np.mean(flatten(get_all_values(comp_dict_v2['chi2'])))
cp_chi2 = np.mean(flatten(get_all_values(comp_dict_cp['chi2'])))
chi2_reduced = (v2_chi2 + cp_chi2) / 2



###########################################################################
# writing the results to a PDF

# Initialize PDF 
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add Title
pdf.set_font("Arial", style="B", size=16)
pdf.cell(0, 10, "RT Pav Image Reconstruction Report", ln=True, align="C")

# Add Paragraph
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, f"my calc. reduced chi2 = {chi2_reduced}", ln=True, align="C")
pdf.multi_cell(0, 20, input_str)

# File names for plots
plot_files = [
    prior_save_path+f"prior_{fid}.png",
    savefig+f"image_reco_w_dirtybeam_{fid}.png",
    savefig+f"v2_obs_vs_imreco_{fid}.png",
    savefig+f"cp_obs_vs_imreco_{fid}.png"
]

# Add Plots
pdf.set_font("Arial", style="B", size=14)
pdf.cell(0, 10, "Plots:", ln=True)

for plot_file in plot_files:
    try:
        # Check if file exists
        with Image.open(plot_file) as img:
            img_width, img_height = img.size
            # Scale image to fit page width
            aspect_ratio = img_height / img_width
            width = 190  # Max width for A4 page
            height = width * aspect_ratio
            pdf.image(plot_file, x=10, y=None, w=width, h=height)
    except FileNotFoundError:
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(0, 10, f"Plot not found: {plot_file}", ln=True)


# Save PDF
output_filename = f"report_{fid}.pdf"
pdf.output(savefig + output_filename)
print(f"PDF report saved as {output_filename}")





##### FUCKING AROUND WITH MIRA
# ymira -initial=Dirac -pixelsize=0.55mas -fov=35mas -flux=1 -min=0 -use_vis=none -use_vis2=none -overwrite -use_t3=all /home/rtc/Documents/long_secondary_periods/data/merged_files/RT_Pav_PIONIER_MERGED_2022-04-29.fits /home/rtc/Downloads/imageReco_PIONIER_H_v2.fits
# 
# -pixelsize=0.55mas -fov=35mas -flux=1 -min=0 -wavemin=1400.0nm -wavemax=1799.9999999999998nm -save_visibilities -overwrite -bootstrap=1 -save_initial -initial=/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/PIONIER_H/priors/UD/best_UD_PRIOR_MATISSE_1.4um.fits -initialhdu=RECO_IMAGE -recenter -use_vis=none -use_vis2=all -use_t3=phi -save_dirty_map -save_dirty_beam -save_residual_map /home/rtc/Documents/long_secondary_periods/data/merged_files/RT_Pav_PIONIER_MERGED_2022-04-29.fits /home/rtc/Downloads/imageReco_PIONIER_H.fits
# from astropy.io import fits
# import matplotlib.pyplot as plt
# #a = fits.open( "/home/rtc/Downloads/imageReco_PIONIER_H_v2.fits")
# #plt.imshow( a[0].data ); plt.show()
# path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
# comp_loc = 'ANU'
# obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
# im_reco_fits =fits.open(image_file)
# # oi is observed, oif is fake observations generated from image reconstruction 
# # at UV samples of the observed data
# oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, image_file )

# docker run --rm -ti --platform linux/amd64 \
#   -v /Users/bencb/Documents/long_secondary_periods/mira_docker_data/data/merged_files:/home/yorick \
#   ferreol/mira \
#   -initial=Dirac -pixelsize=0.55mas -fov=35mas -flux=1 -min=0 -use_vis=none -use_vis2=all -use_t3=all \
#   /home/yorick/RT_Pav_PIONIER_MERGED_2022-04-29.fits \
#   /Users/bencb/Downloads/test.fits


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

sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MAKE SURE THIS IS IN PATH FIRST
# start new terminal
#export PATH="$HOME/easy-yorick/bin/:$PATH"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


plt.ion()

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
parser.add_argument("--plot_logV2", type=bool, default=False,
                    help="plot V^2 in log scale")
parser.add_argument("--plot_image_logscale", type=bool, default=False,
                    help="plot image reco in log scale")
# Parse arguments and run the script
args = parser.parse_args()

path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU' # computer location

# Parameters
ins = args.ins
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
fid = f'{ins}_regul-{regul}_pixelscale-{pixelsize}_fov-{fov}_wavemin-{wavemin}_wavemax-{wavemax}_mu-{mu}_tau-{tau}_eta-{eta}_usev2-{use_vis2}_uset3-{use_t3}'
# output file name
output_imreco_file = savefig + f'imageReco_{fid}.fits'

# send the ccommand to the terminal
input_str = f"ymira -initial=Dirac -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} -gtol={gtol} -ftol={ftol} -flux=1 -min=0 -bootstrap=1 -save_dirty_map -save_dirty_beam -use_vis=none -use_vis2={use_vis2} -overwrite -use_t3={use_t3} {input_files} {output_imreco_file}"
#f"ymira -initial=Dirac -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} {input_files} {output_imreco_file}"
#
os.system(input_str)

# read in the image reconstruction data
plot_util.plot_image_reconstruction( output_imreco_file, single_plot = False , verbose=True, plot_logscale = args.plot_image_logscale, savefig=savefig + f'image_reco_w_dirtybeam_{fid}.png' )

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
pdf.multi_cell(0, 10, input_str)

# File names for plots
plot_files = [
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

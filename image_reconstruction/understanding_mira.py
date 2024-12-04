
import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits 
import os 
import glob 
import json
import pandas as pd

from utilities import plot_util


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



"""
simple script 
dirac prior
small / big FOV
2 x 2 x 2 grid search over mu, gamma, tau? Or just mu ? 
what data - T3 all/none, vis2 all/none
bootstrap - does this impact ? 

document each output and the fits file, 

Also - what happens if we rotate image 

"""

ins = 'pionier'

savefig = '/home/rtc/Documents/long_secondary_periods/'
path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU'
obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')

regul='hyperbolic'
pixelsize=0.68  # 1.6e-6 / 120 / 4 * 1e3 * 3600 * 180/np.pi #mas ( I was using 0.55mas previously!! this is too small)
fov = 35
use_vis2='all'
use_t3 = 'phi' #'none', 'all'
#maxiter=NUMBER, maxeval=NUMBER 

#using parameters from "A typical example" - https://github.com/emmt/MiRA/blob/master/doc/USAGE.md#using-mira-from-the-command-line-via-docker
mu = 3e3 # the relative strength of the prior (compared to the data) 
tau = 1e-5 # the edge threshold, small eta approximates total variation . 
eta = 1  # scale of the finite differences to estimate the local gradient of the image.
ftol=0
gtol=0 

input_files = '/home/rtc/Documents/long_secondary_periods/data/merged_files/RT_Pav_PIONIER_MERGED_2022-04-29.fits'

fid = f'{ins}_regul-{regul}_pixelscale-{pixelsize}_fov-{fov}_mu-{mu}_tau-{tau}_eta-{eta}_usev2-{use_vis2}_uset3-{use_t3}'
output_imreco_file = f'/home/rtc/Downloads/imageReco_{fid}.fits'

input_str = f"ymira -initial=Dirac -regul={regul}-pixelsize={pixelsize}mas -fov={fov}mas  -mu={mu} -tau={tau} -eta={eta} -gtol={gtol} -ftol={ftol} -flux=1 -min=0 -bootstrap=1 -use_vis=none -use_vis2={use_vis2} -overwrite -use_t3={use_t3} {input_files} {output_imreco_file}"
os.system(input_str)

# compare the observed data to the synthetic image reconstruction data
oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, output_imreco_file )

kwargs =  {
    'wvl_lims':[-np.inf, np.inf],\
    'model_col': 'orange',\
    'obs_col':'grey',\
    'fsize':18,\
    'logV2':True,\
    'ylim':[0,1],
    'CPylim':180
    } # 'CP_ylim':180,

v2dict = plot_util.compare_V2_obs_vs_image_reco(oi, oif, return_data=True, savefig=savefig+f'v2_obs_vs_imreco_{fid}.png', **kwargs)
cpdict = plot_util.compare_CP_obs_vs_image_reco(oi, oif, return_data=True, savefig=savefig+f'cp_obs_vs_imreco_{fid}.png', **kwargs)


# put in pdf ?

# im reco parameters
input_str
# im reco 
output_imreco_file
# V2 obs vs im reco 
savefig+f'v2_obs_vs_imreco_{fid}.png'
# CP obs vs im reco 
savefig+f'cp_obs_vs_imreco_{fid}.png'









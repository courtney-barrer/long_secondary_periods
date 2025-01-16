
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

path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU' # computer location
# 1. create parameteric image 
# 2. generate synthetic data from observations 
# 3. reconstruct synthetic data 

# 1 . 
# define parameters and create an image from a parameteric model (saved as fits file)
obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')
pmoired_model = {"p,f": 1, "r,x": 0, "r,y": 0, "p,ud": 5.462567068944827, "r,diamin": 10.3, "r,diamout": 50.1, "r,f": 1.0, "r,incl": 78.30669619040759, "r,projang": -36.67879133231321}
fov = 20
pixelsize = 0.6 
regul = 'hyperbolic'
tau = 1 
mu = 10
eta = 1 
wavemin = 1.4
wavemax = 1.8
use_vis2 = 'all'
gtol = 0 
ftol = 0 

save_path = "/home/rtc/Documents/long_secondary_periods/"
fname = "delme1" 
d_model = plot_util.create_parametric_prior(pmoired_model = pmoired_model ,fov=fov, pixelsize=pixelsize, save_path=save_path, label=fname)


# 2 . 
# oif is the synthetic data (in pmoired format) generated from the parametric 
# image at the UV points defined by obs_files 
oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, image_file = save_path + fname+'.fits', binning=None, insname=None)


# write them back into the original fits file 
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

input_file = 'delme.fits'
data.writeto( input_file , overwrite = True ) 


# With correct image as prior 
prior_file = save_path + fname+'.fits' #"Dirac"
#### Image reconstruction
output_imreco_file = 'delme2.fits'
input_str = f"ymira -initial={prior_file} -regul={regul} -pixelsize={pixelsize}mas -fov={fov}mas -wavemin={wavemin*1e3}nm -wavemax={wavemax*1e3}nm -mu={mu} -tau={tau} -eta={eta} -flux=1 -min=0 -save_initial -bootstrap=1 -save_dirty_map -save_dirty_beam -use_vis=none -use_vis2={use_vis2} -overwrite -use_t3=all {input_file} {output_imreco_file}"
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
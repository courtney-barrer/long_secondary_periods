import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits 
import os 
import glob 
#import json
import pandas as pd
import pmoired 

"""
using 
computeModelImages(self, imFov, model='best', imPix=None,
                           imX=0, imY=0, visibilities=False, debug=False)
visibilities=true should return visibilities of synethic image
"""


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


"""
instrument
# MATISSE, GRAVITY, PIONIER
"""
instr = 'MATISSE' # MATISSE, GRAVITY, PIONIER

prior_type = 'UD'

# MATISSE 
#data_path = '/home/rtc/Documents/long_secondary_periods/data/matisse/reduced_calibrated_data_1/all_merged_N/' #'/home/rtc/Documents/long_secondary_periods/data/merged_files/'
#root_prior_save_path = '/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/MATISSE_N/priors/'

data_path = '/home/rtc/Documents/long_secondary_periods/data/pionier/data/'
root_prior_save_path = '/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/PIONIER_H/priors/'

prior_save_path = root_prior_save_path + prior_type + '/'
if not os.path.exists(prior_save_path ):
    os.makedirs(prior_save_path )



#matisse_files_N = glob.glob( data_path + f'*.fits' )
pionier_files_H = glob.glob( data_path + f'*.fits' )

#merged matisse N-band 2022-07-28T004853 should be in bad data (cannot read in!)
oi = pmoired.OI(pionier_files_H)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# N band - check that this is consistent with what is used in MIRA_image_reconstruction.py 
#imPix = 8* 1e-6/120 * 1e3 * 3600 * 180/3.14  /5 #pixel size (mas)
#imFov = round(128 * imPix)  # FOV (mas) 
#wavegrid = np.arange(8,13.5,0.5) #N-band

imPix = 1.6e-6/120 * 1e3 * 3600 * 180/3.14  /5
imFov = round(64 * pixelsize)  
wavegrid = [1.4, 1.7] 
max_rel_V2_error = 0.4
max_rel_CP_error = 10
wvl_band_dict = {w:[wavegrid[i],wavegrid[i+1]] for i,w in enumerate(wavegrid[:-1])} #{'L':[3.2,3.9],'M':[4.5,5],'N':[[wavegrid[i],wavegrid[i+1]] for i in range(len(wavegrid)-1)]}

bestchi2 = np.inf
sum_dict = {}
for w in wvl_band_dict:
    oi.setupFit({'obs':['V2', 'T3PHI'],
                'min relative error':{'V2':0.0},
                'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                'wl ranges':[wvl_band_dict[w]]})


    for udtmp in np.linspace(0,300,100):
        ud_model = {'*,ud':udtmp}

        oi.doFit(ud_model)
        if oi.bestfit['chi2'] < bestchi2:
            bestchi2 = oi.bestfit['chi2']
            bestud = oi.bestfit['best']['*,ud']

    ud_model = {'*,ud':bestud}
    oi.doFit(ud_model)

    ud_best_fits = pmoiredModel_2_fits( oi, imFov = imFov, imPix = imPix, name=f'bestUD_{w}um')

    ud_best_fits.writeto(prior_save_path + f'best_UD_PRIOR_{instr}_{w}um.fits',overwrite=True)

    sum_dict[w] = {'best ud':oi.bestfit['best'],'chi2':oi.bestfit['chi2']}
    print( w, oi.bestfit['best'] )


# diameter when loading in OImaging is 1". here it should be ~20mas..


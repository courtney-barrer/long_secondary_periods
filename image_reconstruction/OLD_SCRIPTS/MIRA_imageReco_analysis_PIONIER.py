import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits 
import os 
#import glob 
#import json
import pandas as pd 
import glob
import pmoired
import matplotlib.pyplot as plt
import numpy as np
import json 
import importlib

from utilities import plot_util
comp_loc = 'ANU'
path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))

"""
Analysing output from MIRA_image_reconstruction.py

this script plots the reconstructed images, filtering sometimes for the best of them
along with the dirty beam

It also plots the measured observables (V2, CP, etc) against the fitted ones in the 
image reconstruction.

"""
# root path to save best images as png files
what_imgs = 'BEST' #'ALL' #'BEST'
fig_path = path_dict[comp_loc]["root"] + f'image_reconstruction/image_reco/PIONIER_H/{what_imgs}_IMAGES/'

instr = 'PIONIER'
keyword = 'H'
#prior= 'UD'#'Dirac' #'UD' #'random' #'Dirac'
"""
25-7-24 note: I over rode RESULTS_LIGHT_{instr}_{keyword} for Dirac prior - so only have for random prior 
"""
#res_df = pd.read_csv(f'/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/MATISSE_N/RESULTS_LIGHT_{instr}_{keyword}.csv')
for prior in ['UD','random','Dirac']:
    res_df = pd.read_csv(path_dict[comp_loc]["root"] + f'image_reconstruction/image_reco/{instr}_{keyword}/SUMMARY_RESULTS_IMG_RECO_{prior}_{instr}_{keyword}.csv') #RESULTS_LIGHT_{instr}_{keyword}.csv')

    #qick look at distribution of chi2 
    #plt.figure()
    #plt.hist( res_df['chi2'], bins=np.logspace(-3,2,30) );plt.xscale('log');plt.show()

    # per wavelength 
    """fig,ax = plt.subplots(3,4,figsize=(10,6))
    for axx, w in zip(ax.reshape(-1)[:-1], np.unique(res_df['wavemin']) ):
        filt = res_df['wavemin'] == w
        axx.hist( res_df[filt]['chi2'], bins=np.logspace(-3,2,30) );
        axx.set_xscale('log')
        axx.set_title(f'{w}um')
        axx.set_xlabel(r'$\chi_\nu^2$')
    """

    # first iteration forgot to append mu - idiot
    #mu = np.array([float(a.split('mu-')[-1].split('_')[0]) for a in res_df['file'].values]) #res_df['file']
    #res_df['mu'] = mu
    """plt.hist( res_df['chi2'] )

    plt.figure()
    plt.loglog( res_df['mu'], res_df['chi2'],'.')
    plt.xlabel('mu');plt.ylabel('chi2')
    plt.show()
    """

    """
    for each prior
    for each wavelength bin 
    for each regularisation
    """
    # add a distance metric from chi2=1
    res_df['fit_metric'] = abs(res_df['chi2']-1) 
    #plt.ion()

    for w in np.unique(res_df['wavemin']):

        # create a new folder f or wavelength if it doesn't exist
        if not os.path.exists(fig_path + f'{w}um/'):
            os.makedirs(fig_path + f'{w}um/')

        tmp_fig_path = fig_path + f'{w}um/'
        for reg in np.unique(res_df['regul']):
            filt = (res_df['wavemin'] == w) & (res_df['regul'] == reg)
            if what_imgs == 'ALL':
                top10files = res_df[filt].sort_values('fit_metric')['file']
                top10mu = res_df[filt].sort_values('fit_metric')['mu']
                top10chi2 = res_df[filt].sort_values('fit_metric')['chi2']

            elif what_imgs == 'BEST': # get top 10 
                top10files = res_df[filt].sort_values('fit_metric')['file'][:10]
                top10mu = res_df[filt].sort_values('fit_metric')['mu'][:10]
                top10chi2 = res_df[filt].sort_values('fit_metric')['chi2'][:10]
            else:
                raise TypeError('No cases met')
            for i,f in enumerate(top10files):
                with fits.open(f) as h:
                    #initial_image = data['IMAGE-OI INITIAL IMAGE'].data
                    dirty_beam = h['IMAGE-OI DIRTY BEAM'].data
                    #dirty_map = datatmp['IMAGE-OI DIRTY MAP'].data
                    
                    """
                    fig,ax = plt.subplots(1,3,figsize=(15,5))
                    ax[0].imshow( datatmp[0].data )
                    #ax[1].imshow( dirty_map.data )
                    ax[2].imshow( dirty_beam.data )
                    ax[0].set_title( f'RECO {i}, mu={mu}, reg={reg}, wvl={w}')
                    #ax[1].set_title( f'dirty map')
                    ax[2].set_title( f'dirty beam')
                    #plt.show()

                    plt.savefig( tmp_fig_path + f'f.split('/')[-1].split('.fits')[0]}.png', dpi=300, bbox_inches='tight')
                    """


                    r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) )

                    r'$\lambda$ ={}-{}\mu m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 )

                    dx = h[0].header['CDELT1'] #mas * 3600 * 1e3
                    x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])

                    dy = h[0].header['CDELT2'] #mas * 3600 * 1e3
                    y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])

                    origin = 'lower'
                    extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]

                    single_plot = False 
                    # the flipping of the image was cross check with pmoired generated images to make sure coordinates were consistent
                    #im = plt.imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )

                    
                    #plt.pcolormesh(x[::-1], y,  h[0].data /h[0].data.max() , cmap='Reds')#, norm=colors.LogNorm(vmin=1e-2, vmax=1))

                    if single_plot:
                        fig = plt.figure( figsize=(8,8) )
                        im = plt.imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )

                        plt.xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
                        plt.ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
                        plt.gca().tick_params(labelsize=15)

                        plt.text( -x[2], y[2], 'RT Pav', color='k',fontsize=15)
                        plt.text( -x[2], y[4], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
                        plt.text( -x[2], y[6], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)

                        divider = make_axes_locatable(plt.gca())
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cbar = fig.colorbar( im, cax=cax, orientation='vertical')
                        cbar.set_label( 'Normalized flux', rotation=90,fontsize=15)
                        cbar.ax.tick_params(labelsize=15)      

                    else: # we plot dirty beam next to it in subplot
                        fig,ax = plt.subplots( 1,2 , figsize=(12,6) )
                        im = ax[0].imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )
                        ax[1].imshow( np.fliplr(  dirty_beam/np.max(dirty_beam) ),  cmap='Reds',  extent=extent, origin=origin )

                        ax[0].set_ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
                        for axx in [ax[0],ax[1]]:
                            axx.set_xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
                            axx.tick_params(labelsize=15)

                        ax[0].text( x[2], -y[10], 'Image Reco.', color='k',fontsize=15)
                        ax[0].text( x[2], -y[20], 'RT Pav', color='k',fontsize=15)
                        ax[0].text( x[2], -y[30], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
                        ax[0].text( x[2], -y[40], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)
                        ax[1].text( x[2], -y[10], 'Dirty beam', color='k',fontsize=15)

                        divider = make_axes_locatable(ax[0])
                        cax = divider.append_axes('top', size='5%', pad=0.05)
                        cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
                        cax.xaxis.set_ticks_position('top')
                        #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
                        cbar.ax.tick_params(labelsize=15)    

                        divider = make_axes_locatable(ax[1])
                        cax = divider.append_axes('top', size='5%', pad=0.05)
                        cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
                        cax.xaxis.set_ticks_position('top')
                        #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
                        cbar.ax.tick_params(labelsize=15)     
                    #cbar = plt.colorbar()
                    #cbar.ax.set_ylabel('Normalized flux',fontsize=15)
                    #cbar.ax.tick_params(labelsize=12)


                    #cbar.ax.set_yticklabels(ticklabs, fontsize=10)
                    #cbar.ax.set_xlabel(fontsize=15 ) #'Normalize flux')#, rotation=270)
                    tmp_fname = f"{f.split('/')[-1].split('.fits')[0]}.png"
                    plt.savefig( tmp_fig_path + tmp_fname, dpi=300, bbox_inches='tight')
                    plt.close()
                    h.close()


    """
    file = open(f'/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/MATISSE_N/RESULTS_LIGHT_{instr}_{keyword}.json')

    # returns JSON object as 
    # a dictionary
    data = json.load(file)

    regul = np.array( [a.split('reg-')[-1].split('_')[0] for a in data.keys()] )
    chi2 =  np.array( [a['chi2'] for a in data.values()] )
    converged =  np.array( [a['converged'] for a in data.values()] )
    mu =  np.array( [a.split('mu-')[-1].split('_')[0] for a in data.keys()] )

    filt = regul=='compactness'
    plt.plot( mu[filt], chi2[filt]);plt.show()
    file.close()

    """
    
    

#################################
## Testing comparison of model and data from image reconstruction!!!! 
#################################

## 
# root path to save best images as png files
what_imgs = 'ALL' #'ALL' #'BEST'
fig_path = f'/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/PIONIER_H/{what_imgs}_IMAGES/'

instr = 'PIONIER'
keyword = 'H'
#prior= 'UD'#'Dirac' #'UD' #'random' #'Dirac'
"""
25-7-24 note: I over rode RESULTS_LIGHT_{instr}_{keyword} for Dirac prior - so only have for random prior 
"""
#res_df = pd.read_csv(f'/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/MATISSE_N/RESULTS_LIGHT_{instr}_{keyword}.csv')
for prior in ['UD','random','Dirac']:
    res_df = pd.read_csv(f'/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/{instr}_{keyword}/SUMMARY_RESULTS_IMG_RECO_{prior}_{instr}_{keyword}.csv') #RESULTS_LIGHT_{instr}_{keyword}.csv')

    # add a distance metric from chi2=1
    res_df['fit_metric'] = abs(res_df['chi2']-1) 
    #plt.ion()

    for w in np.unique(res_df['wavemin']):

        # create a new folder f or wavelength if it doesn't exist
        if not os.path.exists(fig_path + f'{w}um/'):
            os.makedirs(fig_path + f'{w}um/')

        tmp_fig_path = fig_path + f'{w}um/'
        for reg in np.unique(res_df['regul']):
            filt = (res_df['wavemin'] == w) & (res_df['regul'] == reg)
            if what_imgs == 'ALL':
                top10files = res_df[filt].sort_values('fit_metric')['file'].values
                top10mu = res_df[filt].sort_values('fit_metric')['mu'].values
                top10chi2 = res_df[filt].sort_values('fit_metric')['chi2'].values

            elif what_imgs == 'BEST': # get top 10 
                top10files = res_df[filt].sort_values('fit_metric')['file'][:10].values
                top10mu = res_df[filt].sort_values('fit_metric')['mu'][:10].values
                top10chi2 = res_df[filt].sort_values('fit_metric')['chi2'][:10].values


top10files

#

'image_reconstruction/image_reco/PIONIER_H/priors/UD/best_UD_PRIOR_MATISSE_1.4um.fits'
image_file = 'image_reconstruction/image_reco/PIONIER_H/priors/UD/best_UD_PRIOR_MATISSE_1.4um.fits'#top10files[-1] #'/home/benja/Downloads/imageReco_PIONIER_H_prior-UD_wave-1.4-1.7999999999999998_regul-hyperbolic_mu-46415.888336127726_tau-0.1_fov-35_pixSc-0.6.fits' 
#data_path = #'/home/benja/Documents/long_secondary_periods/data/'
obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')

im_reco_fits =fits.open(image_file)

# oi is observed, oif is fake observations generated from image reconstruction 
# at UV samples of the observed data
oi, oif = plot_util.simulate_obs_from_image_reco( obs_files, image_file )

kwargs =  {
    'wvl_lims':[-np.inf, np.inf],\
    'model_col': 'orange',\
    'obs_col':'grey',\
    'fsize':18,\
    'logV2':True,\
    'ylim':[0,1],
    'CPylim':180
    } # 'CP_ylim':180,

plt.figure(101)
plt.imshow( np.log10( im_reco_fits[0].data ) )
plt.colorbar() 
#plt.title( r'$\chi^2_{\nu}=$' + f'{round(im_reco_fits[2].header['CHISQ'],2)}' ) 
v2dict = plot_util.compare_V2_obs_vs_image_reco(oi, oif, return_data=True, savefig=None, **kwargs)
cpdict = plot_util.compare_CP_obs_vs_image_reco(oi, oif, return_data=True, savefig=None, **kwargs)
plt.show()

def extract_all_values( dictionary , key):
    vals= []
    for k in dictionary[key]:
        for b in dictionary[key][k]:
            vals.append( dictionary[key][k][b] )
            
    flattened_list = [item for sublist in vals for item in sublist]
    return np.array(flattened_list)

v2chi2 = extract_all_values( v2dict, 'chi2')
cpchi2 = extract_all_values( cpdict, 'chi2')
v2res = extract_all_values( v2dict, 'residuals')
cpres = extract_all_values( cpdict, 'residuals')


plt.show()

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits 
import os 
#import glob 
#import json
import pandas as pd 

"""
Analysing output from MIRA_image_reconstruction.py

this script plots the reconstructed images, filtering sometimes for the best of them
along with the dirty beam

It also plots the measured observables (V2, CP, etc) against the fitted ones in the 
image reconstruction.

"""
# root path to save best images as png files
what_imgs = 'BEST' #'ALL' #'BEST'
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
what_imgs = 'BEST' #'ALL' #'BEST'
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

d_model = fits.open( top10files[1] )

visamp = d_model['IMAGE-OI MODEL VISIBILITIES'].data['model_visamp'] 
visphi = d_model['IMAGE-OI MODEL VISIBILITIES'].data['model_visphi'] 
ucoord = d_model['IMAGE-OI MODEL VISIBILITIES'].data['ucoord  '] 
vcoord = d_model['IMAGE-OI MODEL VISIBILITIES'].data['vcoord  '] 
effwvl = d_model['IMAGE-OI MODEL VISIBILITIES'].data['eff_wave'] 
im = d_model[0].data 

  d_model[IMAGE-OI OUTPUT PARAM'].data

plt.figure() 
plt.imshow( im )
plt.show() 


### Now read in the OI data and compare with the model
import glob
import pmoired 
#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = '/home/rtc/Documents/long_secondary_periods/data/' # '/Users/bencb/Documents/long_secondary_periods/data/'

pionier_files = glob.glob(data_path+f'pionier/data/*.fits')

oi = pmoired.OI(pionier_files)

wvl_min = np.min( 1e-3 * d_model[6].data['eff_wave'] )
wvl_max = np.max( 1e-3 * d_model[6].data['eff_wave'] )

oi.setupFit({'obs':['V2', 'T3PHI'], 'wl ranges':[[wvl_min, wvl_max]] })
oi.doFit({'*,ud':3.311})


plt.figure() 
for i in range(len( oi._merged )):
    utmp = oi._merged[i]['OI_VIS2']['all']['u']
    vtmp = oi._merged[i]['OI_VIS2']['all']['v']
    vis2tmp = oi._merged[i]['OI_VIS2']['all']['V2'].reshape(-1)
    vis2errtmp = oi._merged[i]['OI_VIS2']['all']['EV2'].reshape(-1)
    #np.sqrt( utmp**2 + vtmp**2) 
    plt.errorbar( oi._merged[i]['OI_VIS2']['all']['B/wl'].reshape(-1),  vis2tmp , yerr = vis2errtmp,\
        color='k', label='obs',alpha=0.9,fmt='.')
    
plt.figure() 
plt.plot( np.sqrt( (ucoord**2 + vcoord**2 )**0.5 ) /effwvl , abs(visamp)**2, '.', label='model', color='r')
plt.yscale('log')   
plt.show()

plt.figure() 
plt.plot( np.sqrt( (np.unique( ucoord )**2 + np.unique( vcoord )**2 )**0.5/eff_wvl ) , abs(visamp)**2, '.', label='model', color='r')
plt.yscale('log')   
plt.show()

plt.figure() 
plt.plot( effwvl , abs(visamp)**2, '.', label='model', color='r')
plt.yscale('log')   
plt.show()


uu = ucoord.reshape(  6 ,-1) # -1, np.unique( effwvl ).shape [ 0] )
vv = vcoord.reshape(  6, -1)  # -1,np.unique( effwvl ).shape [ 0] )
vamp = visamp.reshape( 6 ,-1) #  -1, np.unique( effwvl ).shape [ 0] )

plt.figure()
for i in [3] : #range(len( uu )):
    plt.plot( uu[i]**2 + vv[i]**2 , vamp[i]  )

plt.show() 

################
# VERIFY PMOIRED AND MIRA RESULTS (COORDINATES) ARE CONSISTENT 
plt.figure()
var = ('v', np.unique( vcoord)  ) 
tmp_list = []
for i in range(len( oi._merged )):
    tmp_list.append( abs(oi._merged[i]['OI_VIS2']['all'][var[0]]) )
   
plt.hist(  [x for xs in tmp_list for x in xs], alpha =0.5,  label = 'pmoired u ')#  oi._merged[i]['OI_VIS2']['all']['B/wl'].reshape(-1) ,label = 'pmoired B/wl' , alpha =0.5 )

plt.hist( abs( var[1] ) ,alpha = 0.5, label='Mira u coord' ) #(ucoord**2 + vcoord**2 )**0.5 / (1e-6 * effwvl)  ,label='Mira model (u^2+v^2)^0.5', alpha =0.5)
plt.legend()
plt.show()


oi._merged[i]['OI_VIS2']['all']['u']



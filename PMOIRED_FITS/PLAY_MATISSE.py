#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:59:11 2024

@author: bcourtne

play 
"""


    

#-- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import pmoired
import glob
import matplotlib.pyplot as plt 
import pandas as pd
import os 
import pickle
from astropy.io import fits 
from mpl_toolkits.axes_grid1 import make_axes_locatable

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
save_path = '/Users/bencb/Documents/long_secondary_periods/PMOIRED_FITS/play_plots/'
ins = 'matisse'

#save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

#feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)


matisse_files_L = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

matisse_files_N = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

#cannot read 2022-07-28T004853_VRTPav_A0B2D0C1_IR-N_LOW_noChop_cal_oifits_0.fits' so ignore it 
#matisse_files_N.pop(9) - this was put in bad data folder 

def pmoiredModel_2_fits( oi, imFov = 200 , name='untitled'):
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
    
    oi.computeModelImages( imFov )

    im = oi.images['cube']
    
    dx = np.mean( np.diff( oi.images['X'], axis = 1) ) #mas 
    
    dy = np.mean( np.diff( oi.images['Y'], axis = 0) ) #mas 
    
    p = fits.PrimaryHDU(  im[0] )
    
    # set headers 
    p.header.set('CRPIX1', oi.images['cube'][0].shape[0] / 2  ) #  Reference pixel  

    p.header.set('CRPIX2', oi.images['cube'][0].shape[0] / 2  ) #  Reference pixel  
    
    p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel     
    
    p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel   
    
    p.header.set('CDELT1', dx * 1e-3  / 3600 * 180/np.pi ) # Coord. incr. per pixel  
    
    p.header.set('CDELT2', dy * 1e-3  / 3600 * 180/np.pi ) # Coord. incr. per pixel  
    
    p.header.set('CUNIT1', 'deg     '   ) # Physical units for CDELT1 and CRVAL1     
    
    p.header.set('CUNIT2', 'deg     '   ) # Physical units for CDELT1 and CRVAL1     
     
    p.header.set('HDUNAME', name  ) # Physical units for CDELT1 and CRVAL1     
    
    h = fits.HDUList([])
    h.append( p ) 

    """
    #example 
    oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

    h = pmoiredModel_2_fits( oi, imFov = 200 , name='bens_test')

    h.writeto( '/Users/bencb/Downloads/hello_rtpav.fits' ) 

    """
    
    return( h )



def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, extent=None, cmap='cool', fontsize=15, cbar_orientation = 'bottom', magnitude = True, axis_off=True, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))
    origin = 'lower'
    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        if magnitude :
            if extent != None:
                im1 = ax1.imshow(  -2.5 * np.log10( np.fliplr( im_list[a] ) ) , extent=extent,cmap=cmap, origin=origin)
            else: 
                im1 = ax1.imshow(  -2.5 * np.log10( np.fliplr( im_list[a] ) ) ,cmap=cmap , origin=origin)
        else: 
            if extent != None:
                im1 = ax1.imshow( np.fliplr(im_list[a] ) , extent=extent ,cmap=cmap , origin=origin)
            else: 
                im1 = ax1.imshow( np.fliplr(im_list[a] ) ,cmap=cmap, origin=origin) 
            
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        
        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
            cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
            cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')           
            cbar.set_label( cbar_label_list[a], rotation=90,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
        
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    plt.show() 



def plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 180,  logV2 = True, savefig_folder=None,savefig_name='plots') :
    model_col = 'orange'
    obs_col= 'grey'
    fsize=18
    fig_inx = 1 
        
    #=========== for plotting 
    # filter for the wavelengths we are looking at 
    wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])

    #===========
        
    # V2
    badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))
    
    wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)
    
    flag_filt = badflag_filt & wvl_plot_filt
    
    
    fig2 = plt.figure(2*fig_inx,figsize=(10,8))
    fig2.set_tight_layout(True)
    
    frame1=fig2.add_axes((.1,.3,.8,.6))
    frame2=fig2.add_axes((.1,.05,.8,.2))  
    
    
    # data 
    frame1.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
    # model
    frame1.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
    
    binned_chi2 = (oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
    frame2.plot( oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )
    
    #frame1.text(10,0.2,feature,fontsize=15)
    
    if logV2:
        frame1.set_yscale('log')
        
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks( [])
    frame1.set_ylim([0,1])
    frame1.legend(fontsize=fsize)
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    
    #plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
    if savefig_folder!=None:
        plt.savefig( savefig_folder + f'{savefig_name}_V2.png' , bbox_inches='tight', dpi=300)
        
    #CP
    badflag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1) ) 
    
    wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)
    
    flag_filt = badflag_filt & wvl_plot_filt
    
    
    fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
    fig3.set_tight_layout(True)
    
    frame1=fig3.add_axes((.1,.3,.8,.6))
    frame2=fig3.add_axes((.1,.05,.8,.2))  
    
    
    # data 
    frame1.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
    # model
    frame1.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
    
    binned_chi2 = (oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
    frame2.plot( oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
    frame2.axhline(1,color='grey',ls=':')
    
    #frame1.text(10,10,feature,fontsize=15)
    
    #if logV2:
    #    plt.yscale('log')
    frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame1.set_ylim([-CP_ylim, CP_ylim])
    frame2.set_yscale('log')
    frame1.legend(fontsize=fsize)
    frame1.set_xticks( [])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    
    if savefig_folder!=None:
        plt.savefig( savefig_folder + f'{savefig_name}_CP.png' , bbox_inches='tight', dpi=300)
    
    

wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_short':[8,9],'N_mid':[9,10],'N_long':[10,13]}

min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 100 
max_rel_CP_error = 20 #deg




#%% #%% Look at Best fit binary in L band 


feature='L'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'


oi = pmoired.OI(matisse_files_L, binning = 5)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 100 
max_rel_CP_error = 20 #deg




# L
best = {'*,ud':3.688,'*,f':1,'c,ud':0,'c,x':-2.431 ,'c,y':1.418,'c,f':0.927}

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 20,  logV2 = False, \
         savefig_folder=savefig_folder , savefig_name=savefig_name)


#%% Look at Best fit binary in M band 

feature='M'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'


oi = pmoired.OI(matisse_files_L, binning = 5)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 100 
max_rel_CP_error = 20 #deg


"""
 \hline\hline 
 \multirow{5}{*}{Binary} 
     & $\theta_{UD,B}$ [mas] & 3.541$\pm$0.011 & 3.4709 $\pm$ 0.006  & 3.688 $\pm$ 0.029 & 8.752 $\pm$ 0.450 \\ \cline{2-6}
     &  $\Delta$RA(x) [mas]  &  0.058$\pm$0.003 & -13.14 $\pm$ 0.20 & -2.431 $\pm$ 0.301 & 0.081 $\pm$ 0.148 \\  \cline{2-6}
     & $\Delta$DEC(y) [mas] & -0.082 $\pm$0.007 & 1.09 $\pm$ 0.29 & 1.418 $\pm$ 0.223 &  0.056 $\pm$ 0.0220  \\ \cline{2-6}
     & $F_c/F_*$ [\%] & 0.062 $\pm$ 0.002 &  0.723  $\pm$ 0.057  & 0.927 $\pm$ 0.345  &  132 $\pm$ 0.148 \\ \cline{2-6}
     & $\chi^2_{\nu, B}$  &  15.7 & 15.0  & 1.6 & 1.1  \\ 
 \hline 
"""

# M 
best = {'*,ud': 8.752,'*,f':1,'c,ud':0,'c,x': 0.081 ,'c,y': 0.056,'c,f':132}

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 20,  logV2 = False, \
         savefig_folder=savefig_folder , savefig_name=savefig_name)

#%% Look at Best fit binary in N short band 
"""
double check UD, ellipse 

oi.doFit({'*,ud':9.935}) # '*,ud':17.64101,, chi2 = 79.0

# UD best
# -- degrees of freedom: 343
# -- reduced chi2: 13.03594203724069
{'*,ud':10.79, # +/- 0.24
}


oi.doFit({'*,ud':9.935,'*,incl':40, '*,projang' :63 })

ellipse best
# -- reduced chi2: 13.08253602681895
{'*,incl':   19.34, # +/- 9.73
'*,projang':88.4, # +/- 41.6
'*,ud':     11.14, # +/- 0.45


"""


feature='N_short'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'


oi = pmoired.OI(matisse_files_N, binning = 5)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.5 
max_rel_CP_error = 20 #deg




best = {'*,ud':9.935,'*,f':1,'c,ud':0,'c,x':-6.45,'c,y':-38.6,'c,f':7/100}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=savefig_folder , savefig_name=savefig_name)

    
plt.figure()
oi.showModel(oi.bestfit['best']  ,imFov=150, showSED=False,imPow=1 )

"""# -- degrees of freedom: 1609
# -- reduced chi2: 11.535197598431155
{'*,ud':10.39, # +/- 0.11
'c,f': 0.0781, # +/- 0.0036
'c,x': -6.88, # +/- 0.20
'c,y': -37.99, # +/- 0.22
'*,f': 1,
'c,ud':0,
}
"""
chiud = 13.5
pmoired.oimodels._nSigmas(chiud, 9 , 5)

sig3_det_limit = 2.8
print('3 sig flux ratio limit', 10**(-sig3_det_limit/2.5))

#%% if we add finite dim
feature='N_short'
model_type = 'resolved_binary'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'


oi = pmoired.OI(matisse_files_N, binning =5)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.5 
max_rel_CP_error = 20 #deg




best = {'*,ud':9.935,'*,f':1,'c,ud':1,'c,x':-6.45,'c,y':-38.6,'c,f':7/100}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f', 'c,f'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=savefig_folder , savefig_name=savefig_name)

    
plt.figure()
oi.showModel(oi.bestfit['best']  ,imFov=150, showSED=False,imPow=0.1 )

"""{'*,ud':9.97, # +/- 0.10
'c,ud':48.35, # +/- 3.22
'c,x': -30.75, # +/- 2.19
'c,y': -19.80, # +/- 2.05
'*,f': 1,
'c,f': 0.07,
}"""
    
    
im = pmoiredModel_2_fits( oi, imFov = 150 , name=f'{ins}_{model_type}_{feature}')
im.writeto( save_path + f'IMAGE_{ins}_{model_type}_{feature}.fits' , overwrite=True ) 


# plot image 
img_list = [  im[0].data ]

xlabel_list = [r'$\Delta$ RA <- E [mas]']
ylabel_list = [r'$\Delta$ DEC -> N [mas]']
title_list = [f'9.0$\mu$m']
cbar_label_list = ['Magnitude']

dx = im[0].header['CDELT1'] * 1e3 * 3600 * np.pi/180 # mas per pixel
extent = [im[0].data.shape[0] * dx / 2, -im[0].data.shape[0] * dx / 2, -im[0].data.shape[1] * dx / 2, im[0].data.shape[1] * dx / 2 ]
cmap='bone'
nice_heatmap_subplots( img_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, extent=extent, cmap=cmap, cbar_orientation = 'right', magnitude = True, axis_off=False, savefig = save_path + f'IMAGE_{ins}_{model_type}_{feature}.png' )



#%%
#%% Look at Best fit binary in N mid band 

"""
double check UD, ellipse 
oi.doFit({'*,ud':9.935}) # '*,ud':17.64101,, chi2 = 79.0

# UD best
# -- degrees of freedom: 801
# -- reduced chi2: 106.82882216565008
{'*,ud':17.65960, # +/- 0.00017
}


oi.doFit({'*,ud':9.935,'*,incl':40, '*,projang' :63 })

ellipse best
{'*,incl':   5.5, # +/- 24.9
'*,projang':92, # +/- 395
'*,ud':     17.67039, # +/- 0.00017
}



plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=None, savefig_name=None)
    
UD has two convertent points, a small diameter to capture average of data, or it just fits something really big elliptical to capture bifurication in data 

"""

feature='N_mid'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'

oi = pmoired.OI(matisse_files_N, binning = 2)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3
max_rel_CP_error = 20 #deg




best = {'*,ud':102.5,'*,f':1,'c,ud':0,'c,x':-0.9,'c,y':1.057,'c,f':62/100}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(best  ,imFov=20, showSED=False)



plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=savefig_folder, savefig_name=savefig_name)


chiud = 73
pmoired.oimodels._nSigmas( chiud, 9.5, 5) # 5 4 parameters and wvl

sig3_det_limit = 3.1
print('3 sig flux ratio limit', 10**(-sig3_det_limit/2.5))


#%% THIS ONE IS INTERESTING - CAPTURES EXACT TREND 
best = {'*,ud':9.935,'*,f':1,'c,ud':0,'c,x':-6.45,'c,y':-38.6,'c,f':7/100}
# try incorporate a disk to bring V2 down 
best = {'*,ud':9.935,'*,f':1,'c,ud':0,'c,x':-100.45,'c,y':-38.6,'c,f':7/100, 'r,diamin':80, 'r,diamout':150,'r,x':-1, 'r,y':0, 'r,f':1 } #,'r2,diamin':80, 'r2,diamout':260,'r2,x':0, 'r2,y':0, 'r2,f':0.1}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

#oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])
oi.doFit(best, doNotFit=['*,f','c,ud', 'r,diamin', 'r,diamout', 'r,x', 'r,y', 'r,f'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=None, savefig_name=None)

    
#plt.figure()
#oi.showModel(oi.bestfit['best']  ,imFov=150, showSED=False,imPow=1 )

"""# -- degrees of freedom: 1609
# -- reduced chi2: 11.535197598431155
{'*,ud':10.39, # +/- 0.11
'c,f': 0.0781, # +/- 0.0036
'c,x': -6.88, # +/- 0.20
'c,y': -37.99, # +/- 0.22
'*,f': 1,
'c,ud':0,
}
"""
chiud = 78
pmoired.oimodels._nSigmas( chiud, 8.4,6)

sig3_det_limit = 3.9
print('3 sig flux ratio limit', 10**(-sig3_det_limit/2.5))



""" grid search for ellipse aspect <- check out matisse_mid_disk_gridsearch
{'p,ud':     10.55, # +/- 0.21
'r,diamin': 150.19, # +/- 6.27
'r,diamout':278.37, # +/- 5.88
'r,incl':   70.10, # +/- 0.91
'r,projang':-68.03, # +/- 0.80
'r,x':      -2.69, # +/- 1.22
'r,y':      0.40, # +/- 0.63
'p,f':      1,
'r,f':      0.9267298700126003}
"""


#%%
#%% Look at Best fit binary in N long band 

"""
double check UD, ellipse 

oi.doFit({'*,ud':9.935}) # '*,ud':17.64101,, chi2 = 79.0


oi.doFit({'*,ud':9.935,'*,incl':40, '*,projang' :63 }) # '*,ud':17.64101,, chi2 = 79.0

#{'*,incl':   6.3, # +/- 18.7
#'*,projang':93, # +/- 262
#'*,ud':     17.65338, # +/- 0.00018
#}

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=None, savefig_name=None)
    
UD has two convertent points, a small diameter to capture average of data, or it just fits something really big elliptical to capture bifurication in data 

"""

feature='N_long'
savefig_folder = save_path
savefig_name = f'matisse_{feature}_best'

oi = pmoired.OI(matisse_files_N,binning = 2)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3
max_rel_CP_error = 20 #deg




best = {'*,ud':106,'*,f':1,'c,ud':0,'c,x':0.9,'c,y':1.2,'c,f':46/100}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(best  ,imFov=20, showSED=False)



#plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
#         savefig_folder=savefig_folder, savefig_name=savefig_name)


chiud = 103
pmoired.oimodels._nSigmas( chiud, 2.8, 6)

sig3_det_limit = 3.9
print('3 sig flux ratio limit', 10**(-sig3_det_limit/2.5))






#%% =============== RING MODELLING 

"""

We do grid search and then final fit on best grid point.
Note inner rad is very sensitive to how we filter uncertain data we use 
min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

Also dfon’t fit primary flux 
oi.doFit( best_model , doNotFit=['p,f'] )


"""



#%% Ring N-short

model_type = 'ring'

feature = 'N_short'

#if __name__ == '__main__':
oi = pmoired.OI(matisse_files_N, binning = 5)

incl_grid = np.linspace( 0, 90 , 15 )
proj_grid = np.linspace( -90, 90 ,20 )
best_grid = [] 
for i, incl in enumerate( incl_grid ) :
    print('progress:',i/len(incl_grid))
    for j, proj in enumerate( proj_grid ):
        
        oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
        best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, \
                      'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                          'r,incl': incl,'r,projang': proj}

        oi.doFit(best_model, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])

        best_grid.append( oi.bestfit )
            

plt.figure() 
chi2_grid = np.array( [a['chi2'] for a in best_grid] ).reshape( len(incl_grid), len( proj_grid))
best_indx = np.unravel_index( np.argmin( chi2_grid ), [len( incl_grid ), len( proj_grid )])

best_incl = incl_grid[ best_indx[0] ]

best_proj = proj_grid[ best_indx[1] ]

plt.pcolormesh( proj_grid, incl_grid, chi2_grid ); 
plt.xlabel('initial projang'); plt.ylabel('initial incl')

best_grid_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, \
              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                  'r,incl': best_incl,'r,projang': best_proj}

# 
"""oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})
"""

oi.doFit( best_grid_model ,doNotFit=['r,f','p,f'])
oi.showModel(oi.bestfit['best'], showSED=False, imFov=400, imPow=0.1)

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=None, savefig_name=None)

    
"""
{'p,ud': 10.47,
 'p,f': 1,
 'r,diamin': 194,
 'r,diamout': 262,
 'r,x': -1.6,
 'r,y': 1,
 'r,f': 0.9267298700126003,
 'r,incl': 83.57142857142857,
 'r,projang': -71.05263157894737}
"""


#%% Ring N-short  Grid serarch , ok now do the big boy search around this point
# from just incl and proj grid search best_incl =  83, bes tproja = -71.05 reduce grid around here
incl_grid = np.linspace( 30, 90 , 3)
proj_grid = np.linspace( -90, -50 , 3 )
rin_grid = np.linspace( 10, 200, 12)
thickness_grid =  np.linspace( 1, 500, 12) # % of inner disk. e.g. outer = in + ridpt * in

best_grid_short = [] # np.zeros( [len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) ] ).astype(list) 


model_type = 'ring'

feature = 'N_short'

    
# # INNER RAD VERY SENSITIVE TO max error
min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


for n, rin in enumerate( rin_grid ):
    for m, thick in enumerate( thickness_grid  ):
        rout = rin + thick 
        
        for i, incl in enumerate( incl_grid ) :
            print('progress:',i/len(incl_grid))
            for j, proj in enumerate( proj_grid ):
                
                oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
                best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': rin,'r,diamout': rout, \
                              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                                  'r,incl': incl,'r,projang': proj}
        
                oi.doFit(best_model, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])
        
                best_grid_short.append( oi.bestfit )  
   


#%% ring N-short - anlsysis
model_type = 'ring'

feature = 'N_short'

chi2grid_short = np.array( [a['chi2'] for a in best_grid_short] ).reshape( len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) )
best_indx = np.unravel_index( np.argmin( chi2grid_short ), [ len(rin_grid ) , len(thickness_grid), len( incl_grid ) , len( proj_grid )])


best_rin = rin_grid[ best_indx[0] ]

best_thickness = thickness_grid[ best_indx[1] ]

best_incl = incl_grid[ best_indx[2] ]

best_proj = proj_grid[ best_indx[3] ]

best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': best_rin,'r,diamout': best_rin + best_thickness, \
              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                  'r,incl': best_incl,'r,projang': best_proj}
    
"""best_model = {'p,ud': 5.56,'p,f': 1, 'r,diamin': 166.5,'r,diamout': 167.7, \
              'r,x': 0.44, 'r,y': -0.01, 'r,f': 0.9267298700126003,\
                  'r,incl': 53,'r,projang':-65.19}"""
    
# fit one more time to see if convergence to same point 


# # INNER RAD VERY SENSITIVE TO max error
min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25 #0.4
max_rel_CP_error = 120 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


#best_model = {'p,f': 1,'p,ud':   15.54, 'r,diamin': 40, 'r,diamout':168, 'r,f': 4, 'r,incl': 53.39, 'r,projang':-65.30, 'r,x':  0.44, 'r,y':  -0.00}

oi.doFit( best_model , doNotFit=['p,f'] )

# best results (chi2=2.1, binning =2)
#{'p,f':    13,'p,ud':   5.54, 'r,diamin': 167, 'r,diamout':168, 'r,f': 4, 'r,incl': 53.39, 'r,projang':-65.30, 'r,x':  0.44, 'r,y':  -0.00}


chi2_fits = fits.PrimaryHDU(  chi2grid_short )
chi2_fits.header.set('what', 'redchi2' ) 

incl_fits = fits.PrimaryHDU(  incl_grid )
incl_fits.header.set('what', 'incl grid' ) 

proj_fits = fits.PrimaryHDU(  proj_grid )
proj_fits.header.set('what', 'proj_grid' ) 

rin_fits = fits.PrimaryHDU(  rin_grid )
rin_fits.header.set('what', 'r_in' ) 

rout_fits = fits.PrimaryHDU(  thickness_grid  )
rout_fits.header.set('what', 'r_out' ) 


h = fits.HDUList( [] ) 
for thing in [chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ]:
    h.append( thing ) 
#h.append( chi2_fits  )
#[chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ])
h.writeto( save_path +  f'disk_GRIDSEARCH_{ins}_{model_type}_{feature}.fits', overwrite=True) 


# get image   
# we add a bit thicker for image prior 
#oi.bestfit['best'] = {'p,f':    1,'p,ud':   5.54, 'r,diamin': 167, 'r,diamout':200, 'r,f': 1, 'r,incl': 53.39, 'r,projang':-65.30, 'r,x':  0.44, 'r,y':  -0.00}

im = pmoiredModel_2_fits( oi, imFov = 500 , name=f'{ins}_{model_type}_{feature}')
#im.writeto( save_path + f'IMAGE_{ins}_{model_type}_{feature}.fits' , overwrite=True ) 


# plot image 
img_list = [ im[0].data ]

xlabel_list = [r'RA <- E [mas]']
ylabel_list = [r'DEC -> N [mas]']
title_list = ['']
cbar_label_list = ['Magnitude']

dx = im[0].header['CDELT1'] * 1e3 * 3600 * np.pi/180 # mas per pixel
extent = [im[0].data.shape[0] * dx / 2, -im[0].data.shape[0] * dx / 2, -im[0].data.shape[1] * dx / 2, im[0].data.shape[1] * dx / 2 ]
cmap='autumn'
nice_heatmap_subplots( img_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, extent=extent, cmap = cmap, cbar_orientation = 'right', magnitude = True, axis_off=False, savefig = None) #save_path + f'matisse/IMAGE_{ins}_{model_type}_{feature}.png' )

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=save_path + 'matisse/', savefig_name=f'{ins}_{model_type}_{feature}')

    

#%% Ring N-mid - GRid fdit
"""
fix ud to L/M band size = 4

"""

model_type = 'ring'
feature = 'N_mid'

#if __name__ == '__main__':
oi = pmoired.OI(matisse_files_N, binning = 2)

# from just incl and proj grid search best_incl =  83, bes tproja = -71.05 reduce grid around here
incl_grid = np.linspace( 30, 90 , 3)
proj_grid = np.linspace( -90, -50 , 3 )
rin_grid = np.linspace( 10, 200, 10)
thickness_grid =  np.linspace( 1, 500, 10) # % of inner disk. e.g. outer = in + ridpt * in

best_grid_mid = [] # np.zeros( [len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) ] ).astype(list) 

min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

for n, rin in enumerate( rin_grid ):
    for m, thick in enumerate( thickness_grid  ):
        rout = rin + thick 
        
        for i, incl in enumerate( incl_grid ) :
            print('progress:',i/len(incl_grid))
            for j, proj in enumerate( proj_grid ):
                
                oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
                best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': rin,'r,diamout': rout, \
                              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                                  'r,incl': incl,'r,projang': proj}
        
                oi.doFit(best_model, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])
        
                best_grid_mid.append( oi.bestfit )  
   

#%%  Ring N-mid - ANslysis
model_type = 'ring'
feature = 'N_mid'

chi2grid = np.array( [a['chi2'] for a in best_grid_mid] ).reshape( len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) )
best_indx = np.unravel_index( np.argmin( chi2grid ), [ len(rin_grid ) , len(thickness_grid), len( incl_grid ) , len( proj_grid )])


best_rin = rin_grid[ best_indx[0] ]

best_thickness = thickness_grid[ best_indx[1] ]

best_incl = incl_grid[ best_indx[2] ]

best_proj = proj_grid[ best_indx[3] ]

best_model = {'p,ud': 10.47,'p,f': 2, 'r,diamin': best_rin,'r,diamout': best_rin + best_thickness, \
              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                  'r,incl': best_incl,'r,projang': best_proj}
    
# # INNER RAD VERY SENSITIVE TO max error
min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


# fit one more time to see if convergence to same point 
oi.doFit( best_model , doNotFit=['p,f'] )

# best results (chi2=7.1, binning =2)
# {'p,f': 1, 'p,ud':     5.74, 'r,diamin': 27.1, 'r,diamout':287.3, 'r,f':1, 'r,incl':   66.59, 'r,projang':-64.31, 'r,x': -4.67, 'r,y': 1.16 }


chi2_fits = fits.PrimaryHDU(  chi2grid )
chi2_fits.header.set('what', 'redchi2' ) 

incl_fits = fits.PrimaryHDU(  incl_grid )
incl_fits.header.set('what', 'incl grid' ) 

proj_fits = fits.PrimaryHDU(  proj_grid )
proj_fits.header.set('what', 'proj_grid' ) 

rin_fits = fits.PrimaryHDU(  rin_grid )
rin_fits.header.set('what', 'r_in' ) 

rout_fits = fits.PrimaryHDU(  thickness_grid  )
rout_fits.header.set('what', 'r_out' ) 




h = fits.HDUList( [] ) 
for thing in [chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ]:
    h.append( thing ) 
#h.append( chi2_fits  )
#[chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ])
h.writeto( save_path +  f'disk_GRIDSEARCH_{ins}_{model_type}_{feature}.fits', overwrite=True) 


# get image   
im = pmoiredModel_2_fits( oi, imFov = 500 , name=f'{ins}_{model_type}_{feature}')
im.writeto( save_path + f'IMAGE_{ins}_{model_type}_{feature}.fits' , overwrite=True ) 


# plot image 
img_list = [ im[0].data ]

xlabel_list = [r'RA <- E [mas]']
ylabel_list = [r'DEC -> N [mas]']
title_list = ['']##[f'10.0$\mu$m']
cbar_label_list = ['Magnitude']

dx = im[0].header['CDELT1'] * 1e3 * 3600 * np.pi/180 # mas per pixel
extent = [im[0].data.shape[0] * dx / 2, -im[0].data.shape[0] * dx / 2, -im[0].data.shape[1] * dx / 2, im[0].data.shape[1] * dx / 2 ]
cmap='autumn'
nice_heatmap_subplots( img_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, extent=extent, cmap=cmap, cbar_orientation = 'right', magnitude = True, axis_off=False, savefig= save_path + f'matisse/IMAGE_{ins}_{model_type}_{feature}.png' )


# ok and what does V2 etc look like 
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=save_path + 'matisse/', savefig_name=f'{ins}_{model_type}_{feature}')

    

# -- reduced chi2: 7.168107154383529
{'p,f':      1, # +/- 2704
'p,ud':     5.74, # +/- 0.69
'r,diamin': 27.1, # +/- 28.8
'r,diamout':287.3, # +/- 10.1
'r,f':      1, # +/- 3549
'r,incl':   66.59, # +/- 1.42
'r,projang':-64.31, # +/- 1.42
'r,x':      -4.67, # +/- 3.09
'r,y':      1.16, # +/- 1.79
}
#%% 
#%% Ring N-long Grid search

model_type = 'ring'
feature = 'N_long'

#if __name__ == '__main__':
oi = pmoired.OI(matisse_files_N, binning = 2)

# from just incl and proj grid search best_incl =  83, bes tproja = -71.05 reduce grid around here
incl_grid = np.linspace( 30, 90 , 3)
proj_grid = np.linspace( -90, -50 , 3 )
rin_grid = np.linspace( 10, 200, 10)
thickness_grid =  np.linspace( 1, 500, 10) # % of inner disk. e.g. outer = in + ridpt * in

best_grid_long = [] # np.zeros( [len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) ] ).astype(list) 



min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


for n, rin in enumerate( rin_grid ):
    for m, thick in enumerate( thickness_grid  ):
        rout = rin + thick 
        
        for i, incl in enumerate( incl_grid ) :
            print('progress:',i/len(incl_grid))
            for j, proj in enumerate( proj_grid ):
                
                oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
                best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': rin,'r,diamout': rout, \
                              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                                  'r,incl': incl,'r,projang': proj}
        
                oi.doFit(best_model, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])
        
                best_grid_long.append( oi.bestfit )  
   


#%%  Ring N-long - ANslysis
model_type = 'ring'
feature = 'N_long'

chi2grid = np.array( [a['chi2'] for a in best_grid_long] ).reshape( len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) )
best_indx = np.unravel_index( np.argmin( chi2grid ), [ len(rin_grid ) , len(thickness_grid), len( incl_grid ) , len( proj_grid )])


best_rin = rin_grid[ best_indx[0] ]

best_thickness = thickness_grid[ best_indx[1] ]

best_incl = incl_grid[ best_indx[2] ]

best_proj = proj_grid[ best_indx[3] ]

"""    
# -- reduced chi2: 1.670997867990985
{'p,f':      1, # +/- 1352
'p,ud':     11.96, # +/- 0.20
'r,diamin': 43.1, # +/- 11.0
'r,diamout':400.64, # +/- 4.45
'r,f':      1, # +/- 1874
'r,incl':   72.92, # +/- 0.28
'r,projang':-44.24, # +/- 0.42
'r,x':      -0.31, # +/- 0.93
'r,y':      -4.00, # +/- 1.28
}
"""
best_model = {'p,ud': 11.96,'p,f': 1, 'r,diamin': best_rin,'r,diamout': best_rin + best_thickness, \
              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                  'r,incl': best_incl,'r,projang': best_proj}

"""
#with actual values
best_model = {'p,ud': 11.96,'p,f': 1, 'r,diamin': 43.1,'r,diamout': 400, \
              'r,x': -0.31, 'r,y': -4, 'r,f': 0.9267298700126003,\
                  'r,incl': 72.92,'r,projang': -44.24 }  """ 
# fit one more time to see if convergence to same point 


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})



oi.doFit( best_model , doNotFit=['p,f'] )

# best results (chi2=7.1, binning =2)
# {'p,f': 1, 'p,ud':     5.74, 'r,diamin': 27.1, 'r,diamout':287.3, 'r,f':1, 'r,incl':   66.59, 'r,projang':-64.31, 'r,x': -4.67, 'r,y': 1.16 }


chi2_fits = fits.PrimaryHDU(  chi2grid )
chi2_fits.header.set('what', 'redchi2' ) 

incl_fits = fits.PrimaryHDU(  incl_grid )
incl_fits.header.set('what', 'incl grid' ) 

proj_fits = fits.PrimaryHDU(  proj_grid )
proj_fits.header.set('what', 'proj_grid' ) 

rin_fits = fits.PrimaryHDU(  rin_grid )
rin_fits.header.set('what', 'r_in' ) 

rout_fits = fits.PrimaryHDU(  thickness_grid  )
rout_fits.header.set('what', 'r_out' ) 


h = fits.HDUList( [] ) 
for thing in [chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ]:
    h.append( thing ) 
#h.append( chi2_fits  )
#[chi2_fits, incl_fits, proj_fits ,rin_fits , rout_fits ])
h.writeto( save_path +  f'disk_GRIDSEARCH_{ins}_{model_type}_{feature}.fits', overwrite=True) 


# get image   
im = pmoiredModel_2_fits( oi, imFov = 500 , name=f'{ins}_{model_type}_{feature}')
im.writeto( save_path + f'IMAGE_{ins}_{model_type}_{feature}.fits' , overwrite=True ) 


# plot image 
img_list = [ im[0].data ]

xlabel_list = [r'RA <- E [mas]']
ylabel_list = [r'DEC -> N [mas]']
title_list = [''] #[f'12.0$\mu$m']
cbar_label_list = ['Magnitude']

dx = im[0].header['CDELT1'] * 1e3 * 3600 * np.pi/180 # mas per pixel
extent = [im[0].data.shape[0] * dx / 2, -im[0].data.shape[0] * dx / 2, -im[0].data.shape[1] * dx / 2, im[0].data.shape[1] * dx / 2 ]
cmap='autumn'
nice_heatmap_subplots( img_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, extent=extent, cmap=cmap, cbar_orientation = 'right', magnitude = True, axis_off=False, savefig=f'IMAGE_{ins}_{model_type}_{feature}.png' )

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, \
         savefig_folder=save_path + 'matisse/', savefig_name=f'{ins}_{model_type}_{feature}')

#amazing agreement with sloan etc !!!!! 

"""    
# -- reduced chi2: 1.670997867990985
{'p,f':      1, # +/- 1352
'p,ud':     11.96, # +/- 0.20
'r,diamin': 43.1, # +/- 11.0
'r,diamout':400.64, # +/- 4.45
'r,f':      1, # +/- 1874
'r,incl':   72.92, # +/- 0.28
'r,projang':-44.24, # +/- 0.42
'r,x':      -0.31, # +/- 0.93
'r,y':      -4.00, # +/- 1.28
}


"""










"""

#T^2 = 4*np.pi**2 / (G * (M1+M2)) * a**3

G = 6.67430 * 1e-11 #m^3 s^-2 kg^-1 
solar_mass =  2e30 #kg
N_sim = 100
# should do Monte carlo with mass ratio
M1 = 1 * solar_mass  # solar_mass * np.random.normal((8-0.5)/2, 1,N_sim )  # (about 0.5 to 8 solar masses)  #2 * (2e30) #kg  2e30kg = solar mass
M2 =  0 #np.array( [np.random.normal(M*1e-3, M*1e-4) for M in M1] ) #np.array( [np.random.normal(M*1e-3, M*1e-4) for M in M1] ) #[np.random.normal(M*1e-3, M*1e-3) for M in M1] # 0.05 * M1 #kg  
P_rtpav = 757 
T = 60 * 60 * 24 * P_rtpav # RT pav period in seconds 
ring_inner_dim = 40  #mas
ring_innder_rad = ring_inner_dim/2 


dist_rtpav = 540 # parsec 
#a = 
ring_sep_parsec =  np.deg2rad( 1e-3/3600 * ring_innder_rad ) * dist_rtpav



m2parsec = 3.24078e-17
a = ( G * (M1+M2) * T**2 / ( 4*np.pi**2 )  )**(1/3)  * m2parsec

sep = np.rad2deg( a / dist_rtpav ) * 3600 * 1e3 
print( sep )

P = np.sqrt( 4*np.pi**2 * (a / m2parsec) **3 / (G*(M1+M2))  )


sep_mean = []
sep_err = []
for i in range( len( candidates['P_LSP'] )):

    if filt[i]:
        m2parsec = 3.24078e-17
        a = ( G * (M1+M2) * T.loc[i]**2 / ( 4*np.pi**2 )  )**(1/3)  * m2parsec

"""





#%% Resolved companion
feature='N_long'
oi = pmoired.OI(matisse_files_N)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3
max_rel_CP_error = 20 #deg


outflow_model = {'*,ud':61.6,'*,f':1,'c,ud':1,'c,x':-10,'c,y':0,'c,f':1}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(outflow_model, doNotFit=['*,f','*,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)
    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, savefig_folder=None,savefig_name='plots')



#%% Disk 
feature='N_mid'
oi = pmoired.OI(matisse_files_N)


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3
max_rel_CP_error = 20 #deg



disk = {'p,ud': 10.47, 'p,f':1, 'r,diamin':40, 'r,diamout':300,'r,x':-1, 'r,y':0, 'r,f':1e-1, }


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(disk, doNotFit=['p,f','p,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, savefig_folder=None,savefig_name='plots')





#oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)


"""
N-mid 
========
initial : disk = {'p,ud': 10.47, 'p,f':1, 'r,diamin':40, 'r,diamout':300,'r,x':-1, 'r,y':0, 'r,f':1e-1, }

min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3
max_rel_CP_error = 20 #deg

oi.doFit(disk, doNotFit=['p,f','p,ud'] )

{'r,diamin': 194.50, # +/- 5.10
'r,diamout':262.96, # +/- 5.12
'r,f':      0.980, # +/- 0.021
'r,x':      -1.61, # +/- 1.02
'r,y':      0.97, # +/- 0.99
'p,f':      1,
'p,ud':     10.47,
}



"""

#%%
#add ellipticity 


disk = {'p,ud': 10.47, 'p,f':1, 'r,diamin':194, 'r,diamout':262,'r,x':-1.6, 'r,y':1, 'r,f':0.9 , 'r,incl':45, 'r,projang':10}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(disk, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])

oi.showModel(oi.bestfit['best'] ,imFov=20, showSED=False)    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, savefig_folder=None,savefig_name='plots')

oi.showModel(oi.bestfit['best'], imFov=400, showSED=False ,imPow=0.1)

#%%
# DO grid search 
# -- define the exploration pattern
expl = {'grid':{'r,incl':(0, 90, 5), 'r,projang':(-90, 90, 5)}}

# -- setup the fit, as usual
oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':wvl_band_dict[feature]})
#oi.setupFit({'obs':['V2', 'T3PHI']})


best_disk = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, 'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003, 'r,incl': 34.44130094553597,'r,projang': -30.053240964807344}
oi.gridFit(expl, model=best_disk, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y','r,f'])
           
"""           , prior=[('c,f', '<', 1)], 
           constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                      ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])"""

  
oi.showGrid(interpolate = True, legend=False,tight=True)




#%% 
#%% large companion 

min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.1
max_rel_CP_error = 20 #deg




outflow_model = {'*,ud':3.311,'*,f':1,'c,ud':1,'c,x':-1,'c,y':0,'c,f':0.06}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(outflow_model, doNotFit=['*,f','*,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)


    
plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 180,  logV2 = False, savefig_folder=None,savefig_name='plots')
# %% point source companion
outflow_model = {'*,ud':3.311,'*,f':1,'c,ud':0,'c,x':-2,'c,y':0,'c,f':0.06}


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(outflow_model, doNotFit=['*,f','*,ud','c,ud'] )#,'e,projang','e,incl'])

#oi.showModel(outflow_model  ,imFov=20, showSED=False)

#plotV2CP( oi ,wvl_band_dict, feature)

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=180, logV2 = True, savefig_folder=save_path, savefig_name=f'pionier_{wvl_band_dict[feature]}_binNone_binary_fit') 

    
#%% Try ellipse where we see chi2 trench 

outflow_model = {'*,ud':3.3,'*,f':1,'e,ud':4,'e,x':-2,'e,y':0,'e,projang':80,'e,incl':60,'e,f':0.9}



oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(outflow_model, doNotFit=['*,f', '*,ud'] )#,'e,projang','e,incl'])

oi.showModel(outflow_model  ,imFov=20, showSED=False)

plotV2CP( oi ,wvl_band_dict, feature)

#%% Try Disk 

outflow_model = {'*,ud':3.3,'*,f':1,'e,ud':4,'e,x':-2,'e,y':0,'e,projang':80,'e,incl':60,'e,f':0.9}



oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

oi.doFit(outflow_model, doNotFit=['*,f', '*,ud'] )#,'e,projang','e,incl'])

oi.showModel(outflow_model  ,imFov=20, showSED=False)

plotV2CP( oi ,wvl_band_dict, feature)






#%%
# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]



fig_inx = 1
for feature in  wvl_band_dict :
    
    print(f'\n======\n{feature}\n')
    min_rel_V2_error = 0.0001
    min_rel_CP_error = 0.1 #deg
    max_rel_V2_error = 100 
    max_rel_CP_error = 20 #deg
    
    #feature = 'continuum'
    ID='play'
    
    ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)
    
    wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])
    
    wvls = oi.data[0]['WL'][wvl_filt][:-1] # avelengths to consider
        
    #model = {'ud':4.9} #, 'x':.5, 'y':-0.2} 
    
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.0},
                 'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                 'wl ranges':[wvl_band_dict[feature]]})
            
    oi.doFit( {'p,ud':3.27,'c,x':-2.2,'c,y':0.2,'c,f':0.01} ,doNotFit=['p,ud'] ) #, doNotFit=['p,ud'] ) 
    
    #oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)
    

    
    model_col = 'orange'
    obs_col= 'grey'
    fsize=18
    
    # seperation in au 
    13 *1e-3/3600 * np.pi/180 * 206265 * 520
    #then from keplers law we can find combined mass 
    
    # V2
    badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))
    
    wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)
    
    flag_filt = badflag_filt & wvl_plot_filt
    
    
    fig2 = plt.figure(2*fig_inx,figsize=(10,8))
    fig2.set_tight_layout(True)
    
    frame1=fig2.add_axes((.1,.3,.8,.6))
    frame2=fig2.add_axes((.1,.05,.8,.2))  
    
    
    # data 
    frame1.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
    # model
    frame1.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
    
    binned_chi2 = (oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
    frame2.plot( oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )
    
    frame1.text(10,0.2,feature,fontsize=15)
    
    #if logV2:
    frame1.set_yscale('log')
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks( [])
    frame1.set_ylim([0,1])
    frame1.legend(fontsize=fsize)
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    
    #plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
    #CP
    badflag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1) ) 
    
    wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)
    
    flag_filt = badflag_filt & wvl_plot_filt
    
    
    fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
    fig3.set_tight_layout(True)
    
    frame1=fig3.add_axes((.1,.3,.8,.6))
    frame2=fig3.add_axes((.1,.05,.8,.2))  
    
    
    # data 
    frame1.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
    # model
    frame1.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
    
    binned_chi2 = (oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
    frame2.plot( oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
    frame2.axhline(1,color='grey',ls=':')
    
    frame1.text(10,10,feature,fontsize=15)
    
    #if logV2:
    #    plt.yscale('log')
    frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame1.set_ylim([-180,180])
    frame2.set_yscale('log')
    frame1.legend(fontsize=fsize)
    frame1.set_xticks( [])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    
    
    fig_inx *= 10

#%% on photosphere 


feature = 'continuum'
ID='play'

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])

wvls = oi.data[0]['WL'][wvl_filt][:-1] # avelengths to consider
    
#model = {'ud':4.9} #, 'x':.5, 'y':-0.2} 


    
oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})
        
#oi.doFit( {'p,ud':3.8,'c,ud':1,'c,x':0,'c,y':0,'c,f':1}, doNotFit=['p,ud'], prior=[('c,x**2 + c,y**2','<=',10)])

oi.doFit( {'p,ud':3.8,'c,ud':2.672,'c,x':0.170,'c,y':0.0181,'c,f':0.403, 'c2,x':-13,'c2,y':1,'c2,f':1}, doNotFit=['p,ud','c,ud','c,x','c,y','c,f']) #, prior=[('c,x**2 + c,y**2','<=',10)])
oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)

fig_inx = 1

model_col = 'orange'
obs_col= 'grey'
fsize=18

# seperation in au 
13 *1e-3/3600 * np.pi/180 * 206265 * 520
#then from keplers law we can find combined mass 

# V2
badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig2 = plt.figure(2*fig_inx,figsize=(10,8))
fig2.set_tight_layout(True)

frame1=fig2.add_axes((.1,.3,.8,.6))
frame2=fig2.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )

frame1.text(10,0.2,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$V^2$',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame2.set_yscale('log')
frame1.set_xticks( [])
frame1.set_ylim([0,1])
frame1.legend(fontsize=fsize)
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )
frame2.axhline(1,color='grey',ls=':')

#plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
  
#CP
badflag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1) ) 

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
fig3.set_tight_layout(True)

frame1=fig3.add_axes((.1,.3,.8,.6))
frame2=fig3.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
frame2.axhline(1,color='grey',ls=':')

frame1.text(10,10,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame1.set_ylim([-10,10])
frame2.set_yscale('log')
frame1.legend(fontsize=fsize)
frame1.set_xticks( [])
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )
    
#%% add disk 


oi.setupFit({'obs':['T3PHI','V2','PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})

model_2 = {'p,ud': 3.47,  'r,diamin':40, 'r,diamout':150,'r,x':-0, 'r,y':0, 'r,f':1e0, } #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
#model_2 = {'p,ud': 3.47,  'r,diamin':0, 'r,diamout':150,'r,x':-13, 'r,y':0, 'r,f':1e0, } #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
# if just fit cCP
"""{'p,ud':     4.1348, # +/- 0.0088
'r,diamin': 42.80, # +/- 0.42
'r,diamout':149.41, # +/- 0.18
'r,f':      2.25, # +/- 0.29
'r,x':      0.25, # +/- 0.11
'r,y':      -1.767, # +/- 0.073
}"""

oi.doFit( model_2, doNotFit=[{'p,ud'}] ) 


oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)


# V2
badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig2 = plt.figure(2*fig_inx,figsize=(10,8))
fig2.set_tight_layout(True)

frame1=fig2.add_axes((.1,.3,.8,.6))
frame2=fig2.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )

frame1.text(10,0.2,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$V^2$',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame2.set_yscale('log')
frame1.set_xticks( [])
frame1.set_ylim([0,1])
frame1.legend(fontsize=fsize)
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )
frame2.axhline(1,color='grey',ls=':')


#CP
badflag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1) ) 

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
fig3.set_tight_layout(True)

frame1=fig3.add_axes((.1,.3,.8,.6))
frame2=fig3.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
frame2.axhline(1,color='grey',ls=':')

frame1.text(10,10,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame1.set_ylim([-10,10])
frame2.set_yscale('log')
frame1.legend(fontsize=fsize)
frame1.set_xticks( [])
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )



#%% disk model (TO DO FOR MATISSE N short


oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


model_2 = {'disk,diamin':36.4, 
         'disk,diamout':36.8, 
         'disk,profile':'$R**-2', 
         'disk,az amp1':1, 
         'disk,az projang1':60, 
         'disk,projang':45, 
         'disk,incl':30, 
         'disk,x':-1.8, 
         'disk,y':1.1, 
         'disk,spectrum':'0.8*($WL)**-2',
         'star,ud':3.8, 
         'star,spectrum':'$WL**-3', 
        }
oi.doFit( model_2, doNotFit=[{'p,ud'}] ) 


oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)


# V2
badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig2 = plt.figure(2*fig_inx,figsize=(10,8))
fig2.set_tight_layout(True)

frame1=fig2.add_axes((.1,.3,.8,.6))
frame2=fig2.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )

frame1.text(10,0.2,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$V^2$',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame2.set_yscale('log')
frame1.set_xticks( [])
frame1.set_ylim([0,1])
frame1.legend(fontsize=fsize)
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )
frame2.axhline(1,color='grey',ls=':')


#CP
badflag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1) ) 

wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[0]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)

flag_filt = badflag_filt & wvl_plot_filt


fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
fig3.set_tight_layout(True)

frame1=fig3.add_axes((.1,.3,.8,.6))
frame2=fig3.add_axes((.1,.05,.8,.2))  


# data 
frame1.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
frame1.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

binned_chi2 = (oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
frame2.plot( oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
frame2.axhline(1,color='grey',ls=':')

frame1.text(10,10,feature,fontsize=15)

#if logV2:
#    plt.yscale('log')
frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
frame1.set_ylim([-10,10])
frame2.set_yscale('log')
frame1.legend(fontsize=fsize)
frame1.set_xticks( [])
frame1.tick_params( labelsize=fsize )
frame2.tick_params( labelsize=fsize )
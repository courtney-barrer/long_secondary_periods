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
    
    
    
#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
save_path = '/Users/bencb/Documents/long_secondary_periods/PMOIRED_FITS/play_plots/'

ins = 'gravity'

#save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

#feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)



gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')



oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning = 50 ) 

wvl_band_dict =  {'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}

    
    

    
    
#%% Start with ellipse model (best fit) - recalibrate redchi2 reference from this due to different spectral binning 

feature = 'continuum'

oi.setupFit({'obs':['T3PHI','V2','PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':1, 'CP':1},
             'wl ranges':[wvl_band_dict[feature]]})

model_1 = {'p,ud': 3.47,  'p,projang':80, 'p,incl':12 } #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
#model_2 = {'p,ud': 3.47,  'r,diamin':0, 'r,diamout':150,'r,x':-13, 'r,y':0, 'r,f':1e0, } #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
# if just fit cCP
"""{'p,ud':     4.1348, # +/- 0.0088
'r,diamin': 42.80, # +/- 0.42
'r,diamout':149.41, # +/- 0.18
'r,f':      2.25, # +/- 0.29
'r,x':      0.25, # +/- 0.11
'r,y':      -1.767, # +/- 0.073
}"""

oi.doFit( model_1, doNotFit=[{}] ) 


#oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)


#plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=18, logV2 = False, savefig_folder=None,savefig_name='plots') 

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=18, logV2 = False, savefig_folder=save_path, savefig_name=f'gravity_{wvl_band_dict[feature]}_bin50_ellipse_fit') 

"""
# -- reduced chi2: 10.43701238058854
{'p,incl':   11.36, # +/- 0.43
'p,projang':84.68, # +/- 2.37
'p,ud':     3.5141, # +/- 0.0040
}
"""

#%% Binary 

feature = 'continuum'

# THIS IS INCLUDING DIFFERENTIAL PHASE 
oi.setupFit({'obs':['T3PHI','V2','PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':1, 'CP':10},
             'wl ranges':[wvl_band_dict[feature]]})

model_2 = {'p,ud': 3.47,   'p,f':1, 'c,ud':0., 'c,x':-13, 'c,y':1, 'c,f':0.1} #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
#model_2 = {'p,ud': 3.47,  'r,diamin':0, 'r,diamout':150,'r,x':-13, 'r,y':0, 'r,f':1e0, } #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
# if just fit cCP
"""{'p,ud':     4.1348, # +/- 0.0088
'r,diamin': 42.80, # +/- 0.42
'r,diamout':149.41, # +/- 0.18
'r,f':      2.25, # +/- 0.29
'r,x':      0.25, # +/- 0.11
'r,y':      -1.767, # +/- 0.073
}"""

oi.doFit( model_2 , doNotFit={'c,ud','p,f'} ) #, doNotFit={'c,ud','p,f'} ) 


oi.showModel(oi.bestfit['best'], imFov=40, showSED=False ,imPow=0.1)


#plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=18, logV2 = False, savefig_folder=save_path, savefig_name=f'gravity_{wvl_band_dict[feature]}_bin50_binary_fit') 

"""

#redchi2 = 8.7  . 

oi.doFit( model_2 )  - no constraints on parameters

{'c,f': 1, # +/- 31378
'c,ud':3.56, # +/- 0.18
'c,x': -13.375, # +/- 0.072
'c,y': 1.130, # +/- 0.078
'p,f': 120, # +/- 3342750
'p,ud':3.4537, # +/- 0.0020
}

if constraining parameters (doNotFit={'c,ud','p,f'} )
# -- reduced chi2: 9.044327346742447
{'c,f': 0.00564, # +/- 0.00024
'c,x': -13.198, # +/- 0.060
'c,y': 1.043, # +/- 0.047
'p,ud':3.4627, # +/- 0.0017
'c,ud':0.1,
'p,f': 1,
}

if we constrain to point source 
# -- reduced chi2: 9.044814303687398
{'c,f': 0.00563, # +/- 0.00024
'c,x': -13.197, # +/- 0.060
'c,y': 1.043, # +/- 0.047
'p,ud':3.4627, # +/- 0.0017
'c,ud':0.0,
'p,f': 1,
}

"""


#%% Try ellipse where we see chi2 trench 


outflow_model = {'*,ud':3.3,'*,f':1,'e,ud':4,'e,x':-2,'e,y':0,'e,projang':80,'e,incl':60,'e,f':0.9}

oi.setupFit({'obs':['T3PHI','V2','PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':1, 'CP':10},
             'wl ranges':[wvl_band_dict[feature]]})


oi.doFit(outflow_model, doNotFit=['*,f', '*,ud'] )#,'e,projang','e,incl'])

oi.showModel(outflow_model  , imFov=20, showSED=False)

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=20, logV2 = False, savefig_folder=None,savefig_name='plots') 



#%% Try ring

disk_model = {'p,ud': 3.47, 'p,f':1, 'r,diamin':0, 'r,diamout':15,'r,x':0, 'r,y':0., 'r,f':0.1, } 


oi.setupFit({'obs':['T3PHI','V2'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':1, 'CP':10},
             'wl ranges':[wvl_band_dict[feature]]})


oi.doFit(disk_model, doNotFit=['p,f', 'p,ud'] )#,'e,projang','e,incl'])

oi.showModel(oi.bestfit['best'],  imFov=200, showSED=False)

plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=20, logV2 = False, savefig_folder=None,savefig_name='plots') 






#%% lets include an offset 
feature = 'continuum'

oi.setupFit({'obs':['T3PHI','V2','PHI'], 
             'min relative error':{'V2':0.01},
             'max relative error':{'V2':1, 'CP':1},
             'wl ranges':[wvl_band_dict[feature]]})

model_2 = {'p,ud': 3.47,  'p,projang':80, 'p,incl':12, 'p,x':-2, 'p,y':1} #, 'c,f': 0.00771640731798301,'c,x': -13.306142852954801,'c,y': 1.1878697066255672}
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


plotV2CP( oi ,wvl_band_dict, feature, CP_ylim=20, logV2 = False, savefig_folder=None,savefig_name='plots') 


    
#%%
# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]


for feature in  wvl_band_dict :
    
    print(f'\n======\n{feature}\n')
    min_rel_V2_error = 0.01
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
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                 'wl ranges':[wvl_band_dict[feature]]})
            
    oi.doFit( {'p,ud':3.47,'c,ud':10,'c,x':-13,'c,y':1,'c,f':0.1}, doNotFit=['p,ud'] ) 
    
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
        
oi.doFit( {'p,ud':3.8,'c,ud':1,'c,x':0,'c,y':0,'c,f':1}, doNotFit=['p,ud'], prior=[('c,x**2 + c,y**2','<=',10)])

#oi.doFit( {'p,ud':3.8,'c,ud':2.672,'c,x':0.170,'c,y':0.0181,'c,f':0.403, 'c2,x':-13,'c2,y':1,'c2,f':1}, doNotFit=['p,ud','c,ud','c,x','c,y','c,f']) #, prior=[('c,x**2 + c,y**2','<=',10)])
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
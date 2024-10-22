#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:09:25 2024

@author: bencb


"""

#-- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import pmoired
import glob
import matplotlib.pyplot as plt 
import pickle
import os 
import pandas as pd

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
#save_path = '/Users/bencb/Documents/long_secondary_periods/pmoired_fits/binary/'
#data_path = '/home/bcourtne/data/rt_pav/'
#save_path = '/home/bcourtne/data/rt_pav/pmoired_fits/BINARY/'
data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'

ID='binary' #type of fit 
save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

OUTSIDE_UD = False # do we constrain the fit to outside the measured UD diameter at the central wavelength?


wvl_band_dict = {'H':[1,2]}

feature='H'


logV2 = True

if __name__ == '__main__': 
    
    fig_inx = 1 # index figs
    ins = 'pionier'
    
    pionier_files = glob.glob(data_path+f'{ins}/*.fits')
    
    
    oi = pmoired.OI(pionier_files)
    
    
    #=========== for plotting 
    # filter for the wavelengths we are looking at 
    wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])
    
    wvls = oi.data[0]['WL'][wvl_filt][:-1] # avelengths to consider
    #===========
        
    # check wls 
    wvl0 = 1.6
    wl_thresh = 1 
    ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

    print(f'number wavelengths to fit = {sum( (oi.data[0]["WL"] > wvl0  - wl_thresh) & (oi.data[0]["WL"] < wvl0  + wl_thresh) )}')
    
    
    # -- smallest lambda/B in mas (first data set) 
    step = 180*3600*1000e-6/np.pi/(  max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']]) )
    
    # -- spectral resolution (first data set) 
    R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])
    
    print('step: %.1fmas, range: +- %.1fmas'%(step, R/2*step))
    
    # ======== NAMING FOR OUTPUT FILES 
    ID = f'{ins}_OUTSIDE_UD-{OUTSIDE_UD}_step-{round(step,2)}_R-{round(R)}_wvlRange-{round(wvl0-wl_thresh,3)}_{round(wvl0+wl_thresh,3)}'
    
    mat_feat_folder = f'{ins}_{feature}/'
    if not os.path.exists(save_path_0 + mat_feat_folder):
        os.makedirs(save_path_0 + mat_feat_folder)
    else:
        print('path already exists - overriding data in this folder! ')
        
    
    save_path = save_path_0+mat_feat_folder
    current_folder = save_path
    
    """
    if not os.path.exists(save_path + ID):
        os.makedirs(save_path + ID +'/')
    else:
        print('path already exists - overriding data in this folder! ')
        
    current_folder = save_path + ID +'/'"""
    # ========================
    
    
    # -- initial model dict: 'c,x' and 'c,y' do not matter, as they will be explored in the fit
    param = {'*,ud':ud_wvl, '*,f':1, 'c,f':0.01, 'c,x':4, 'c,y':4, 'c,ud':0.}
    
    # -- define the exploration pattern
    expl = {'grid':{'c,x':(-R/2*step, R/2*step, step), 'c,y':(-R/2*step, R/2*step, step)}}
    
    # -- setup the fit, as usual
    oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[(wvl0-wl_thresh, wvl0+wl_thresh)]})
    #oi.setupFit({'obs':['V2', 'T3PHI']})
    
    # -- actual grid fit
    if not OUTSIDE_UD:
        oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
                   constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                              ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])
    
    else:
        # TO fit outside UD diameter 
        oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
                   constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                              ('np.sqrt(c,x**2+c,y**2)', '>', ud_wvl) ])
        
    
    """ grid fit on resolved companion 
    
    oi.gridFit(expl, model=param, doNotFit=[], prior=[('c,f', '<', 1)], 
               constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                          ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])
    """
    # -- show chi2 grid
    oi.showGrid(interpolate = True, legend=False,tight=True)
    plt.savefig( current_folder + f'{ins}_pmoired_binary_GRIDSEARCH_{ID}.png', bbox_inches='tight', dpi=300 )
    
    
    # fit the model 
    oi.doFit(doNotFit=['*,f', 'c,ud'])
    
    
    
         
    # ============  PLOT BEST FIT VS V2 AND CP
    
    model_col = 'orange'
    obs_col= 'grey'
    fsize=18
    
    # V2
    badflag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) & ((oi._merged[0]['OI_VIS2']['all']['EV2']<0.6).reshape(-1))
    
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
    
    
    if logV2:
        frame1.set_yscale('log')
    #frame1.text(10,0.2,feature,fontsize=15)
    
    
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks( [])
    frame1.legend(fontsize=fsize)
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    plt.savefig( current_folder + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
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
    
    
    #frame1.text(10,10,feature,fontsize=15)
    #if logV2:
    #    plt.yscale('log')
    frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    
    frame2.set_yscale('log')
    frame1.legend(fontsize=fsize)
    frame1.set_xticks( [])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')    
    
    plt.savefig( current_folder + f'{ins}_{feature}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
           
        
    
    # bootstrapping ! 
    oi.bootstrapFit(300)
    
    oi.showBootstrap()
    plt.savefig( current_folder + f'{ins}_pmoired_binary_BOOTSTRAPPING_{ID}.png', bbox_inches='tight', dpi=300 )
    
    
    with open(current_folder+f'{ins}_pmoired_binary_model_BESTFIT_{ID}.pickle', 'wb') as handle:
        pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(current_folder+f'{ins}_pmoired_binary_model_{ID}.pickle', 'wb') as handle:
        pickle.dump(oi._model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(current_folder+f'{ins}_pmoired_binary_BOOTSTRAPPED_{ID}.pickle', 'wb') as handle:
        pickle.dump(oi.boot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    print('---- detection limit')
    # best model
    best_test_injection = oi.bestfit['best'] |  {'3,ud':0, '3,x':-15, '3,y':5, '3,f':0.01}
    
    expl = {'rand':{'3,x':(-R*step/2, R*step/2), '3,y':(-R*step/2, R*step/2)}}

    oi.detectionLimit(expl, '3,f', model=best_test_injection, Nfits=500, nsigma=3,\
                 constrain=[('np.sqrt(3,x**2+3,y**2)', '<=', R*step/2 ),\
                            ('np.sqrt(3,x**2+3,y**2)', '>', step/2) ])
    
    oi.showLimGrid(mag=1)
    plt.savefig( current_folder + f'{ins}_pmoired_DETECTION_LIMIT_{ID}.png', bbox_inches='tight', dpi=300)
    
    
    
    
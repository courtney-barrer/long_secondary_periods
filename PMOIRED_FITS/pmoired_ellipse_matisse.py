#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:35:46 2024

@author: bcourtne

https://github.com/amerand/PMOIRED/blob/master/Model%20definitions%20and%20examples.ipynb
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

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = '/home/rtc/Documents/long_secondary_periods/data/'

ins = 'MATISSE'

ID='ellipse' #type of fit 

save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

#feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)


matisse_files_L = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

matisse_files_N = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

#cannot read 2022-07-28T004853_VRTPav_A0B2D0C1_IR-N_LOW_noChop_cal_oifits_0.fits' so ignore it 
#matisse_files_N.pop(9) - this was put in bad data folder 




# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]


wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_short':[8,9],'N_mid':[9,10],'N_long':[10,13]}

min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 100 
max_rel_CP_error = 20 #deg

# Check constraints !!! 
"""
for proj in np.linspace(,90,8):
    
    for incl in np.linspace(-90,90,5):
        
            oi.showModel({'ud':4.0, 'incl':incl+90, 'projang':proj},imFov=20,showSED=False)

# -90 < projang < 90, 0 < inc < 90
# not this is only tru if uniform!!!!
"""
prior = [('inc', '<=', 90),('inc', '>=', 0),('projang', '<', 90),('projang', '>=', -90)]


cont_fits = [] # to hold bestfit from each feature to print at end 
fig_inx = 1
for feature in wvl_band_dict  :

    # ============ SETUP
    if 'N_' in feature:
        matisse_files = matisse_files_N

    else:
        matisse_files = matisse_files_L


    print(f'\n\n=============\nFITTING {feature} FEATURE\n\n==============\n')
    
    mat_feat_folder = f'{ins}_{feature}/'
    if not os.path.exists(save_path_0 + mat_feat_folder):
        os.makedirs(save_path_0 + mat_feat_folder)
    else:
        print('path already exists - overriding data in this folder! ')
        
    save_path = save_path_0+mat_feat_folder
        

    oi = pmoired.OI(matisse_files , binning = 1 ) # insname='GRAVITY_SC_P1',

    # filter for the wavelengths we are looking at 
    wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])

    wvls = oi.data[0]['WL'][wvl_filt][:-1] # avelengths to consider
    
    # ============ FIT PER WVL 
    res = {}
    
    
    for wvl0, wvl1 in zip( oi.data[0]['WL'][wvl_filt][:-1], oi.data[0]['WL'][wvl_filt][1:]):
        oi.setupFit({'obs':['V2', 'T3PHI'], 
                     'min relative error':{'V2':0.01},
                     'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                     'wl ranges':[(wvl0, wvl1)]})
        
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - 1e-6 * wvl0)) ]
        oi.doFit( {'ud':ud_wvl, 'incl':45, 'projang':45} , prior=prior ) # {'ud':1.0, 'incl':60, 'projang':30} )#{'ud':8.5})
        res[wvl0] = oi.bestfit
        
        
    # ============ SAVE RESULTS IN CSV

    
    fig1 = plt.figure(1*fig_inx,figsize=(10,8))
    fig1.set_tight_layout(True)
    
    ud = np.array( [res[w]['best']['ud'] for w in res])   
    uderr = np.array( [res[w]['uncer']['ud'] for w in res] )
    inc = np.array( [res[w]['best']['incl'] for w in res])  
    incerr = np.array( [res[w]['uncer']['incl'] for w in res] )
    proj = np.array( [res[w]['best']['projang'] for w in res])  
    projerr = np.array( [res[w]['uncer']['projang'] for w in res] )
    redchi2 = np.array( [res[w]['chi2'] for w in res])   
    
    df = pd.DataFrame([wvls, ud, uderr, inc, incerr, proj, projerr, redchi2], index = ['wvls', 'ud', 'uderr', 'inc', 'incerr', 'proj', 'projerr', 'redchi2'])
    df.T.to_csv(save_path + f'{ID}_fit_pmoired_results_{ins}_{feature}.csv')
    
    fig1 = plt.figure(1*fig_inx,figsize=(10,12))
    fig1.set_tight_layout(True)
    
    fontsize=20
    
    frame1 = fig1.add_axes((.1,.7,.8,.3))
    frame2 = fig1.add_axes((.1,.4,.8,.3))
    frame3 = fig1.add_axes((.1,.1,.8,.3))
    frame4 = fig1.add_axes((.1,.0,.8,.1))
    
    frame1.errorbar(wvls, ud , yerr=uderr, color = 'k', lw = 2)
    frame1.set_ylabel(r'$\theta_{UD}$\n[mas]',fontsize=fontsize)
    frame1.tick_params(labelsize=fontsize)
    #frame1.set_yscale('log')
    frame2.errorbar(wvls, inc  , yerr=uderr, color = 'k', lw = 2)
    frame2.set_ylabel(r'inclination\n[deg]',fontsize=fontsize)
    frame2.tick_params(labelsize=fontsize)
    
    frame3.errorbar(wvls, proj  , yerr=uderr, color = 'k', lw = 2)
    frame3.set_ylabel(r'projection\nangle [deg]',fontsize=fontsize)
    frame3.tick_params(labelsize=fontsize)
    frame3.set_ylim( [-90,130] )
        
    frame4.errorbar(wvls, redchi2  , yerr=uderr, color = 'k', lw = 2)
    frame4.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
    frame4.tick_params(labelsize=fontsize)
    
    plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_PARAMS_v_WVL_{ID}.png', bbox_inches='tight', dpi=300)  
 
    # ============  PLOT BEST FIT VS V2 AND CP
    
    model_col = 'orange'
    obs_col= 'grey'
    fsize=18
    
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
    
    plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
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
    
    
    plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
           

    
    #final fit
    
    #  ============  FIT FINAL MODEL IN CONTIUUM 
    
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                     'min relative error':{'V2':min_rel_V2_error, 'CP':min_rel_CP_error},
                     'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                     'wl ranges':[wvl_band_dict[feature]]})
    
    

    w_idx  = np.argmin( abs(np.mean(wvl_band_dict[feature])-wvls))
    oi.doFit( res[wvls[w_idx]]['best'], prior = prior )
    
    oi.showModel( oi.bestfit['best'], imFov = oi.bestfit['best']['ud'] * 2, showSED=False)
    plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_IM_{ID}.png', bbox_inches='tight', dpi=300)
    
    with open(save_path+f'{ins}_{feature}_pmoired_model_BESTFIT.pickle', 'wb') as handle:
        pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    cont_fits.append( oi.bestfit )

    #_=input(f'{feature} results - copy')
    
    fig_inx *= 10
    
for k , a in zip ( wvl_band_dict, cont_fits):
    print(k)
    print('best parameters',a['best'])
    print('uncert parameters',a['uncer'])
    print('redchi2',a['chi2'])
    
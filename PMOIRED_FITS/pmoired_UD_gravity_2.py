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
data_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/'


ins = 'gravity'


ID='UD' #type of fit 


save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'


#feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)


# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]


wvl_band_dict =  {'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.5 
max_rel_CP_error = 20 #deg

cont_fits = [] # to hold bestfit from each feature to print at end 
fig_inx = 1
for feature in wvl_band_dict  :




    print(f'\n\n=============\nFITTING {feature} FEATURE\n\n==============\n')
    
    mat_feat_folder = f'{ins}_{feature}/'
    if not os.path.exists(save_path_0 + mat_feat_folder):
        os.makedirs(save_path_0 + mat_feat_folder)
    else:
        print('path already exists - overriding data in this folder! ')
        
    save_path = save_path_0+mat_feat_folder
        

    oi = pmoired.OI(gravity_files, binning = 1,insname='GRAVITY_SC_P1' ) # insname='GRAVITY_SC_P1',

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
        oi.doFit( {'ud':ud_wvl} ) # {'ud':1.0, 'incl':60, 'projang':30} )#{'ud':8.5})
        res[wvl0] = oi.bestfit
        
        
    # ============ SAVE RESULTS IN CSV
    
    ud = np.array( [res[w]['best']['ud'] for w in res])    
    uderr = np.array( [res[w]['uncer']['ud'] for w in res])   
    redchi2 = np.array( [res[w]['chi2'] for w in res])      

    df = pd.DataFrame([wvls, ud, uderr, redchi2], index = ['wvls', 'ud', 'uderr', 'redchi2'])
    df.T.to_csv(save_path + f'{ID}_fit_pmoired_results_{ins}_{feature}.csv')
    
    
    fig1 = plt.figure(1*fig_inx,figsize=(10,8))
    fig1.set_tight_layout(True)
    
    frame1=fig1.add_axes((.1,.3,.8,.6))
    frame2=fig1.add_axes((.1,.05,.8,.2))   
    
    
            
    # ============ PLOT FITTED PARAMETERS PER WVL 
    frame1.errorbar(wvls, ud, yerr=uderr, color = 'k', fmt='-o', lw = 2)
    #frame1.set_yscale('log')
    frame2.plot(wvls, redchi2, '-',lw=2, color='k')
    
    
    fontsize=20
    #frame1.set_title('RT Pav Uniform Disk Fit vs Wavelength')
    frame1.grid()
    frame1.legend(fontsize=fontsize)
    frame1.set_ylabel(r'$\theta_{UD}$ [mas]',fontsize=fontsize)
    frame1.tick_params(labelsize=fontsize)
    frame1.set_xticklabels([]) 
    
    frame2.grid()
    frame2.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
    frame2.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
    frame2.tick_params(labelsize=fontsize)
    
    plt.tight_layout()
    
    
 
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
    frame1.text(10,0.2,feature,fontsize=15)
    
    
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks( [])
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
    
    
    frame1.text(10,10,feature,fontsize=15)
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
    
    plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
           

    
    #final fit
    
    #  ============  FIT FINAL MODEL IN CONTIUUM 
    
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                     'min relative error':{'V2':min_rel_V2_error, 'CP':min_rel_CP_error},
                     'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                     'wl ranges':[wvl_band_dict[feature]]})
    

    w_idx  = np.argmin( abs(np.mean(wvl_band_dict[feature])-wvls))
    oi.doFit( res[wvls[w_idx]]['best'] )
    
    with open(save_path+f'{ins}_{feature}_pmoired_model_BESTFIT.pickle', 'wb') as handle:
        pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    cont_fits.append( oi.bestfit )

    #_=input(f'{feature} results - copy')
    
    fig_inx *= 10
    
for a in cont_fits:

    print('best parameters',a['best'])
    print('uncert parameters',a['uncer'])
    print('redchi2',a['chi2'])
    
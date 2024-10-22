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
save_path_0 = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/UD/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'


ins='pionier'
feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

pionier_files = glob.glob(data_path+f'{ins}/*.fits')


oi = pmoired.OI(pionier_files , binning = 1 ) # insname='GRAVITY_SC_P1',



#[{'ud':0.5, 'x':.5, 'y':-0.2}, 
#              {'ud':1.0, 'incl':60, 'projang':30} 
#             ]:
    


grav_feat_folder = f'{ins}_{feature}/'
if not os.path.exists(save_path_0 + grav_feat_folder):
    os.makedirs(save_path_0 + grav_feat_folder)
else:
    print('path already exists - overriding data in this folder! ')
    
save_path = save_path_0 +  grav_feat_folder


res = {}
for wvl0, wvl1 in zip( oi.data[0]['WL'][:-1], oi.data[0]['WL'][1:]):
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[(wvl0, wvl1)]})
    
    oi.doFit( {'ud':4.5} ) # {'ud':1.0, 'incl':60, 'projang':30} )#{'ud':8.5})
    res[wvl0] = oi.bestfit


#%%

spectral_features = {'continuum':[1,2]}
                                   
wvls = oi.data[0]['WL'][:-1]
ud = np.array( [res[w]['best']['ud'] for w in res])    
uderr = np.array( [res[w]['uncer']['ud'] for w in res])   
redchi2 = np.array( [res[w]['chi2'] for w in res])  



"""
plt.figure(figsize=(8,5))
plt.errorbar( wvls, ud , yerr=uderr, color='k')
#for f,wR in spectral_features.items():
#    plt.gca().fill_between(wvls , min(ud), max(ud), where=(wvls < wR[1]) & (wvls > wR[0]), alpha=0.3,label=f)

plt.legend(fontsize=15,bbox_to_anchor=(1,1))
plt.xlabel('wavelength [$\mu$m]',fontsize=15)
plt.ylabel('UD [mas]',fontsize=15)
plt.gca().tick_params(labelsize=15) 
#plt.savefig() 
"""


fig1 = plt.figure(1,figsize=(10,8))
fig1.set_tight_layout(True)

frame1=fig1.add_axes((.1,.3,.8,.6))
frame2=fig1.add_axes((.1,.05,.8,.2))   


        
# plot it
frame1.errorbar(wvls, ud, yerr=uderr, color = 'k', fmt='-o', lw = 2)
frame1.set_yscale('log')
frame2.semilogy(wvls, redchi2, '-',lw=2, color='k')


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


      

ID = feature.split('_')[0]
model_col = 'orange'
obs_col= 'grey'
fsize=18
plt.figure(figsize=(8,5)) 
flag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

#wvl_filt0 = (oi._model[0]['WL'] > spectral_features['continuum'][0]) & (oi._model[0]['WL'] < spectral_features['continuum'][1]) #
#wvl_filt = np.array( [wvl_filt0 for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] )
#flag_filt*= wvl_filt.reshape(-1) 

# data 
plt.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
# model
plt.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

#if logV2:
#    plt.yscale('log')
plt.xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
plt.ylabel(r'$V^2$',fontsize=fsize)
plt.legend(fontsize=fsize)
plt.gca().tick_params( labelsize=fsize )

plt.savefig( save_path + f'{ins}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)


plt.figure(figsize=(8,5))
flag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1)) 
# data 
plt.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col,label='obs',alpha=0.9,fmt='.')
# model
plt.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
plt.xlabel(r'$B_{max}/\lambda [M rad^{-1}]$',fontsize=fsize)
plt.ylabel('CP [deg]',fontsize=fsize)
plt.legend(fontsize=fsize)
plt.gca().tick_params( labelsize=fsize )

plt.savefig( save_path + f'{ins}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)


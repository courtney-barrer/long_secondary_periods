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
save_path_0 = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/LD/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')

ins = 'pionier'
feature = 'LD_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

pionier_files = glob.glob(data_path+f'{ins}/*.fits')

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)



oi = pmoired.OI(pionier_files , binning = 1 ) 



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
    
    ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]
    
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[(wvl0, wvl1)]})
    
    oi.doFit( {'diam':ud_wvl, 'profile':'$MU**$alpha', 'alpha':0.5} ) # {'ud':1.0, 'incl':60, 'projang':30} )#{'ud':8.5})
    res[wvl0] = oi.bestfit


#%%

spectral_features = {'continuum':[1,2]}
                                   
wvls = oi.data[0]['WL'][:-1]
ud = np.array( [res[w]['best']['diam'] for w in res])    
alpha = np.array( [res[w]['best']['alpha'] for w in res])    
uderr = np.array( [res[w]['uncer']['diam'] for w in res])   
alphaerr = np.array( [res[w]['uncer']['alpha'] for w in res])   
redchi2 = np.array( [res[w]['chi2'] for w in res])  

df = pd.DataFrame( [wvls, ud, uderr, alpha, alphaerr, redchi2] , index = ['wvls', 'ud', 'uderr', 'alpha', 'alphaerr', 'redchi2'] ).T

df.to_csv(save_path + f'LD_per_wvl_pmoired_{ins}_results.csv') 

#filt = abs(alpha
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


fig1 = plt.figure(1,figsize=(10,12))
fig1.set_tight_layout(True)

frame1 = fig1.add_axes((.1,.6,.8,.4))
frame2 = fig1.add_axes((.1,.2,.8,.4))
#frame3 = fig1.add_axes((.1,.1,.8,.3))
frame3 = fig1.add_axes((.1,.0,.8,.2))

#param_labels=['alpha','theta']

        
# plot it
frame1.errorbar(wvls, ud, yerr=uderr, color = 'k', fmt='-o', lw = 2, alpha=0.5)
#frame1.set_yscale('log')
frame2.errorbar(wvls, alpha, yerr=alphaerr, color = 'k', fmt='-o', lw = 2, alpha=0.5)
frame3.plot(wvls, redchi2, '-',lw=2, color='k', alpha=1)


fontsize=20
#frame1.set_title('RT Pav Uniform Disk Fit vs Wavelength')
frame1.grid()
frame1.legend(fontsize=fontsize)
frame1.set_ylabel(r'$\theta_{LD}$ [mas]',fontsize=fontsize)
frame1.tick_params(labelsize=fontsize)
frame1.set_xticklabels([]) 
frame1.set_ylim([2,20])

frame2.grid()
frame2.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
frame2.set_ylabel(r'$\alpha$',fontsize=fontsize)
frame2.tick_params(labelsize=fontsize)
frame2.set_ylim([-1,10])

frame3.grid()
frame3.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
frame3.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
frame3.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig(save_path + f'LD_per_wvl_pmoired_{ins}.png',dpi=300, bbox_inches='tight')
  


#  FIT IN CONTIUUM

oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[spectral_features['continuum']]})

#constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2)
oi.doFit( res[wvls[np.argmin( abs(np.mean(oi.data[0]['WL'])-oi.data[0]['WL']))]]['best'] )

with open(save_path +f'{ins}_pmoired_LD_model_BESTFIT.pickle', 'wb') as handle:
    pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""

# -- reduced chi2: 14.888636233344217
{'alpha':  1.757, # +/- 0.025
'diam':   4.277, # +/- 0.010
'profile':'$MU**$alpha',

"""


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

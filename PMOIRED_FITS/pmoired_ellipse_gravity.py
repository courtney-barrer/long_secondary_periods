#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:35:46 2024

@author: bcourtne

https://github.com/amerand/PMOIRED/blob/master/Model%20definitions%20and%20examples.ipynb



Need to do this just for the conti using all wavelengths 
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
import json
comp_loc = 'ANU'
path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = path_dict[comp_loc]["data"] #'/home/rtc/Documents/long_secondary_periods/data/'
save_path_0 = path_dict[comp_loc]["root"] + "PMOIRED_FITS/ellipse/" #'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/ellipse/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

gravity_files = glob.glob(data_path+'gravity/data/*.fits')

ins='GRAVITY'
feature='ellipse_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)



oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning = 100 ) 



#[{'ud':0.5, 'x':.5, 'y':-0.2}, 
#              {'ud':1.0, 'incl':60, 'projang':30} 
#             ]:
    

spectral_features = {'continuum':[2.1,2.19],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
    
    
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
                 'wl ranges':[(wvl0, wvl1)]}),
                #constrain=[('inc', '<=', 90),('inc', '>=', 90),('projang', '<=', 90),('projang', '>=', -90)])
    
    oi.doFit( {'ud':4.0, 'incl':45, 'projang':45} , prior=[('inc', '<=', 90),('inc', '>=', -90),('projang', '<=', 90),('projang', '>=', 0)]) #{'ud':8.5})
    res[wvl0] = oi.bestfit



#for w in  oi.data[0]['WL'][::100]:
#    oi.showModel( res[wvl0]['best'], showSED=False, imFov=20 )



#%%
                                   
wvls = oi.data[0]['WL'][:-1]
ud = np.array( [res[w]['best']['ud'] for w in res])   
uderr = np.array( [res[w]['uncer']['ud'] for w in res] )
inc = np.array( [res[w]['best']['incl'] for w in res])  
incerr = np.array( [res[w]['uncer']['incl'] for w in res] )
proj = np.array( [res[w]['best']['projang'] for w in res])  
projerr = np.array( [res[w]['uncer']['projang'] for w in res] )
redchi2 = np.array( [res[w]['chi2'] for w in res])   

df = pd.DataFrame([wvls, ud, uderr, inc, incerr, proj, projerr, redchi2], index = ['wvls', 'ud', 'uderr', 'inc', 'incerr', 'proj', 'projerr', 'redchi2'])
df.T.to_csv(save_path + 'ellipse_fit_pmoired_results.csv')

fig1 = plt.figure(1,figsize=(10,12))
fig1.set_tight_layout(True)

fontsize=20

frame1 = fig1.add_axes((.1,.7,.8,.3))
frame2 = fig1.add_axes((.1,.4,.8,.3))
frame3 = fig1.add_axes((.1,.1,.8,.3))
frame4 = fig1.add_axes((.1,.0,.8,.1))

frame1.errorbar(wvls, ud , yerr=uderr, color = 'k', lw = 2)
frame1.set_ylabel(r'$\theta_{UD}$ [mas]',fontsize=fontsize)
frame1.tick_params(labelsize=fontsize)
#frame1.set_yscale('log')
frame2.errorbar(wvls, inc  , yerr=uderr, color = 'k', lw = 2)
frame2.set_ylabel(r'inclination [deg]',fontsize=fontsize)
frame2.tick_params(labelsize=fontsize)

frame3.errorbar(wvls, proj  , yerr=uderr, color = 'k', lw = 2)
frame3.set_ylabel(r'projection angle [deg]',fontsize=fontsize)
frame3.tick_params(labelsize=fontsize)
frame3.set_ylim( [-90,130] )
    
frame4.errorbar(wvls, redchi2  , yerr=uderr, color = 'k', lw = 2)
frame4.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
frame4.tick_params(labelsize=fontsize)

#plt.savefig(save_path + f'ellipse_per_wvl_pmoired_{ins}.png',dpi=300, bbox_inches='tight')


# looking at best fit images 
#for w in wvls[::50]:
#    oi.showModel(res[w]['best'], WL=w, imFov=8, showSED=False)

#constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2)


 
# FIT FINAL MODEL IN CONTIUUM 

oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[spectral_features['continuum']]})

#constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2)
oi.doFit( res[wvls[np.argmin( abs(2.14-oi.data[0]['WL']))]]['best'] )

with open(save_path+f'{ins}_pmoired_ELLIPSE_model_BESTFIT.pickle', 'wb') as handle:
    pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)



        # e.g. can plot / saave original data vs model

ID = feature.split('_')[0]
model_col = 'orange'
obs_col= 'grey'
fsize=18
plt.figure(figsize=(8,5)) 
flag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

wvl_filt0 = (oi._model[0]['WL'] > spectral_features['continuum'][0]) & (oi._model[0]['WL'] < spectral_features['continuum'][1]) #
wvl_filt = np.array( [wvl_filt0 for _ in range(oi._merged[0]['OI_VIS2']['all']['FLAG'].shape[0] )] )

flag_filt*= wvl_filt.reshape(-1) 
# data 
plt.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.8,fmt='.')
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
plt.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col,label='obs',alpha=0.1,fmt='.')
# model
plt.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
plt.xlabel(r'$B_{max}/\lambda [M rad^{-1}]$',fontsize=fsize)
plt.ylabel('CP [deg]',fontsize=fsize)
plt.legend(fontsize=fsize)
plt.gca().tick_params( labelsize=fsize )

plt.savefig( save_path + f'{ins}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)

#%% 


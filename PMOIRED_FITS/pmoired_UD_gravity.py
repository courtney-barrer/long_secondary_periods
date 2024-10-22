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

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')

ins='GRAVITY'
feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)



oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning = 1 ) 



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

spectral_features = {'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
                                   
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


"""
{'continuum':[2.1,2.19],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
"""        



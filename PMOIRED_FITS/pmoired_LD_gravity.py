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
save_path_0 = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/LD/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')

ins='GRAVITY'
feature='LD_per_wvl'
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
    
    ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]
    
    oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[(wvl0, wvl1)]})
    
    oi.doFit( {'diam':ud_wvl, 'profile':'$MU**$alpha', 'alpha':0.5} ) # {'ud':1.0, 'incl':60, 'projang':30} )#{'ud':8.5})
    res[wvl0] = oi.bestfit


#%%

spectral_features = {'continuum':[2.1,2.19],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
                                   
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

"""
{'continuum':[2.1,2.19],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
"""        


#%%  FIT IN CONTIUUM

oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[spectral_features['continuum']]})

#constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2)
oi.doFit( res[wvls[np.argmin( abs(2.14-oi.data[0]['WL']))]]['best'] )

with open(save_path +f'{ins}_pmoired_LD_model_BESTFIT.pickle', 'wb') as handle:
    pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""

# -- reduced chi2: 14.888636233344217
{'alpha':  1.757, # +/- 0.025
'diam':   4.277, # +/- 0.010
'profile':'$MU**$alpha',

"""

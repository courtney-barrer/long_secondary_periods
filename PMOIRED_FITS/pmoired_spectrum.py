#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:35:46 2024

@author: bcourtne

https://github.com/amerand/PMOIRED/blob/master/Model%20definitions%20and%20examples.ipynb


1. FIT  UD PER WVL FOR ALL WVLS , GRAPH RESULTS HIGHLIGHTING SPECTRAL FEATURES
2. FIT UD IN CONTIUUM [2.1,2.19]um FOR ALL WAVELENGTHS. PUT RESULTS IN TABLE



thesis here:
    Koutoulaki, Maria-Kalliopi
    A closer look at protoplanetary discs : the inner few au probed with spectroscopy and optical interferometry
    
    https://researchrepository.ucd.ie/server/api/core/bitstreams/72b35026-1cb5-40c6-a266-4b93cd243935/content
    
    https://researchrepository.ucd.ie/server/api/core/bitstreams/72b35026-1cb5-40c6-a266-4b93cd243935/content
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
from pmoired import tellcorr



def showTellurics(filename, fig=99, plot=False):
    h = fits.open(filename)
    if not 'TELLURICS' in h:
        print('no tellurics model found')
        return 
    if not fig is None:
        plt.close(fig)
        plt.figure(fig, figsize=(9,6))

    if plot:
        ax1 = plt.subplot(211)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['RAW_SPEC'],
                 '-', alpha=0.5, label='raw spectrum', color='orange', lw=2)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['CORR_CONT'],
                 ':g', alpha=0.5, label='estimated continuum', lw=1)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['RAW_SPEC']/h['TELLURICS'].data['TELL_TRANS'],
                 '-k', label='corrected spectrum', lw=1)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['TELL_TRANS']*h['TELLURICS'].data['CORR_CONT'], 
                 '-b', label='telluric*continuum (PWV=%.2fmm)'%h['TELLURICS'].header['PWV'], 
                 alpha=0.5, lw=1)
    
        plt.legend()
        plt.ylabel("flux (arb. unit)")
    
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
             h['TELLURICS'].data['RAW_SPEC']/h['TELLURICS'].data['CORR_CONT'],
              '-', alpha=0.5, label='raw normalised spectrum', color='orange', lw=2)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['TELL_TRANS'],
                 '-b', label='telluric model (PWV=%.2fmm)'%h['TELLURICS'].header['PWV'], 
                 alpha=0.5, lw=1)
        plt.plot(h['TELLURICS'].data['EFF_WAVE']*1e6,
                 h['TELLURICS'].data['RAW_SPEC']/
                 h['TELLURICS'].data['TELL_TRANS']/h['TELLURICS'].data['CORR_CONT'],
                 '-k', label='corrected spectrum', lw=1)
        plt.legend()
        plt.ylabel("normalised flux")
        plt.xlabel('wavelength ($\mu$m)')
        plt.suptitle(os.path.basename(filename), fontsize=8)
    
        plt.tight_layout()

    
    wvls =  h['TELLURICS'].data['EFF_WAVE']*1e6
    corr_norm_flux = h['TELLURICS'].data['RAW_SPEC']/ h['TELLURICS'].data['TELL_TRANS']/h['TELLURICS'].data['CORR_CONT']

    h.close()

    return( wvls, corr_norm_flux )
    
#from multiprocessing import Pool

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
save_path_0 = '/Users/bencb/Documents/long_secondary_periods/PMOIRED_FITS/GRAVITY_telluric/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')

ins='GRAVITY'
feature='telluric'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

#ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

### =========== CORRECTING HERE 
#for f in gravity_files :
#    tellcorr.gravity( f )
    
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning = 1 ) 


spectral_features = {'FeII':[1.98,2.02], 'HeI':[2.038, 2.078], 'H2':[2.105, 2.118], 'MgII':[2.130, 2.150],'Brg':[2.160, 2.172],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261],  'CO(2-0)':[2.292, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}


file = gravity_files[1]

plt.figure(figsize=(8,5))
for file in gravity_files: 
    print(file)
    wvls, corr_norm_flux = showTellurics(file, fig=99, plot=False)
    #wvl_list.append( wvls )
    #norm_flux_list.append( norm_flux_list )
    plt.plot( wvls, corr_norm_flux , color='k',alpha=0.9 , lw=0.5 ) 
    
for f,wR in spectral_features.items():
    plt.gca().fill_between(wvls , np.min(0), np.max(2), where=(wvls < wR[1]) & (wvls > wR[0]), alpha=0.3,label=f)
plt.legend(bbox_to_anchor=(1,1),fontsize=15)
plt.axhline( 1 , color='k')
plt.ylabel( 'Normalized flux',fontsize=15)
plt.xlabel( r'wavelength [$\mu$m]',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.xlim([1.99,2.41])
plt.ylim([0.6,1.4])

plt.savefig( save_path_0 + 'NORMALIZED_FLUX_WITH_SPECTRAL_LINES.png', dpi=300, bbox_inches = 'tight')


    
#%%
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


"""# -- remove tellurics
with Pool(pmoired.MAX_THREADS) as p:
    p.map(tellcorr.removeTellurics, files[:1])

# -- (re)compute tellurics
with Pool(pmoired.MAX_THREADS) as p:
    p.map(tellcorr.gravity, files[:1])
    """
    
#%% 

# 
# emmission 2.06, 
# abs 1.987, 2.294, 2.315, 2.370

spectral_features = {'Fe II':[1.98,2.02], 'HeI':[2.038, 2.078], 'HeI-other':[2.105, 2.118], 'MgII':[2.130, 2.150],'Brg':[2.160, 2.172],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
    
fidx = 0
#for fidx in range(len(oi.data)):
plt.figure() 
wvls = oi.data[0]['WL']
flux = oi._merged[0]['OI_FLUX']['all']['FLUX'].T 
plt.plot(wvls, flux  ,color='k',alpha=0.5)

for f,wR in spectral_features.items():
    plt.gca().fill_between(wvls , np.min(flux ), np.max(flux ), where=(wvls < wR[1]) & (wvls > wR[0]), alpha=0.3,label=f)


plt.legend(fontsize=15,bbox_to_anchor=(1,1))




# look at brg

for f in spectral_features:
    fidx = 0
    #for fidx in range(len(oi.data)):
    plt.figure() 
    wvls = oi.data[0]['WL']
    flux = oi._merged[fidx]['OI_FLUX']['all']['FLUX'].T 
    plt.plot(wvls, flux  ,color='k',alpha=0.5)
    
    #for f,wR in spectral_features.items():
    #    plt.gca().fill_between(wvls , np.min(flux ), np.max(flux ), where=(wvls < wR[1]) & (wvls > wR[0]), alpha=0.3,label=f)
    
    plt.title(f)
    plt.legend(fontsize=15,bbox_to_anchor=(1,1))
    
    if 'CO' in f:
        plt.xlim( [spectral_features[f][0]-0.02,spectral_features[f][0]+0.02] ) #spectral_features['Brg'] )
        
    else:
         plt.xlim( spectral_features[f] )   





# CO bandheads
fidx = 0
#for fidx in range(len(oi.data)):
plt.figure() 
wvls = oi.data[0]['WL']
flux = oi._merged[fidx]['OI_FLUX']['all']['FLUX'].T 
plt.plot(wvls, flux/np.mean(flux)  ,color='k',alpha=0.5)
plt.ylabel('flux [normalised]')
plt.xlabel('wavelength [um]')
plt.xlim( [2.280, 2.4] ) 
plt.ylim([0,1.5])
#plt.savefig( save_path + 'CO_bands_absorption_gravity.png', dpi=300)

# He 
fidx = 0
#for fidx in range(len(oi.data)):
plt.figure() 
plt.plot(wvls, flux/np.mean(flux)  ,color='k',alpha=0.5)
plt.ylabel('flux [normalised]')
plt.xlabel('wavelength [um]')
plt.xlim( [2.03, 2.08] ) 
#plt.savefig( save_path + 'HeI_emission_gravity.png', dpi=300)



# look at CLosure phases here 
plt.figure() 
wvls = oi.data[0]['WL']
CP = oi._merged[0]['OI_T3']['all']['T3PHI']
V2 = oi._merged[0]['OI_VIS2']['all']['V2']

#CP 
plt.figure()
plt.plot(wvls, CP.T ,color='k',alpha=0.5)
plt.ylabel('CP [deg]')
plt.xlabel('wavelength [um]')
plt.xlim( [2.03, 2.08] ) 
#plt.savefig( save_path_0 + 'HeI_emission_CP_gravity.png', dpi=300)

plt.figure()
plt.plot(wvls, CP.T ,color='k',alpha=0.5)
plt.ylabel('CP [deg]')
plt.xlabel('wavelength [um]')
plt.xlim( [2.280, 2.4] ) 
#plt.savefig( save_path_0 + 'CO_bands_emission_CP_gravity.png', dpi=300)
#V2
plt.figure()
plt.plot(wvls, V2.T ,color='k',alpha=0.5)
plt.ylabel('V2')
plt.xlabel('wavelength [um]')
plt.xlim( [2.03, 2.08] ) 
#plt.savefig( save_path_0 + 'HeI_emission_V2_gravity.png', dpi=300)

plt.figure()
plt.plot(wvls, V2.T ,color='k',alpha=0.5)
plt.ylabel('V2')
plt.xlabel('wavelength [um]')
plt.xlim( [2.280, 2.4] ) 
#plt.savefig( save_path_0 + 'CO_bands_emission_V2_gravity.png', dpi=300)

#%%





"""
{'continuum':[2.1,2.19],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                 'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
"""        



#%% NOW JUST FIT CONTIUUM FOR ALL WAVELENGTHS 


oi.setupFit({'obs':['V2', 'T3PHI'], 
                 'min relative error':{'V2':0.01},
                 'max relative error':{'V2':0.2},
                 'wl ranges':[(2.1,2.19)]})

oi.doFit( {'ud':4.5} )

oi.bestfit
#ud = oi.bestfit['best']['ud']  
#uderr = oi.bestfit['uncer']['ud'] 
#redchi2 = oi.bestfit['chi2']['ud'] 

with open(save_path +f'{ins}_pmoired_UD_model_cont_BESTFIT.pickle', 'wb') as handle:
    pickle.dump(oi.bestfit, handle, protocol=pickle.HIGHEST_PROTOCOL)

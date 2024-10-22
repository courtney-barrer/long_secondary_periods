#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:09:25 2024

@author: bencb

GRAVITY - have to work out how to bin the data, do contiumn and  H20, C0 bandheads 

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
save_path_0 = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/binary/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

gravity_files = glob.glob(data_path+'gravity/my_reduction_v3/*.fits')


#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

OUTSIDE_UD = True # do we constrain the fit to outside the measured UD diameter at the central wavelength?

logV2 = False # scale log of V^2 in plot

# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]



"""
for d in oi.data:
    print(d['filename'],'\n')
    for b in d['OI_VIS2']:
        print(b, 'mean err',np.mean( d['OI_VIS2'][b]['EV2']) )
        
# bad files with large V2 uncertainties 
GRAVI.2022-06-24T07:54:20.635_singlescivis_singlesciviscalibrated.fits 
GRAVI.2022-09-01T02:24:46.061_singlescivis_singlesciviscalibrated.fits 

# 2 nan values on uncertainty for J2A0, G1A0 baselines 
GRAVI.2022-09-01T02:10:40.025_singlescivis_singlesciviscalibrated.fits 

# RUN with these files and without them .. see difference 

colors= ['k','grey','b','g','r','y','m','pink','purple','brown']
for col,d in zip(colors,oi.data):
    print(d['filename'],'\n')
    plt.figure()
    plt.title(d['filename'].split('/')[-1])
    for i,b in enumerate(d['OI_VIS2']):
        plt.plot(d['OI_VIS2'][b]['B/wl'].reshape(-1),  d['OI_VIS2'][b]['V2'].reshape(-1),'.',color=col) #,label=d['filename'])
        #plt.plot(d['OI_VIS2'][b]['B/wl'].reshape(-1),  d['OI_VIS2'][b]['FLAG'].reshape(-1),'.',color=col) #,label=d['filename'])
        
        print( d['OI_VIS2'][b]['V2'].shape )
    
plt.legend()

"""        
if __name__ == '__main__': 
    #
    spectral_feature_dictionary_k = {'continuum':[2.1,2.3],'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                     'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO(2-0)':[2.2934, 2.298],\
                                       'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
        
    # DID I PUT CONTIUUM BACK IN DICTIONARY? 
    
    cont_binning = 60 #60 #spectral binnning in contiuum 
    
    for feature in spectral_feature_dictionary_k :

        ins = 'GRAVITY'
        
        print(f'\n\n=============\nFITTING {feature} FEATURE\n\n==============\n')
        
        grav_feat_folder = f'{ins}_{feature}/'
        if not os.path.exists(save_path_0 + grav_feat_folder):
            os.makedirs(save_path_0 + grav_feat_folder)
        #else:
        #    print('path already exists - overriding data in this folder! ')
            
        save_path = save_path_0 +  grav_feat_folder
            
        if feature == 'continuum': #if looking at contiuum  with use spectral binning
        
            
            oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning = cont_binning ) #, 
            
        else: 
          
            binning = 1 #np.max( [1, round(  cont_binning * abs(spectral_feature_dictionary_k[feature][1]- spectral_feature_dictionary_k[feature][0])/(abs(2.038 - 2.078)) ) ])
            
            print(f'\n\n======== using {binning} binning for {feature} \n\n')
            oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=binning ) #no spectral binning 
            

        
        # check wls 
        wvl0 = np.mean( spectral_feature_dictionary_k[feature] ) #2.166 # central wvl 
        wl_thresh = np.diff(spectral_feature_dictionary_k[feature] )[0]/2 #0.005  #+/- this value 
        
        # set the uniform disk diameter for the current central wavelength 
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]
        
        print(f'\n=====\nnumber wavelengths to fit = {sum( (oi.data[0]["WL"] > wvl0  - wl_thresh) & (oi.data[0]["WL"] < wvl0  + wl_thresh) )}\n====')
    
    
    
    
    
        # use this to constrain wavelength range... wrote in cont. and band wvls.. 
        
        
        # -- smallest lambda/B in mas (first data set) 
        step = 180*3600*1000e-6/np.pi/max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']])
        
        # -- spectral resolution (first data set) 
        R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])
        
        searchRad = 200 #mas 
        print('step: %.1fmas, range: +- %.1fmas'%(step,searchRad))
        
        # -- initial model dict: 'c,x' and 'c,y' do not matter, as they will be explored in the fit
        param = {'*,ud':ud_wvl, '*,f':1, 'c,f':0.01, 'c,x':0, 'c,y':0, 'c,ud':0.0}
        
        # -- define the exploration pattern
        expl = {'grid':{'c,x':(-searchRad, searchRad, step), 'c,y':(-searchRad, searchRad, step)}}
        
        # -- setup the fit, as usual
        oi.setupFit({'obs':['V2', 'T3PHI'],'wl ranges':[(wvl0 -wl_thresh, wvl0 +wl_thresh)]})
        # -- actual grid fit
        oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
                   constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                              ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])
        
        
        oi.showGrid()
        
        # fit the model 
        oi.doFit(doNotFit=['*,f', 'c,ud'])
    
    
    
    
        # ======== NAMING FOR OUTPUT FILES 
        ID = f'{ins}_spectralline_{feature}_OUTSIDE_UD-{OUTSIDE_UD}_step-{round(step,2)}_R-{round(R)}_wvlRange-{round(wvl0-wl_thresh,3)}_{round(wvl0+wl_thresh,3)}'
        
        if not os.path.exists(save_path + ID):
            os.makedirs(save_path + ID +'/')
        else:
            print('path already exists - overriding data in this folder! ')
            
        current_folder = save_path + ID +'/'
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
                                  ('np.sqrt(c,x**2+c,y**2)', '>', ud_wvl/2) ])
            
        
        """ grid fit on resolved companion 
        
        oi.gridFit(expl, model=param, doNotFit=[], prior=[('c,f', '<', 1)], 
                   constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                              ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])
        """
        # -- show chi2 grid
        oi.showGrid()
        plt.savefig( current_folder + f'{ins}_pmoired_binary_GRIDSEARCH_{ID}.png', bbox_inches='tight', dpi=300 )
        
        
        # fit the model 
        oi.doFit(doNotFit=['*,f', 'c,ud'])
        
        
        
        # e.g. can plot / saave original data vs model
        
        model_col = 'orange'
        obs_col= 'grey'
        fsize=18
        plt.figure(figsize=(8,5)) 
        flag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))
        # data 
        plt.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.8,fmt='.')
        # model
        plt.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

        if logV2:
            plt.yscale('log')
        plt.xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
        plt.ylabel(r'$V^2$',fontsize=fsize)
        plt.legend(fontsize=fsize)
        plt.gca().tick_params( labelsize=fsize )
        
        plt.savefig( current_folder + f'{ins}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
        
        
        plt.figure(figsize=(8,5))
        flag_filt = (~oi._merged[0]['OI_T3']['all']['FLAG'].reshape(-1)) 
        # data 
        plt.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col,label='obs',alpha=0.8,fmt='.')
        # model
        plt.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
        plt.xlabel(r'$B_{max}/\lambda [M rad^{-1}]$',fontsize=fsize)
        plt.ylabel('CP [deg]',fontsize=fsize)
        plt.legend(fontsize=fsize)
        plt.gca().tick_params( labelsize=fsize )
        
        plt.savefig( current_folder + f'{ins}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
        
        
        
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
        #best_test_injection = oi.bestfit['best'] |  {'3,ud':0, '3,x':-15, '3,y':5, '3,f':0.01}
        best_test_injection = {**oi.bestfit['best'] ,  **{'3,ud':0, '3,x':-15, '3,y':5, '3,f':0.01} }
        expl = {'rand':{'3,x':(-R*step/2, R*step/2), '3,y':(-R*step/2, R*step/2)}}
    
        oi.detectionLimit(expl, '3,f', model=best_test_injection, Nfits=500, nsigma=3,\
                     constrain=[('np.sqrt(3,x**2+3,y**2)', '<=', R*step/2 ),\
                                ('np.sqrt(3,x**2+3,y**2)', '>', step/2) ])
        
        oi.showLimGrid(mag=1)
        plt.savefig( current_folder + f'{ins}_pmoired_DETECTION_LIMIT_{ID}.png', bbox_inches='tight', dpi=300)
    
    
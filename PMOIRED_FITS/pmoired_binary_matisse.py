#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:09:25 2024

@author: bencb

matisse - have to work out how to bin the data, do contiumn and  H20, C0 bandheads 

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
save_path_0 = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/binary/'
#results_path = '/Users/bencb/Documents/long_secondary_periods/parameter_modelling/RESOLVED_BINARY_RESULTS/'

matisse_files_L = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

matisse_files_N = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

#cannot read 2022-07-28T004853_VRTPav_A0B2D0C1_IR-N_LOW_noChop_cal_oifits_0.fits' so ignore it 
#matisse_files_N.pop(9)


ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

OUTSIDE_UD = False # do we constrain the fit to outside the measured UD diameter at the central wavelength?

logV2 = False # scale log of V^2 in plot

# to check where errors are bad 
#[np.mean( [np.nanmedian( x['OI_VIS2'][b]['EV2'] ) for b in x['OI_VIS2']] ) for x in oi.data]


wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_short':[8,9],'N_mid':[9,10],'N_long':[10,13]}

ins = 'MATISSE'
        
"""
for d in oi.data:
    print(d['filename'],'\n')
    for b in d['OI_VIS2']:
        print(b, 'mean err',np.mean( d['OI_VIS2'][b]['EV2']) )

# RUN with these files and without them .. see difference 

#V2
for d in oi.data:
    print(d['filename'],'\n')
    plt.figure()
    plt.title(d['filename'].split('/')[-1])
    for i,b in enumerate(d['OI_VIS2']):

        flag = ~d['OI_VIS2'][b]['FLAG'].reshape(-1)

        Lfilt = (d['WL'] < 3.9) &  (d['WL'] > 3.2) 
        Mfilt = (d['WL'] < 5) &  (d['WL'] > 4.5) 
        try: 
            plt.plot(d['OI_VIS2'][b]['B/wl'].reshape(-1)[flag & Lfilt ],  d['OI_VIS2'][b]['V2'].reshape(-1)[flag & Lfilt],'.',color='orange') #,label=d['filename'])
            plt.plot(d['OI_VIS2'][b]['B/wl'].reshape(-1)[flag & Mfilt],  d['OI_VIS2'][b]['V2'].reshape(-1)[flag & Mfilt],'.',color='r') #,label=d['filename'])
            #plt.plot(d['OI_VIS2'][b]['B/wl'].reshape(-1),  d['OI_VIS2'][b]['FLAG'].reshape(-1),'.',color=col) #,label=d['filename'])
        except:
            print( f'2 col data shape issue with {d} on {b}')

#CP
for d in oi.data:
    print(d['filename'],'\n')
    plt.figure()
    plt.title(d['filename'].split('/')[-1])
    for i,b in enumerate(d['OI_T3']):

        flag = ~d['OI_T3'][b]['FLAG'].reshape(-1)

        Lfilt = (d['WL'] < 3.9) &  (d['WL'] > 3.2) 
        Mfilt = (d['WL'] < 5) &  (d['WL'] > 4.5) 
        #try: 
        #plt.plot(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Lfilt ],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Lfilt],'.',color='orange') #,label=d['filename'])
        #plt.plot(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Mfilt],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Mfilt],'.',color='r') #,label=d['filename'])

        plt.errorbar(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Lfilt ],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Lfilt], yerr= d['ET3PHI'][b]['ET3PHI'].reshape(-1)[flag & Lfilt],fmt='.',color='orange') #,label=d['filename'])
        plt.errorbar(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Mfilt],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Mfilt],yerr= d['ET3PHI'][b]['ET3PHI'].reshape(-1)[flag & Mfilt], fmt='.',color='r') #,label=d['filename'])

        #except:
        #    print( f'2 col data shape issue with {d} on {b}')

# N band 


#V2
for d in oi.data:
    print(d['filename'],'\n')
    plt.figure()
    plt.title(d['filename'].split('/')[-1])
    for i,b in enumerate(d['OI_VIS2']):

        flag = ~d['OI_VIS2'][b]['FLAG'].reshape(-1)

       
        try: 
            plt.errorbar(d['OI_VIS2'][b]['B/wl'].reshape(-1)[flag ],  d['OI_VIS2'][b]['V2'].reshape(-1)[flag ], yerr = d['OI_VIS2'][b]['EV2'].reshape(-1)[flag ],fmt = '.',color='orange') #,label=d['filename'])

        except:
            print( f'2 col data shape issue with {d} on {b}')

#CP
for d in oi.data:
    print(d['filename'],'\n')
    plt.figure()
    plt.title(d['filename'].split('/')[-1])
    for i,b in enumerate(d['OI_T3']):

        flag = ~d['OI_T3'][b]['FLAG'].reshape(-1)

        #Lfilt = (d['WL'] < 3.9) &  (d['WL'] > 3.2) 
        #Mfilt = (d['WL'] < 5) &  (d['WL'] > 4.5) 
        #try: 
        #plt.plot(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Lfilt ],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Lfilt],'.',color='orange') #,label=d['filename'])
        #plt.plot(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag & Mfilt],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag & Mfilt],'.',color='r') #,label=d['filename'])

        plt.errorbar(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag ], yerr= d['OI_T3'][b]['ET3PHI'].reshape(-1)[flag ],fmt='.',color='orange') #,label=d['filename'])
        plt.errorbar(d['OI_T3'][b]['Bmax/wl'].reshape(-1)[flag],  d['OI_T3'][b]['T3PHI'].reshape(-1)[flag ],yerr= d['OI_T3'][b]['ET3PHI'].reshape(-1)[flag ], fmt='.',color='r') #,label=d['filename'])

        #except:
        #    print( f'2 col data shape issue with {d} on {b}')


get rid of (bad data) in L band 
- 2022-08-06T011029_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits 
- 2022-07-27T232634_VRTPav_A0B2D0C1_IR-LM_LOW_Chop_cal_oifits_0.fits 
- 2022-07-28T004853_VRTPav_A0B2D0C1_IR-LM_LOW_Chop_cal_oifits_0.fits 
. Due to huge uncertainty etc 

and in N band 
 2022-07-28T004853_VRTPav_A0B2D0C1_IR-LM_LOW_Chop_cal_oifits_0.fits 
"""        




#%%
fig_inx = 1
binning = 20
version = 3 # to keep seperate iterations where  made changes
if __name__ == '__main__': 

    
    for feature in wvl_band_dict :
        
        fig_inx *= 10

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
            
        binning = 5 #np.max( [1, round(cont_binning * abs(wvl_band_dict[feature][1]- wvl_band_dict[feature][0])/(abs(2.038 - 2.078)) ) ])
        
        print(f'\n\n======== using {binning} binning for {feature} \n\n')
        oi = pmoired.OI(matisse_files, binning=binning ) #, binning=binning ) #no spectral binning ,  insname='GRAVITY_SC_P1',
           
        #=========== for plotting 
        # filter for the wavelengths we are looking at 
        wvl_filt = (oi.data[0]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[0]['WL'] <= wvl_band_dict[feature][1])
        
        wvls = oi.data[0]['WL'][wvl_filt][:-1] # avelengths to consider
        #===========
        
        wvl0 = np.mean( wvl_band_dict[feature] ) #2.166 # central wvl 
        wl_thresh = np.diff(wvl_band_dict[feature] )[0]/2 #0.005  #+/- this value 
        
        # set the uniform disk diameter for the current central wavelength 
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - 1e-6 * wvl0)) ]
        
        print(f'\n=====\nnumber wavelengths to fit = {sum( (oi.data[0]["WL"] > wvl0 - wl_thresh) & (oi.data[0]["WL"] < wvl0 + wl_thresh) )}\n====')
    

    
        # use this to constrain wavelength range... wrote in cont. and band wvls.. 
        
        
        # -- smallest lambda/B in mas (first data set) 
        step = 180*3600*1000e-6/np.pi/max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']])
        
        # -- spectral resolution (first data set) 
        R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])
        
        searchRad = 500 #mas 
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
        #fit the model         
        oi.doFit(doNotFit=['*,f', 'c,ud'])
    
    
    
    
        # ======== NAMING FOR OUTPUT FILES 
        ID = f'{ins}_spectralline_{feature}_step-{round(step,2)}_R-{round(R)}_wvlRange-{round(wvl0-wl_thresh,3)}_{round(wvl0+wl_thresh,3)}_vers{version}'
        
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
            oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
                   constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                              ('np.sqrt(c,x**2+c,y**2)', '>', ud_wvl) ])
            
        # -- show chi2 grid
        oi.showGrid()
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
        
        
        #if logV2:
        #    plt.yscale('log')
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
               
            
        
        # e.g. can plot / saave original data vs model
        """ OLD PLOTTING
        model_col = 'orange'
        obs_col= 'grey'
        fsize=18
        
        plt.figure(figsize=(8,5)) 
        flag_filt = (~oi._merged[0]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))
        # data 
        plt.errorbar(oi._merged[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt], oi._merged[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.8,fmt='.')
        # model
        plt.plot(oi._model[0]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt], oi._model[0]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)

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
        plt.errorbar(oi._merged[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], oi._merged[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[0]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col,label='obs',alpha=0.8,fmt='.')
        # model
        plt.plot(oi._model[0]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], oi._model[0]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
        plt.xlabel(r'$B_{max}/\lambda [M rad^{-1}]$',fontsize=fsize)
        plt.ylabel('CP [deg]',fontsize=fsize)
        plt.legend(fontsize=fsize)
        plt.gca().tick_params( labelsize=fsize )
        
        plt.savefig( current_folder + f'{ins}_pmoired_BESTFIT_CP_PLOT_{ID}.png', bbox_inches='tight', dpi=300)
        """
        
        
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
        best_test_injection = {**oi.bestfit['best'], **{'3,ud':0, '3,x':-15, '3,y':5, '3,f':0.01} }
        expl = {'rand':{'3,x':(-R*step/2, R*step/2), '3,y':(-R*step/2, R*step/2)}}
    
        oi.detectionLimit(expl, '3,f', model=best_test_injection, Nfits=500, nsigma=3,\
                      constrain=[('np.sqrt(3,x**2+3,y**2)', '<=', R*step/2 ),\
                                 ('np.sqrt(3,x**2+3,y**2)', '>', step/2) ])
        
        oi.showLimGrid(mag=1)
        plt.savefig( current_folder + f'{ins}_pmoired_DETECTION_LIMIT_{ID}.png', bbox_inches='tight', dpi=300)
    
    
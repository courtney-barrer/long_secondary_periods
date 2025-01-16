#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 01:59:34 2024

@author: bencb
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
import re
import fnmatch
import argparse

plt.ion()

if __name__=="__main__":

    path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))


    comp_loc = 'ANU' # computer location

    #data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
    data_path = path_dict[comp_loc]['data'] #+  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits") #'/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
    save_path = '/Users/bencb/Documents/long_secondary_periods/PMOIRED_FITS/play_plots/'

    ud_fits = pd.read_csv(path_dict[comp_loc]['data'] + 'UD_fit.csv',index_col=0)

    gravity_bands = {'continuum':[2.1,2.29], 'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                                    'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261], 'CO2-0':[2.2934, 2.298],\
                                    'CO3-1':[2.322,2.324],'CO4-2':[2.3525,2.3555]}

    parser = argparse.ArgumentParser(description="Script to fit parametric models.")
        
    # Add arguments
    parser.add_argument("--ins", type=str, default="pionier",
                        help="Instrument name (default: pionier)")
    parser.add_argument("--model", type=str, default="UD",
                        help="parameteric model to fit with Pmoired. (default UD)")
    parser.add_argument("--wavemin", type=float, default=None,
                        help="minimum wavelength in microns")
    parser.add_argument("--wavemax", type=float, default=None,
                        help="maximum wavelength in microns")
    parser.add_argument("--binning", type=int, default=None,
                        help="spectral binning (how many wavelength bins should be aggregated?)")
    parser.add_argument("--plot_logV2", action='store_true',
                        help="plot V^2 in log scale")
    parser.add_argument("--plot_image_logscale", action='store_true',
                        help="plot image reco in log scale")
    # Parse arguments and run the script
    args = parser.parse_args()

    ins = args.ins
    model = args.model
    wavemin = args.wavemin #um
    wavemax = args.wavemax #um
    binning = args.binning

    # map instrument to particular data files and parameters if no user specification
    if ins == 'pionier':

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')

        if (wavemin is None) or (wavemax is None):
            wavemin = 1.5
            wavemax = 1.8  

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

    elif ins == 'gravity':

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
        

        if (wavemin is None) or (wavemax is None):
            wavemin = 2.100#2.05
            wavemax = 2.102#2.40  

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]


    #The uniform disk fits within these absorption regions give diameters of 3.90$\pm$0.05mas, 4.06 $\pm$0.07mas, 3.94 $\pm$ at wavelengths 2.294, 2.322, 2.353$\mu$m, corresponding to CO(2-1) , CO(3-1), CO(4-2) band heads respectively. 
    elif fnmatch.fnmatch(ins, 'gravity_line_*'):

        #extract the band 
        band_label = ins.split('gravity_line_')[-1]
        # get the wavelength limits 
        wavemin = gravity_bands[band_label][0]
        wavemax = gravity_bands[band_label][1]

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]


                
    elif ins == 'matisse_LM':

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')

        if (wavemin is None) or (wavemax is None):
            wavemin = 3.1 
            wavemax = 4.9       
                
        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

    elif ins == 'matisse_L':
        print("Using CHOPPED MATISSE_LM data (not full merged data)")

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')


        if (wavemin is None) or (wavemax is None):
            wavemin = 3.3 #2.8
            wavemax = 3.6 #3.8   
            
        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

    elif ins == 'matisse_M':
        
        print("Using CHOPPED MATISSE_LM data (not full merged data)")

        obs_files = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits')

        if (wavemin is None) or (wavemax is None):
            wavemin = 4.6 
            wavemax = 4.9    

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]


    elif ins == 'matisse_N':
        
        # using flipped phases and CP phases ( )
        """
        modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
        in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
        """
        obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")

        if (wavemin is None) or (wavemax is None):
            raise UserWarning('for --ins matisse_N --wavemin and --wavemax must be specified. Inbuilt binnings options can be selected with (for example matisse_N_*um, or matisse_N_short, matisse_N_mid, matisse_N_long) ')

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

    elif ins == 'matisse_N_short':
        
        # using flipped phases and CP phases ( )
        """
        modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
        in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
        """
        obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")

        if (wavemin is None) or (wavemax is None):
            wavemin = 8.0#7.5
            wavemax = 9.0 #13.0

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

        #if model == 'binary':
        #    binning = 5 

    elif ins == 'matisse_N_mid':
        
        # using flipped phases and CP phases ( )
        """
        modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
        in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
        """
        obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")

        if (wavemin is None) or (wavemax is None):
            wavemin = 9.0#7.5
            wavemax = 10.0 #13.0

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]
        #if model == 'binary':
        #    binning = 5 

    elif ins == 'matisse_N_long':
        
        # using flipped phases and CP phases ( )
        """
        modified the visibility phase and closure phases - taking negative sign  “wrong sign of the phases, including closiure phase, in the N-band, causing an image or model rotation of 180 degrees.” -   https://www.eso.org/sci/facilities/paranal/instruments/ 
        in /home/rtc/Documents/long_secondary_periods/data/swap_N_band_CP.py  we take negative sign of visibility phase and closure phases in the individual reduced and merged data 
        """
        obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")

        if (wavemin is None) or (wavemax is None):
            wavemin = 10.0#7.5
            wavemax = 13.0 #13.0

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]

        #if model == 'binary':
        #    binning = 5 

    elif fnmatch.fnmatch(ins, 'matisse_N_*um'): 
        # using 0.5um binning in N-band for reconstruction
        wvl_bin = 0.5 #um 
        pattern = r"^matisse_N_([\d.]+)um$"
        match = re.match( pattern, ins)
        wmin = round(float(match.group(1)) , 1) 
        
        obs_files = glob.glob(path_dict[comp_loc]['data'] +  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits")

        if (wavemin is None) or (wavemax is None):
            wavemin = wmin 
            wavemax = wmin + wvl_bin

        wvl0 = (wavemin + wavemax ) / 2
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index -  1e-6 * wvl0)) ]


    else: 
        raise UserWarning('input instrument (ins) is not a valid options' )




    #save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

    #feature='ud_per_wvl'
    #oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 


    #matisse_files_L = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

    #matisse_files_N = glob.glob(data_path+"matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits") 

    #cannot read 2022-07-28T004853_VRTPav_A0B2D0C1_IR-N_LOW_noChop_cal_oifits_0.fits' so ignore it 
    #matisse_files_N.pop(9) - this was put in bad data folder 


    #wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_short':[8,9],'N_mid':[9,10],'N_long':[10,13]}


    #wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_8':[8,8.5],'N_9.5':[9.5,10],'N_10':[10,10.5]}


    min_rel_V2_error = 0.01
    min_rel_CP_error = 0.1 #deg
    max_rel_V2_error = 0.2
    max_rel_CP_error = 10 #deg



    #if __name__ == '__main__':
    oi = pmoired.OI(obs_files , binning = binning)

    # Define the grid for parameter exploration
    # expl = {
    #     'grid': {
    #         'r,incl': (30, 90, 10),  # Inclination
    #         'r,projang': (-90, -50, 20),  # Position angle
    #         'r,diamin': (10, 200, 20),  # Inner diameter
    #         'r,diamout': (20, 500, 40),  # Outer diameter
    #     }
    # }



    ############################
    # Grid search 
    ############################
    if model == 'UD':

        param = {'ud':ud_wvl}

        expl = {'grid':{'ud':(0, 300, 5)} }


        oi.setupFit({
            'obs': ['V2', 'T3PHI'], 
            'wl ranges': [[wavemin, wavemax]],
            'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
            'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
        })

        oi.gridFit(expl, model=param, doNotFit=[] ) 


    # elif model == 'LD':

        # param = {'ud':ud_wvl, 'profile':'$MU**$alpha', 'alpha':0.5}

        # expl = {'grid':{'ud':(0, 300, 5), 'alpha':(0.1, 0.4, 0.1)}}
            

        # oi.setupFit({
        #     'obs': ['V2', 'T3PHI'], 
        #     'wl ranges': [[wavemin, wavemax]],
        #     'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
        #     'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
        # })

        # #oi.doFit( param )
        # #oi.gridFit(expl, model=param, doNotFit=[])


    elif model == 'ellipse':

        param = {'ud':ud_wvl, 'incl':45, 'projang':45}
        
        expl = {'grid':{'ud':(0, 200, 10), 'incl':(1, 89, 10), 'projang':(-89,89,10)}}
        
        oi.setupFit({
            'obs': ['V2', 'T3PHI'], 
            'wl ranges': [[wavemin, wavemax]],
            'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
            'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
        })

        oi.gridFit(expl, model=param, doNotFit=[] ) #, prior=[('inc', '<=', 90),('inc', '>=', 0),('projang', '<=', 90),('projang', '>=', -90)])


    elif model == 'binary':

        step = 180*3600*1000e-6/np.pi/ 120 #60
        R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])

        param = {'*,ud':ud_wvl, '*,f':1, 'c,f':0.01, 'c,x':4, 'c,y':4, 'c,ud':0.}
        
        # -- define the exploration pattern
        expl = {'grid':{'c,x':(-R/2*step, R/2*step, step), 'c,y':(-R/2*step, R/2*step, step)}}
        
        oi.setupFit({
            'obs': ['V2', 'T3PHI'], 
            'wl ranges': [[wavemin, wavemax]],
            'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
            'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
        })


        oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
                    constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                                ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])

        #oi.doFit(param, doNotFit=['*,f', 'c,ud'])

    elif model == 'disk':
        #
        # param = {"p,ud": 10.47, 
        #         "p,f": 1, 
        #         "r,diamin": 194, 
        #         "r,diamout": 262, 
        #         "r,x": 0, 
        #         "r,y": 0, 
        #         "r,f": 0.9267298700126003, 
        #         "r,incl": 64.28571428571429, 
        #         "r,projang": -80.52631578947368
        #         }
        
        expl = {
            'grid': {
                'r,incl': (50, 80, 5),  # Inclination
                'r,projang': (-90, -70, 5),  # Position angle
                'r,diamin': (10, 60, 5), # Inner diameter
                'r,diamout': (100, 300, 40),  # Outer diameter
                'p,ud': (10, 100, 20)
            },
            # 'randn': {
            #     'r,incl': (64,50),  # Inclination
            #     'r,projang': (-80,50),  # Position angle
            #     'r,diamin': (100,100),  # Inner diameter
            #     #'r,diamout': (150, 50),  # Outer diameter
            #     'p,ud': (50,30),
            # }
        }
        #        'r,f': (0.8, 0.3),  # Outer diameter


        # Initial model parameters
        param = {
            'p,ud': 100.47, 'p,f': 1,
            'r,diamin': 50, 'r,diamout': 262,
            'r,x': 0, 'r,y': 0, 'r,f': 0.93,
            'r,incl': 64, 'r,projang': -80
        }

        # Ensure the fit setup includes 'obs'
        oi.setupFit({
            'obs': ['V2', 'T3PHI'], 
            'wl ranges': [[wavemin, wavemax]],
            'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
            'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
        })


        # Perform the grid fit
        results = oi.gridFit(
            model=param, expl=expl, fitOnly=None, doNotFit=['p,f',  'r,x', 'r,y'],
            verbose=2,constrain = [('r,diamout', '>=', 'r,diamin'),('r,diamout', '<=', '400')], multi=True, maxfev=50000
        )



    #oi.bestfit['best']

    #oi.doFit( oi.bestfit['best'] )

    #oi.showModel(oi.bestfit['best'], showSED=False, imFov=80, imPow=0.1)

    #oi.showGrid(interpolate = True, legend=False,tight=True)

    # save the json 
    write_dict = oi.bestfit['best']
    write_dict['chi2'] = oi.bestfit['chi2']

    with open(path_dict[comp_loc]['root'] + 'PMOIRED_FITS/best_models/'+f'bestparamodel_{model}_{ins}_{round(wavemin,1)}-{round(wavemax,1)}um.json', 'w') as f:
        json.dump(write_dict, f)







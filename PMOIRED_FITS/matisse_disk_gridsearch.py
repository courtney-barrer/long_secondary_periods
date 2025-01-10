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

path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU' # computer location

#data_path = '/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
data_path = path_dict[comp_loc]['data'] #+  "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits") #'/Users/bencb/Documents/long_secondary_periods/rt_pav_data/'
save_path = '/Users/bencb/Documents/long_secondary_periods/PMOIRED_FITS/play_plots/'
ins = 'matisse'

#save_path_0 = f'/Users/bcourtne/Documents/ANU_PHD2/RT_pav/PMOIRED_FITS/{ID}/'

#feature='ud_per_wvl'
#oi = pmoired.OI(gravity_files , insname='GRAVITY_SC_P1', binning=20 ) #, 

ud_fits = pd.read_csv(data_path + 'UD_fit.csv',index_col=0)

matisse_files_L = glob.glob(data_path+'matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

matisse_files_N = glob.glob(data_path+"matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits") 

#cannot read 2022-07-28T004853_VRTPav_A0B2D0C1_IR-N_LOW_noChop_cal_oifits_0.fits' so ignore it 
#matisse_files_N.pop(9) - this was put in bad data folder 


wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_short':[8,9],'N_mid':[9,10],'N_long':[10,13]}


#wvl_band_dict = {'L':[3.2,3.9],'M':[4.5,5],'N_8':[8,8.5],'N_9.5':[9.5,10],'N_10':[10,10.5]}


min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.3 
max_rel_CP_error = 20 #deg



feature = 'N_mid'
model_type = 'disk'


#if __name__ == '__main__':
oi = pmoired.OI(matisse_files_N, binning = 5)

# Define the grid for parameter exploration
# expl = {
#     'grid': {
#         'r,incl': (30, 90, 10),  # Inclination
#         'r,projang': (-90, -50, 20),  # Position angle
#         'r,diamin': (10, 200, 20),  # Inner diameter
#         'r,diamout': (20, 500, 40),  # Outer diameter
#     }
# }





#
#{"p,ud": 10.47, "p,f": 1, "r,diamin": 194, "r,diamout": 262, "r,x": -1.6, "r,y": 1, "r,f": 0.9267298700126003, "r,incl": 64.28571428571429, "r,projang": -80.52631578947368}
expl = {
    'grid': {
        'r,incl': (50, 80, 5),  # Inclination
        'r,projang': (-90, -70, 5),  # Position angle
        'r,diamin': (10, 200, 10), # Inner diameter
        #'r,diamout': (100, 300, 40),  # Outer diameter
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
    'r,diamin': 194, 'r,diamout': 262,
    'r,x': 0, 'r,y': 0, 'r,f': 0.93,
    'r,incl': 64, 'r,projang': -80
}

# Ensure the fit setup includes 'obs'
oi.setupFit({
    'obs': ['V2', 'T3PHI'], 
    'wl ranges': [wvl_band_dict[feature]],
    'min relative error': {'V2': min_rel_V2_error, 'CP': min_rel_CP_error},
    'max relative error': {'V2': max_rel_V2_error, 'CP': max_rel_CP_error}
})


# Perform the grid fit
results = oi.gridFit(
    model=param, expl=expl, fitOnly=None, doNotFit=['p,f',  'r,x', 'r,y','r,diamout'],
    verbose=2,constrain = [('r,diamout', '>=', 'r,diamin'),('r,diamout', '<=', '300')], multi=True, maxfev=50000
)
























incl_grid = np.linspace( 30, 90 , 3)
proj_grid = np.linspace( -90, -50 , 3 )
rin_grid = np.linspace( 10, 200, 10)
thickness_grid =  np.linspace( 1, 500, 5) # % of inner disk. e.g. outer = in + ridpt * in

best_grid = [] # np.zeros( [len( incl_grid ) , len( proj_grid ), len(rin_grid ) , len(thickness_grid) ] ).astype(list) 

    
# # INNER RAD VERY SENSITIVE TO max error
min_rel_V2_error = 0.01
min_rel_CP_error = 0.1 #deg
max_rel_V2_error = 0.25
max_rel_CP_error = 20 #deg

oi.setupFit({'obs':['V2', 'T3PHI'], 
             'min relative error':{'V2':0.0},
             'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
             'wl ranges':[wvl_band_dict[feature]]})


for n, rin in enumerate( rin_grid ):
    for m, thick in enumerate( thickness_grid  ):
        rout = rin + thick 
        
        for i, incl in enumerate( incl_grid ) :
            print('progress:',i/len(incl_grid))
            for j, proj in enumerate( proj_grid ):
                
                oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
                best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': rin,'r,diamout': rout, \
                              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                                  'r,incl': incl,'r,projang': proj}
        
                oi.doFit(best_model, doNotFit=['p,f','p,ud','r,x', 'r,y'] )#,'e,projang','e,incl'])
        
                best_grid.append( oi.bestfit )  
   


results = pd.DataFrame([
    {
        'rin': fit['best']['r,diamin'],
        'thickness': fit['best']['r,diamout'] - fit['best']['r,diamin'],
        'incl': fit['best']['r,incl'],
        'proj': fit['best']['r,projang'],
        'chi2': fit['chi2']
    }
    for fit in best_grid
])

plt.figure() ;plt.semilogy( results['rin'], results['chi2'], '.', alpha =0.4 ) ; plt.savefig('delme.png')


# "r,incl": 64.28571428571429, "r,projang": -80.52631578947368} were best on wide grid search keeping other parameters fixed 
# Define parameter grids
incl_grid = np.linspace(50, 70, 3)  # Inclination angle grid
proj_grid = np.linspace(-90, 70, 3)  # Position angle grid
diamin_grid = np.linspace( 20, 200, 5)  # Fixed inner diameter
thickness_grid = np.linspace(0.1, 2, 5)   # Fixed outer diameter
diamout_grid = 
f_grid = [0.9267298700126003]  # Fixed ring flux

# Prepare grid to scan
param_grid = [
    {"r,incl": incl, "r,projang": proj, "r,diamin": diamin, "r,diamout": diamout, "r,f": r_f}
    for incl in incl_grid
    for proj in proj_grid
    for diamin in diamin_grid
    for diamout in diamout_grid
    for r_f in f_grid
]

# Set up observables and wavelength ranges
oi.setupFit({"obs": ["V2", "T3PHI"], "wl ranges": [wvl_band_dict[feature]]})

# Prepare a list to store results
results = []

# Perform grid fit
for i, params in enumerate(param_grid):
    print(f"Progress: {i+1}/{len(param_grid)}")
    
    # Base model parameters
    best_model = {
        "p,ud": 10.47,  # Uniform disk parameter
        "p,f": 1,  # Flux scaling
        "r,x": -1.6,  # Ring center x-offset
        "r,y": 1,  # Ring center y-offset
        **params,  # Update model with current parameter set
    }

    # Perform the fit with fixed parameters
    oi.doFit(
        best_model,
        doNotFit=["p,f", "p,ud", "r,diamin", "r,diamout", "r,x", "r,y"]
    )

    # Save the best fit result
    results.append({
        "params": params,
        "chi2": oi.bestfit["chi2"],  # Extract reduced chi-squared value
        "best_model": oi.bestfit  # Full best-fit model parameters
    })

# Find the best-fit result
best_result = min(results, key=lambda x: x["chi2"])
print("Best Fit Parameters:", best_result["params"])
print("Best Chi2:", best_result["chi2"])























plt.figure() 
chi2_grid = np.array( [a['chi2'] for a in best_grid] ).reshape( len(incl_grid), len( proj_grid))
best_indx = np.unravel_index( np.argmin( chi2_grid ), [len( incl_grid ), len( proj_grid )])

best_incl = incl_grid[ best_indx[0] ]

best_proj = proj_grid[ best_indx[1] ]

plt.pcolormesh( proj_grid, incl_grid, chi2_grid ); 
plt.xlabel('initial projang'); plt.ylabel('initial incl'); plt.colorbar()

best_grid_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 20,'r,diamout': 262, \
                      'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
                          'r,incl': best_incl,'r,projang': best_proj}
# {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, \
#               'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
#                   'r,incl': best_incl,'r,projang': best_proj}

# 
oi.doFit( best_grid_model ,doNotFit=['r,f','p,f'])
oi.showModel(oi.bestfit['best'], showSED=False, imFov=80, imPow=0.1)


# save the json 
with open(path_dict[comp_loc]['root'] + 'PMOIRED_FITS/best_models/'+f'bestparamodel_{model_type}_{feature}.json', 'w') as f:
    json.dump(best_grid_model, f)

#

best_grid_model




"""

{'p,ud':     10.55, # +/- 0.21
'r,diamin': 150.19, # +/- 6.27
'r,diamout':278.37, # +/- 5.88
'r,incl':   70.10, # +/- 0.91
'r,projang':-68.03, # +/- 0.80
'r,x':      -2.69, # +/- 1.22
'r,y':      0.40, # +/- 0.63
'p,f':      1,
'r,f':      0.9267298700126003,
}
"""
# add wvls dependence or azimuth brightness profile 

#plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 50,  logV2 = False, savefig_folder=None,savefig_name='plots')


        
"""
# DO grid search 
# -- define the exploration pattern
expl = {'grid':{'r,incl':(0, 90, 5), 'r,projang':(-90, 90, 8)}}

# -- setup the fit, as usual
oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':wvl_band_dict[feature]})
#oi.setupFit({'obs':['V2', 'T3PHI']})


best_disk = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, 'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003, 'r,incl': 34.44130094553597,'r,projang': -30.053240964807344}
oi.gridFit(expl, model=best_disk, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'])


  
oi.showGrid(interpolate = True, legend=False,tight=True)

print( 'best fit: ', oi.bestfit['best'], '\nchi2:' , oi.bestfit['chi2'])


# fit the model 
oi.doFit( )
"""



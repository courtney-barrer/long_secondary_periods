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

min_rel_V2_error = 0.05 #0.01
min_rel_CP_error = 1 #0.1 #deg
max_rel_V2_error = 0.4 #100 
max_rel_CP_error = 40 #20 #deg



feature = 'N_mid'
model_type = 'disk'

#if __name__ == '__main__':
oi = pmoired.OI(matisse_files_N, binning = 5)

incl_grid = np.linspace( 0, 90 , 15 )
proj_grid = np.linspace( -90, 90 ,20 )
best_grid = [] 
for i, incl in enumerate( incl_grid ) :
    print('progress:',i/len(incl_grid))
    for j, proj in enumerate( proj_grid ):
        
        oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]})
        
        best_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 20,'r,diamout': 262, \
                      'r,x': 0, 'r,y': 0, 'r,f': 0.9267298700126003,\
                          'r,incl': incl,'r,projang': proj}
        

        #{'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, \
        #              'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
        #                'r,incl': incl,'r,projang': proj}

        oi.doFit(best_model, doNotFit=['p,f','p,ud','r,diamin', 'r,diamout','r,x', 'r,y'] )#,'e,projang','e,incl'])

        best_grid.append( oi.bestfit )
            

plt.figure() 
chi2_grid = np.array( [a['chi2'] for a in best_grid] ).reshape( len(incl_grid), len( proj_grid))
best_indx = np.unravel_index( np.argmin( chi2_grid ), [len( incl_grid ), len( proj_grid )])

best_incl = incl_grid[ best_indx[0] ]

best_proj = proj_grid[ best_indx[1] ]

plt.pcolormesh( proj_grid, incl_grid, chi2_grid ); 
plt.xlabel('initial projang'); plt.ylabel('initial incl'); plt.colorbar()
plt.savefig('delme.png')

oi.setupFit({'obs':['V2', 'T3PHI'] ,'wl ranges':[wvl_band_dict[feature]]}) #,constrain=['r,diamin','>','p,ud'])
        
# best_incl = 70
# best_proj = -60 
best_grid_model = {'p,ud': 10.47,'p,f': 1, 'r,diamin': 20,'r,diamout': 262, \
                      'r,x': 0, 'r,y': 0, 'r,f': 0.9267298700126003,\
                          'r,incl': best_incl,'r,projang': best_proj}

#'r,x': -1.6, 'r,y': 1, 

#oi.showModel( best_grid_model ,showSED=False, imFov=300, imPow=0.1 )

# {'p,ud': 10.47,'p,f': 1, 'r,diamin': 194,'r,diamout': 262, \
#               'r,x': -1.6, 'r,y': 1, 'r,f': 0.9267298700126003,\
#                   'r,incl': best_incl,'r,projang': best_proj}

# 
oi.doFit( best_grid_model ,doNotFit=['r,f','p,f'] )
oi.showModel(oi.bestfit['best'], showSED=False, imFov=300, imPow=0.1)
plt.savefig('delme.png')

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



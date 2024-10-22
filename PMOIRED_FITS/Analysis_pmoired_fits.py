#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 05:16:22 2024

@author: bencb

analyse outputs of pmoired_binary_[INS].py that should be in long_secondary_periods/pmoired_fits/binary/'
"""


#-- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import pmoired
import glob
import matplotlib.pyplot as plt 
import pickle
import os 
import pandas as pd

data_path = '/Users/bencb/Documents/long_secondary_periods/pmoired_fits/binary/'
save_path = '/Users/bencb/Documents/long_secondary_periods/pmoired_fits/binary/'
#data_path = '/home/bcourtne/data/rt_pav/'
#save_path = '/home/bcourtne/data/rt_pav/pmoired_fits/BINARY/'

#fit_folder = 'pionier_OUTSIDE_UD-False_step-2.42_R-34_wvlRange-0.6_2.6/'
fit_folder = 'pionier_OUTSIDE_UD-True_step-2.42_R-34_wvlRange-0.6_2.6/'


files = glob.glob( data_path+fit_folder +'*.pickle' )

for f in files:
    if 'BESTFIT' in f:
        bestfit = pd.read_pickle(f)

    elif 'BOOTSTRAP' in f:
        boot = pd.read_pickle(f)
        

for k in bestfit['best']:
    print( f'{k}= {boot["best"][k]} +/-  {boot["uncer"][k]}\n')
print('chi2', boot['chi2'] )


"""ax[0].hist( boot['all best']['c,y'] )
ax[0].ylabel('c,y')


ax[1].hist( boot['all best']['c,x'] )
ax[1].ylabel('c,x')

ax[1].hist( boot['all best']['c,x'] )
ax[1].ylabel('c,x')"""


plt.figure(figsize=(5,5))
#plt.scatter(0, 0, marker='o', s=3.3,alpha = 0.3)
plt.gca().add_patch(plt.Circle((0, 0), 3.3/2, color='r', alpha=0.5,label='UD best fit'))
plt.plot( [0.058],[-0.081], 'x', label='Pionier best fit - no inner radius constraint')
plt.plot( [-2.241],[0.253], 'x', label='Pionier best fit - inner radius constraint')
plt.plot( [-13.17],[1.14], 'x', label='Gravity best fit - inner radius constraint')

plt.xlim([-20,20])
plt.ylim([-20,20])
plt.legend()
plt.xlabel(r'$\Delta RA$')
plt.ylabel(r'$\Delta DEC$')



"""
best fit corr when secondary inside UD - correlated flux - diam 
array([[ 1.        ,  0.96947893, -0.61868892,  0.40282414],
       [ 0.96947893,  1.        , -0.5945931 ,  0.36695715],
       [-0.61868892, -0.5945931 ,  1.        , -0.47602251],
       [ 0.40282414,  0.36695715, -0.47602251,  1.        ]])


"""

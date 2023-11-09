#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:14:49 2023

@author: bcourtne
"""

import os 
import pyoifits as oifits
from astropy.io import fits

#import pyoifits as oifits
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import scipy
from  scipy.interpolate import interp1d 
import pandas as pd
import matplotlib.colors as colors

path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/image_recos/alg-mira_rgl-compact_fit-V2CP'

files = glob.glob( os.path.join(path, '*image.fits') )


sort_filt = [float(f.split('/')[-1].split('-')[0].split('_')[-1]) for f in files] 

sorted_files = [x for _, x in sorted(zip(sort_filt, files))]

for f in sorted_files:
    
    h=oifits.open(f)
    
    r'$\chi^2$={}'.format( round( h[-1].header['CHISQ'] , 2) )
    
    r'$\lambda$ ={}-{}\mu m'.format( h[-2].header['WAVE_MIN']*1e6  , h[-2].header['WAVE_MAX']*1e6 )
    
    dx = h[0].header['CDELT1'] * 3600 * 1e3
    x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])
    
    dy = h[0].header['CDELT2'] * 3600 * 1e3
    y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])
    
    
    plt.figure( figsize=(8,8) )
    plt.pcolormesh(x, y[::-1],  h[0].data /h[0].data.max() )#, norm=colors.LogNorm(vmin=1e-2, vmax=1))
    plt.xlabel('RA [mas]',fontsize=15)
    plt.ylabel('DEC [mas]',fontsize=15)
    plt.gca().tick_params(labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normalized flux',fontsize=15)
    cbar.ax.tick_params(labelsize=12) 
    #cbar.ax.set_yticklabels(ticklabs, fontsize=10)
    #cbar.ax.set_xlabel(fontsize=15 ) #'Normalize flux')#, rotation=270)
    
    plt.text( x[5],y[-10], 'RT Pav', color='w',fontsize=15)
    plt.text( x[5],y[-20], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h[-2].header['WAVE_MIN']*1e6  , h[-2].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='w')
    plt.text( x[5],y[-30], r'$\chi^2$={}'.format( round( h[-1].header['CHISQ'] , 2) ), color='w',fontsize=15)

    plt.tight_layout()
    filename = f.split('/')[-1].split('fit')[0][:-1]+'.png'
    plt.savefig(os.path.join(path, filename))
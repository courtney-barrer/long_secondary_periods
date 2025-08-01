#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:21:27 2023

@author: bcourtne

#%% Dirty beam of VLTI data 
"""






import glob
import numpy as np
import pyoifits as oifits
import matplotlib.pyplot as plt 
import json

def dirty_beam(files,du=None,dv=None,verbose=True):
    u,v = [],[]
    for f_indx, f in enumerate( files ):
        
        h = oifits.open(f)
        
        #spatial frequencies
        wvl = h['OI_WAVELENGTH'].data['EFF_WAVE'][::]
        u.append( [B/wvl  for B in h['OI_VIS2'].data['UCOORD'] ]) # East rad^-1
        v.append( [B/wvl  for B in h['OI_VIS2'].data['VCOORD'] ])  # North rad^-1
        
    
    
    v_pts =  np.array(v).ravel() 
    u_pts =  np.array(u).ravel() 
    
    if verbose:
        plt.figure()
        plt.scatter( u_pts, v_pts)
        plt.xlabel(r'u [rad$^{-1}$]')   
        plt.ylabel(r'v [rad$^{-1}$]')    
    
    if dv == None:
        dv = 0.5e6 #np.quantile( np.diff(np.sort(v_pts))[np.diff(np.sort(v_pts))!=0] , 0.5 )
    if du == None:
        du = 0.5e6 #np.quantile(  np.diff(np.sort(u_pts))[np.diff(np.sort(u_pts))!=0], 0.5 )
    
    u_grid = np.arange( min( min(v_pts), min(u_pts) ) ,max( max(v_pts), max(u_pts) ), du)
    v_grid = np.arange( min( min(v_pts), min(u_pts) ) ,max( max(v_pts), max(u_pts) ), dv)
    
    
    uu, vv = np.meshgrid( u_grid, v_grid )
    
    
    #def func(x, y):
    
        #return (uu - upt)**2 + (vv-vpt)**2 < (du+dv)**2
    v2=[]
    for count, (upt,vpt) in enumerate(zip(u_pts,v_pts)):
        if verbose:
            print('completed (%) = ', 100* count/ len(u_pts))
    
        v2.append((uu - upt)**2 + (vv-vpt)**2 < (du**2+dv**2)/4 )
    
    if verbose:
        plt.figure()
        plt.pcolormesh(np.sum(v2, axis=0)>0)
        plt.gca().set_aspect('equal')
        
        
    uv = np.sum(v2, axis=0)>0
    
    dirty_beam = abs(np.fft.fftshift(np.fft.fft2( uv ) ) )
    
    xE = np.fft.fftshift( np.fft.fftfreq(uv.shape[0], du) )
    xN = np.fft.fftshift( np.fft.fftfreq(uv.shape[1], dv) )
    
    
    north_mas = np.rad2deg( xN  ) * 3600 * 1e3
    east_mas = np.rad2deg( xE ) * 3600 * 1e3

    return(north_mas, east_mas, dirty_beam )





path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU'

pionier_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits' ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')


gravity_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
#glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits' ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob(path_dict[comp_loc]['data'] + "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits" ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]



#%% MATISSE LM BAND
#files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

north_mas_mat_LM, east_mas_mat_LM, dirty_beam_mat_LM = dirty_beam(matisse_files_L,du=None,dv=None,verbose=True)


mask=np.ones(dirty_beam_mat_LM.shape)
ca=2
bias = 0
mask[len(dirty_beam_mat_LM)//2-ca:len(dirty_beam_mat_LM )//2+ca, len(dirty_beam_mat_LM )//2-ca+bias:len(dirty_beam_mat_LM )//2+ca+bias] = 0
    
plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_mat_LM, east_mas_mat_LM, mask*dirty_beam_mat_LM   ) 
plt.gca().set_aspect('equal')
plt.ylabel(r'$\Delta$RA (mas) - [North]',fontsize=16)
plt.xlabel(r'$\Delta$DEC (mas) - [East]',fontsize=16)
plt.title('MATISSE LM BAND DIRTY BEAM (MASKED)')
plt.colorbar()
 
plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_mat_LM, east_mas_mat_LM, dirty_beam_mat_LM   ) 
plt.gca().set_aspect('equal')
plt.ylabel(r'$\Delta$RA (mas) - [North]',fontsize=16)
plt.xlabel(r'$\Delta$DEC (mas) - [East]',fontsize=16)
plt.title('MATISSE LM BAND DIRTY BEAM')

#%% MATISSE N BAND
#files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

north_mas_mat_N, east_mas_mat_N, dirty_beam_mat_N = dirty_beam(matisse_files_N,du=1e5,dv=1e5,verbose=True)


mask=np.ones(dirty_beam_mat_N.shape)
ca=2
bias = 0
mask[len(dirty_beam_mat_N)//2-ca:len(dirty_beam_mat_N )//2+ca, len(dirty_beam_mat_N )//2-ca+bias:len(dirty_beam_mat_N )//2+ca+bias] = 0
    
ll=80
plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_mat_N, east_mas_mat_N, mask*dirty_beam_mat_N / dirty_beam_mat_N.max()  ) 
plt.gca().set_aspect('equal')
plt.colorbar()
plt.ylabel(r'$\Delta$DEC (mas) - [North]',fontsize=16)
plt.xlabel(r'$\Delta$RA (mas) - [East]',fontsize=16)
plt.title('MATISSE LM BAND DIRTY BEAM (MASKED)')
plt.xlim([-ll,ll])
plt.ylim([-ll,ll])

plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_mat_N, east_mas_mat_N, dirty_beam_mat_N  / dirty_beam_mat_N.max()  ) 
plt.gca().set_aspect('equal')
plt.ylabel(r'$\Delta$DEC (mas) - [North]',fontsize=16)
plt.xlabel(r'$\Delta$RA (mas) - [East]',fontsize=16)
plt.title('MATISSE N BAND DIRTY BEAM')
plt.colorbar()
plt.xlim([-ll,ll])
plt.ylim([-ll,ll])
plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/N_band_dirty_beam.png')



#%% PIONIER H BAND 
#files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')

north_mas_pio_H, east_mas_pio_H, dirty_beam_pio_H = dirty_beam(pionier_files,du=1e5,dv=1e5,verbose=True)

ll=20
mask=np.ones(dirty_beam_pio_H.shape)
ca=2
bias = 0
mask[len(dirty_beam_pio_H)//2-ca:len(dirty_beam_pio_H )//2+ca, len(dirty_beam_pio_H )//2-ca+bias:len(dirty_beam_pio_H )//2+ca+bias] = 0
    
plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_pio_H, east_mas_pio_H, mask*dirty_beam_pio_H   ) 
plt.gca().set_aspect('equal')
plt.ylabel(r'$\Delta$RA (mas) - [East]',fontsize=16)
plt.xlabel(r'$\Delta$DEC (mas) - [North]',fontsize=16)
plt.title('MATISSE LM BAND DIRTY BEAM (MASKED)')
plt.xlim([-ll,ll])
plt.ylim([-ll,ll])


plt.figure(figsize=(8,8))
plt.pcolormesh( north_mas_pio_H, east_mas_pio_H, dirty_beam_pio_H   ) 
plt.gca().set_aspect('equal')
plt.ylabel(r'$\Delta$RA (mas) - [North]',fontsize=16)
plt.xlabel(r'$\Delta$DEC (mas) - [East]',fontsize=16)
plt.title('PIONIER H BAND DIRTY BEAM')
plt.xlim([-ll,ll])
plt.ylim([-ll,ll])




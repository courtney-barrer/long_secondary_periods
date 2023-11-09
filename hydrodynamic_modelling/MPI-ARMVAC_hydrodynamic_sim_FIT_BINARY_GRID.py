#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:27:45 2023

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
import pmoired

par2au = 206265
rad2mas = 180/np.pi * 3600 * 1e3

#%% 
def sort_baseline_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:] ] ))
    return(n_key)

def sort_triangle_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:4],B_string[4:]  ] ))
    return(n_key)

def flatten_comprehension(matrix):
    return([item for row in matrix for item in row])
    
pionier_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')

gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

matisse_files_N = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

model_file_grid = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/AMRVAC/*')
model_file_grid = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/AMRVAC/*best*')

noise = {'|V|':0.01, # fractional error on visibility amplitude
         'V2':0.01, # fractional error on visibility squared amplitude
         'T3AMP':0.01, # fractional error on triple product amplitude
         'T3PHI':1.0, # error on triple product phase, in degrees
         'PHI':1.0, # error on phase, in degrees
         'FLUX':0.01 # fractional error on flux
        }

master_dict = {}

#master_dict[fname][observable][config]={[model:{wvl} [] , 'obs':]['wvl']
for f in pionier_files: #matisse_files_L: #
    
    ext = None # 12
    # we get the data from the observations 
    obs = oifits.open(f)
    
    station_list = list( obs['OI_ARRAY'].data['STA_NAME'] )
    indx_list = list( obs['OI_ARRAY'].data['STA_INDEX'] )
    indx2sta = { i:n for i,n in zip(indx_list,  station_list) } 
    
    RA = obs[0].header['RA'] * 24/360
    DEC = obs[0].header['DEC']
    LST = obs[0].header['LST'] / 60 / 60
    
    fname = f.split('/')[-1]
    
    master_dict[fname] = {}
    
    #master_dict['model'][obs_name] = {}
    #master_dict['obs'][obs_name] = {}
    
    obs_wvl = obs['OI_WAVELENGTH',ext].data['EFF_WAVE']
    obs_v2 = obs['OI_VIS2',ext].data['VIS2DATA']
    obs_v2err = obs['OI_VIS2',ext].data['VIS2ERR'] ###ADD IN TO MSTER DICT
    obs_Cu = obs['OI_VIS2',ext].data['UCOORD'] # [(u,v) for u,v in zip(obs['OI_VIS2'].data['UCOORD'], obs['OI_VIS2'].data['VCOORD'])]
    obs_Cv = obs['OI_VIS2',ext].data['VCOORD']

    obs_T3PHI = obs['OI_T3'].data['T3PHI']
    obs_T3PHIerr = obs['OI_T3'].data['T3PHIERR']
    obs_Cu1 = obs['OI_T3'].data['U1COORD']
    obs_Cu2 = obs['OI_T3'].data['U2COORD']
    obs_Cv1 = obs['OI_T3'].data['V1COORD']
    obs_Cv2 = obs['OI_T3'].data['V2COORD']

    obs_v2_indx = obs['OI_VIS2'].data['STA_INDEX']
    obs_T3_indx = obs['OI_T3'].data['STA_INDEX']
    
    
    
    
    obs_baselines = [[indx2sta[x] for x in obs_v2_indx[i]] for i in range(len(obs_v2_indx))]
    obs_triangles = [[indx2sta[x] for x in obs_T3_indx[i]] for i in range(len(obs_T3_indx))]
    
    baseline_keys = [''.join( sorted(a) ) for a in obs_baselines]
    triangle_keys = [''.join( sorted(a) ) for a in obs_triangles]
    
    #init some dictionaries 
    master_dict[fname]['VIS2DATA'] = {}
    master_dict[fname]['VIS2ERR'] = {}
    master_dict[fname]['T3'] = {}
    master_dict[fname]['VIS2DATA']['OBS'] = {}
    master_dict[fname]['VIS2ERR']['OBS'] = {}
    master_dict[fname]['T3']['OBS'] = {}
    
    for i,B in enumerate(baseline_keys):
        master_dict[fname]['VIS2DATA']['OBS'][B] = {}
        master_dict[fname]['VIS2ERR']['OBS'][B] = {}
        
        master_dict[fname]['VIS2DATA']['OBS'][B]['uv'] = (obs_Cu[i], obs_Cv[i])
        master_dict[fname]['VIS2DATA']['OBS'][B]['wvl'] = obs_wvl
        master_dict[fname]['VIS2DATA']['OBS'][B]['V2'] = obs_v2[i] # CHECK THE INDEXING ON THIS
        master_dict[fname]['VIS2ERR']['OBS'][B]['V2err'] = obs_v2err[i] 
        
        
    for i, T in enumerate(triangle_keys):
        master_dict[fname]['T3']['OBS'][T] = {}
        master_dict[fname]['T3']['OBS'][T]['wvl'] = obs_wvl
        master_dict[fname]['T3']['OBS'][T]['T3PHI'] = obs_T3PHI[i]
        master_dict[fname]['T3']['OBS'][T]['T3PHIerr'] = obs_T3PHIerr[i]
        master_dict[fname]['T3']['OBS'][T]['uv_1'] = (obs_Cu1[i], obs_Cv1[i])
        master_dict[fname]['T3']['OBS'][T]['uv_2'] = (obs_Cu2[i], obs_Cv2[i])
    
    # data from the model
    for ml in model_file_grid: 
        
        model = fits.open(ml)
        model_name = ml.split('/')[-1]
        
        master_dict[fname]['VIS2DATA'][model_name] = {}
        master_dict[fname]['T3'][model_name] = {}
        
        
        number_pixels = len(model[0].data[0, 0, :])
        half_domain = model[0].header['HALF_DOM']   # in AU
        distance = par2au * model[0].header['DISTANCE']    # in AU
        x = np.linspace(-half_domain/(distance), half_domain/distance, number_pixels) # rad
        y = np.linspace(-half_domain/(distance), half_domain/distance, number_pixels) # rad
        input_array = model[0].data
        cube = {}
        cube['scale'] = rad2mas * (np.max(x)-np.min(x)) / number_pixels     # 0.0788  mas / pixel
        cube['X'], cube['Y'] = np.meshgrid(rad2mas * x, rad2mas * y)
        cube['image'] = input_array
        cube['WL'] = model[1].data
        wl = cube['WL']
        print('spectral channel resolution: R=%.1f'%(np.mean(cube['WL']/np.gradient(cube['WL']))))
        
        model_obs = pmoired.oifake.makeFakeVLTI(station_list ,
                                            (RA, DEC), # Simbad name or sky coordiates as (ra_h, dec_d)
                                            [LST], # list of LST for observations
                                            wl, # list of wavelength, in um
                                            cube=cube, # cube dictionnary (see above)
                                            noise = noise,
                                            )
        
        oi = pmoired.OI() # create an empty PMOIRED OI object
        oi.data = model_obs
        
        
        # VISIBILITIES
        
        #enforce correct ordering of baseline keys 
        for baseline_key in oi.data['OI_VIS2'].copy().keys():
        
            new_key = sort_baseline_string(baseline_key )
            oi.data['OI_VIS2'][new_key] = oi.data['OI_VIS2'].pop(baseline_key)
            
        
        for B in oi.data['OI_VIS2'].keys():
            
            # sometimes baseline string has different order - so we put it in a set to compare
            if B in master_dict[fname]['VIS2DATA']['OBS']:
                
                master_dict[fname]['VIS2DATA'][model_name][B] = {}
                
                utmp = np.array([oi.data['OI_VIS2'][B]['u'][0] for B in oi.data['OI_VIS2']]).reshape(1,-1)[0]
                vtmp = np.array([oi.data['OI_VIS2'][B]['v'][0] for B in oi.data['OI_VIS2']]).reshape(1,-1)[0]
                
                master_dict[fname]['VIS2DATA'][model_name][B]['uv'] = [(u,v) for u,v in zip(utmp,vtmp)] 
                master_dict[fname]['VIS2DATA'][model_name][B]['wvl'] = oi.data['WL']
                master_dict[fname]['VIS2DATA'][model_name][B]['V2'] = oi.data['OI_VIS2'][B]['V2'][0]
                
            else: 
                raise TypeError("check why baseline keys don't match")
                
                
        # CLOSURE PHASES

        #enforce correct ordering of baseline keys 
        for triangle_key in oi.data['OI_T3'].copy().keys():
        
            new_key = sort_triangle_string( triangle_key )
            oi.data['OI_T3'][new_key] = oi.data['OI_T3'].pop(triangle_key)
            
          
        for T in oi.data['OI_T3'].keys():
            
            if T in master_dict[fname]['T3']['OBS'] :
                
                master_dict[fname]['T3'][model_name][T] = {}
                
                u1tmp = np.array([oi.data['OI_T3'][T]['u1'][0] for T in oi.data['OI_T3']]).reshape(1,-1)[0]
                u2tmp = np.array([oi.data['OI_T3'][T]['u2'][0] for T in oi.data['OI_T3']]).reshape(1,-1)[0]
                
                v1tmp = np.array([oi.data['OI_T3'][T]['v1'][0] for T in oi.data['OI_T3']]).reshape(1,-1)[0]
                v2tmp = np.array([oi.data['OI_T3'][T]['v2'][0] for T in oi.data['OI_T3']]).reshape(1,-1)[0]
                
                master_dict[fname]['T3'][model_name][T]['wvl'] = oi.data['WL']
                
                master_dict[fname]['T3'][model_name][T]['T3PHI'] = oi.data['OI_T3'][T]['T3PHI'][0]
                
                master_dict[fname]['T3'][model_name][T]['uv_1'] = [(u,v) for u,v in zip(u1tmp,v1tmp)]
                
                master_dict[fname]['T3'][model_name][T]['uv_2'] = [(u,v) for u,v in zip(u2tmp,v2tmp)]
                
            else: 
                raise TypeError("check why baseline keys don't match")
        #master_dict['model'][obs_name][model_name] = oi
        #oi.data[0]['OI_VIS2'][B]['u']
        
        #obs['OI_VIS2'].data['STA_INDEX']
    
    
    # oi.data[0]['OI_T3']['A0B2C1']['T3PHI']
    # obs['OI_T3'].data['STA_INDEX']
    # indx2station = {obs['OI_ARRAY'].data['STA_INDEX'][i]:obs['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(obs['OI_ARRAY'].data['STA_NAME']))}
    # indx2station

#plt.figure()
#plt.scatter(  -np.array([oi.data['OI_VIS2'][B]['u'] for B in oi.data['OI_VIS2']]).reshape(1,-1)[0] ,-np.array([oi.data['OI_VIS2'][B]['v'] for B in oi.data['OI_VIS2']]).reshape(1,-1)[0] )
#plt.scatter( obs['OI_VIS2'].data['UCOORD'],obs['OI_VIS2'].data['VCOORD'] )
#%% Visibilities (model vs obs)

v2_obs_list = []
v2_err_list = []
v2_mod_list = []
B = []

model = 'PIO_m2.5_b2.fits'

for model in [a.split('/')[-1] for a in model_file_grid]:
    if model!='OBS':
        for f in master_dict:
            for B in master_dict[f]['VIS2DATA']['OBS']:
                
                wvl_obs = 1e6 * master_dict[f]['VIS2DATA']['OBS'][B]['wvl'] 
                
                for wvl, v2_mod in zip( master_dict[f]['VIS2DATA'][model][B]['wvl'], master_dict[f]['VIS2DATA'][model][B]['V2']):    
                    
                    
                    wvl_indx = np.argmin(abs(wvl-wvl_obs))
                    print(wvl_indx, wvl, wvl_obs)
                    v2_obs_list.append(  master_dict[f]['VIS2DATA']['OBS'][B]['V2'][wvl_indx]  )
                    v2_err_list.append( master_dict[f]['VIS2ERR']['OBS'][B]['V2err'][wvl_indx]  )
                    v2_mod_list.append( v2_mod ) 
        
        #plt.figure()
        #plt.plot( v2_mod - v2_obs)
        plt.figure()
        plt.plot( np.array( v2_obs_list  ), np.array( v2_mod_list  ) ,'.'); plt.plot( np.array( v2_obs_list  ),np.array( v2_obs_list  ), color='red',label='1:1'); plt.legend(); plt.xlabel('observed V2'); plt.ylabel('modelled V2')
        plt.title(model)
        
      
chisqr = np.sum( ( np.array( v2_obs_list ) - np.array(v2_mod_list) )**2 / np.array(v2_err_list)**2 )
        
redchisqr = chisqr/ (len(v2_obs_list)-1)

#%% closure phase  (model vs obs)

# NOTE MODEL ONLY USES 3 WAVELENGTHS 
#all_cp_model = [[master_dict[fname]['T3'][model_name][T]['T3PHI'] for T in master_dict[fname]['T3'][model_name]] for fname in master_dict]
#all_cp_obs = [[master_dict[fname]['T3']['OBS'][T]['T3PHI'] for T in master_dict[fname]['T3']['OBS']] for fname in master_dict]
#plt.figure()
#plt.hist( flatten_comprehension( flatten_comprehension( all_cp_model ) ) , alpha=0.5, label='MODEL')
#plt.hist( flatten_comprehension( flatten_comprehension( all_cp_obs ) ) , alpha=0.5, label='OBS' )


T3_obs_list = []
T3_err_list = []
T3_mod_list = []
T3_uv1=[]
T3_uv2= []
B = []

for model in [a.split('/')[-1] for a in model_file_grid]:
    if model!='OBS':
        for f in master_dict:
            for T in master_dict[f]['T3']['OBS']:
                
                wvl_obs = 1e6 * master_dict[f]['T3']['OBS'][T]['wvl'] 
                
                
                
                for wvl, T3PHI_mod in zip( master_dict[f]['T3'][model][T]['wvl'], master_dict[f]['T3'][model][T]['T3PHI']):    
                    
                    T3_uv1.append( master_dict[f]['T3']['OBS'][T]['uv_1'] )
                    T3_uv2.append( master_dict[f]['T3']['OBS'][T]['uv_2']  )
                    
                    wvl_indx = np.argmin(abs(wvl-wvl_obs))
                    print(wvl_indx, wvl, wvl_obs)
                    T3_obs_list.append(  master_dict[f]['T3']['OBS'][T]['T3PHI'][wvl_indx]  )
                    T3_err_list.append( master_dict[f]['T3']['OBS'][T]['T3PHIerr'][wvl_indx]  )
                    T3_mod_list.append( T3PHI_mod ) 
                    
                    
                
        #plt.figure()
        #plt.plot( v2_mod - v2_obs)
        plt.figure()
        plt.plot( np.array( T3_obs_list ), np.array( T3_mod_list  ) ,'.'); plt.plot( np.array( T3_obs_list  ),np.array( T3_obs_list ), color='red',label='1:1'); plt.legend(); plt.xlabel('observed CP [deg]'); plt.ylabel('modelled CP [deg]')
        plt.title(model)


        #plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/AMRVAC/CP_best_model_vs_obs.png',dpi=300)

plt.figure()
plt.hist(  T3_mod_list  , alpha=0.5, label='MODEL', bins=np.linspace(-200,200,80))
plt.hist( T3_obs_list , alpha=0.5, label='OBS', bins=np.linspace(-200,200,80) )
plt.legend()
plt.xlabel('closure phase [deg]')
plt.ylabel('counts')
plt.tight_layout() 
plt.yscale('log')
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/AMRVAC/hist_CP_best_model_vs_obs.png',dpi=300)



fig1 = plt.figure(1,figsize=(10,8))
fig1.set_tight_layout(True)

frame1=fig1.add_axes((.1,.3,.8,.6))
frame2=fig1.add_axes((.1,.05,.8,.2))  

#plt.figure(figsize=(8,5))
Bmax = [np.sqrt( (uv1[0]+uv2[0])**2 + (uv1[1]+uv2[1])**2)  for uv1,uv2 in zip( T3_uv1 , T3_uv2)]
frame1.errorbar( Bmax, T3_obs_list ,yerr=T3_err_list , fmt='.', label='OBS',alpha=0.8)
frame1.plot( Bmax, T3_mod_list , '.', label='MODEL',alpha=0.8)
frame1.legend(fontsize=25)
frame1.set_ylabel('closure phase [deg]',fontsize=20)

frame2.plot(Bmax, abs(np.array(T3_mod_list)-np.array(T3_obs_list))/ np.array(T3_err_list ),'.',color='k')
frame2.set_xlabel(r'$B_{max}$',fontsize=20)
frame2.set_ylabel(r'$\chi^2$',fontsize=20)

frame1.tick_params(labelsize=20)
frame2.tick_params(labelsize=20)
frame1.set_xticklabels([]) 
#plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/AMRVAC/CP_vs_Bmax.png',dpi=300)

#%%


obs = oifits.open(pionier_files[0])

#print([obs[x].header['EXTNAME'] for x in range(1,8)])

# index to station mapping 
indx2station = {obs['OI_ARRAY'].data['STA_INDEX'][i]:obs['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(obs['OI_ARRAY'].data['STA_NAME']))}

obs_v_model_dict = {'obs':{},'model':{}}

# ++++++++++++++++++++++++++++++++++++++++
# FOR A GIVEN OBSERVATION PUT OUR OBSERVATIONS IN THE obs_v_model_dict 

extname = 'OI_T3'
for i in range(obs[extname].data['T3PHI'].shape[0]):
    
    stations_tmp = list( np.sort([indx2station[obs[extname].data['STA_INDEX'][i][ss]] for ss in range(len(obs[extname].data['STA_INDEX'][0]))]) )
    stations_tmp = ''.join(stations_tmp)
    flags = obs[extname].data['FLAG'][i,:]
    
    obs_v_model_dict['obs'][stations_tmp] = {}
    for v in obs[extname].header.values():
        try: 
            if obs[extname].data[v].shape ==  obs[extname].data['T3PHI'].shape: #.shape:
                    obs_v_model_dict['obs'][stations_tmp][v] = obs[extname].data[v][i,:][~flags] #{'T3': obs[extname].data['T3PHI'][i,:][~flags] , 'T3PHIERR':obs[extname].data['T3PHIERR'][i,:][~flags]}
        except:
            print('did not work for ', v)
  
    #add wavelength 
    obs_v_model_dict['obs'][stations_tmp]['EFF_WAVE'] = obs['OI_WAVELENGTH'].data['EFF_WAVE'][~flags]
    
    
extname =  'OI_VIS2'
for i in range(obs[extname].data['VIS2DATA'].shape[0]):
    
    stations_tmp = list( np.sort([indx2station[obs[extname].data['STA_INDEX'][i][ss]] for ss in range(len(obs[extname].data['STA_INDEX'][0]))]) )
    stations_tmp = ''.join(stations_tmp)
    flags = obs[extname].data['FLAG'][i,:]
    
    obs_v_model_dict['obs'][stations_tmp] = {}
    for v in obs[extname].header.values():
        try: 
            if obs[extname].data[v].shape ==  obs[extname].data['VIS2DATA'].shape: #.shape:
                    obs_v_model_dict['obs'][stations_tmp][v] = obs[extname].data[v][i,:][~flags] #{'T3': obs[extname].data['T3PHI'][i,:][~flags] , 'T3PHIERR':obs[extname].data['T3PHIERR'][i,:][~flags]}
        except:
            print('did not work for ', v)
        
    #add wavelength 
    obs_v_model_dict['obs'][stations_tmp]['EFF_WAVE'] = obs['OI_WAVELENGTH'].data['EFF_WAVE'][~flags]
  


#%% 

par2au = 206265
rad2mas = 180/np.pi * 3600 * 1e3
d0={'PIO_no_dust':0.8, 'PIO_quasiHigh_dust':3, 'PIO_low_dust':1, 'PIO_high_dust':40}  # priors for fit
diam_fits = {}
oi_dict = {}

# -- simplistic noise model 
noise = {'|V|':0.01, # fractional error on visibility amplitude
         'V2':0.01, # fractional error on visibility squared amplitude
         'T3AMP':0.01, # fractional error on triple product amplitude
         'T3PHI':1.0, # error on triple product phase, in degrees
         'PHI':1.0, # error on phase, in degrees
         'FLUX':0.01 # fractional error on flux
        }


fits_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/fitted_radius_sensitivity_dust_study/*.fits')
model_grid = [f.split('/')[-1].split('.')[0] for f in fits_files]
#labels = ['PIO_no_dust', 'PIO_low_dust','PIO_quasiHigh_dust', 'PIO_high_dust']
massloss = [0, 1e-5,1e-8, 1e-4] # solar mass per year  ( for  ['PIO_no_dust', 'PIO_quasiHigh_dust', 'PIO_low_dust', 'PIO_high_dust'] respectively )

model_dict = {}
for l,f in zip(model_grid,fits_files):
    model_dict[l]=fits.open(f)
    
for l in model_grid:

    fits_file = model_dict[l]
    
    number_pixels = len(fits_file[0].data[0, 0, :])
    half_domain = fits_file[0].header['HALF_DOM']   # in AU
    distance = par2au * fits_file[0].header['DISTANCE']    # in AU
    x = np.linspace(-half_domain/(distance), half_domain/distance, number_pixels) # rad
    y = np.linspace(-half_domain/(distance), half_domain/distance, number_pixels) # rad
    input_array = fits_file[0].data
    cube = {}
    cube['scale'] = rad2mas * (np.max(x)-np.min(x)) / number_pixels     # 0.0788  mas / pixel
    cube['X'], cube['Y'] = np.meshgrid(x, y)
    cube['image'] = input_array
    cube['WL'] = fits_file[1].data
    wl = cube['WL']
    print('spectral channel resolution: R=%.1f'%(np.mean(cube['WL']/np.gradient(cube['WL']))))
    
    station = list(obs_v_model_dict['obs'])[0]
    
    data = [pmoired.oifake.makeFakeVLTI(['A0', 'B2', 'C1', 'D0'],
                                        (18, -69), # Simbad name or sky coordiates as (ra_h, dec_d)
                                        [19.22, 22.5], # list of LST for observations
                                        wl, # list of wavelength, in um
                                        cube=cube, # cube dictionnary (see above)
                                        noise =noise,
                                        ),
            pmoired.oifake.makeFakeVLTI( ['K0', 'G2', 'D0', 'J3'],
                            (18, -69), # Simbad name or sky coordiates as (ra_h, dec_d)
                            [19.22], # list of LST for observations
                            wl, # list of wavelength, in um
                            cube=cube, # cube dictionnary (see above)
                            ),
           ]
    
    oi = pmoired.OI() # create an empty PMOIRED OI object
    oi.data = data # note that data needs to be a list, even if it has only a single element!
    #oi.setupFit({'obs':['V2'],'min relative error':{'V2':0.01}}) # observable to display
    # oi.doFit({'ud':d0[l]})
    
    oi_dict[l]=oi
      
#%%
#e.g. 
#plt.plot( obs_v_model_dict['obs']['A0G1J2']['EFF_WAVE'] , obs_v_model_dict['obs']['A0G1J2']['T3PHI'] ,'.')

# ++++++++++++++++++++++++++++++++++++++++
# FOR A GIVEN MODEL OUR MODEL IN THE obs_v_model_dict 
"""
model = {'1,ud':10,  
         '1,x':0, 
         '1,y':0,
         '1,f':1,
         '2,ud':2, 
         '2,x':2, 
         '2,y':2,
         '2,f':0.8
        }"""

model = {'1,ud':3,  
         '1,x':0, 
         '1,y':0,
         '1,f':1,
         '2,ud':3, 
         '2,x':2, 
         '2,y':0,
         '2,f':0.1
        }

"""model = {'disk,diamin':4, 
         'disk,diamout':5, 
         'disk,profile':'$R**-1', 
         'disk,az amp1':1, 
         'disk,az projang1':60, 
         'disk,projang':45, 
         'disk,incl':-30, 
         'disk,x':-0.05, 
         'disk,y':-0.15, 
         'disk,spectrum':'($WL)**-2',
         'star,ud':10, 
         'star,spectrum':'($WL)**-3', 
        }"""

#model = {'diamin':3, 'diamout':5, 'profile':'$R**-2', 'az amp1':0.8, 'az projang1':60, 'projang':45, 'incl':-30, 'x':-0.2, 'y':0.1}

# -- wavelength vector to simulate data (in um)
WL =  1e6 * obs['OI_WAVELENGTH'].data['EFF_WAVE'][~flags] 
# -- list of sidereal time of observations (hours): must be a list or np.ndarray
lst = [QC_table.loc[date]['LST_chopped[hr]']]

coord = (QC_table.loc[date]['RA[deg]'], QC_table.loc[date]['DEC[deg]'])

station = obs['OI_ARRAY'].data['STA_NAME'] #[QC_table.loc[date]['config'][0+n:2+n] for n in [0,2,4,6]]
# show image
oi = pmoired.OI()
oi.showModel(model, WL=WL, imFov=10, showSED=False)

oi.data = [pmoired.oifake.makeFakeVLTI(station, coord, lst, WL, model=model)]
#oi.data[0]['OI_T3']['A0G1J2']['T3AMP'][0]

obs_v_model_dict['model'] = {}
for s in obs_v_model_dict['obs'].keys():
    obs_v_model_dict['model'][s] = {}
    if len(s)==6: #triangles
        for k in oi.data[0]['OI_T3'][s]:
            obs_v_model_dict['model'][s][k] = oi.data[0]['OI_T3'][s][k]


    elif len(s)==4: #baselines
        for k in oi.data[0]['OI_VIS2'][s]:
            obs_v_model_dict['model'][s][k] = oi.data[0]['OI_VIS2'][s][k]
    

    else:
        raise TypeError('something wrong here') 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 07:56:59 2023

@author: bcourtne


fit UD diameters for pionier, gravity and matisse data 

-for each instrument 
make a wvl grid 
get V2 V2err and interpolate onto grid 
get unique baselines 
put in dataframe 

-merge all dataframes to one master 

- fit ellipse model per wavelength bin

To Do 
===
read
https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html

it could also be interesting to fit the center of the ellipse with phase or closure phase information.. 

#The u and v coordinates describe E-W and N-S components of the projected interferometer baseline.

Because brightness is real, each observation provides us a second point,
where: V(-u,-v) = V*(u,v) !!!


#include flag df !!! 

"""
import glob
import numpy as np
import pandas as pd 
import pyoifits as oifits
from scipy.interpolate import interp1d
from scipy import special
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import emcee
import corner
import os
from matplotlib import colors


def baseline2uv_matrix( h, d):
    """
    

    Parameters
    ----------
    Not used wvl : TYPE  need to divide by this later
        DESCRIPTION. wavelength (m)
    h : TYPE
        DESCRIPTION. target hour angle (radian)
    d : TYPE
        DESCRIPTION. declination of target (radian)

    Returns
    -------
    None.

    """
    
    #from https://web.njit.edu/~gary/728/Lecture6.html
    
    # https://www.icrar.org/wp-content/uploads/2018/11/Perley_Basic_Radio_Interferometry_Geometry.pdf
    #wvl in m
    #location should be tuple of (latitude, longitude) coordinates degrees 
    #telescope coordinates should be astropy.coordinates  SkyCoord class
    
    #hour angle (radian)
    #h = ( get_LST( location,  datetime ) - telescope_coordinates.ra ).radian #radian
    
    # convert dec to radians also
    #d = telescope_coordinates.dec.radian #radian
    
    mat =  np.array([[np.sin(h), np.cos(h), 0],\
           [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],\
           [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d) ]] )
        
    return(mat)   




def B_L(wvl,T): #Spectral density as function of temperature
    #wvl is wavelength vector to calculate spectral density at
    #T is temperature of the black body 
    c = 3e8 #speed of light m/s
    h = 6.63e-34 # plank's constant J.s
    kB = 1.38e-23 #boltzman constant m2 kg /s^2/K
    Bb = 2*h*c**2/wvl**5 * 1 / (np.exp(h*c/(wvl*kB*T)) - 1)
    return(Bb)


def disk_v2(rho, theta):
    #rho = B/wvl (m/m), theta is angular diamter (radians)
    v2 = ( 2*special.j1( np.pi*rho*theta ) / (np.pi*rho*theta) )**2 
    return v2


    
    
def chi2(y_model,y_true,yerr):
    return(sum( (y_model-y_true)**2/yerr**2 ) )

    
def rad2mas(rad):
    return(rad*180/np.pi * 3600 * 1e3)

def mas2rad(mas):
    return(mas*np.pi/180 / 3600 / 1e3)


def fit_prep(files, EXTVER=None,flip=True):    
    # pionier data is [wvl, B], while gravity is [B,wvl ] (so gravity we want flip =Tue)              
    
    if EXTVER==None:
        wvl_EXTNAME = 'OI_WAVELENGTH'
        v2_EXTNAME = 'OI_VIS2'
    
    else:
        wvl_EXTNAME = ('OI_WAVELENGTH',EXTVER)
        v2_EXTNAME = ('OI_VIS2',EXTVER)
        
    hdulists = [oifits.open(f) for f in files]
    
    print( len( hdulists) ,'\n\n\n')
    wvls = [ h[wvl_EXTNAME].data['EFF_WAVE'] for h in hdulists]
    wvl_grid = np.median( wvls , axis=0) # grid to interpolate wvls 
    
    data_dict = {} 
    for ii, h in enumerate( hdulists ):
        
        
        file = files[ii].split('/')[-1]
        print(f'looking at file {ii}/{len(hdulists)}, which is \n {file} \n')
        #Bx = h['OI_VIS2'].data['UCOORD'] # east-west 
        #By = h['OI_VIS2'].data['VCOORD'] # north-south
                
        dec = np.deg2rad(h[0].header['DEC'])
        ha = np.deg2rad( h[0].header['LST']/60/60 )
        B = [] # to holdprojected baseline !
        for Bx,By in zip( h['OI_VIS2'].data['UCOORD'],h['OI_VIS2'].data['VCOORD'] ): # U=east-west , V=north-sout
            #lambda_u, lambda_v, _ = baseline2uv_matrix(ha, dec) @ np.array( [Bx,By,0] ) # lambda_u has to be multiplied by lambda to get u±!!!
            #B.append( (lambda_u, lambda_v) ) # projected baseline !
            B.append( (Bx,By) ) # projected baseline !
        #B = [(a,b) for a,b in zip(lambda_u, lambda_v) ] # projected baseline ! #(h[v2_EXTNAME].data['UCOORD'], h[v2_EXTNAME].data['VCOORD']) #np.sqrt(h[v2_EXTNAME].data['UCOORD']**2 + h[v2_EXTNAME].data['VCOORD']**2)
        
        v2_list = []
        v2err_list = []
        flag_list = []
        dwvl = []
        obs_time = []

        for b in range(len(B)):
            
            #for each baseline make interpolation functions 
            V2Interp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2DATA'][b,:] ,kind='linear', fill_value =  "extrapolate" )
            
            V2errInterp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2ERR'][b,:] ,kind='linear', fill_value =  "extrapolate" )
            
            FlagInterp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['FLAG'][b,:] ,kind='nearest', fill_value =  "extrapolate" )

            dwvl.append( np.max( [1e9 * ( abs( ww -  wvl_grid ) ) for ww in h[wvl_EXTNAME].data['EFF_WAVE'] ] ) )
            
            obs_time.append( [h[0].header['DATE-OBS'],h[0].header['LST']/60/60 ,h[0].header['RA'], h[0].header['DEC'] ] )   #LST,ec,ra should be in deg#
            
            v2_list.append(  V2Interp_fn( wvl_grid ) )
            
            v2err_list.append( V2errInterp_fn( wvl_grid ) )
            
            flag_list.append( FlagInterp_fn( wvl_grid ) )
          
        print('max wvl difference in interpolatation for {} = {}nm'.format(file, np.max(dwvl)))
        
        # Put these in dataframes 
        v2_df = pd.DataFrame( v2_list , columns = wvl_grid , index = B )
        
        v2err_df = pd.DataFrame( v2err_list , columns = wvl_grid , index = B)
        
        time_df = pd.DataFrame( obs_time , columns = ['DATE-OBS','LST', 'RA','DEC'] , index = B)
        
        flag_df = pd.DataFrame( np.array(flag_list).astype(bool) , columns = wvl_grid , index = B )
        
        data_dict[file] = {'v2':v2_df, 'v2err':v2err_df, 'flags' : flag_df,'obs':time_df}
        
        v2_df = pd.concat( [data_dict[f]['v2'] for f in data_dict] , axis=0)
        
        v2err_df = pd.concat( [data_dict[f]['v2err'] for f in data_dict] , axis=0)
        
        flag_df = pd.concat( [data_dict[f]['flags'] for f in data_dict] , axis=0)
        
        obs_df = pd.concat( [data_dict[f]['obs'] for f in data_dict] , axis=0)

           
    return( v2_df , v2err_df , flag_df,  obs_df)


pionier_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')


gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]


pion_v2_df , pion_v2err_df  , pion_flag_df,  pion_obs_df = fit_prep(pionier_files)

grav_p1_v2_df , grav_p1_v2err_df, grav_p1_flag_df , grav_p1_obs_df= fit_prep(gravity_files, EXTVER = 11 )
grav_p2_v2_df , grav_p2_v2err_df , grav_p2_flag_df , grav_p2_obs_df = fit_prep(gravity_files, EXTVER = 12 )

mati_L_v2_df , mati_L_v2err_df , mati_L_flag_df, mati_L_obs_df = fit_prep(matisse_files_L )
mati_N_v2_df , mati_N_v2err_df , mati_N_flag_df, mati_N_obs_df = fit_prep(matisse_files_N )


#%% Testing 
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ellipse_v2_for_plot(u,v,a,phi,theta):
    #rho = B/wvl (m/m), theta is angular diamter (radians)
    #u,v = x[:,0], x[:,1]
    #a, phi, theta  = params
    #a, phi  = params
    #theta = ud_wvl 
    rho = np.sqrt( (u/a*np.cos(phi) - v/a*np.sin(phi))**2  + (v*a*np.cos(phi) + u*a*np.sin(phi))**2 )
    
    v2 = ( 2*special.j1( np.pi*rho*theta ) / (np.pi*rho*theta) )**2 

    return v2
# plot v2 vs u v position 
colormap = plt.cm.bwr #or any other colormap
normalize = colors.Normalize(vmin=0, vmax=1)

#include flag df !!! 

v2_df_list = [pion_v2_df.copy() , grav_p1_v2_df.copy(), mati_L_v2_df.copy(), mati_N_v2_df.copy()]
v2err_df_list = [pion_v2err_df.copy() , grav_p1_v2err_df.copy(), mati_L_v2err_df.copy(), mati_N_v2err_df.copy()]
flag_df_list =  [pion_flag_df.copy() , grav_p1_flag_df.copy(), mati_L_flag_df.copy(), mati_N_flag_df.copy()]

fig,ax = plt.subplots(1,len(v2_df_list),figsize=(32,8),sharex=True,sharey=True)

for i, (df, df_mask, df_err) in enumerate(zip( v2_df_list , flag_df_list,  v2err_df_list)):
#df = mati_L_v2_df.copy() #grav_p2_v2_df.copy()  #mati_L_v2_df.copy() #mati_N_v2_df.copy()
#df_mask = mat_L_flag_df.copy()
    
    df_mask = df_mask * ((df<=1) & (df>=0) & (abs( df_err )<0.5)).values

    Bx = np.array( [bx[0] for bx in df.index] )  #projected east-west component 
    By = np.array( [by[1] for by in df.index] )  #projected north-south compone
    for wvl in df.columns: #wavelengths (m)
        filt = ~df_mask[wvl]
        fff = ax[i].scatter( Bx[filt]/wvl, By[filt]/wvl, c = df[wvl].values[filt] ,norm = colors.Normalize(vmin=0, vmax=1), cmap = plt.cm.coolwarm, alpha=0.9)#,
        #Because brightness is real, each observation provides us a second point, where: V(-u,-v) = V*(u,v) !!!
        ax[i].scatter( -Bx[filt]/wvl, -By[filt]/wvl, c = df[wvl].values[filt] ,norm = colors.Normalize(vmin=0, vmax=1), cmap = plt.cm.coolwarm, alpha=0.9)

    ax[i].set_xlabel(r'$B_x^{proj}/\lambda E->W (rad^{-1})$',fontsize=20)
    #ax[i].set_ylabel(r'B_x^{proj}/\lambda N->S (rad$^{-1})$',fontsize=20)
    ax[i].tick_params(labelsize=20)
    ax[i].grid()
    
ax[0].set_ylabel(r'$B_x^{proj}/\lambda N->S (rad^{-1})$',fontsize=20)   

divider = make_axes_locatable(ax[-1])
cax = divider.append_axes('right', size='5%', pad=0.2)
fig.colorbar(fff, cax=cax, orientation='vertical')

# v2 map of ellipse vs disk .. ellipse function works (i.e. as a->1, we recover disk model)
b = np.linspace(-100,100,100)
XX,YY = np.meshgrid(b,b)


fig,ax = plt.subplots(1,2,figsize=(14,7))

ud_wvl = mas2rad(10)
wvl=2.2e-6
ax[0].pcolormesh(XX/wvl, YY/wvl, disk_v2( 1/wvl *(XX**2+YY**2)**0.5, ud_wvl ) )
ax[0].set_title(r'$|V|^2$ uniform disk, $\theta$=10mas, $\lambda$={}um'.format(wvl*1e6))

"""xx=[]
for x in b:
    for y in b:
        xx.append( [x/wvl,y/wvl]  )"""

ax[1].pcolormesh(XX/wvl, YY/wvl, ellipse_v2_for_plot(u=1/wvl*XX,v=1/wvl*YY,a=1.7,phi=np.pi/4,theta=ud_wvl ) )  
ax[1].set_title(r'$|V|^2$ ellipse, a=1.7, $\phi=45 deg$, $\theta$=10mas, $\lambda$={}um'.format(wvl*1e6))
for i in [0,1]:
    ax[i].set_xlabel(r'u (rad$^{-1}$)')
    ax[i].set_ylabel(r'v (rad$^{-1}$)')
    
    
# manual fitto convince myself 
df=mati_L_v2_df.copy()
wvl = list(df.columns)[-3]
y=df[wvl]
filt = (y>0) & (y<=1)
# Plottimg the visibilities 
Bx = np.array( [bx[0] for bx in df.index[filt]] )  #projected east-west component 
By = np.array( [by[1] for by in df.index[filt]] ) 

y_model = ellipse_v2_for_plot(u=1/wvl*Bx,v=1/wvl*By,a=2.2,phi=np.pi/4,theta=ud_wvl )

plt.plot( ellipse_v2_for_plot(u=1/wvl*Bx,v=1/wvl*By,a=1,phi=0,theta=mas2rad(3.5) ) , y[filt],'.');plt.plot(y[filt],y[filt],'-',color='r')

#%%  FITTING ELLIPSE INCLUDING UD AS FREE PARAMETER 

def ellipse_v2(x,params):
    #rho = B/wvl (m/m), theta is angular diamter (radians)
    u,v = x[:,0], x[:,1]
    a, phi, theta  = params
    #a, phi  = params
    #theta = ud_wvl  # global parameter defined outside of function 
    rho = np.sqrt( (u/a*np.cos(phi) - v/a*np.sin(phi))**2  + (v*a*np.cos(phi) + u*a*np.sin(phi))**2 )
    
    v2 = ( 2*special.j1( np.pi*rho*theta ) / (np.pi*rho*theta) )**2 

    return v2



def log_likelihood(params, x, y, yerr):
    #u,v = x[:,0], x[:,1] #x is list with [u,v]
    model = ellipse_v2(x,params)
    sigma2 = yerr**2  #+ model**2 * np.exp(2 * log_f)
    return( -0.5 * np.sum((y - model) ** 2 / sigma2 ) )#+ np.log(sigma2)) )



def log_prior(params):
    a, phi, theta = params
    #a, phi = params  
    
    if not (0 <= phi < np.pi): #& (a>1): #  uniform prior on rotation of ellipse between 0-180 degrees
        return(-np.inf)


    if not 1 <= a < 100: # uniform prior on a (major / minor axis ratio) between 1-100 (note we don't go below 1 to avoid degenerancy with rotation phi)
        return(-np.inf)
    
    else:
        #gaussian prior on a
        mu = mas2rad( ud_wvl ) #rad - note this is an external variable that should be defined 
        sigma = mas2rad( 2 ) #* ud_wvl_err ) # 
        return(np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta-mu)**2/sigma**2)

    #if 0 < theta < mas2rad(200):# and -10.0 < log_f < 1.0:
    #    return 0.0
    #else:
    #    return -np.inf
    
    """if (0 < phi < np.pi): #& (a>1): #  uniform prior on rotation of ellipse between 0-180 degrees
        return 0.0
    else:
        return -np.inf

    if 1 < a < 100: # uniform prior on a (major / minor axis ratio) between 1-100 (note we don't go below 1 to avoid degenerancy with rotation phi)
        return 0.0
    else:
        return -np.inf"""

def log_probability(params, x, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(params, x, y, yerr)
    
plot=True
fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/ellipse_fit/'
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
param_labels=['a','phi','theta'] #['a','phi'] #param_labels=['a','phi','theta'] #a, phi,theta = params

# filters 
grav_B_filt = grav_p1_v2_df.index.values !=0 
grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6)

# matisse wvl limits from https://www.eso.org/sci/facilities/paranal/instruments/matisse.html
mat_L_wvl_filt = (mati_L_v2_df.columns > 3.2e-6) & (mati_L_v2_df.columns < 3.9e-6) #| (mati_L_v2_df.columns > 4.5e-6) 
mat_M_wvl_filt = (mati_L_v2_df.columns > 4.5e-6) &  (mati_L_v2_df.columns <= 5e-6)
mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) & (mati_N_v2_df.columns <= 12.1e-6)#| (mati_L_v2_df.columns > 4.5e-6)

# instrument vis tuples 
# instrument vis tuples 
pion_tup = (pion_v2_df , pion_v2err_df)
grav_p1_tup = (grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2err_df.columns[::50]][grav_B_filt] )
grav_p2_tup = (grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2err_df.columns[::50]][grav_B_filt] )
mati_L_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2err_df.columns[mat_L_wvl_filt][::5]] )
mati_M_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2err_df.columns[mat_M_wvl_filt][::5]] )
mati_N_tup = (mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2err_df.columns[mat_N_wvl_filt][::5]] )

"""pion_tup = (pion_v2_df , pion_v2err_df)
grav_p1_tup = (grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2_df.columns[::50]][grav_B_filt] )
grav_p2_tup = (grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2_df.columns[::50]][grav_B_filt] )
mati_L_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] )
mati_M_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] )
mati_N_tup = (mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] )
"""

ins_vis_dict = {'Pionier (H)':pion_tup, 'Gravity P1 (K)' : grav_p1_tup, \
                'Gravity P2 (K)' : grav_p2_tup, 'Matisse (L)':mati_L_tup,\
                    'Matisse (M)':mati_M_tup, 'Matisse (N)':mati_N_tup }

#v2_df, v2err_df = pion_v2_df , pion_v2err_df

#grav_B_filt = grav_p1_v2_df.index.values !=0  #sometimes we get zero baseline in data that screws things up
#grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6) # e.g.
#v2_df, v2err_df = grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2_df.columns[::50]][grav_B_filt]
#v2_df, v2err_df = grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2_df.columns[::50]][grav_B_filt]

#mat_L_wvl_filt = (mati_L_v2_df.columns < 4.1e-6) | (mati_L_v2_df.columns > 4.5e-6) # e.g.
#v2_df, v2err_df = mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]]

#mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) #| (mati_L_v2_df.columns > 4.5e-6) # e.g.
#v2_df, v2err_df = mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]]



ellipse_fit_per_ins = {} # to hold fitting results per instrument photometric band

for ins in ins_vis_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to Ellipse model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_vis_dict[ins]
    
    ellipse_fit_results = {}
    
    redchi2 = []
    #params = [] #best fit
    ellipse_result_per_wvl_dict = {xxx:{ 'mean' :[], 'median' : [], 'err' : [] } for xxx in param_labels}
    
    intermediate_results_dict = {}
    
    for wvl_indx, wvl in enumerate(v2_df.columns ):
        
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        #ud_wvl_err = ud_fits['ud_err'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        
        intermediate_results_dict[wvl] = {  } #{'rho':[], 'v2_obs':[], 'v2_obs_err':[],\'v2_model':[],'samplers':[] }

        # u,v coorindates ### #
        #SOMETHING WRONG HERE 
        rho = 1/wvl  *  np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).T # np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).reshape(len(v2_df.index.values),2)
        # Check rho matches v2_df.index.values tuples
        v2 = v2_df[wvl].values  # |V|^2
        v2_err = v2err_df[wvl].values # uncertainty 
        
        # filter out unphysical V2 values 
        v2_filt = (v2>0) & (v2<1.1)
        
        # short hand model notation 
        x, y, yerr = rho[v2_filt] , v2[v2_filt], v2_err[v2_filt]
        
        # TO CHECK WE CAN PLOT (SQRT(X[0]**2+X[1]**2), Y) SHOULD SEE NICE V2 CURVE FROM DATA! 
    
        """
        plt.figure()
        plt.semilogy( [ -log_likelihood(mas2rad(i), x, y, yerr) for i in range(10) ] )
        
        np.random.seed(42)
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([ mas2rad(5) ]) #, np.log(0.1)]) #+ 0.1 * np.random.randn(2)
        soln = minimize(nll, initial, args=(x, y, yerr),tol=mas2rad(0.1))
        """
        
        
        
        #pos = soln.x + 1e-4 * np.random.randn(32, 3)
        #pos = [mas2rad(5)]
        #nwalkers, ndim = 32,1
        
        nwalkers = 150 #32
        
        # initialize at UD fit (save a table)quick grid search 
        theta0 = mas2rad( ud_wvl ) #rad
        
        #do rough grid search 
        best_chi2 = np.inf # initialize at infinity 
        for a_tmp in np.linspace( 1,10,20):
            for phi_tmp in np.linspace( 0, np.pi, 10):
                
                params_tmp=[a_tmp, phi_tmp, theta0]
                #params_tmp=[a_tmp, phi_tmp]
                y_model_cand = ellipse_v2(x, params_tmp) 
                
                chi2_tmp = chi2(y_model_cand  , y, yerr)
                if chi2_tmp < best_chi2:
                    best_chi2 = chi2_tmp 
                    initial = params_tmp
                    
        print(f'best initial parameters = {initial} with chi2={best_chi2}')
        #a0 = 1 #squeeze/stretching (1=circle) - no units
        #phi0 = 0 #rotation (rad)

        #initial = np.array([ a0, phi0, theta0 ])
        ndim = len(initial)
        

        p0 = [initial + np.array([0.1, np.deg2rad(10), theta0/10 ]) * np.random.rand(3)  for i in range(nwalkers)]
        #p0 = [initial + np.array([0.1, np.deg2rad(10)]) * np.random.rand(ndim)  for i in range(nwalkers)]
        

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=( x, y, yerr )
        )
        sampler.run_mcmc(p0, 1000, progress=True);
        
        #samples = sampler.get_chain(flat=True)
        
        #plt.hist(np.log10(samples) ) , bins = np.logspace(-9,-7,100)) #[-1,:,0])
        
        #plt.hist( np.log10( samples ) , bins=np.linspace(-9,-5,100 ))
        
        
        # use sampler.get_autocorr_time()
        # The samples in an MCMC chain are not independent. consider every n samples to thin out to integrated autocorrelation time without loss of statistical power       
        tau = sampler.get_autocorr_time() #                                                                                                                                 
        thin = int(0.5 * np.min(tau))
        print(f'based on sampler autocorrelation time we are considering every {thin} samples')
        flat_samples = sampler.get_chain(discard=100, thin=thin, flat=True)
        
        
      
        if plot:
            flat_samples4plot = flat_samples.copy()
            flat_samples4plot[:,1] = np.rad2deg(flat_samples4plot[:,1])
            flat_samples4plot[:,2] = rad2mas(flat_samples4plot[:,2])
            
            plt.figure()
            #fig=corner.corner( flat_samples ,labels=['a',r'$\phi$',r'$\theta$'],quantiles=[0.16, 0.5, 0.84],\
            #           show_titles=True, title_kwargs={"fontsize": 12})
            fig=corner.corner( flat_samples4plot ,labels=['a',r'$\phi$ [deg]',r'$\theta$ [mas]'],quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

            fig.gca().annotate(f'{ins} - {round(1e6*wvl,2)}um',xy=(1.0, 1.0),xycoords="figure fraction", xytext=(-20, -10), textcoords="offset points", ha="right", va="top")
            
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            plt.savefig(os.path.join(fig_path,f'ellipse_mcmc_corner_{ins.split()[0]}_{round(1e6*wvl,2)}um.jpeg'))
            
        """plt.figure() 
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ')
        plt.xlabel('Baseline (m)')
        plt.ylabel(r'$V^2$')
        plt.plot(v2_df.columns,  disk_v2( rho, np.mean( rad2mas( flat_samples[:, :] ) ) *1e-3 * np.pi/180 / 3600  ) ,'.')
        """
        
        #y_model = np.median( rad2mas( flat_samples[:, :] ) ) * 1e-3 * np.pi/180 / 3600
        
        #for i,k in enumerate(ellipse_result_per_wvl_dict):
        #    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
        #    q = np.diff(mcmc)
        
        
        #    ellipse_result_per_wvl_dict[k]['mean'].append( np.mean(  flat_samples[:, i] ) )
        #    ellipse_result_per_wvl_dict[k]['median'].append( mcmc[1] )
        #    ellipse_result_per_wvl_dict[k]['err'].append( q )
        
        """
        fig1 = plt.figure(1)
        #Plot Data-model
        frame1=fig1.add_axes((.1,.3,.8,.6))
        fig1.set_tight_layout(True)
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ',color='darkred',alpha=0.6,label='measured')
        plt.plot(v2_df.columns,  disk_v2( rho, mas2rad(diam_median[-1])  ) ,'.',color='darkblue',alpha=0.6, label='model')
        plt.ylabel(r'$V^2$', fontsize=15)
        plt.gca().tick_params(labelsize=13)
        plt.legend(fontsize=15)
        frame2=fig1.add_axes((.1,.1,.8,.2))    
          
        plt.plot(v2_df.columns, ((y-y_model.values)/yerr),'o',color='k')
        plt.axhline(0,color='g',linestyle='--')
        plt.ylabel(r'$\chi_i$',fontsize=15 ) #r'$\Delta V^2 /\sigma_{V2}$',fontsize=15)
        plt.gca().tick_params(labelsize=13)
        plt.xlabel('Baseline (m)',fontsize=15)"""
        
        

        for i,k in enumerate(param_labels):
            
            intermediate_results_dict[wvl][k]={}
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
            q = np.diff(mcmc)
            
            intermediate_results_dict[wvl][k]['mean'] = np.mean(  flat_samples[:, i] ) 
            intermediate_results_dict[wvl][k]['median'] = mcmc[1] 
            intermediate_results_dict[wvl][k]['err'] = q 
            
            #ellipse_result_per_wvl_dict
            
        #best fit
        best_params_wvl = [intermediate_results_dict[wvl][k]['median'] for k in param_labels] 
        
        #
        y_model = ellipse_v2(x, best_params_wvl) #disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
        redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df[wvl])-ndim  ))
        
              
        intermediate_results_dict[wvl]['rho'] = x
        intermediate_results_dict[wvl]['v2_obs'] = y
        intermediate_results_dict[wvl]['v2_obs_err'] = yerr
        intermediate_results_dict[wvl]['v2_model'] = y_model
        intermediate_results_dict[wvl]['samplers'] = flat_samples
        
        #reduced chi2 
        #redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df.iloc[wvl_indx])-1))
        
        print('reduced chi2 = {}'.format(chi2(y_model, y, yerr) / (len(v2_df[wvl])-ndim )) )
    
    #for i,k in enumerate(ellipse_result_per_wvl_dict):
    #    ellipse_fit_results[k]={}
    #    ellipse_fit_results[k]['mean'] = diam_mean
    #    ellipse_fit_results[k]['median'] = diam_median
    #    ellipse_fit_results[k]['err'] = diam_err
        
    ellipse_fit_results['redchi2'] = redchi2
    
    ellipse_fit_results['intermediate_results'] = intermediate_results_dict
    
    ellipse_fit_per_ins[ins] = ellipse_fit_results




#%% Plot Ellipse results 
fig1 = plt.figure(1,figsize=(10,12))
fig1.set_tight_layout(True)

frame1 = fig1.add_axes((.1,.7,.8,.3))
frame2 = fig1.add_axes((.1,.4,.8,.3))
frame3 = fig1.add_axes((.1,.1,.8,.3))
frame4 = fig1.add_axes((.1,.0,.8,.1))

fontsize=20
#for ins, col in zip(ellipse_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
for ins, col in zip(ellipse_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
    if 1: #ins!='Matisse (N)':
        wvl_grid = np.array( list( ellipse_fit_per_ins[ins]['intermediate_results'].keys() ) )
        
        redchi2 = ellipse_fit_per_ins[ins]['redchi2']
        frame4.semilogy(1e6*wvl_grid, redchi2, '-',lw=2, color=col)
        frame4.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
        frame4.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
        frame4.tick_params(labelsize=fontsize)
        
        for fig , k in zip( [frame1,frame2,frame3], param_labels):
            median = np.array( [ellipse_fit_per_ins[ins]['intermediate_results'][wvl][k]['median'] for wvl in wvl_grid] )
            err = np.array( [ellipse_fit_per_ins[ins]['intermediate_results'][wvl][k]['err'] for wvl in wvl_grid] )
            
            #fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
            fig.set_ylabel(k)
            if k=='a':
                fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(1,2)
                fig.tick_params(labelsize=fontsize)
                fig.grid()
                fig.set_ylabel('a [unitless]\n',fontsize=fontsize)
            if k=='theta':
                fig.errorbar(1e6*wvl_grid, rad2mas(median), yerr=np.array(rad2mas(err)).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(0,200)
                fig.set_yscale('log')
                fig.tick_params(labelsize=fontsize)
                fig.grid()
                fig.set_ylabel(r'$\theta$ [mas]',fontsize=fontsize)
            if k=='phi':
                fig.errorbar(1e6*wvl_grid, np.rad2deg( median), yerr=np.array(np.rad2deg(err)).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(0,180)
                fig.tick_params(labelsize=fontsize)
                fig.grid()
                fig.set_ylabel(r'$\phi$ [deg]',fontsize=fontsize)
                
                



frame1.legend(fontsize=fontsize)
frame1.set_title('RT Pav Ellipse Fit vs Wavelength')
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/FIT_Ellipse_logscale.pdf',bbox_inches='tight',dpi=400)




#plt.tight_layout()







#%%  FITTING ELLIPSE FIXING AREA TO FITTED UD AREA 

def ellipse_v2(x, params, theta):
    #rho = B/wvl (m/m), theta is angular diamter (radians)
    u,v = x[:,0], x[:,1]
    a, phi = params
    #a, phi  = params
    #theta = ud_wvl  # global parameter defined outside of function 
    rho = np.sqrt( (u/a*np.cos(phi) - v/a*np.sin(phi))**2  + (v*a*np.cos(phi) + u*a*np.sin(phi))**2 )
    
    v2 = ( 2*special.j1( np.pi*rho*theta ) / (np.pi*rho*theta) )**2 

    return v2



def log_likelihood(params, x, y, yerr, theta):
    #u,v = x[:,0], x[:,1] #x is list with [u,v]
    model = ellipse_v2(x,params, theta)
    sigma2 = yerr**2  #+ model**2 * np.exp(2 * log_f)
    return( -0.5 * np.sum((y - model) ** 2 / sigma2 ) )#+ np.log(sigma2)) )



def log_prior(params, theta):
    a, phi = params
    #a, phi = params  
    
    if not (0 < phi < np.pi): #& (a>1): #  uniform prior on rotation of ellipse between 0-180 degrees
        return(-np.inf)


    if not 1 < a < 100: # uniform prior on a (major / minor axis ratio) between 1-100 (note we don't go below 1 to avoid degenerancy with rotation phi)
        return(-np.inf)
    
    #gaussian prior on a
    mu = mas2rad( ud_wvl ) #rad - note this is an external variable that should be defined 
    sigma = mas2rad( 2 ) #* ud_wvl_err ) # 
    return(np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta-mu)**2/sigma**2)
    #gaussian prior on a
    mu = mas2rad( theta ) #rad - note this is an external variable that should be defined 
    sigma = mas2rad( 2 ) #* ud_wvl_err ) # 
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta-mu)**2/sigma**2



def log_probability(params, x, y, yerr, theta):
    lp = log_prior(params, theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(params, x, y, yerr, theta)
    
plot=True
fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/ellipse_fit/'
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
param_labels=['a','phi'] #['a','phi'] #param_labels=['a','phi','theta'] #a, phi,theta = params

# filters 
grav_B_filt = grav_p1_v2_df.index.values !=0 
grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6)

# matisse wvl limits from https://www.eso.org/sci/facilities/paranal/instruments/matisse.html
mat_L_wvl_filt = (mati_L_v2_df.columns > 3.2e-6) & (mati_L_v2_df.columns < 3.9e-6) #| (mati_L_v2_df.columns > 4.5e-6) 
mat_M_wvl_filt = (mati_L_v2_df.columns > 4.5e-6) &  (mati_L_v2_df.columns <= 5e-6)
mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) & (mati_N_v2_df.columns <= 12.1e-6)#| (mati_L_v2_df.columns > 4.5e-6)

# instrument vis tuples 
pion_tup = (pion_v2_df , pion_v2err_df)
grav_p1_tup = (grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2_df.columns[::50]][grav_B_filt] )
grav_p2_tup = (grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2_df.columns[::50]][grav_B_filt] )
mati_L_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] )
mati_M_tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] )
mati_N_tup = (mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] )


ins_vis_dict = {'Pionier (H)':pion_tup, 'Gravity P1 (K)' : grav_p1_tup, \
                'Gravity P2 (K)' : grav_p2_tup, 'Matisse (L)':mati_L_tup,\
                    'Matisse (M)':mati_M_tup, 'Matisse (N)':mati_N_tup }

#v2_df, v2err_df = pion_v2_df , pion_v2err_df

#grav_B_filt = grav_p1_v2_df.index.values !=0  #sometimes we get zero baseline in data that screws things up
#grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6) # e.g.
#v2_df, v2err_df = grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2_df.columns[::50]][grav_B_filt]
#v2_df, v2err_df = grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2_df.columns[::50]][grav_B_filt]

#mat_L_wvl_filt = (mati_L_v2_df.columns < 4.1e-6) | (mati_L_v2_df.columns > 4.5e-6) # e.g.
#v2_df, v2err_df = mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]]

#mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) #| (mati_L_v2_df.columns > 4.5e-6) # e.g.
#v2_df, v2err_df = mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]]



ellipse_fit_per_ins = {} # to hold fitting results per instrument photometric band

for ins in ins_vis_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to Ellipse model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_vis_dict[ins]
    
    ellipse_fit_results = {}
    
    redchi2 = []
    #params = [] #best fit
    ellipse_result_per_wvl_dict = {xxx:{ 'mean' :[], 'median' : [], 'err' : [] } for xxx in param_labels}
    
    intermediate_results_dict = {}
    
    for wvl_indx, wvl in enumerate(v2_df.columns ):
        
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        #ud_wvl_err = ud_fits['ud_err'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        
        intermediate_results_dict[wvl] = {  } #{'rho':[], 'v2_obs':[], 'v2_obs_err':[],\'v2_model':[],'samplers':[] }

        # u,v coorindates
        rho = 1/wvl  * np.array([[a[0] for a in v2_df.index.values],[a[1] for a in v2_df.index.values]]).reshape(len(v2_df.index.values),2)

        v2 = v2_df[wvl].values  # |V|^2
        v2_err = v2err_df[wvl].values # uncertainty 
        
        # filter out unphysical V2 values 
        v2_filt = (v2>0) & (v2<1.1)
        
        # short hand model notation 
        x, y, yerr = rho[v2_filt] , v2[v2_filt], v2_err[v2_filt]
    
        """
        plt.figure()
        plt.semilogy( [ -log_likelihood(mas2rad(i), x, y, yerr) for i in range(10) ] )
        
        np.random.seed(42)
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([ mas2rad(5) ]) #, np.log(0.1)]) #+ 0.1 * np.random.randn(2)
        soln = minimize(nll, initial, args=(x, y, yerr),tol=mas2rad(0.1))
        """
        
        
        
        #pos = soln.x + 1e-4 * np.random.randn(32, 3)
        #pos = [mas2rad(5)]
        #nwalkers, ndim = 32,1
        
        nwalkers = 150 #32
        
        # initialize at UD fit (save a table)quick grid search 
        theta0 = mas2rad( ud_wvl ) #rad
        
        #do rough grid search 
        best_chi2 = np.inf # initialize at infinity 
        for a_tmp in np.linspace( 1,10,20):
            for phi_tmp in np.linspace( 0, np.pi, 10):
                
                params_tmp=[a_tmp, phi_tmp]
                #params_tmp=[a_tmp, phi_tmp]
                y_model_cand = ellipse_v2(x, params_tmp, theta0) 
                
                chi2_tmp = chi2(y_model_cand  , y, yerr)
                if chi2_tmp < best_chi2:
                    best_chi2 = chi2_tmp 
                    initial = params_tmp
                    
        print(f'best initial parameters = {initial} with chi2={best_chi2}')
        #a0 = 1 #squeeze/stretching (1=circle) - no units
        #phi0 = 0 #rotation (rad)

        #initial = np.array([ a0, phi0, theta0 ])
        ndim = len(initial)
        

        p0 = [initial + np.array([0.1, np.deg2rad(10)]) * np.random.rand(2)  for i in range(nwalkers)]
        #p0 = [initial + np.array([0.1, np.deg2rad(10)]) * np.random.rand(ndim)  for i in range(nwalkers)]
        

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=( x, y, yerr, theta0 )
        )
        sampler.run_mcmc(p0, 1000, progress=True);
        
        #samples = sampler.get_chain(flat=True)
        
        #plt.hist(np.log10(samples) ) , bins = np.logspace(-9,-7,100)) #[-1,:,0])
        
        #plt.hist( np.log10( samples ) , bins=np.linspace(-9,-5,100 ))
        
        
        # use sampler.get_autocorr_time()
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        
        
      
        if plot:
            plt.figure()
            #fig=corner.corner( flat_samples ,labels=['a',r'$\phi$',r'$\theta$'],quantiles=[0.16, 0.5, 0.84],\
            #           show_titles=True, title_kwargs={"fontsize": 12})
            fig=corner.corner( flat_samples ,labels=['a',r'$\phi$'],quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

            fig.gca().annotate(f'{ins} - {round(1e6*wvl,2)}um',xy=(1.0, 1.0),xycoords="figure fraction", xytext=(-20, -10), textcoords="offset points", ha="right", va="top")
            
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            plt.savefig(os.path.join(fig_path,f'ellipse_fixed_ud_mcmc_corner_{ins.split()[0]}_{round(1e6*wvl,2)}um.jpeg'))
            
        """plt.figure() 
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ')
        plt.xlabel('Baseline (m)')
        plt.ylabel(r'$V^2$')
        plt.plot(v2_df.columns,  disk_v2( rho, np.mean( rad2mas( flat_samples[:, :] ) ) *1e-3 * np.pi/180 / 3600  ) ,'.')
        """
        
        #y_model = np.median( rad2mas( flat_samples[:, :] ) ) * 1e-3 * np.pi/180 / 3600
        
        #for i,k in enumerate(ellipse_result_per_wvl_dict):
        #    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
        #    q = np.diff(mcmc)
        
        
        #    ellipse_result_per_wvl_dict[k]['mean'].append( np.mean(  flat_samples[:, i] ) )
        #    ellipse_result_per_wvl_dict[k]['median'].append( mcmc[1] )
        #    ellipse_result_per_wvl_dict[k]['err'].append( q )
        
        """
        fig1 = plt.figure(1)
        #Plot Data-model
        frame1=fig1.add_axes((.1,.3,.8,.6))
        fig1.set_tight_layout(True)
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ',color='darkred',alpha=0.6,label='measured')
        plt.plot(v2_df.columns,  disk_v2( rho, mas2rad(diam_median[-1])  ) ,'.',color='darkblue',alpha=0.6, label='model')
        plt.ylabel(r'$V^2$', fontsize=15)
        plt.gca().tick_params(labelsize=13)
        plt.legend(fontsize=15)
        frame2=fig1.add_axes((.1,.1,.8,.2))    
          
        plt.plot(v2_df.columns, ((y-y_model.values)/yerr),'o',color='k')
        plt.axhline(0,color='g',linestyle='--')
        plt.ylabel(r'$\chi_i$',fontsize=15 ) #r'$\Delta V^2 /\sigma_{V2}$',fontsize=15)
        plt.gca().tick_params(labelsize=13)
        plt.xlabel('Baseline (m)',fontsize=15)"""
        
        

        for i,k in enumerate(param_labels):
            
            intermediate_results_dict[wvl][k]={}
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
            q = np.diff(mcmc)
            
            intermediate_results_dict[wvl][k]['mean'] = np.mean(  flat_samples[:, i] ) 
            intermediate_results_dict[wvl][k]['median'] = mcmc[1] 
            intermediate_results_dict[wvl][k]['err'] = q 
            
            #ellipse_result_per_wvl_dict
            
        #best fit
        best_params_wvl = [intermediate_results_dict[wvl][k]['median'] for k in param_labels] 
        
        #
        y_model = ellipse_v2(x, best_params_wvl, theta0) #disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
        redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df[wvl])-ndim  ))
        
              
        intermediate_results_dict[wvl]['rho'] = x
        intermediate_results_dict[wvl]['v2_obs'] = y
        intermediate_results_dict[wvl]['v2_obs_err'] = yerr
        intermediate_results_dict[wvl]['v2_model'] = y_model
        intermediate_results_dict[wvl]['samplers'] = flat_samples
        
        #reduced chi2 
        #redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df.iloc[wvl_indx])-1))
        
        print('reduced chi2 = {}'.format(chi2(y_model, y, yerr) / (len(v2_df[wvl])-ndim )) )
    
    #for i,k in enumerate(ellipse_result_per_wvl_dict):
    #    ellipse_fit_results[k]={}
    #    ellipse_fit_results[k]['mean'] = diam_mean
    #    ellipse_fit_results[k]['median'] = diam_median
    #    ellipse_fit_results[k]['err'] = diam_err
        
    ellipse_fit_results['redchi2'] = redchi2
    
    ellipse_fit_results['intermediate_results'] = intermediate_results_dict
    
    ellipse_fit_per_ins[ins] = ellipse_fit_results



#save a summary table

ellipse_table = {'wvl_list':[],'ellipse_redchi2':[]} 
# add in parameter mean and err 
for _,k in enumerate(param_labels):
    ellipse_table[f'{k}_mean'] = []
    ellipse_table[f'{k}_median'] = []
    ellipse_table[f'{k}_err_16'] = []
    ellipse_table[f'{k}_err_84'] = []
    
for ins in ellipse_fit_per_ins:
    wvls_tmp = list( ellipse_fit_per_ins[ins]['intermediate_results'].keys() )
    
    ellipse_table['wvl_list'].append( wvls_tmp )
    ellipse_table['ellipse_redchi2'].append( ellipse_fit_per_ins[ins]['redchi2']  ) 
    for i,k in enumerate(param_labels):
        
        ellipse_table[f'{k}_mean'].append( [ellipse_fit_per_ins[ins]['intermediate_results'][w][k]['mean'] for w in wvls_tmp] )
        ellipse_table[f'{k}_median'].append( [ellipse_fit_per_ins[ins]['intermediate_results'][w][k]['median'] for w in wvls_tmp] )
        ellipse_table[f'{k}_err_16'].append( [ellipse_fit_per_ins[ins]['intermediate_results'][w][k]['err'][0] for w in wvls_tmp]  ) 
        ellipse_table[f'{k}_err_84'].append( [ellipse_fit_per_ins[ins]['intermediate_results'][w][k]['err'][1] for w in wvls_tmp]  ) 
    
    
for k in ellipse_table:
    ellipse_table[k] = [item for sublist in ellipse_table[k] for item in sublist]

ellipse_table = pd.DataFrame( ellipse_table )
ellipse_table = ellipse_table.set_index('wvl_list')
#ellipse_table.to_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/ellipse_fit.csv')










#%% OLD BELOW

#%% Plot UD results 
fig1 = plt.figure(1,figsize=(10,12))
fig1.set_tight_layout(True)

frame1 = fig1.add_axes((.1,.7,.8,.3))
frame2 = fig1.add_axes((.1,.4,.8,.3))
frame3 = fig1.add_axes((.1,.1,.8,.3))
frame4 = fig1.add_axes((.1,.0,.8,.1))


#for ins, col in zip(ellipse_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
for ins, col in zip(ellipse_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
    if 1: #ins!='Matisse (N)':
        wvl_grid = np.array( list( ellipse_fit_per_ins[ins]['intermediate_results'].keys() ) )
        
        redchi2 = ellipse_fit_per_ins[ins]['redchi2']
        frame4.semilogy(1e6*wvl_grid, redchi2, '-',lw=2, color=col)
        
        for fig , k in zip( [frame1,frame2,frame3], param_labels):
            median = np.array( [ellipse_fit_per_ins[ins]['intermediate_results'][wvl][k]['median'] for wvl in wvl_grid] )
            err = np.array( [ellipse_fit_per_ins[ins]['intermediate_results'][wvl][k]['err'] for wvl in wvl_grid] )
            
            #fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
            fig.set_ylabel(k)
            if k=='a':
                fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(0,20)
                #fig.set_ylabel'a')
            if k=='theta':
                fig.errorbar(1e6*wvl_grid, rad2mas(median), yerr=np.array(rad2mas(err)).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(0,200)
                fig.set_yscale('log')
            if k=='phi':
                fig.errorbar(1e6*wvl_grid, np.rad2deg( median), yerr=np.array(rad2mas(err)).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(-180,180)
                
                #fig.set_ylabel'a')
            #frame1.set_yscale('log')
            
            #fig.set_ylim(0,180)

"""
ins =  'Pionier (H)'

wvl= 1.6e-6
#dict_keys(['Pionier (H)', 'Gravity P1 (K)', 'Gravity P2 (K)', 'Matisse (L)', 'Matisse (M)', 'Matisse (N)']
wvl_indx = list(ellipse_fit_per_ins[ins]['intermediate_results'].keys())[np.argmin( abs(np.array( list(ellipse_fit_per_ins[ins]['intermediate_results'].keys() ) ) - wvl) )]
plt.figure()
fig = corner.corner(
            ellipse_fit_per_ins[ins]['intermediate_results'][wvl_indx]['samplers'], labels=param_labels
        )
"""
#%%
fontsize=20
frame1.set_title('RT Pav Uniform Disk Fit vs Wavelength')
frame1.grid()
frame1.legend(fontsize=fontsize)
frame1.set_ylabel(r'$\theta$ [mas]',fontsize=fontsize)
frame1.tick_params(labelsize=fontsize)
frame1.set_xticklabels([]) 

frame2.grid()
frame2.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
frame2.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
frame2.tick_params(labelsize=fontsize)

plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/FIT_UDs_logscale.pdf',bbox_inches='tight')

#plt.title('RT Pav\nuniform disk diameter')

def plot_sampler(ins, wvl ,fontsize=14, xlabel=r'$\theta$ [mas]'):
    
    plt.figure()
    plt.title(f'MCMC Uniform Disk fit')
    plt.hist( rad2mas(ellipse_fit_per_ins[ins]['intermediate_results'][wvl]['samplers']),\
             label=f'{ins} at {round(1e6*wvl,3)}um',histtype='step',color='k',lw=3 )
    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.gca().tick_params(labelsize=fontsize)
    plt.show()
    

ins = list( ellipse_fit_per_ins.keys() )[3] 
wvls =  list( ellipse_fit_per_ins[ins]['intermediate_results'].keys() ) 
plot_sampler(ins, wvls[3] , xlabel=r'$\theta$ [mas]')



#%%










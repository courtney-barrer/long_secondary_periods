#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:11:15 2023

@author: bcourtne

the binary has coordinates [RA_c, DEC_c] relative to primary [RA_p, DEC_p].
or relative seperation [RA_p, DEC_p] - [RA_c, DEC_c] = [dRA, dDEC]

uv coordinates of primary 

[u,v] = [ B_x/wvl, B_y/wvl ] = M @ [B_E, B_W]

where M = 

star vector between primary and companion: S = [ rho * cos(Theta), rho * sin(Theta) ]

https://www.icrar.org/wp-content/uploads/2018/11/Perley_Basic_Radio_Interferometry_Geometry.pdf
- X points to H=0, d=0 (intersection of meridian and celestial equator)
– Y points to H = -6, d = 0 (to east, on celestial equator)
– Z points to d = 90 (to NCP)


we got uv ground coordinates (B_E, B_N) - note  +u is east, +v is north
from 2005PASP..117.1255P - A Data Exchange Standard for Optical (Visible/IR) Interferometry, Pauls 2005
    "The coordinate u is the east component and is the northv
    component of the projection of the baseline vector onto the
    plane normal to the direction of the phase center, which is
    assumed to be the pointing center."

The key parameter for interferometric observation of a double star is the projection of
the star vector on the direction of the baseline vector

    rho_p = rho * np.cos(theta - Theta)

the star vector is :
     S = [S_x, S_y] = [rho*cos(theta) ,rho*sin(theta)] 
     where rho*cos(theta) is projection of star vector onto N, rho*sin(theta) is projection of star vector  onto E. 
     
rho is radial distance between primary and companion 
theta is the real angle between the primary and companion in the NE sky reference frame
Theta is the angle of the sstar vector projected on the baseline vector 

we can calculate Theta :
    
    Theta = np.atan( B_y / B_x ) = np.atan( u / v ) 


u,v = 1/wvl * [B_y,B_x] is given in the data , where Bx and By are already projected on the sky by the coordinates, hr angle etc 


V^2(f) = (v_1**2 + (R*v_2)**2 + (2*R*abs(v_1)*abs(v_2) * np.cos(2 * np.pi * f * (rho * np.cos(theta - Theta) ) ) ) ) / (1+r)**2


# OR WE CAN JUST DO IT USING CANDID METHOD 

TO DO 
========
Its clear I just need a function take takes the binary parameters , u1,u2,v2,v2 (same format as oi.fits data - note u3,v3 are implicitly implied u3=-u2-u1 to close the loop)

I can include functions to calculate u1,u2,v1,v2 from telescope positions and then I can play with configurations etc myslef , just need to take into account uv plane rotation (do projected baselines)
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
import itertools
from itertools import repeat

def binary_v2__m1(u, v, F, sep, theta): #https://hal.archives-ouvertes.fr/hal-01221306/document
    """
    u = projected  east fourier plane coorindate (B_y/lambda)  (rad-1)
    v=  projected north fourier plane coordinate (B_x/lambda)  (rad-1)
    theta = projected angle from north between primary and companion 
    sep = radial seperation of primary and comapnion 
    F = flux ratio ( companion/primary )
    relating coordinates back to delta RA and DEC
    Note X=sep* cos(theta) = dRA * cos(DEC), Y=sep * sin(theta) = dDEC 
    """
    
    # Bx = v/wvl, By = u/wvl (Bx is in north same as v!!)
    omega = np.arctan2( u, v )
    
    rho = np.sqrt( u**2 + v**2 )
    
    SB = sep * np.cos( theta - omega )
    
    v2 = (v_1**2 + (F*v_2)**2 + (2*F*abs(v_1)*abs(v_2) * np.cos(2 * np.pi * SB * rho ) ) ) / (1+F)**2
    
    return( v2 )



def chi2(y_model,y_true,yerr):
    return(sum( (y_model-y_true)**2/yerr**2 ) )

    
def rad2mas(rad):
    return(rad*180/np.pi * 3600 * 1e3)

def mas2rad(mas):
    return(mas*np.pi/180 / 3600 / 1e3)


def binary_V(u, v, dRA, dDEC, F , ud, R=None): 
    """
    
    Parameters
    ----------
    u : TYPE. float or array of floats  
        DESCRIPTION. u coordinate = B_y/wvl, where B_y is projected baseline vector to the East, units = rad^-1, 
    v : TYPE. float or array of floats
        DESCRIPTION. v coordinate = B_x/wvl, where B_x is projected baseline vector to the North, units = rad^-1
    dRA : TYPE. float 
        DESCRIPTION. RA seperation between companion and primary, units = rad
    dDEC : TYPE. float 
        DESCRIPTION. DEC seperation between companion and primary, units = rad
    F : TYPE. float 
        DESCRIPTION. Flux ratio (companion/primary)
    ud : TYPE. float 
        DESCRIPTION. Uniform disk diameter of primary, units = rad
    
    R : TYPE, optional
        DESCRIPTION. spectral resolution of observation to calculate bandwidth smearing correction factor. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    Visibility

    """
    
    x = np.pi * ud  * np.sqrt(u**2 + v**2) 
    
    #primary Visibility (uniform disk)
    V_s = 2 * special.j1(x) / x
    
    #companion Visibility (point source)
    V_c = np.exp(-2*np.pi*1j * (u * dRA + v * dDEC) )
    
    # bandwidth smearing correction
    if (R==None):
        G = 1
    elif (R!=None):
        eta = np.pi * (u * dRA + v * dDEC) / (R)
        G = abs( np.sin(eta) / eta )

    V = (V_s + G * F * V_c) / (1+F)

    return(V)




def binary_bispectrum(u_mat, v_mat, wvls, dRA, dDEC, F , ud, R=None):
    """
    

    Parameters
    ----------
    u_mat : TYPE, list of lists
        DESCRIPTION. u coordinates = B_y/wvl, where B_y is projected baseline vector to the East, units = rad^-1, 
        rows correspond to baselines, columns to wavelength
    v_mat : TYPE, list of lists
        DESCRIPTION. v coordinate = B_x/wvl, where B_x is projected baseline vector to the North, units = rad^-1
        rows correspond to baselines, columns to wavelength
    wvls : TYPE, array like
        DESCRIPTION. wavelengths of observations 
    dRA : TYPE. float 
        DESCRIPTION. RA seperation between companion and primary, units = rad
    dDEC : TYPE. float 
        DESCRIPTION. DEC seperation between companion and primary, units = rad
    F : TYPE. float 
        DESCRIPTION. Flux ratio (companion/primary)
    ud : TYPE. float 
        DESCRIPTION. Uniform disk diameter of primary, units = rad
    R : TYPE, optional
        DESCRIPTION. spectral resolution of observation to calculate bandwidth smearing correction factor. The default is None.

    Returns
    -------
    triangles, bispectrum.
        triangles is a dictionary with wavelength key holding baseline coordinates (vertices) that form the triangle
        bispectrum is a dictionary with wavelength key holding the bispectrum for each respective triangle (should be ordered the same between triangle and bispectrum dictionary)

    """
    
    bispectrum={}  
    triangles={}
    for i,w in enumerate( wvls ): 
        # fun fact: number of triangles from n points is n*(n-1)*(n-2)/6 ! 
        
        utmp = u_mat[:,i]
        vtmp = v_mat[:,i]
        Bcoord_at_wvl = list( w * np.array( [utmp, vtmp] ).T )
        V_binary_at_wvl = binary_V(utmp, vtmp, dRA, dDEC, F , ud, R=R)
        
        B_triangles = itertools.combinations(Bcoord_at_wvl, 3) 
        V_triangles = itertools.combinations(V_binary_at_wvl, 3) 
        
        bispectrum[w] = [v[0]*v[1]*np.conjugate(v[2]) for v in V_triangles]  
        triangles[w] = list( B_triangles ) 
        
    return( triangles, bispectrum )
  

def binary_bispectrum(Bx_proj1, Bx_proj2, By_proj1, By_proj2, wvls, dRA, dDEC, F , ud, R=None):
    """
    

    Parameters
    ----------
    Bx_proj1 : TYPE, number or array like 
        DESCRIPTION. Bx_proj1 is projected baseline for one of the telescope triangles edge (North component of baseline vector), units = m 

    Bx_proj12 : TYPE, number or array like 
        DESCRIPTION. Bx_proj2 is projected baseline for one of the telescope triangles edge (North component of baseline vector), units = m 

    By_proj1 : TYPE, number or array like 
        DESCRIPTION. By_proj1 is projected baseline for one of the telescope triangles edge (East component of baseline vector), units = m 

    By_proj12 : TYPE, number or array like 
        DESCRIPTION. By_proj2 is projected baseline for one of the telescope triangles edge (East component of baseline vector), units = m 

    wvls : TYPE, array like
        DESCRIPTION. wavelengths of observations 
    dRA : TYPE. float 
        DESCRIPTION. RA seperation between companion and primary, units = rad
    dDEC : TYPE. float 
        DESCRIPTION. DEC seperation between companion and primary, units = rad
    F : TYPE. float 
        DESCRIPTION. Flux ratio (companion/primary)
    ud : TYPE. float 
        DESCRIPTION. Uniform disk diameter of primary, units = rad
    R : TYPE, optional
        DESCRIPTION. spectral resolution of observation to calculate bandwidth smearing correction factor. The default is None.

    Returns
    -------
    triangles, bispectrum.
        triangles is a dictionary with wavelength key holding baseline coordinates (vertices) that form the triangle
        bispectrum is a dictionary with wavelength key holding the bispectrum for each respective triangle (should be ordered the same between triangle and bispectrum dictionary)

    """
    
    bispectrum={}  
    triangles={}
    for i,w in enumerate( wvls ): 
        # fun fact: number of triangles from n points is n*(n-1)*(n-2)/6 ! 
        
        u1 = By_proj1/w
        u2 = By_proj2/w
        u3 = -u2-u1 #close triangle
        
        v1 = Bx_proj1/w
        v2 = Bx_proj2/w
        v3 = -v2-v1
        
        number_of_triangles = len(u1)
        #uv1 = np.array([u1,v1]).T
        #uv2 = np.array([u2,v2]).T
        #uv3 = np.array([u3,v3]).T
        
        triangle_matrix = np.array( [ [(px1,py1), (px2, py2), (px3,py3)] for px1, px2, px3, py1, py2, py3 in zip(u1,u2,u3,v1,v2,v3) ] )
        # format:
        # triangle 0 | (u1,v1)_0, (u2,v2)_0, (u3,v3)_0 |
        #   ...      |                ...              | 
        # triangle N | (u1,v1)_N, (u2,v2)_N, (u3,v3)_N | etc
        
        # now map binary_V function over each uv point map
        #V_triangle = np.array( list( map( binary_V , *triangles.reshape(-1), repeat(wvls), repeat(dRA), repeat(dDEC), repeat(F ), repeat(ud) ) ) )
        V_triangle = 1j*np.ones([number_of_triangles,3]) 
        for i in range(triangle_matrix.shape[0]):
            for j in range(triangle_matrix.shape[1]):
                V_triangle[i,j] = binary_V(u = triangle_matrix[i,j][0], v = triangle_matrix[i,j][1],  dRA=dRA, dDEC=dDEC, F=F, ud=ud,R=R)
        # [ V(u1,v1)_0, V(u2,v2)_0, V(u3,v3)_0 ... V(u1,v1)_N, V(u2,v2)_N, V(u3,v3)_N ]

        # put to original shape
        #V_triangle = V_triangle.reshape(triangles.shape)
        
        # now calculate the bispectrum for each triangles in our wavelength bin
        bispectrum[w] = [v[0]*v[1]*np.conjugate(v[2]) for v in V_triangle  ]
        triangles[w] = triangle_matrix
        
    return( triangles, bispectrum )
  

def get_trianlge_baselines_from_telescope_coordinates( tel_x, tel_y ):
    
    # need to deal with signs properly here!
    tri_x = list(itertools.combinations(tel_x,3))
    tri_y = list(itertools.combinations(tel_y,3))

    B_x_12 = np.diff( tri_x, axis=1 ) # difference between T2-T1, T3-T2 
    B_x_3 = -B_x_12[:,0] - B_x_12[:,1] # last edge in triangle to close it 
    B_x = np.hstack([B_x_12, B_x_3.reshape(-1,1)]) # our three baselines in the triangle  (x component)
    # check np.sum(B_x)==0
    B_y_12 = np.diff( tri_y, axis=1 ) 
    B_y_3 = -B_y_12[:,0] - B_y_12[:,1] 
    B_y = np.hstack([B_y_12, B_y_3.reshape(-1,1)]) # our three baselines in the triangle (y component)
    
    return(B_x, B_y)


def baseline2uv_matrix( h, d):
    """
    

    Parameters
    ----------
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
    
    # rows = X (N), Y (E), Z (UP)
    mat =  np.array([[np.sin(h), np.cos(h), 0],\
           [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],\
           [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d) ]] )
        

    return(mat)   

def get_projected_baselines( B_x, B_y , h, d):
    """
    

    Parameters
    ----------
    B_x : TYPE
        DESCRIPTION. baseline component in the North direction 
    B_y : TYPE
        DESCRIPTION. baseline component in the East direction 
    h : TYPE
        DESCRIPTION. target hour angle (radian)
    d : TYPE
        DESCRIPTION. declination of target (radian)
        
    Returns
    -------
    projected baseline

    """
    
    #hour angle (radian)
    #h = ( get_LST( location,  datetime ) - telescope_coordinates.ra ).radian #radian
    
    # convert dec to radians also
    #d = telescope_coordinates.dec.radian #radian
    
    M = baseline2uv_matrix( h, d )
    
    B_proj = M @ [B_x, B_y, 0]
    
    Bu_proj = B_proj[0]
    Bv_proj = B_proj[0]
    return(Bu_proj, Bv_proj)


def plot_uv_triangles( triangles ):
    """
    plots baseline triangles

    Parameters
    ----------
    triangles : TYPE, list of tuples 
        DESCRIPTION. vectors corresponding to each baseline coordinate in triangle. e.g. 
        triangles = [ ([Bx_1, By_1], [Bx_2, By_2], [Bx_3, By_3])_0 , ... , ([Bx_1, By_1], [Bx_2, By_2], [Bx_3, By_3])_n ] 
        
        for a given wvl , triangles input should be the same format as is returned in biinary_bispectrum(u_mat, v_mat, wvls, dRA, dDEC, F , ud).. e.g.
        
            triangles, _ = biinary_bispectrum(u_mat, v_mat, wvls, dRA, dDEC, F , ud)
            
            then we can plot triangles for a given wavelength :
                
            plot_uv_triangles( triangles[wvls[0]] )
            
    Returns
    -------
    fig, ax and shows the plot

    """
    fig,ax = plt.subplots()
    for t in triangles:
        p = plt.Polygon(t, alpha=0.2)
        ax.add_patch(p)
        
    ax.set_xlim([-100,100])
    ax.set_ylim([-100,100])
    
    plt.show() 

    return(fig, ax)
    

def get_angular_frequency_from_traingles(triangles):
    # assumes triangles are already in uv plane 
    triangle_angular_freq = {}
    for w in triangles: # for each wavelength in triangle dictionary
        triangle_angular_freq[w]=[] #init list to hold baseline lengths
        for edge in triangles[w]: # for each set of triangle vertices in the uv plane
            
            triangle_angular_freq[w].append( [np.sqrt( b[0]**2 + b[1]**2) for b in edge] )

    return(triangle_angular_freq)


def get_baseline_lengths_from_traingles(triangles):
    # assumes triangles are already in uv plane 
    triangle_baselines = {}
    for w in triangles: # for each wavelength in triangle dictionary
        triangle_baselines[w]=[] #init list to hold baseline lengths
        for edges in triangles[w]: # for each set of triangle vertices in the uv plane
            # we calculate the baseline length np.sqrt( Bx**2 + By**2)
            triangle_baselines[w].append( [w * np.sqrt( b[0]**2 + b[1]**2) for b in edges] )

    return(triangle_baselines)






# now how to prep data to fit  - I just need u_mat, v_mat from data , and corresponding to CP...
"""
indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
stations = [[indx2station[h['OI_T3'].data['STA_INDEX'][tri][tel]] for tel in range(3)] for tri in range(4)]  
h['OI_T3'].data['U1COORD '] , h['OI_T3'].data['U2COORD '] # but why is there only 2 for u and v?? 
# u3 = -u1-u2 !! closing triangle!!

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
"""
#---- Test 
# set up observational  wavelengths and baselines 

wvls = 1e-6 * np.linspace(1.4,2.6,10)


tel_x = 150 *(0.5-np.random.rand(4)) # North coordinate
tel_y = 150 *(0.5-np.random.rand(4)) # East coorindate

# the baselines vectors [(Bx,By)..(Bx,By)] formed by telescopes 
Bx = np.diff( list(itertools.combinations(tel_x,2 ) ) ).reshape(-1) #north 
By = np.diff( list(itertools.combinations(tel_y,2 )) ).reshape(-1) #east
u = By[:,np.newaxis]/wvls #east
v = Bx[:,np.newaxis]/wvls #north 
baseline_coords = [(x,y) for x,y in zip( Bx, By) ]

# x,y coorindates of baselines vector in each triangle !! 
B_tx , B_ty = get_trianlge_baselines_from_telescope_coordinates( tel_x, tel_y )


# project them 
# get_projected_baselines( B_x, B_y , 0, np.deg2rad(-100)) # need to deal with input format 

# set-up binary parameters 
F = 0.05
ud = mas2rad(3)
dRA = mas2rad(10)
dDEC = mas2rad(1)


V_binary = binary_V(u, v, dRA, dDEC, F , ud, R=None)

# 4 telescopes => 4*3*2/6 = 4 closing triangles => 4 bispectrums,  V_matrix should be 4 x 3
triangles, bispectrum = binary_bispectrum(B_tx[:,0], B_tx[:,1], B_ty[:,0], B_ty[:,1], wvls, dRA, dDEC, F , ud, R=None)
        
# get baseline lengths sqrt(Bx^2 + By^2) for each triangle
triangle_baseline_lengths = get_baseline_lengths_from_traingles(triangles)

# get angular frequency lengths sqrt(u^2 + v^2) for each triangle
triangle_angularfreq_lengths = get_angular_frequency_from_traingles(triangles)

# max baseline lengths for each triangle 
triangle_Bmax = np.array( [[np.max(b) for b in triangle_baseline_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 

# max angular frequency lengths for each triangle 
triangle_uvmax = np.array( [[np.max(b) for b in triangle_angularfreq_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 


CP = np.array( [np.rad2deg( np.angle( bispectrum[w] ) ) for w in wvls] ).T # transpose to keep columns corresponding to wvl 


fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].plot(u, v,'.')
ax[0].set_xlabel('U [rad$^-1$]')
ax[0].set_ylabel('V [rad$^-1$]')

ax[1].set_title( f'F={F}, ud = {round(rad2mas(ud),1)}mas, dRA = {round(rad2mas(dRA),1)}mas, dDEC = {round(rad2mas(dDEC),1)}mas')
ax[1].plot(  np.sqrt(u**2 + v**2), abs(V_binary)**2 ,'.')
ax[1].set_xlabel('projected baseline (m)')
ax[1].set_ylabel(r'$|V|^2$')

ax[2].plot( triangle_uvmax , np.array( CP ) ,'.' )
ax[2].set_xlabel('max projected  in triangle (rad^-1)')
ax[2].set_ylabel('CP [deg]')





















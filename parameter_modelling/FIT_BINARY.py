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



def chi2(y_model, y_true, yerr):
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
        # format:
        # triangle 0 | V(u1,v1)_0, V(u2,v2)_0, V(u3,v3)_0 |
        #   ...      |                ...              | 
        # triangle N | V(u1,v1)_N, V(u2,v2)_N, V(u3,v3)_N | etc
        
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
    
    if np.any( np.sum( B_x , axis=1) ) or np.any( np.sum( B_y , axis=1) ):
        raise TypeError('baselines x and/or y components do no sum to zero in at least one of the telescope triangles')
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

#indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
#stations = [[indx2station[h['OI_T3'].data['STA_INDEX'][tri][tel]] for tel in range(3)] for tri in range(4)]  
#h['OI_T3'].data['U1COORD '] , h['OI_T3'].data['U2COORD '] # but why is there only 2 for u and v?? 
# u3 = -u1-u2 !! closing triangle!!

def V2_prep(files, EXTVER=None):    
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
        
        # Baselines
        B = [] # to holdprojected baseline !
        for Bx,By in zip( h['OI_VIS2'].data['UCOORD'],h['OI_VIS2'].data['VCOORD'] ): # U=east-west , V=north-sout
            #lambda_u, lambda_v, _ = baseline2uv_matrix(ha, dec) @ np.array( [Bx,By,0] ) # lambda_u has to be multiplied by lambda to get u±!!!
            #B.append( (lambda_u, lambda_v) ) # projected baseline !
            B.append( (Bx,By) ) # projected baseline !
        #B = [(a,b) for a,b in zip(lambda_u, lambda_v) ] # projected baseline ! #(h[v2_EXTNAME].data['UCOORD'], h[v2_EXTNAME].data['VCOORD']) #np.sqrt(h[v2_EXTNAME].data['UCOORD']**2 + h[v2_EXTNAME].data['VCOORD']**2)
        
        # squared visibilities
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








def CP_prep(files,EXTVER=None):
    
    if EXTVER==None:
        wvl_EXTNAME = 'OI_WAVELENGTH'
        T3_EXTNAME = 'OI_T3'
    
    else:
        wvl_EXTNAME = ('OI_WAVELENGTH',EXTVER)
        T3_EXTNAME = ('OI_T3',EXTVER)
    
    hdulists = [oifits.open(f) for f in files]
    
    print( len( hdulists) ,'\n\n\n')
    wvls = [ h[wvl_EXTNAME].data['EFF_WAVE'] for h in hdulists]
    wvl_grid = np.median( wvls , axis=0) # grid to interpolate wvls 
    
    data_dict = {} 
    for ii, h in enumerate( hdulists ):
        
        file = files[ii].split('/')[-1]
        print(f'looking at file {ii}/{len(hdulists)}, which is \n {file} \n')
            
        dec = np.deg2rad( h[0].header['DEC'] )
        ha = np.deg2rad( h[0].header['LST']/60/60 )
                
        Uc1 = h[T3_EXTNAME ].data['U1COORD'] # East
        Vc1 = h[T3_EXTNAME ].data['V1COORD'] # North
        Vc2 = h[T3_EXTNAME ].data['V2COORD']
        Uc2 = h[T3_EXTNAME ].data['U2COORD']
        
        B_triangle = [] # to holdprojected baseline !
        for By1,Bx1, By2, Bx2 in zip( Uc1,Vc1, Uc2, Vc2 ):
            B_triangle.append([(By1,Bx1), (By2,Bx2)])

        # CP 
        CP_list = []
        CPerr_list = []
        flag_list = []
        dwvl = []
        obs_time = []
        
        for T in range(len(B_triangle)):
            #for each baseline make interpolation functions 
            CPInterp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[T3_EXTNAME].data['T3PHI'][T,:] ,kind='linear', fill_value =  "extrapolate" )
            
            CPerrInterp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[T3_EXTNAME].data['T3PHIERR'][T,:] ,kind='linear', fill_value =  "extrapolate" )
            
            FlagInterp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[T3_EXTNAME].data['FLAG'][T,:] ,kind='nearest', fill_value =  "extrapolate" )
        
            dwvl.append( np.max( [1e9 * ( abs( ww -  wvl_grid ) ) for ww in h[wvl_EXTNAME].data['EFF_WAVE'] ] ) )
            
            obs_time.append( [h[0].header['DATE-OBS'],h[0].header['LST']/60/60 ,h[0].header['RA'], h[0].header['DEC'] ] )   #LST,ec,ra should be in deg#
            
            CP_list.append(  CPInterp_fn ( wvl_grid ) )
            
            CPerr_list.append( CPerrInterp_fn ( wvl_grid ) )
            
            flag_list.append( FlagInterp_fn( wvl_grid ) )
        
        # Put these in dataframes 
        
        # multi index in df. (u1, v1, u2, v2)
        index = pd.MultiIndex.from_tuples([(p1[0],p1[1], p2[0], p2[1]) for p1, p2 in B_triangle], names=["u1", "v1","u2","v2"])
        
        CP_df = pd.DataFrame( CP_list , columns = wvl_grid , index = index )
        
        CPerr_df = pd.DataFrame( CPerr_list , columns = wvl_grid , index = index)
        
        time_df = pd.DataFrame( obs_time , columns = ['DATE-OBS','LST', 'RA','DEC'] , index = index)
        
        flag_df = pd.DataFrame( np.array(flag_list).astype(bool) , columns = wvl_grid , index = index )
        
        data_dict[file] = {'CP':CP_df, 'CPerr':CPerr_df, 'flags' : flag_df,'obs':time_df}
        
        CP_df = pd.concat( [data_dict[f]['CP'] for f in data_dict] , axis=0)
        
        CPerr_df = pd.concat( [data_dict[f]['CPerr'] for f in data_dict] , axis=0)
        
        flag_df = pd.concat( [data_dict[f]['flags'] for f in data_dict] , axis=0)
        
        obs_df = pd.concat( [data_dict[f]['obs'] for f in data_dict] , axis=0)
    
    return( CP_df , CPerr_df , flag_df,  obs_df)




def CP_binary(params, **x):
    """
    
    assuming single wvl float input! so put wvls in list [wvls] - this is stupid - i should optimize it
    
    Parameters
    ----------
    x: TYPE, list
        DESCRIPTION. [Bx_proj1, Bx_proj2, By_proj1, By_proj2, wvls, R ]. Note x is North, y is East 
        
    params: TYPE, list 
        DESCRIPTION. [dRA, dDEC, F, ud]
        
    see binary_bispectrum() for description of parameters 
    
    Returns
    -------
    CP
    
    """
    
    #wvls, V2, V2err, CP, CPerr, uv, Bx1, By1, Bx2, By2, R = x
    
    dRA, dDEC, F, ud  = params 
    
    # should include if statement about if wvls is float or array 
    if not hasattr( x['wvls'], "__len__") : # if wvls are just a scalar then we reshape CP to have a single column 
        _, bispectrum  = binary_bispectrum(x['Bx1'], x['Bx2'], x['By1'], x['By2'], [x['wvls']], dRA, dDEC, F , ud, x['R'] )
    
        CP = np.array( [np.rad2deg( np.angle( bispectrum[w] ) ) for w in [x['wvls']] ] ).T
        CP = CP.reshape(-1)
    else: 
        _, bispectrum  = binary_bispectrum(x['Bx1'], x['Bx2'], x['By1'], x['By2'], x['wvls'], dRA, dDEC, F , ud, x['R'] )
    
        CP = np.array( [np.rad2deg( np.angle( bispectrum[w] ) ) for w in x['wvls']] )
        
    return( CP )
  

        
def V2_binary( params, **x):
    """
    

    Parameters
    ----------
    x: TYPE, list
        DESCRIPTION. [Bx_proj1, Bx_proj2, By_proj1, By_proj2, wvls, R ]. Note x is North, y is East 
        
    params: TYPE, list 
        DESCRIPTION. [dRA, dDEC, F, ud]
        
    see binary_bispectrum() for description of parameters 
    
    Returns
    -------
    CP
    
    """
    #wvls, V2, V2err, CP, CPerr, uv, Bx1, By1, Bx2, By2, R = x['wvls'], x['V2'],x['V2err'], x['CP'], x['CPerr'], x['uv'], x['Bx1'] , x['By1'], x['Bx2'], x['By2'], x['R'] 
    u,v = x['uv']
    R= x['R']
    
    dRA, dDEC, F , ud = params 
    
    V2 = abs( binary_V(u, v, dRA, dDEC, F , ud, R) )**2
    
    return( V2 )

#def arg_dict2tuple( x ): # we put emcee arguments in dictionary for easy tracking , then pass them as a tuple (cqannot use dictionary in emcee)
#    tup = ( x['wvls'], x['V2'],x['V2err'], x['CP'], x['CPerr'], x['uv'], x['Bx1'] , x['By1'], x['Bx2'], x['By2'], x['R'] )
#    return(tup)


def log_likelihood(params, **x):
    #u,v = x[:,0], x[:,1] #x is list with [u,v]

    sigma2_CP = x['CPerr']**2
    sigma2_V2 = x['V2err']**2  
    
    model_CP = CP_binary( params, **x)
    model_V2 = V2_binary( params, **x)

    return( -0.5 * (np.sum((V2 - model_V2) ** 2 / sigma2_V2 ) + np.sum((CP - model_CP) ** 2 / sigma2_CP )  ) )#+ np.log(sigma2)) )



def log_prior(params):
    dRA, dDEC, F, ud = params
    #a, phi = params  
    
    if (-8888 <= dRA < np.pi) & (-8888 <= dDEC < np.pi) & (1e-4 <= F)  : #& (a>1): #  uniform prior on rotation of ellipse between 0-180 degrees

        #gaussian prior on a
        mu = mas2rad( ud_wvl ) #rad - note this is an external variable that should be defined 
        sigma = mas2rad( 2 ) #* ud_wvl_err ) # 
        return(np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(ud-mu)**2/sigma**2)
    
    else:
        return(-np.inf)

def log_probability(params, **x):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(params, **x)
    
    
#%% PREP

pionier_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')


gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]

#VIS
pion_v2_df , pion_v2err_df  , pion_flag_df,  pion_obs_df = V2_prep(pionier_files)

grav_p1_v2_df , grav_p1_v2err_df, grav_p1_flag_df , grav_p1_obs_df= V2_prep(gravity_files, EXTVER = 11 )
grav_p2_v2_df , grav_p2_v2err_df , grav_p2_flag_df , grav_p2_obs_df = V2_prep(gravity_files, EXTVER = 12 )

mati_L_v2_df , mati_L_v2err_df , mati_L_flag_df, mati_L_obs_df = V2_prep(matisse_files_L )
mati_N_v2_df , mati_N_v2err_df , mati_N_flag_df, mati_N_obs_df = V2_prep(matisse_files_N )

#CP
pion_CP_df , pion_CPerr_df  , pion_flag_df,  pion_obs_df = CP_prep(pionier_files)

grav_p1_CP_df , grav_p1_CPerr_df, grav_p1_flag_df , grav_p1_obs_df= CP_prep(gravity_files, EXTVER = 11 )
grav_p2_CP_df , grav_p2_CPerr_df , grav_p2_flag_df , grav_p2_obs_df = CP_prep(gravity_files, EXTVER = 12 )

mati_L_CP_df , mati_L_CPerr_df , mati_L_flag_df, mati_L_obs_df = CP_prep(matisse_files_L )
mati_N_CP_df , mati_N_CPerr_df , mati_N_flag_df, mati_N_obs_df = CP_prep(matisse_files_N )

# filters 
grav_B_filt = grav_p1_v2_df.index.values !=0 
grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6)

# matisse wvl limits from https://www.eso.org/sci/facilities/paranal/instruments/matisse.html
mat_L_wvl_filt = (mati_L_v2_df.columns > 3.2e-6) & (mati_L_v2_df.columns < 3.9e-6) #| (mati_L_v2_df.columns > 4.5e-6) 
mat_M_wvl_filt = (mati_L_v2_df.columns > 4.5e-6) &  (mati_L_v2_df.columns <= 5e-6)
mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) & (mati_N_v2_df.columns <= 12.1e-6)#| (mati_L_v2_df.columns > 4.5e-6)


# instrument vis tuples 
pion_V2tup = (pion_v2_df , pion_v2err_df)
grav_p1_V2tup = (grav_p1_v2_df[grav_p1_v2_df.columns[::50]][grav_B_filt] , grav_p1_v2err_df[grav_p1_v2err_df.columns[::50]][grav_B_filt] )
grav_p2_V2tup = (grav_p2_v2_df[grav_p2_v2_df.columns[::50]][grav_B_filt] , grav_p2_v2err_df[grav_p2_v2err_df.columns[::50]][grav_B_filt] )
mati_L_V2tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_L_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2err_df.columns[mat_L_wvl_filt][::5]] )
mati_M_V2tup = (mati_L_v2_df[mati_L_v2_df.columns[mat_M_wvl_filt][::5]] , mati_L_v2err_df[mati_L_v2err_df.columns[mat_M_wvl_filt][::5]] )
mati_N_V2tup = (mati_N_v2_df[mati_N_v2_df.columns[mat_N_wvl_filt][::5]] , mati_N_v2err_df[mati_N_v2err_df.columns[mat_N_wvl_filt][::5]] )


# instrument vis tuples 
pion_CPtup = (pion_CP_df , pion_CPerr_df)
grav_p1_CPtup = (grav_p1_CP_df[grav_p1_CP_df.columns[::50]] , grav_p1_CPerr_df[grav_p1_CPerr_df.columns[::50]] )
grav_p2_CPtup = (grav_p2_CP_df[grav_p2_CP_df.columns[::50]] , grav_p2_CPerr_df[grav_p2_CPerr_df.columns[::50]] )
mati_L_CPtup = (mati_L_CP_df[mati_L_CP_df.columns[mat_L_wvl_filt][::5]] , mati_L_CPerr_df[mati_L_CPerr_df.columns[mat_L_wvl_filt][::5]] )
mati_M_CPtup = (mati_L_CP_df[mati_L_CP_df.columns[mat_M_wvl_filt][::5]] , mati_L_CPerr_df[mati_L_CPerr_df.columns[mat_M_wvl_filt][::5]] )
mati_N_CPtup = (mati_N_CP_df[mati_N_CP_df.columns[mat_N_wvl_filt][::5]] , mati_N_CPerr_df[mati_N_CPerr_df.columns[mat_N_wvl_filt][::5]] )

ins_V2_dict = {'Pionier (H)':pion_V2tup, 'Gravity P1 (K)' : grav_p1_V2tup, \
                'Gravity P2 (K)' : grav_p2_V2tup, 'Matisse (L)':mati_L_V2tup,\
                    'Matisse (M)':mati_M_V2tup, 'Matisse (N)':mati_N_V2tup }
    
ins_CP_dict = {'Pionier (H)':pion_CPtup, 'Gravity P1 (K)' : grav_p1_CPtup, \
                'Gravity P2 (K)' : grav_p2_CPtup, 'Matisse (L)':mati_L_CPtup,\
                    'Matisse (M)':mati_M_CPtup, 'Matisse (N)':mati_N_CPtup }

    
#%%
#---- Test 
# set up observational  wavelengths and baselines 

wvls = 1e-6 * np.linspace(2,2.4,40)


tel_x = 130 *(0.5-np.random.rand(10)) # North coordinate
tel_y = 130 *(0.5-np.random.rand(10)) # East coorindate

# the baselines vectors [(Bx,By)..(Bx,By)] formed by telescopes 
Bx = np.diff( list(itertools.combinations(tel_x,2 ) ) ).reshape(-1) #north 
By = np.diff( list(itertools.combinations(tel_y,2 )) ).reshape(-1) #east
u = By[:,np.newaxis]/wvls #east, columns correspond to wavelength
v = Bx[:,np.newaxis]/wvls #north, columns correspond to wavelengt 
baseline_coords = [(x,y) for x,y in zip( Bx, By) ]

# x,y coorindates of baselines vector in each triangle !! 
B_tx , B_ty = get_trianlge_baselines_from_telescope_coordinates( tel_x, tel_y )


# project them 
# get_projected_baselines( B_x, B_y , 0, np.deg2rad(-100)) # need to deal with input format 

# set-up binary parameters 
F = 0.02
ud = mas2rad(3)
dRA = mas2rad(3)
dDEC = mas2rad(0)


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


ax[2].set_ylim(-40,40)


#%% Try brute force optimization 

fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/binary_fit/'
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
param_labels=['dRA', 'dDEC', 'F', 'ud'] #['a','phi'] #param_labels=['a','phi','theta'] #a, phi,theta = params

bestParams = []
redchi2 = []
wvls = []
ndim = len(param_labels)
for ins in ins_V2_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to UD model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_V2_dict[ins]
    CP_df, CPerr_df = ins_CP_dict[ins]
    
    binary_fit_results = {}
    

    #params = [] #best fit
    binary_result_per_wvl_dict = {xxx:{ 'mean' :[], 'median' : [], 'err' : [] } for xxx in param_labels}
    
    intermediate_results_dict = {}
    x = {}
    
    for wvl_indx, wvl in enumerate( v2_df.columns ):
        
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        #ud_wvl_err = ud_fits['ud_err'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        
        intermediate_results_dict[wvl] = { } #{'rho':[], 'v2_obs':[], 'v2_obs_err':[],\'v2_model':[],'samplers':[] }

        # u,v coorindates ### #
        uv_unfilt = 1/wvl  *  np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).T # np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).reshape(len(v2_df.index.values),2)
        
        # baselines in  tiagnle 
        Bx1 = CP_df.index.get_level_values('u1')
        By1 = CP_df.index.get_level_values('v1')
        Bx2 = CP_df.index.get_level_values('u2')
        By2 = CP_df.index.get_level_values('v2')
        #B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )


        # Check rho matches v2_df.index.values tuples
        v2 = v2_df[wvl].values  # |V|^2
        v2_err = v2err_df[wvl].values # uncertainty 
        
        CP = CP_df[wvl].values
        CP_err = CPerr_df[wvl].values
        # filter out unphysical V2 values 
        v2_filt = (v2>0) & (v2<1.1)
        CP_filt = np.isfinite( CP )
        
        # short hand model notation 
        uv, V2, V2err = uv_unfilt[v2_filt].T , v2[v2_filt], v2_err[v2_filt]
        
        CP, CPerr = CP[CP_filt], CP_err[CP_filt]
        
        
        x['wvls'] = wvl
        x['V2'] = V2
        x['V2err'] = V2err
        x['CP'] = CP
        x['CPerr'] = CPerr
        x['R'] = None
        x['uv'] = uv
        
        x['uv'] = uv
        x['Bx1'] = np.array(Bx1)
        x['By1'] = np.array(By1)
        x['Bx2'] = np.array(Bx2)
        x['By2'] = np.array(By2)
        
        #arg_tup = arg_dict2tuple( x )
        print(' begin fit' )
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
        
        nwalkers = 500 #32
        
        # initialize at UD fit (save a table)quick grid search 
        theta0 = mas2rad( ud_wvl ) #rad
        
        #do rough grid search 
        best_chi2 = np.inf # initialize at infinity 
        for dRA in np.linspace( -10, 10, 10):
            for dDEC in np.linspace( -10,10,10):
                for F in np.logspace( -4, -2, 10):

                    params_tmp=[mas2rad(dRA), mas2rad(dDEC), F, theta0]
                    #params_tmp=[a_tmp, phi_tmp]
                    V2_model_cand =  V2_binary( params_tmp, **x) 
                    CP_model_cand =  CP_binary( params_tmp, **x) # should fix things to get rid of reshape necisity  
                    chi2_tmp = chi2(V2_model_cand  , x['V2'], x['V2err']) + chi2(CP_model_cand, x['CP'], x['CPerr'])
                    if chi2_tmp < best_chi2:
                        best_chi2 = chi2_tmp 
                        initial = params_tmp
                    

        #
        V2_model = V2_binary( initial,  **x) #disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
        CP_model = CP_binary( initial, **x ) 
        
        wvls.append( wvl )
        bestParams.append( initial )
        redchi2.append( ( chi2(V2_model , x['V2'], x['V2err']) + chi2(CP_model , x['CP'], x['CPerr']) ) / (len(x['V2']) + len(x['CP']) - ndim  ))
        
bestParams = np.array(bestParams)        

plt.figure()
plt.scatter( rad2mas(bestParams[:,0]), rad2mas(bestParams[:,1]) ,alpha=0.1)


plt.plot( np.array( [a**2 + b**2 for a,b in x['uv'].T] )**0.5 , x['V2'] ,'.',color='r'); plt.plot( np.array( [a**2 + b**2 for a,b in x['uv'].T] )**0.5 , V2_model ,'.',color='g')
#%%
plot=True
fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/binary_fit/'
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
param_labels=['dRA', 'dDEC', 'F', 'ud'] #['a','phi'] #param_labels=['a','phi','theta'] #a, phi,theta = params

binary_fit_per_ins = {} # to hold fitting results per instrument photometric band

for ins in ins_V2_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to UD model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_V2_dict[ins]
    CP_df, CPerr_df = ins_CP_dict[ins]
    
    binary_fit_results = {}
    
    redchi2 = []
    
    #params = [] #best fit
    binary_result_per_wvl_dict = {xxx:{ 'mean' :[], 'median' : [], 'err' : [] } for xxx in param_labels}
    
    intermediate_results_dict = {}
    x = {}
    for wvl_indx, wvl in enumerate(v2_df.columns ):
        
        ud_wvl = ud_fits['ud_mean'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        #ud_wvl_err = ud_fits['ud_err'].iloc[ np.argmin(abs(ud_fits.index - wvl)) ] #best ud fit at wavelength (mas)
        
        intermediate_results_dict[wvl] = { } #{'rho':[], 'v2_obs':[], 'v2_obs_err':[],\'v2_model':[],'samplers':[] }

        # u,v coorindates ### #
        uv_unfilt = 1/wvl  *  np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).T # np.array([[aa[0] for aa in v2_df.index.values],[aa[1] for aa in v2_df.index.values]]).reshape(len(v2_df.index.values),2)
        
        # baselines in  tiagnle 
        Bx1 = CP_df.index.get_level_values('u1')
        By1 = CP_df.index.get_level_values('v1')
        Bx2 = CP_df.index.get_level_values('u2')
        By2 = CP_df.index.get_level_values('v2')
        #B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )


        # Check rho matches v2_df.index.values tuples
        v2 = v2_df[wvl].values  # |V|^2
        v2_err = v2err_df[wvl].values # uncertainty 
        
        CP = CP_df[wvl].values
        CP_err = CPerr_df[wvl].values
        # filter out unphysical V2 values 
        v2_filt = (v2>0) & (v2<1.1)
        CP_filt = np.isfinite( CP )
        
        # short hand model notation 
        uv, V2, V2err = uv_unfilt[v2_filt].T , v2[v2_filt], v2_err[v2_filt]
        
        CP, CPerr = CP[CP_filt], CP_err[CP_filt]
        
        
        x['wvls'] = wvl
        x['V2'] = V2
        x['V2err'] = V2err
        x['CP'] = CP
        x['CPerr'] = CPerr
        x['R'] = None
        x['uv'] = uv
        
        x['uv'] = uv
        x['Bx1'] = np.array(Bx1)
        x['By1'] = np.array(By1)
        x['Bx2'] = np.array(Bx2)
        x['By2'] = np.array(By2)
        
        #arg_tup = arg_dict2tuple( x )
        print(' begin fit' )
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
        for dRA in np.linspace( 0, 20, 20):
            for dDEC in np.linspace( 0,20,20):
                for F in np.logspace( -4, -2, 15):

                    params_tmp=[mas2rad(dRA), mas2rad(dDEC), F, theta0]
                    #params_tmp=[a_tmp, phi_tmp]
                    V2_model_cand =  V2_binary( params_tmp, **x) 
                    CP_model_cand =  CP_binary( params_tmp, **x) # should fix things to get rid of reshape necisity  
                    chi2_tmp = chi2(V2_model_cand  , x['V2'], x['V2err']) + chi2(CP_model_cand, x['CP'], x['CPerr'])
                    if chi2_tmp < best_chi2:
                        best_chi2 = chi2_tmp 
                        initial = params_tmp
                    
        print(f'best initial parameters = {initial} with chi2={best_chi2}')
        #a0 = 1 #squeeze/stretching (1=circle) - no units
        #phi0 = 0 #rotation (rad)

        #initial = np.array([ a0, phi0, theta0 ])
        ndim = len(initial)
        
        p0 = [initial + np.array([mas2rad(2), mas2rad(2), 1e-2 , theta0/10 ]) * np.random.randn(4)  for i in range(nwalkers)]
        #p0 = [initial + np.array([0.1, np.deg2rad(10)]) * np.random.rand(ndim)  for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, kwargs = x
        )
        sampler.run_mcmc(p0, 1000, progress=True);
        
        #samples = sampler.get_chain(flat=True)
        
        #plt.hist(np.log10(samples) ) , bins = np.logspace(-9,-7,100)) #[-1,:,0])
        
        #plt.hist( np.log10( samples ) , bins=np.linspace(-9,-5,100 ))
        
        
        # use sampler.get_autocorr_time()
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        
        
      
        if plot:
            flat_samples4plot = flat_samples.copy()
            flat_samples4plot[:,2] = rad2mas(flat_samples4plot[:,2])
            flat_samples4plot[:,2] = np.rad2deg(flat_samples4plot[:,1])
            plt.figure()
            #fig=corner.corner( flat_samples ,labels=['a',r'$\phi$',r'$\theta$'],quantiles=[0.16, 0.5, 0.84],\
            #           show_titles=True, title_kwargs={"fontsize": 12})
            fig=corner.corner( flat_samples4plot ,labels=['dRA', 'dDEC', 'F', 'ud'],quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

            fig.gca().annotate(f'{ins} - {round(1e6*wvl,2)}um',xy=(1.0, 1.0),xycoords="figure fraction", xytext=(-20, -10), textcoords="offset points", ha="right", va="top")
            
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            plt.savefig(os.path.join(fig_path,f'binary_mcmc_corner_{ins.split()[0]}_{round(1e6*wvl,2)}um.jpeg'))
            
        """plt.figure() 
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ')
        plt.xlabel('Baseline (m)')
        plt.ylabel(r'$V^2$')
        plt.plot(v2_df.columns,  disk_v2( rho, np.mean( rad2mas( flat_samples[:, :] ) ) *1e-3 * np.pi/180 / 3600  ) ,'.')
        """
        
        #y_model = np.median( rad2mas( flat_samples[:, :] ) ) * 1e-3 * np.pi/180 / 3600
        
        #for i,k in enumerate(binary_result_per_wvl_dict):
        #    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
        #    q = np.diff(mcmc)
        
        
        #    binary_result_per_wvl_dict[k]['mean'].append( np.mean(  flat_samples[:, i] ) )
        #    binary_result_per_wvl_dict[k]['median'].append( mcmc[1] )
        #    binary_result_per_wvl_dict[k]['err'].append( q )
        
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
            
            #binary_result_per_wvl_dict
            
        #best fit
        best_params_wvl = [intermediate_results_dict[wvl][k]['median'] for k in param_labels] 
        
        #
        V2_model = V2_binary( best_params_wvl,  **x) #disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
        CP_model = CP_binary(best_params_wvl, **x) 
        
        redchi2.append( ( chi2(V2_model , x['V2'], ['V2err']) + chi2(CP_model , x['CP'], x['CPerr']) ) / (len(v2_df[wvl])-ndim  ))
        
              
        intermediate_results_dict[wvl]['uv'] = x['uv']
        intermediate_results_dict[wvl]['Bx1'] = x['Bx1'] #x is north 
        intermediate_results_dict[wvl]['By1'] = x['By1'] # y is East
        intermediate_results_dict[wvl]['Bx2'] = x['Bx2']
        intermediate_results_dict[wvl]['By2'] = x['By2']
        intermediate_results_dict[wvl]['v2_obs'] = x['V2']
        intermediate_results_dict[wvl]['v2_obs_err'] = x['V2err']
        intermediate_results_dict[wvl]['v2_model'] = V2_model
        intermediate_results_dict[wvl]['CP_obs'] = x['CP']
        intermediate_results_dict[wvl]['CP_obs_err'] = x['CPerr']
        intermediate_results_dict[wvl]['CP_model'] = CP_model
        intermediate_results_dict[wvl]['samplers'] = flat_samples
        
        #reduced chi2 
        #redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df.iloc[wvl_indx])-1))
        
        print('reduced chi2 = {}'.format( redchi2[-1]) )
    
    #for i,k in enumerate(binary_result_per_wvl_dict):
    #    binary_fit_results[k]={}
    #    binary_fit_results[k]['mean'] = diam_mean
    #    binary_fit_results[k]['median'] = diam_median
    #    binary_fit_results[k]['err'] = diam_err
        
    binary_fit_results['redchi2'] = redchi2
    
    binary_fit_results['intermediate_results'] = intermediate_results_dict
    
    binary_fit_per_ins[ins] = binary_fit_results



#%% PLOTTING GRAVITY COLORCODING SPECTRAL FEATURE 




# PIONIER
fig, ax = plt.subplots(1,1, figsize=(10,8) )
CP_df , CPerr_df , CP_flag_df,  CP_obs_df = CP_prep(pionier_files, EXTVER=None)

B1 = np.sqrt( CP_df.index.get_level_values('u1')**2 + CP_df.index.get_level_values('v1')**2 )
B2 = np.sqrt( CP_df.index.get_level_values('u2')**2 + CP_df.index.get_level_values('v2')**2 )
B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )

Bmax = np.max([(b1,b2,b3) for b1,b2,b3 in zip(B1,B2,B3)],axis=1)

# I should also use flags! 
wvl_grid = np.array(list(CP_df.columns))


tmp_filt = wvl_grid > 0.1e-6 #(wvl_grid > 8e-6)  & (wvl_grid<9e-6)

plt.plot( Bmax[:,np.newaxis]/wvl_grid[tmp_filt] , CP_df.values[:,tmp_filt],'.',color='k',alpha=0.5)

ax.legend()
#plt.ylim(-40,40)
ax.set_ylim(-180,180)

ax.set_xlabel(r'$B_{max}/\lambda$ [rad$^{-1}$]')
ax.set_ylabel('CP [deg]')



# GRAVITY 
fig, ax = plt.subplots(1,1, figsize=(10,8) )
CP_df , CPerr_df , CP_flag_df,  CP_obs_df = CP_prep(gravity_files, EXTVER=11)
#CP_df , CPerr_df , CP_flag_df,  CP_obs_df = CP_prep(matisse_files_L, EXTVER=None)

B1 = np.sqrt( CP_df.index.get_level_values('u1')**2 + CP_df.index.get_level_values('v1')**2 )
B2 = np.sqrt( CP_df.index.get_level_values('u2')**2 + CP_df.index.get_level_values('v2')**2 )
B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )

Bmax = np.max([(b1,b2,b3) for b1,b2,b3 in zip(B1,B2,B3)],axis=1)

# I should also use flags! 
wvl_grid = np.array(list(CP_df.columns))

CO1_filter = ( (wvl_grid<2.298e-6) & (wvl_grid>2.2934e-6 ) ) 
CO2_filter = ( (wvl_grid< 2.324e-6 ) & (wvl_grid>2.3226e-6) ) 
CO3_filter = ( (wvl_grid< 2.3555e-6 ) & (wvl_grid>2.3525e-6 ) ) 
bg_filter =  ( (wvl_grid< 2.167e-6 ) & (wvl_grid>2.165e-6 ) ) 
#tmp_filt = wvl_grid > 0.1e-6 #(wvl_grid > 8e-6)  & (wvl_grid<9e-6)
# ( (wvl_grid<) & (wvl_grid<) ) or ( (wvl_grid<) & (wvl_grid<) )
#H20_filter = 
#brackagamma_filter = 
ax.plot( Bmax[:,np.newaxis]/wvl_grid[(~CO1_filter) & (~CO2_filter) & (~CO3_filter)] , CP_df.values[:,(~CO1_filter) & (~CO2_filter) & (~CO3_filter)],'.',color='k',alpha=0.1)
#plt.plot( Bmax[:,np.newaxis]/wvl_grid[tmp_filt] , CP_df.values[:,tmp_filt],'.',color='k',alpha=0.1)
ax.plot( Bmax[:,np.newaxis]/wvl_grid[CO1_filter] , CP_df.values[:,CO1_filter],'.',color='red',alpha=0.9)
ax.plot( Bmax[:,np.newaxis]/wvl_grid[CO2_filter] , CP_df.values[:,CO2_filter],'.',color='orange',alpha=0.9)
ax.plot( Bmax[:,np.newaxis]/wvl_grid[CO3_filter] , CP_df.values[:,CO3_filter],'.',color='yellow',alpha=0.9)
ax.plot( Bmax[:,np.newaxis]/wvl_grid[bg_filter ] , CP_df.values[:,bg_filter ],'.',color='green',alpha=0.9)
ax.plot( 0, 200,'.',color='red',alpha=0.9,label='CO bandhead (2-0)')
ax.plot( 0, 200,'.',color='orange',alpha=0.9,label='CO bandhead (3-1)')
ax.plot( 0, 200,'.',color='yellow',alpha=0.9,label='CO bandhead (4-2)')
ax.plot( 0, 200,'.',color='green',alpha=0.9,label=r'$br\gamma$')
ax.legend()
#plt.ylim(-40,40)
ax.set_ylim(-40,40)

ax.set_xlabel(r'$B_{max}/\lambda$ [rad$^{-1}$]')
ax.set_ylabel('CP [deg]')




# MATISSE L
fig, ax = plt.subplots(1,1, figsize=(10,8) )
CP_df , CPerr_df , CP_flag_df,  CP_obs_df = CP_prep(matisse_files_L, EXTVER=None)

B1 = np.sqrt( CP_df.index.get_level_values('u1')**2 + CP_df.index.get_level_values('v1')**2 )
B2 = np.sqrt( CP_df.index.get_level_values('u2')**2 + CP_df.index.get_level_values('v2')**2 )
B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )

Bmax = np.max([(b1,b2,b3) for b1,b2,b3 in zip(B1,B2,B3)],axis=1)

# I should also use flags! 
wvl_grid = np.array(list(CP_df.columns))


tmp_filt = wvl_grid > 0.1e-6 #(wvl_grid > 8e-6)  & (wvl_grid<9e-6)

plt.plot( Bmax[:,np.newaxis]/wvl_grid[tmp_filt] , CP_df.values[:,tmp_filt],'.',color='k',alpha=0.5)

ax.legend()
#plt.ylim(-40,40)
ax.set_ylim(-180,180)

ax.set_xlabel(r'$B_{max}/\lambda$ [rad$^{-1}$]')
ax.set_ylabel('CP [deg]')



# MATISSE N
fig, ax = plt.subplots(1,1, figsize=(10,8) )
CP_df , CPerr_df , CP_flag_df,  CP_obs_df = CP_prep(matisse_files_N, EXTVER=None)

B1 = np.sqrt( CP_df.index.get_level_values('u1')**2 + CP_df.index.get_level_values('v1')**2 )
B2 = np.sqrt( CP_df.index.get_level_values('u2')**2 + CP_df.index.get_level_values('v2')**2 )
B3 = np.sqrt( (-CP_df.index.get_level_values('u2')-CP_df.index.get_level_values('u1'))**2 + (- CP_df.index.get_level_values('v2')- CP_df.index.get_level_values('v2'))**2 )

Bmax = np.max([(b1,b2,b3) for b1,b2,b3 in zip(B1,B2,B3)],axis=1)

# I should also use flags! 
wvl_grid = np.array(list(CP_df.columns))


tmp_filt = wvl_grid > 0.1e-6 #(wvl_grid > 8e-6)  & (wvl_grid<9e-6)

plt.plot( Bmax[:,np.newaxis]/wvl_grid[tmp_filt] , CP_df.values[:,tmp_filt],'.',color='k',alpha=0.5)

ax.legend()
#plt.ylim(-40,40)
ax.set_ylim(-180,180)

ax.set_xlabel(r'$B_{max}/\lambda$ [rad$^{-1}$]')
ax.set_ylabel('CP [deg]')





#%% [old to discard]



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







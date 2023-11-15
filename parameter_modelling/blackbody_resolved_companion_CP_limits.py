#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:41:04 2023

@author: bcourtne


RT Pav has a primary pulsation period of 85 days and a long secondary period 
of 757 days (see figure \ref{fig:RT_pav_ASAS_lightcurve}. The peak-to-peak 
amplitude of the LSP in the V-band varies between periods, but is typically 
between 0.5 - 1.4 mag, corresponding to the flux dropping by up to a factor 
of 1/3.8 of its peak over the LSP (i.e. 26\% of its peak value). 
Therefore assuming a hypothetical  dusty companion is causing the LSP from 
eclipsing the primary star, we can estimate a lower bound of the expected 
projected area of the dusty companion as a fraction of the projected area of 
the primary star with some assumptions about the optical depth of the companion. 
Making the reasonable assumption that the dust is optically very thick and has 
negligible emission in the visible - from simple arguments the fractional 
projected area of the companion should be roughly equal to the fractional 
drop in the observed visible flux, i.e. 26\%. Therefore to confirm the binary 
hypothesis we would expect to spatially resolve a dense region of dust at 
least 26\% the projected area of the primary star, with some uncertainty 
coming from the 3D geometry of the dusty companion. Assuming different 
temperature profiles and separations we can then constrain upper limits of 
expected contrast, visibility's and closure phases at different wavelengths 
in the observations. `
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

fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/'

paranal_coordinate = (-24.62794830, -70.40479659) #(lat, long) degrees 

#VLTI station coordinates (https://www.eso.org/observing/etc/doc/viscalc/vltistations.html)

#_ID_______P__________Q__________E__________N____
vlti_stations = 'A0    -32.001    -48.013    -14.642    -55.812\
 A1    -32.001    -64.021     -9.434    -70.949 nl\
 B0    -23.991    -48.019     -7.065    -53.212 nl\
 B1    -23.991    -64.011     -1.863    -68.334 nl\
 B2    -23.991    -72.011      0.739    -75.899 nl\
 B3    -23.991    -80.029      3.348    -83.481 nl\
 B4    -23.991    -88.013      5.945    -91.030 nl\
 B5    -23.991    -96.012      8.547    -98.594 nl\
 C0    -16.002    -48.013      0.487    -50.607 nl\
 C1    -16.002    -64.011      5.691    -65.735 nl\
 C2    -16.002    -72.019      8.296    -73.307 nl\
 C3    -16.002    -80.010     10.896    -80.864 nl\
 D0      0.010    -48.012     15.628    -45.397 nl\
 D1      0.010    -80.015     26.039    -75.660 nl\
 D2      0.010    -96.012     31.243    -90.787 nl\
 E0     16.011    -48.016     30.760    -40.196 nl\
 G0     32.017    -48.0172    45.896    -34.990 nl\
 G1     32.020   -112.010     66.716    -95.501 nl\
 G2     31.995    -24.003     38.063    -12.289 nl\
 H0     64.015    -48.007     76.150    -24.572 nl\
 I1     72.001    -87.997     96.711    -59.789 nl\
 J1     88.016    -71.992    106.648    -39.444 nl\
 J2     88.016    -96.005    114.460    -62.151 nl\
 J3     88.016      7.996     80.628     36.193 nl\
 J4     88.016     23.993     75.424     51.320 nl\
 J5     88.016     47.987     67.618     74.009 nl\
 J6     88.016     71.990     59.810     96.706 nl\
 K0     96.002    -48.006    106.397    -14.165 nl\
 L0    104.021    -47.998    113.977    -11.549 nl\
 M0    112.013    -48.000    121.535     -8.951 nl\
 U1    -16.000    -16.000     -9.925    -20.335 nl\
 U2     24.000     24.000     14.887     30.502 nl\
 U3     64.0013    47.9725    44.915     66.183 nl\
 U4    112.000      8.000    103.306     43.999'
 
station_pos_dict=dict()
for s in vlti_stations.split('nl'):
    #keys = station , values = (N,E) (units = meters)
    ss=s.split()
    station_pos_dict[ss[0]]=(float(ss[-1]), float(ss[-2])) # (N,E) coordinate
    

AT_config_dict = {'small':['A0','B2','D0','C1'], 'medium':['K0','G2','D0','J3'],\
                  'large':['A0','G1','J2','J3'],'astrometric':['A0','G1','J2','K0']}


def B_L(wvl,T): #Spectral density as function of temperature
    c = 3e8 #speed of light m/s
    h = 6.63e-34 # plank's constant J.s
    kB = 1.38e-23 #boltzman constant m2 kg /s^2/K

    #wvl is wavelength vector to calculate spectral density at
    #T is temperature of the black body 
    Bb = 2*h*c**2/wvl**5 * 1 / (np.exp(h*c/(wvl*kB*T)) - 1)
    return(Bb)

    
def resolved_binary_V(u, v, **visibility_args): 
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
    

    dRA = visibility_args['dRA']
    dDEC = visibility_args['dDEC']
    F = visibility_args['F']
    ud_p = visibility_args['ud_p']
    ud_c = visibility_args['ud_c']
    R = visibility_args['R']
    
    x_p = np.pi * ud_p  * np.sqrt(u**2 + v**2) 
    
    x_c = np.pi * ud_c  * np.sqrt(u**2 + v**2) 
    
    #primary Visibility (uniform disk)
    V_s = 2 * special.j1(x_p) / x_p
    
    #companion Visibility (point source)
    V_c = np.exp(-2*np.pi*1j * (u * dRA + v * dDEC) ) * 2 * special.j1(x_c) / x_c
    
    # bandwidth smearing correction
    if (R==None):
        G = 1
    elif (R!=None):
        eta = np.pi * (u * dRA + v * dDEC) / (R)
        G = abs( np.sin(eta) / eta )

    V = (V_s + G * F * V_c) / (1+F)

    return(V)




def clumpy_photosphere(u, v, **visibility_args):
    """
    Using Zernike modes across a uniform disk to simulate clumpiness 
    image = Sum_i(Z_i(rho,theta))
    V = F[Image] = Sum_i(F[Z_i(rho,theta)])
    Noll derives analytic expressions for the Fourier transform of Zernike modes 
    
    It would be better reconstruction purpose to use KL modes so covariance is diagonalized (can optimize modes independently )
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    **visibility_args : TYPE dictionary holding information about each mode 
        DESCRIPTION. 
            The number of top level keys indicates the # modes considred. 
            Each mode should hold another dictionary with
                - a uniform disk diameter (ud),  
                - Noll index of Zernike mode (j) 
                - and flux ratio (F). 
        e.g. visibility_args['mode_1'] = {'j':5,'ud':mas2rad(5),'F':0.5}

    Returns
    -------
    VISIBILITY
    
    #Test
    #Bx = np.linspace(-120,120,10); By = np.linspace(-120,120,10); wvls 
    u,v = np.meshgrid( np.linspace(-1e8,1e8,20),np.linspace(-1e8,1e8,20))  #np.meshgrid( Bx[:,np.newaxis]/wvls , Bx[:,np.newaxis]/wvls)
    
    visibility_args = {'mode1':{'j':1,'ud':mas2rad(5),F:1}} # UD of 5mas
    V=clumpy_photosphere(u, v, **visibility_args)
    
    plt.pcolormesh(u,v, np.abs( V ) )
    
    #OR
    
    u = np.linspace(1e5,1e8,100); v = np.zeros(100) 
    visibility_args = {'mode1':{'j':5,'ud':mas2rad(5),'F':1}}
    V=clumpy_photosphere(u, v, **visibility_args)
    plt.plot(np.sqrt(u**2+v**2), V)
    
    # for a disk with some small mode onto with flux ratio F
    
    u,v = np.meshgrid( np.linspace(-1e8,1e8,20),np.linspace(-1e8,1e8,20))  #np.meshgrid( Bx[:,np.newaxis]/wvls , Bx[:,np.newaxis]/wvls)
    
    visibility_args_1 = {'j':1,'ud':mas2rad(5),'F':1} #piston 
    visibility_args_2 = {'j':5,'ud':mas2rad(5),'F':0.5} #astig
    
    visibility_args={'mode1':visibility_args_1, 'mode2':visibility_args_2}
    
    V = clumpy_photosphere(u, v, **visibility_args) 
    plt.pcolormesh(u,v,abs(V))
    
    """
    rho = np.sqrt(u**2 + v**2)
    phi = np.arctan2(u,v)

    for i, mode in enumerate(visibility_args):
        
        j = visibility_args[f'{mode}']['j']
        ud = visibility_args[f'{mode}']['ud']
        F = visibility_args[f'{mode}']['F']
        
        if i==0:
            V = F * zernike_visibility(j, ud, rho, phi) 
            if not hasattr(V,'__len__'):
                V+=0j #force number to complex type
            else:
                V=np.array(V,dtype=complex) #force to array with complex type
                
        else:
            V += F * zernike_visibility(j, ud, rho, phi)
            
        #normalization so V is between 1 and 0
        V *= 1/sum([visibility_args[c]['F'] for c in visibility_args])     
            
    
    return(V)

def noll_indices(j):
    """
    COPIED FROM PYZELDA!!
    Convert from 1-D to 2-D indexing for Zernikes or Hexikes.
    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.
    """

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if np.mod(j,2)==1:
            sign = -1
        else:
            sign = 1

        if np.mod(n,2)==1:
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    #_log.debug("J=%d:\t(n=%d, m=%d)" % (j, n, m))
    return(n, m)
    

def zernike_visibility(j, ud, rho, phi):
    """
    NEED TO TEST

    Parameters
    ----------
    j : TYPE, float
        DESCRIPTION. Noll index
    ud : TYPE, float
        DESCRIPTION. uniform disk diameter of Zernike mode
    rho : TYPE, array like
        DESCRIPTION. radius in Fourier plane (1/wvl * sqrt( Bx**2 + By**2 ) )
    phi : TYPE array like
        DESCRIPTION. angle in Fourier plane arctan(Bx/By). Bx is in North, By in East - so angle subtended from north axis

    Returns
    -------
    visibility

    """

    
    # Fourier transform of Zernike mode Noll index J (from original Noll paper 1976)
    
    # OSA [4] and ANSI single-index Zernike polynomials using
    n, m = noll_indices(j)
    
    if (np.mod(j,2)==0)  & (m!=0):
        V = np.sqrt(n+1) * (2*abs( special.jv( n+1, np.pi* ud * rho)/(np.pi* ud * rho )) ) * (-1)**((n-m)/2) * (1j)**m * 2**0.5 * np.cos(m * phi)
        
    elif (np.mod(j,2)==1) & (m!=0):
        V = np.sqrt(n+1) * (2*abs( special.jv( n+1, np.pi* ud * rho)/(np.pi* ud * rho )) ) * (-1)**((n-m)/2) * (1j)**m * 2**0.5 * np.sin(m * phi)
    
    else: 
        V = np.sqrt(n+1) * (2*abs( special.jv( n+1, np.pi* ud * rho)/(np.pi* ud * rho )) ) * (-1)**(n/2)
        
    return(V) 

    
def binary_bispectrum(Bx_proj1, Bx_proj2, By_proj1, By_proj2, wvls, visibility_func, **visibility_args):
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
    
    visibility_func : TYPE, function
        DESCRIPTION. function to use to calculate the visibilities at the u=By/wvl,v=Bx/wvl coordinates
    
    **visibility_args: dictionary holding the relevant arguments of the visibility_func.
    
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
                V_triangle[i,j] = visibility_func(u = triangle_matrix[i,j][0], v = triangle_matrix[i,j][1],   **visibility_args)
        # format:
        # triangle 0 | V(u1,v1)_0, V(u2,v2)_0, V(u3,v3)_0 |
        #   ...      |                ...              | 
        # triangle N | V(u1,v1)_N, V(u2,v2)_N, V(u3,v3)_N | etc
        
        # now calculate the bispectrum for each triangles in our wavelength bin
        bispectrum[w] = [v[0]*v[1]*np.conjugate(v[2]) for v in V_triangle  ]
        triangles[w] = triangle_matrix
        
    return( triangles, bispectrum )

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



def chi2(y_model, y_true, yerr):
    return(sum( (y_model-y_true)**2/yerr**2 ) )

    
def rad2mas(rad):
    return(rad*180/np.pi * 3600 * 1e3)

def mas2rad(mas):
    return(mas*np.pi/180 / 3600 / 1e3)
"""bispectrum={}  
triangles={}
for i,w in enumerate( wvls ): 
    # fun fact: number of triangles from n points is n*(n-1)*(n-2)/6 ! 
    
    u1 = B_ty[:,0]/w
    u2 = B_ty[:,1]/w
    u3 = -u2-u1 #close triangle
    
    v1 = B_tx[:,0]/w
    v2 = B_tx[:,1]/w
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
    V_triangle = [] #1j*np.ones([number_of_triangles,3]) 
    for i in range(triangle_matrix.shape[0]):
        for j in range(triangle_matrix.shape[1]):
            V_triangle.append( resolved_binary_V(u = triangle_matrix[i,j][0], v = triangle_matrix[i,j][1],   **visibility_args) )
    # format:
    # triangle 0 | V(u1,v1)_0, V(u2,v2)_0, V(u3,v3)_0 |
    #   ...      |                ...              | 
    # triangle N | V(u1,v1)_N, V(u2,v2)_N, V(u3,v3)_N | etc
    
    # now calculate the bispectrum for each triangles in our wavelength bin
    bispectrum[w] = [v[0]*v[1]*np.conjugate(v[2]) for v in V_triangle  ]
    triangles[w] = triangle_matrix"""



def prepare_CP(files, EXTVAR=None):
    
    cp_sigma = {} #all cp sigmas away from zero 
    cp_values = {} # all cp
    cp_mask = {} #bad data flag 
    AT_config = []
    for f in files :    
        h = oifits.open(f)
                
        AT_config.append( h['OI_ARRAY'].data['STA_NAME'] )
        
        indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
        
        current_config = ''.join(list(np.sort(h['OI_ARRAY'].data['STA_NAME'])))
        
        #effective wavelength
        wvl = h['OI_WAVELENGTH',EXTVAR].data['EFF_WAVE']
        
        cp = h['OI_T3',EXTVAR].data['T3PHI'][:,:]
        
        cp_err = h['OI_T3',EXTVAR].data['T3PHIERR'][:,:]
        
        cp_flag = h['OI_T3',EXTVAR].data['FLAG']
        
        triangles = [[indx2station[h['OI_T3',EXTVAR].data['STA_INDEX'][tri][tel]] for tel in range(3)] for tri in range(4)]
    
        filt = ~cp_flag
        for i,w in enumerate(wvl):
            if w not in cp_values:
                cp_values[w] = [list( cp[:,i] ) ]
                cp_sigma[w] = [list( cp[:,i] / cp_err[:,i] ) ]
                cp_mask[w] = [list( filt[:,i] ) ]
            else:
                cp_values[w].append( list( cp[:,i] ) )
                cp_sigma[w].append( list( cp[:,i] / cp_err[:,i] ) )
                cp_mask[w].append( list( filt[:,i] ) )


    return( cp_values, cp_sigma , cp_mask, AT_config)



def scatter_hist(x, y, ax,  ax_histy, color='k'):
    # no labels
    #ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,color=color,alpha=0.4, s=0.45)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    #ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal',color=color,alpha=0.4)

    
def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]


# %% BINARY HYPOTHESIS 
# set up observational  wavelengths and baselines 
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
wvls = np.array( list( ud_fits.index.astype(float) ) )#1e-6 * np.linspace(2,2.4,40)

configurations = ['small','medium','large']

mast_dict_binary = {} 
for configuration in configurations:
    
    print(f'looking at {configuration} configuration')
    tel_x = [station_pos_dict[c][0] for c in AT_config_dict[configuration]] # north
    tel_y = [station_pos_dict[c][1] for c in AT_config_dict[configuration]] # east
    #tel_x = 130 *(0.5-np.random.rand(10)) # North coordinate
    #tel_y = 130 *(0.5-np.random.rand(10)) # East coorindate
    
    # UV ROTATION 
    
    # the baselines vectors [(Bx,By)..(Bx,By)] formed by telescopes 
    Bx = np.diff( list(itertools.combinations(tel_x,2 ) ) ).reshape(-1) #north 
    By = np.diff( list(itertools.combinations(tel_y,2 )) ).reshape(-1) #east
    u = By[:,np.newaxis]/wvls #east, columns correspond to wavelength
    v = Bx[:,np.newaxis]/wvls #north, columns correspond to wavelengt 
    baseline_coords = [(x,y) for x,y in zip( Bx, By) ]
    
    # x,y coorindates of baselines vector in each triangle !! 3 baselines per triangle , n(n-1)(n-2)/6 triangles per n telescope
    B_tx , B_ty = get_trianlge_baselines_from_telescope_coordinates( tel_x, tel_y )
    
    
    # project them 
    # get_projected_baselines( B_x, B_y , 0, np.deg2rad(-100)) # need to deal with input format 
    
    # set-up binary parameters 
    
    
    V2_binary=[]
    CP = []
    triangle_Bmax = []
    triangle_uvmax = []
    
                
    T_p = 3200 #primary temp, K
    T_c = 1000 #companion temp, K
    ud_p = ud_fits['ud_mean'].values[0] # we just take fit at shortest wavelength !! 
    flux_ratios = B_L(wvls,T_c) / B_L(wvls,T_p) #contrast
    obs_dict = {}   
    #companion outside the photosphere
    for dRA in list(np.linspace(mas2rad(-30),mas2rad(-ud_p), 10)) + list(np.linspace(mas2rad(ud_p),mas2rad(30), 10)):
        for dDEC in list(np.linspace(mas2rad(-30),mas2rad(-ud_p), 10)) + list(np.linspace(mas2rad(ud_p),mas2rad(30), 10)):
            
            coord = (dRA,dDEC)
            
            obs_dict[coord]={}
            V2_binary=[]
            CP = []
            triangle_Bmax = []
            triangle_uvmax = []
            
            for i,w in enumerate( wvls ):
    
                ud_c = 0.27 * ud_p
                F = flux_ratios[i]
                
                visibility_args = {'dRA':dRA,'dDEC':dDEC,'F':F,'R':None, 'ud_p':mas2rad(ud_p), 'ud_c':mas2rad(ud_c)}
                
                V2_binary.append( resolved_binary_V(u[:,i], v[:,i], **visibility_args) )
                
                # 4 telescopes => 4*3*2/6 = 4 closing triangles => 4 bispectrums,  V_matrix should be 4 x 3
                triangles, bispectrum = binary_bispectrum(B_tx[:,0], B_tx[:,1], B_ty[:,0], B_ty[:,1], [w], resolved_binary_V, **visibility_args)
             
                CP.append( np.rad2deg( np.angle( bispectrum[w] ) ) )  # transpose to keep columns corresponding to wvl 
                
                       
                # get baseline lengths sqrt(Bx^2 + By^2) for each triangle
                triangle_baseline_lengths = get_baseline_lengths_from_traingles(triangles)
                
                # get angular frequency lengths sqrt(u^2 + v^2) for each triangle
                triangle_angularfreq_lengths = get_angular_frequency_from_traingles(triangles)
                
                # max baseline lengths for each triangle 
                triangle_Bmax.append( np.max( list( triangle_baseline_lengths.values() )[0] , axis=1) ) #np.array([np.max(b) for b in triangle_baseline_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 
                
                # max angular frequency lengths for each triangle 
                triangle_uvmax.append(  np.max( list( triangle_angularfreq_lengths.values() )[0] , axis=1) ) # np.array( [np.max(b) for b in triangle_angularfreq_lengths[w]] )  # transpose to keep columns corresponding to wvl 
        
            obs_dict[coord]['V2'] = V2_binary
            obs_dict[coord]['CP'] = CP
            obs_dict[coord]['u'] = u
            obs_dict[coord]['v'] = v
            obs_dict[coord]['triangle_Bmax'] = triangle_Bmax
            obs_dict[coord]['triangle_uvmax'] = triangle_uvmax
            
    mast_dict_binary[configuration] = obs_dict  
    




fig_kwargs = {'fontsize':14}

# get all CPs for a given wavelength 

"""fig,ax = plt.subplots(3, 1, figsize=(10,15),sharex=True) 

for axx, conf in zip(ax.reshape(-1),configurations): 
    for w in [1.65e-6, 2.2e-6, 3.6e-6, 4.1e-6,8e-6, 11e-6]: 
        axx.hist( flatten_comprehension( [np.array( mast_dict_binary[conf][coord]['CP'])[np.argmin(abs(w-wvls))] for coord in mast_dict_binary[conf]] ), bins=20, alpha=0.3, label=f'{round(1e6*w,2)}um' )
    axx.set_ylabel(r'frequency', **fig_kwargs)
    axx.tick_params(labelsize=14)

plt.legend()    
plt.xlabel(r'CP [deg]',**fig_kwargs )"""



fig,ax = plt.subplots(3, 1, figsize=(10,15),sharex=True,sharey=True) 
for axx, conf in zip(ax.reshape(-1),configurations): 
    for coord in mast_dict_binary[conf]:
        axx.plot( wvls, mast_dict_binary[conf][coord]['CP'] ,'.')
    axx.set_ylabel(r'CP [deg]',**fig_kwargs )
    axx.set_title(f'{conf} configuration' )
axx.set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )

# contrast plot 
plt.figure()
plt.semilogy( wvls , B_L(wvls,T_c) / B_L(wvls,T_p) ,'.')
plt.xlabel(r'wavelength [$\mu$m]',**fig_kwargs )
plt.ylabel(r'contrast', **fig_kwargs)
plt.gca().tick_params(labelsize=14)

# contrast plot 
plt.figure()
plt.plot( triangle_uvmax , CP,'.')
plt.xlabel(r'$B_{max}/\lambda$ [rad$^{-1}$]',**fig_kwargs )
plt.ylabel(r'CP [deg]', **fig_kwargs )
plt.gca().tick_params(labelsize=14)

"""
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
"""


#%% DIPOLE MODEL
mast_dict_dipole = {} 
for configuration in configurations:
    
    print(f'looking at {configuration} configuration')
    tel_x = [station_pos_dict[c][0] for c in AT_config_dict[configuration]] # north
    tel_y = [station_pos_dict[c][1] for c in AT_config_dict[configuration]] # east
    #tel_x = 130 *(0.5-np.random.rand(10)) # North coordinate
    #tel_y = 130 *(0.5-np.random.rand(10)) # East coorindate
    
    # UV ROTATION 
    
    # the baselines vectors [(Bx,By)..(Bx,By)] formed by telescopes 
    Bx = np.diff( list(itertools.combinations(tel_x,2 ) ) ).reshape(-1) #north 
    By = np.diff( list(itertools.combinations(tel_y,2 )) ).reshape(-1) #east
    u = By[:,np.newaxis]/wvls #east, columns correspond to wavelength
    v = Bx[:,np.newaxis]/wvls #north, columns correspond to wavelengt 
    baseline_coords = [(x,y) for x,y in zip( Bx, By) ]
    
    # x,y coorindates of baselines vector in each triangle !! 3 baselines per triangle , n(n-1)(n-2)/6 triangles per n telescope
    B_tx , B_ty = get_trianlge_baselines_from_telescope_coordinates( tel_x, tel_y )
    
    
    # project them 
    # get_projected_baselines( B_x, B_y , 0, np.deg2rad(-100)) # need to deal with input format 
    
    # set-up binary parameters 
    
    
    V2_binary=[]
    CP = []
    triangle_Bmax = []
    triangle_uvmax = []
    
                
    T_p = 3200 #primary temp, K
    
    ud_p = ud_fits['ud_mean'].values[0] # we just take fit at shortest wavelength !! 
    #ud_c = 0.27 * ud_p
    
    obs_dict = {}    
    
    visibility_args_ud = {'j':1,'ud':mas2rad(ud_p),'F':1} #piston 
    
    for j in range(6,10):
        for delta_T in np.linspace(-150, 150,30) : #companion temp, K
            T_mode = T_p + delta_T
            flux_ratios = 1 - B_L( wvls, T_mode ) / B_L( wvls, T_p ) #needs to be 1-F so that when delta_T=0 F=0
            coord = (j, T_mode )
        
        
            obs_dict[coord]={}
            V2_binary=[]
            CP = []
            triangle_Bmax = []
            triangle_uvmax = []
            for i,w in enumerate( wvls ):
    
                
                F = flux_ratios[i]
                
                visibility_args_mode = {'j':j,'ud':mas2rad(ud_p),'F':F} #astig
                visibility_args = {'mode1':visibility_args_ud, 'mode2':visibility_args_mode }
                
                V = clumpy_photosphere(u, v, **visibility_args) 
    
                # 4 telescopes => 4*3*2/6 = 4 closing triangles => 4 bispectrums,  V_matrix should be 4 x 3
                triangles, bispectrum = binary_bispectrum(B_tx[:,0], B_tx[:,1], B_ty[:,0], B_ty[:,1], [w], clumpy_photosphere, **visibility_args)
             
                CP.append( np.rad2deg( np.angle( bispectrum[w] ) ) )  # transpose to keep columns corresponding to wvl 
                
                       
                # get baseline lengths sqrt(Bx^2 + By^2) for each triangle
                triangle_baseline_lengths = get_baseline_lengths_from_traingles(triangles)
                
                # get angular frequency lengths sqrt(u^2 + v^2) for each triangle
                triangle_angularfreq_lengths = get_angular_frequency_from_traingles(triangles)
                
                # max baseline lengths for each triangle 
                triangle_Bmax.append( np.max( list( triangle_baseline_lengths.values() )[0] , axis=1) ) #np.array([np.max(b) for b in triangle_baseline_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 
                
                # max angular frequency lengths for each triangle 
                triangle_uvmax.append(  np.max( list( triangle_angularfreq_lengths.values() )[0] , axis=1) ) # np.array( [np.max(b) for b in triangle_angularfreq_lengths[w]] )  # transpose to keep columns corresponding to wvl 
        
            obs_dict[coord]['V2'] = V2_binary
            obs_dict[coord]['CP'] = CP
            obs_dict[coord]['u'] = u
            obs_dict[coord]['v'] = v
            obs_dict[coord]['triangle_Bmax'] = triangle_Bmax
            obs_dict[coord]['triangle_uvmax'] = triangle_uvmax
                
    mast_dict_dipole[configuration] = obs_dict  
    

fig,ax = plt.subplots(3, 1, figsize=(10,15),sharex=True,sharey=True) 
for axx, conf in zip(ax.reshape(-1),configurations): 
    for coord in mast_dict_dipole[conf]:
        axx.plot( 1e6*wvls, mast_dict_dipole[conf][coord]['CP'] ,'.',color='k')
    axx.set_ylabel(r'CP [deg]',**fig_kwargs )
    axx.text( 9, 150,f'{conf} configuration' )
axx.set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )



#%% putting them together
fig,ax = plt.subplots(3, 2, figsize=(15,10),sharex=True,sharey=True) 
for row, conf in zip(range(len(ax)),configurations): 
    for coord in mast_dict_dipole[conf]:
        ax[row,0].plot( 1e6*wvls, mast_dict_dipole[conf][coord]['CP'] ,'.',color='k',alpha=0.5)
    ax[row,0].set_ylabel(r'CP [deg]',**fig_kwargs )
    ax[row,0].text( 8, 150,f'{conf} configuration' ,**fig_kwargs)
ax[row,0].set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )
ax[0,0].set_title('oscilatory mode hypothesis')

for row, conf in zip(range(len(ax)),configurations): 
    for coord in mast_dict_binary[conf]:
        ax[row,1].plot( 1e6*wvls, mast_dict_binary[conf][coord]['CP'] ,'.',color='k',alpha=0.5)
    ax[row,1].text( 8, 150,f'{conf} configuration' , **fig_kwargs)
ax[row,1].set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )
ax[0,1].set_title('binary hypothesis')

#plt.tight_layout()
plt.savefig(fig_path + 'MC_simulation_CP_binary_vs_dipole_hypoth.png',dpi=200)



#%% 
# matisse wvl limits from https://www.eso.org/sci/facilities/paranal/instruments/matisse.html
#mat_L_wvl_filt = (mati_L_v2_df.columns > 3.2e-6) & (mati_L_v2_df.columns < 3.9e-6) #| (mati_L_v2_df.columns > 4.5e-6) 
#mat_M_wvl_filt = (mati_L_v2_df.columns > 4.5e-6) &  (mati_L_v2_df.columns <= 5e-6)
#mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) & (mati_N_v2_df.columns <= 12.1e-6)#| (mati_L_v2_df.columns > 4.5e-6)


pionier_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')

gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]

file_set_labels = ['H','K','LM','N']
obs_CP_dict = {}
configs = []
for lab , files in zip(file_set_labels, [pionier_files, gravity_files, matisse_files_L, matisse_files_N]):
    print(lab)
    if lab=='K':
        EXTVAR = 11
    else:
        EXTVAR = None
    cp_values, cp_sigma , cp_mask, at_config = prepare_CP(files,EXTVAR)
    
    # flattening out 
    cp_flat =np.array([])
    cpZ_flat =np.array([])
    wvl_flat =np.array([])
    configs.append(  at_config  )
    for w in cp_values:
        
        mask_tmp = np.array( cp_mask[w] )
    
        v_tmp = np.array( cp_values[w] )
        s_tmp = np.array( cp_sigma[w] )
        wvl_tmp = w * np.ones(np.array(cp_values[w]).shape)
    
        v_filt = v_tmp[mask_tmp]
        s_filt = s_tmp[mask_tmp]
        wvl_filt = wvl_tmp[mask_tmp]
        
        wvl_flat=np.concatenate([wvl_flat,wvl_filt])
        cp_flat=np.concatenate([cp_flat,v_filt])
        cpZ_flat=np.concatenate([cpZ_flat,s_filt])

    obs_CP_dict[lab] = {'wvl_flat':wvl_flat,'cp_flat':cp_flat,'cpZ_flat':cpZ_flat}

all_wvls = flatten_comprehension( [obs_CP_dict[lab]['wvl_flat'] for lab in obs_CP_dict] )
all_cp = flatten_comprehension( [obs_CP_dict[lab]['cp_flat'] for lab in obs_CP_dict] )
np.array( [set(x) for x in at_config] ) == set(['A0', 'J2', 'G1', 'K0'])



fig,ax = plt.subplots(1,1,figsize=(8,5))
ax.plot( 1e6* np.array(all_wvls), np.array(all_cp),'.')
ax.set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )
ax.set_ylabel(r'CP [deg]',**fig_kwargs )



spectral_feature_dictionary_k = {'HeI':[2.038, 2.078], 'MgII':[2.130, 2.150],'Brg':[2.136, 2.196],\
                               'NaI':[2.198, 2.218], 'NIII': [2.237, 2.261],'CO(2-0)':[2.2934, 2.298],\
                                   'CO(3-1)':[2.322,2.324],'CO(4-2)':[2.3525,2.3555]}
    
spectral_feature_dictionary_LM = {'Pfe':[3.010, 3.070], 'Pfd':[3.270, 3.330],'Pfg': [3.710, 3.770],\
                               'HeI':[3.858, 3.918], 'Bra': [4.020, 4.080],'Fe I':[4.354, 4.414],\
                                   'CO Ice':[4.64, 4.70]}
    
spectral_feature_dictionary_N = {'OI':[8.416, 8.476], 'PAHs':[8.5, 8.8], 'Si amorphous silicate':[9.6, 9.8],\
                                 'PAHs/SiC':[11.15, 11.35], 'H2O':[11.4, 11.6] ,'Hua':[12.3, 12.6]}

fig,ax = plt.subplots(1,1,figsize=(8,5))
filt_cont = np.ones(len(all_wvls),dtype=bool)
for sp in spectral_feature_dictionary_N:
    filt_tmp = (1e6* np.array(all_wvls) > spectral_feature_dictionary_N[sp][0]) &  (1e6* np.array(all_wvls) < spectral_feature_dictionary_N[sp][1])

    filt_cont *=   ~filt_tmp  
    ax.plot( 1e6* np.array(all_wvls)[filt_tmp], np.array(all_cp)[filt_tmp],'.',label=sp)
    ax.set_xlabel(r'wavelength [$\mu$m]',**fig_kwargs )
    ax.set_ylabel(r'CP [deg]',**fig_kwargs )

ax.plot( 1e6* np.array(all_wvls)[filt_cont], np.array(all_cp)[filt_cont],'.',color='k',label='continuum')    
ax.legend()

"""    
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
#ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)


scatter_hist(1e6 * wvl_flat ,cp_flat , ax,  ax_histy)

ax.set_ylabel('CP [deg]',fontsize=14)
ax.set_xlabel(r'wavelength [$\mu$m]',fontsize=14)
ax_histy.set_xlabel('counts',fontsize=14)
ax_histy.set_xscale('log')
ax.grid()
ax_histy.grid()
"""

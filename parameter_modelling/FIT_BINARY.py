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
  

    
def plot_uv_triangles( triangles ):
    """
    plots baseline triangles

    Parameters
    ----------
    triangles : TYPE, list of tuples 
        DESCRIPTION. vertices corresponding to each baseline coordinate in triangle. e.g. 
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
    

def get_baseline_lengths_from_traingles(triangles):
    triangle_baselines = {}
    for w in triangles: # for each wavelength in triangle dictionary
        triangle_baselines[w]=[] #init list to hold baseline lengths
        for vertices in triangles[w]: # for each set of triangle vertices in the uv plane
            # we calculate the baseline length np.sqrt( Bx**2 + By**2)
            triangle_baselines[w].append( [np.sqrt( b[0]**2 + b[1]**2) for b in vertices] )

    return(triangle_baselines)

def get_angular_frequency_from_traingles(triangles):
    triangle_angular_freq = {}
    for w in triangles: # for each wavelength in triangle dictionary
        triangle_angular_freq[w]=[] #init list to hold baseline lengths
        for vertices in triangles[w]: # for each set of triangle vertices in the uv plane
            # we calculate the baseline length np.sqrt( Bx**2 + By**2)
            triangle_angular_freq[w].append( [np.sqrt( b[0]**2 + b[1]**2)/w for b in vertices] )

    return(triangle_angular_freq)





# now how to prep data to fit  - I just need u_mat, v_mat from data , and corresponding to CP...




#---- Test 
# set up observational  wavelengths and baselines 
wvls = 1e-6 * np.linspace(1.4,1.6,5)
B_x = 150 *(0.5-np.random.rand(15))  # baselines no positions 
B_y = 150 * (0.5-np.random.rand(15))  # baselines no positions 
B_coords = np.array( [B_x,B_y] ).T

u_mat = B_y[:,np.newaxis]/wvls # (baseline, wvl)
v_mat = B_x[:,np.newaxis]/wvls
u = np.array( [item for row in u_mat for item in row] )
v = np.array( [item for row in v_mat for item in row] )

# look at where we are sampling in uv plane
#plt.figure() 
#plt.scatter(u, v)

# set-up binary parameters 
F = 0.01
ud = mas2rad(3)
dRA = mas2rad(20)
dDEC = mas2rad(1)

# calculate visibilities , baseline triangles and bispectrum (for CP)
V_binary = binary_V(u_mat, v_mat, dRA, dDEC, F , ud, R=None)
triangles, bispectrum = binary_bispectrum(u_mat, v_mat, wvls, dRA, dDEC, F , ud)

# get baseline lengths sqrt(Bx^2 + By^2) for each triangle
triangle_baseline_lengths = get_baseline_lengths_from_traingles(triangles)

# get angular frequency lengths sqrt(u^2 + v^2) for each triangle
triangle_angularfreq_lengths = get_angular_frequency_from_traingles(triangles)

# max baseline lengths for each triangle 
triangle_Bmax = np.array( [[np.max(b) for b in triangle_baseline_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 

# max angular frequency lengths for each triangle 
triangle_uvmax = np.array( [[np.max(b) for b in triangle_angularfreq_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 

triangle_uvmean = np.array( [[np.max(b) for b in triangle_angularfreq_lengths[w]] for w in wvls] ).T # transpose to keep columns corresponding to wvl 

CP = np.array( [np.rad2deg( np.angle( bispectrum[w] ) ) for w in wvls] ).T # transpose to keep columns corresponding to wvl 

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].plot(u_mat, v_mat,'.')
ax[0].set_xlabel('U [rad$^-1$]')
ax[0].set_ylabel('V [rad$^-1$]')

ax[1].set_title( f'F={F}, ud = {round(rad2mas(ud),1)}mas, dRA = {round(rad2mas(dRA),1)}mas, dDEC = {round(rad2mas(dDEC),1)}mas')
ax[1].plot(  np.sqrt(u_mat**2 + v_mat**2), abs(V_binary)**2 ,'.')
ax[1].set_xlabel('projected baseline (m)')
ax[1].set_ylabel(r'$|V|^2$')

ax[2].plot( triangle_uvmean , np.array( CP ) ,'.' )
ax[2].set_xlabel('max projected  in triangle (m)')
ax[2].set_ylabel('CP [deg]')





















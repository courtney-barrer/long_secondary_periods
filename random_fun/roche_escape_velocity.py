#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:02:26 2023

@author: bcourtne



"""
import numpy as np 
import matplotlib.pyplot as plt


def vbet(r, r0, v_inf, css, beta):
    v= css + (v_inf-css)*(1-r0/r)**beta
    return(v)

def KE(v,m):
    KE = 0.5 * m * v**2
    return(KE)

def gPE(m, M,r):
    PE=-G* m * M/r
    return(PE)

def rochelobe_radius(s,q):
    r = s * 0.49 * q**(2/3) / (0.6*q**(2/3) + np.log(1+q**(1/3)))
    return(r)

G = 6.674e-11 #N⋅m2/kg2
au2m = 1.496e+11 #m/au
km2m = 1e3 
solar2kg = 1.989e+30 #kg/M*
m2parsec = 3.24078e-17 #m/parsec

T =  60 * 60 * 24 *757 # RT pav period (seconds)

# what Davide uses in thesis 
#css = 2.65 # km/s. calculated from ideal gas law assume sonic point is at dust condensation radius and T=1500K at dust condensation 

eta2 = 6.5 #3.77 # ratio of  initial (@ dust cond. ) to terminal velocity
v_inf = 10 * km2m #m/s 
css =  v_inf  * eta2 # speed of sound

# radial unitas au
q = 10 #primary mass over companion 
M = 3 * solar2kg
m = M/q 
s = ( G * (m+M) * T**2 / ( 4*np.pi**2 )  )**(1/3) #* m2parsec

beta = 2
FF = 0.8

rl = rochelobe_radius(s,q)
r_c = FF*rl


#set initial radius to condensation radius 
r0 = r_c.copy()

r_inf = 100 * au2m
r = np.linspace(r0, r_inf, 1000)


ratio = 0.5  * (css + (v_inf-css)*(1-r0/r)**beta)**2 / (G * M/r)


plt.loglog( r/au2m, ratio); plt.axhline( 1 ,color='r')
plt.axvline( rl /au2m ,color='k',linestyle=':',label= 'roche lobe radius')
#plt.axvline( r0 /au2m ,color='k')
plt.xlabel('radius [au]')
plt.ylabel('KE/PE')
plt.legend()
#KE(vbet(r, r0, v_inf, css, 2),m) / gPE(m, M,r)

#%% 


# fixed parameters 
beta = 2
FF = 0.8
M = 3 * solar2kg

# free parameters 
eta2_ar = np.linspace(1,10,5) 
M_ar = np.linspace(2,5,4)* solar2kg
q_ar = np.logspace(0,2,3) #primary mass over companion 
ratio_grid=[]
for eta2 in eta2_ar: 
    for M in M_ar:
        for q in q_ar:
            # derived parameters 
            m = M/q 
            s = ( G * (m+M) * T**2 / ( 4*np.pi**2 )  )**(1/3) #* m2parsec
            
            rl = rochelobe_radius(s,q) #roche lobe radius 
            r_c = FF*rl # condensation radius 
            
            ratio_grid.append( 0.5  * (css + (v_inf-css)*(1-r0/rl)**beta)**2 / (G * M/rl) )
            
Mgrid,eta2grid,qgrid = np.meshgrid(M_ar, eta2_ar , q_ar )
threshold_filt = np.array( ratio_grid ).reshape(5,4,3) > 1

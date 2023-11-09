#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 01:03:30 2023

@author: bcourtne

my attempt at basic orbital simulation to fit light curves,
inspired by elliptical orbit wikipedia 

OBJECTIVE:
    to constrain the seperation and projected area ratio of primary to 
    hypothetical companion 
    
    this can be used together with Matisse data (and general VLTI) data to constrain LSP theoies 
"""

import numpy as np
import matplotlib.pyplot as plt 
import pmoired

def ellipse_radius(a,e, theta):
    r = a*(1-e**2) / (1-e*np.cos(theta)) # radius
    return(r)

def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

# simulation parameters 
dt = 20 #days

# orbital parameters (to do grid search over)
a= 10 # au
e = 0.6 # scalar ellipticity 
c = a*e #F1 = 0, F2 = 2c

P = 400 # period (days)
R1 = 2 #primary radius (au)
R2 = 1 #secondary radius (au) to be fitted!
flux_ratio = 0.1 # flux ratio to be fitted!


#elliptical orbit in polar coordinates
theta = np.linspace(0,3*np.pi,1000)  # angle (linear grid alittle more the 2pi)
dtheta = np.diff( theta )[0] #differential element 

r = ellipse_radius(a,e,theta) #+ 2*c # radius



# cumulative area swept out by companion around primary
area_cum = np.cumsum( 1/2 * r**2 * dtheta )

A = area_cum[ np.argmin( abs(theta-2*np.pi) ) ] #total area of ellipes

dA = A/P * dt # incremetal area covered during dt using Keplers law (equal areas covered in equal time)

# find thetas that partition the cumulative area into equal blocks of dA
theta_i = [theta[0]] # to hold thetas that partition the cumulative area into equal blocks of dA

# keplers law : theta_(i+j) - theta_i = dA, find j that minimizes |theta_(i+j) - theta_i - dA|
i = 0 
j = 0 
indx = []
while theta_i[-1] < 2*np.pi:
    obj_fn = [abs(area_cum[j] - area_cum[i] - dA) for j in range(len( theta[:] ) )]
    i = np.argmin( obj_fn ) 
    theta_i.append( theta[i] )
    indx.append(i)


plt.figure()
plt.plot(theta, area_cum)
for tt in theta_i:
    plt.axvline(tt,color='r')
plt.xlabel('theta (rad)')
plt.ylabel('cumulative area')

# also check that this is almost constat plt.plot( np.diff( area_cum[ indx ] ), '.' )


#now calculate radius at these points 
r_i = ellipse_radius(a,e,theta_i)

#convert cartessian coordinates 
x_i, y_i = r_i*np.cos(theta_i) - 2*c, r_i*np.sin(theta_i)  ## need primary star at origin !! why is focus at F2 and not F1 and I have to apply translation by 2c

# lets look 

plt.figure()
plt.plot(x_i, y_i, 'x',label=f'a={a}au, e={e}')
plt.plot([0],[0],'x',lw=8,color='r',label='primary')
plt.xlabel('x (au)',fontsize=15)
plt.ylabel('y (au)',fontsize=15)
plt.xlim([-2*a,2*a])
plt.ylim([-2*a,2*a])
plt.gca().set_aspect(1)
plt.legend()
plt.title('base coordinates')


# turn into vectors 
pos_i = np.array([[x,y,0] for x,y in zip(x_i,y_i)])

# now apply 3d rotation 
rot_mat = Rz(np.pi/3) @ Ry(np.pi/2) @ Ry(0) 
pos_proj = rot_mat @  pos_i.T 


#take a peak at projection
plt.figure()
plt.plot(pos_proj[0], pos_proj[1], 'x',label=f'a={a}au, e={e}')
plt.plot( [0],[0],'x',lw=8,color='r',label='primary')
plt.xlabel('x (au)',fontsize=15)
plt.ylabel('y (au)',fontsize=15)
plt.xlim([-2*a,2*a])
plt.ylim([-2*a,2*a])
plt.gca().set_aspect(1)
plt.legend()
plt.title('rotated (projected) coordinates')


"""
To constrain the problem and keep simple for fitting light curves
 - primary and secondary are always SPHERES
 - they are both completely opaque 
 - they have uniform temperature profiles
 
 projection onto line of sight will always be a circle, 
 just assume circle with flat temperature profile (ignore limb darkening )
 use targets z coordinate at its center to determine which one is in front 
 
 given period, primary radius R1, paramemters to fit :
     a - semi major axis
     e - eccentricity
     Rx - rotation angle around z 
     Ry - rotation angle around z 
     Rz - rotation angle around z 
     
     R2/R1 - radis ratio of obscuring companion (dust cloud?)
     flux_ratio - flux ratio f2/f1

"""
#make sure primary is always at (0,0,0)!!!!! secondary_in_front filter won't work otherwise
secondary_in_front = pos_proj[1]>=0 # boolean indicating if secondary is in front of primary along line of sight 

#creat image grid
x = np.linspace(-2*a, 2*a, 400)
y = np.linspace(-2*a, 2*a, 400)
xx,yy = np.meshgrid(x,y)

plot=True #True

images = []
for it in range(pos_proj.shape[1]):
    
    #imagge of primary (have to keep in loop to re-init each iteration (incase of shared pixels with secondary))
    img_1 = xx**2 + yy**2 < R1**2 
    
    #image of secondary 
    img_2 = (xx - pos_proj[0][it])**2 + (yy - pos_proj[1][it])**2 < R2**2 
    
    
    #find any position where both images == 1 (shared pixels)
    shared_pixels_filt = img_1.reshape(-1) & img_2.reshape(-1) 
    
    if sum(shared_pixels_filt): # if there are shared pixels
    
        if secondary_in_front[it]:
        
            #then put shared pixels in primary to zero 
            img_1.reshape(-1)[shared_pixels_filt] = 0
            
        elif not secondary_in_front[it]:
            
            #then put shared pixels in secondary to zero
            img_2.reshape(-1)[shared_pixels_filt] = 0
        
        
        else:
            raise TypeError('something went wrong right here!!')
    
    
    images.append( img_1 + flux_ratio * img_2 )
    
    if plot:
        plt.figure()
        plt.imshow(img_1 + flux_ratio * img_2)


intensity = np.sum(images, axis=(1,2))

plt.figure()
plt.plot(np.linspace(0,P,len(intensity)), intensity)
plt.xlabel('days')
plt.ylabel('intensity')
plt.title('SIMULATED LIGHT CURVE')
"""

NOW DOWNLOAD LIGHT CURVES AND FIT THE ORBITAL PARAMETERS, FLUX RATIO AND 
AREA RATIO UNDER SPHERICAL CONSTRAINTS.

THIS WILL PUT STRONG CONSTRAINTS ON FITTING VISIBILITIES & CP. 
EITHER:
- THERE IS A SOLUTION THAT FITS BOTH LIGHT CURVE AND VLTI OBSERVABLES, THEREFORE POSSIBLE COMPANION CAUSING LSP
OR
- THERE IS NO COMPANION THAT CAUSES THE LSP (THERE MAY STILL HOWEVER BE A COMPANION BELOW OUR DETECTION LIMITS, WHAT IS IT?)
- THERE IS A COMPANION THAT CAUSES THE LSP WITH SURROUNDING GEOMETRY THAT VARIES SIGNIFICANTLY FROM SPHERICAL SYMMETRY


"""

#%%  now interface with PMOIRED package to get VLTI observables from images (we can also just feed coordinates to this model)
# https://github.com/amerand/PMOIRED/blob/master/examples/Be%20model%20comparison%20with%20AMHRA.ipynb

it = 1 # what iteration in orbit do we run it for?
model = {'1,ud':2,  
         '1,x':0, 
         '1,y':0,
         '1,f':1,
         '2,ud':1, 
         '2,x':pos_proj[0][it], 
         '2,y':pos_proj[1][it],
         '2,f':0.1
        }
#   

# -- wavelength vector to simulate data (in um)
WL_L = np.linspace(3.2, 4.6, 30)
# -- list of sidereal time of observations (hours): must be a list or np.ndarray
lst = [0] 

coord = (0, -32)

# show image
oi = pmoired.OI()
oi.showModel(model, WL=WL_L, imFov=8, showSED=False)


# get VLTI observations of this 
# -- create empty OI object
oi = pmoired.OI()
oi.fig=100
# -- note that oi.data must be a list! 
oi.data = [pmoired.oifake.makeFakeVLTI(['A0','G1','J3','J2'], coord, lst, WL_L, model=model),
           pmoired.oifake.makeFakeVLTI(['D0','G1','J2','K0'], coord, lst, WL_L, model=model)]
oi.show()



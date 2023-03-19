#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:07:07 2023

@author: bcourtne
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pmoired
import phoebe

#from phoebe import u # units

"""
fit light curve and CP & visibilities simultaneously 

create binary mesh , mass,
generate light curve  


"""

#%%  Tutorial with Michael! Thanks :) 
run_dict = {}

compute_times = np.linspace(0,1,11)

b = phoebe.default_binary()

print( b['period'] )

b['syncpar@primary@component'].set_value(5)

print(b['teff'])

b['teff@secondary'].set_value(1500)
b['teff@primary'].set_value(3300)


b.add_dataset('orb', dataset='orb1', compute_times =compute_times)

b.add_dataset('lc', dataset='lc1', compute_times = compute_times)

b.add_dataset('mesh',dataset='mesh1', compute_times = compute_times)

b['columns'] = ['*@lc1','teffs', 'mus', 'us', 'vs', 'ws']

# set how limb darkening coefficient 
b['ld_mode'].set_value_all('manual') # making it manual 
b['ld_func'].set_value_all('logarithmic') # power law (log) limb darkening model from LUT 

b['ld_mode_bol'].set_value_all('manual') # making it manual 
b['ld_func_bol'].set_value_all('logarithmic') # power law (log) limb darkening model from LUT 
 
# set atmospheres to blackbody
b['atm'].set_value_all('blackbody') 
 
# Need to set passband (default = V (print(b['passband'].choices)) 
b['passband'].set_value('Johnson:K')



b.run_compute()



print(b['times@model@mesh'])

#plot our mesh 
b.plot(kind = 'mesh', show=True, time=float(  b.times[2]) , fc='intensities', ec='intensities') #fc=face colors, ec=edge colors 



# do things manually to get 2d array to input into PMOIRED 
# now make our grid and search for 
u = np.linspace( -5, 5,  51)

v = np.linspace( -3, 3, 31)

grid = np.zeros( [len(u)-1, len(v)-1]  )


t = '00.200000'

# use * so we include both components: secondary, primary
mus = b[f'mus@*@{t}'] #emergent angle (positive if coming towards me)
vs = b[f'vs@*@{t}'] # y coordinate default = solar radius
us = b[f'us@*@{t}'] # x coordinate default = solar radius
intensities = b[f'intensities@*@{t}'] # intensities (W / m3)




for component in ['primary','secondary']:
    #emergent angle, filter for components coming to us:
    filt = mus[component] > 0
    # now getting coordinates and intensities of the given component filtered for mus
    us_filt = us[component].value[filt] 
    vs_filt = vs[component].value[filt]
    intensities_filt = intensities[component].value[filt]
    
    
    
    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):
            filt_u_tmp = ( vs_filt <= v[i+1] ) & ( vs_filt > v[i] ) 
            filt_v_tmp = ( us_filt <= u[j+1] ) & ( us_filt > u[j] ) 
            filt_tmp = filt_u_tmp & filt_v_tmp
            if len(filt_tmp):
                grid[j,i] += np.sum( intensities_filt[filt_tmp] )
            #else:
            #    grid[j,i] = 0
# for x,y, 
plt.figure()
plt.pcolormesh(u,v,grid.T)

plt.figure()
plt.title('logscale')
plt.imshow( np.log10(1e-11*np.max(grid.T ) + grid.T) )

# lightcurve
plt.figure()
plt.plot(b['times@lc1@model'].value, b['fluxes@lc1@model'].value)

b.plot(kind='lc', show=True)



#%% get VLTI observables from grid 

#pmoired.oifake.visImage(grid, scale=0.2, u=[10], v=[50], wl=2, debug=False)


"""


To be understood by PMOIRED, the model needs to be organised in a dictionnary cube containing at least:

image contains the 2D image or the 3D cube (3rd dim is wavelength). Pixels must be square!
X, Y: 2D coordianates of the image X= in mas towards East; Y= in mas towards North
WL: the wavelength table, in microns
scale: the pixel size, in mas

"""


def showCube(cube, imWl0=None, fig=0, imPow=1.0, imMax='100', cmap='inferno', 
             contour=False, lvl=None):
    """
    simple plot for cubes
    imWl0: list of wavelength to show. If integers, index of slices
    imPow: power law for image brightness scaling
    imMax: max cut for image (percentile if a string)
    
    """
    if imWl0 is None:
        imWl0 = [len(cube['WL'])//2]
        
    plt.close(fig)
    plt.figure(fig, figsize=(min(3.5*len(imWl0), 9.5),4.5))
    axs = []
    for i, iwl in enumerate(imWl0):
        if i==0:
            axs.append(plt.subplot(1,len(imWl0),1+i, aspect='equal'))
            plt.ylabel(r'y $\rightarrow$ N (mas)')
        else:
            axs.append(plt.subplot(1,len(imWl0),1+i, aspect='equal', sharex=axs[0], sharey=axs[0]))
        plt.xlabel(r'E $\leftarrow$ x (mas)')

        # -- if actual wavelength, convert to index
        if type(iwl)!=int:
            iwl = np.argmin(np.abs(np.array(cube['WL'])-iwl))
        # -- truncate the brightest part of the image if need be
        if type(imMax)==str:
            vmax=np.percentile(cube['image'][iwl]**imPow, float(imMax))
        elif not imMax is None:
            vmax = imMax
            
        if contour:
            plt.contour(cube['X'], cube['Y'], cube['image'][iwl]**imPow, 
                        lvl*vmax, # relative levels
                        cmap=cmap, vmax=vmax, vmin=0)
        else:
            plt.pcolormesh(cube['X'], cube['Y'], cube['image'][iwl]**imPow, 
                           cmap=cmap, vmax=vmax, vmin=0)
            cb = plt.colorbar(orientation='horizontal')
            Xcb = np.linspace(0,1,5)*vmax
            XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
            XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
            cb.set_ticks(Xcb)
            cb.set_ticklabels(XcbL)
            cb.ax.tick_params(labelsize=6)
                
        plt.title('$\lambda$=%.5f$\mu$m'%cube['WL'][iwl], fontsize=8)
    axs[0].invert_xaxis()
    plt.tight_layout()
    return axs


#plt.imshow( abs( np.fft.fftshift( np.fft.fft2( grid.T ) )) )
wl = np.array( [1.0,1.9,2.0,2.1] )
rad2mas = np.pi/180 * 3600 * 1e3

distance  = 8.251e+7/1000 #au  (1 parsec = 206265au)
cube = {'image':np.array( [grid.T for i in range(len(wl))] ),\
         'X': rad2mas * u / distance ,\
         'Y': rad2mas * v / distance ,\
         'WL':wl  ,\
         'scale':np.diff(rad2mas * u / distance)[0]    
        } 

showCube(cube, imWl0=cube['WL'])

    
# -- list of sidereal time of observations (hours): must be a list or np.ndarray
lst = [0] 

coord = (0, -32)

"""# show image
oi = pmoired.OI()
oi.showModel(model, WL=model['WL'], imFov=8, showSED=False)


# get VLTI observations of this 
# -- create empty OI object
oi = pmoired.OI()
oi.fig=100

wl = np.array([1.9,2.0,2.1])
# -- note that oi.data must be a list! 
oi.data = [pmoired.oifake.makeFakeVLTI(['A0','G1','J3','J2'], coord, lst, wl , model=model),\
           pmoired.oifake.makeFakeVLTI(['A0','G1','J3','J2'], coord, lst, wl, model=model),\
           pmoired.oifake.makeFakeVLTI(['A0','G1','J3','J2'], coord, lst, wl, model=model),
           pmoired.oifake.makeFakeVLTI(['D0','G1','J2','K0'], coord, lst, wl, model=model)]
oi.show()
"""

# -- no noise -> cannot fit as the chi2 will be ~1/0 !!!
noise = 0

# -- simplistic noise model 
noise = {'|V|':0.01, # fractional error on visibility amplitude
         'V2':0.01, # fractional error on visibility squared amplitude
         'T3AMP':0.01, # fractional error on triple product amplitude
         'T3PHI':1.0, # error on triple product phase, in degrees
         'PHI':1.0, # error on phase, in degrees
         'FLUX':0.01 # fractional error on flux
        }

noise = {k:noise[k]/100 for k in noise}

data = [pmoired.oifake.makeFakeVLTI(['A0', 'G1', 'K0', 'J2'], # VLTI telescope configuration (use "U" for UTs)
                                    (0, -24), # Simbad name or sky coordiates as (ra_h, dec_d)
                                    [0], # list of LST for observations
                                    wl, # list of wavelength, in um
                                    cube=cube, # cube dictionnary (see above)
                                    noise=noise, # noise dictionnary (see above)
                                    ),   
       ]

"""
         pmoired.oifake.makeFakeVLTI(['A0', 'B5', 'J2', 'J6'], # VLTI telescope configuration (use "U" for UTs)
                                    (0, -24), # Simbad name or sky coordiates as (ra_h, dec_d)
                                    [0], # list of LST for observations
                                    wl, # list of wavelength, in um
                                    cube=cube, # cube dictionnary (see above)
                                    noise=noise, # noise dictionnary (see above)
                                    #doubleDL=True, # for the extended array, doubling of DL is used
                                  ),     
"""


oi = pmoired.OI() # create an empty PMOIRED OI object
oi.data = data # note that data needs to be a list, even if it has only a single element!
oi.setupFit({'obs':['T3PHI', 'DPHI', '|V|']}) # observable to display

oi.show()

# or we can plot things manually : 
#plt.plot( oi.data[0]['OI_VIS2']['K0J2']['V2'][0] ,'.')


#%% RT pav data 
P_lsp_rtpav = 757 #days
P_p_rtpav = 85

star_ids = ['RT_pav']

path = '/Users/bcourtne/Documents/ANU_PhD2/RT_pav/LSP_vlti_proposal_p109/LSP_light_curves_data/'

light_curves = {}
light_curves[star_ids[0]] = pd.read_csv(path + f'{star_ids[0]}_asassn.csv',header=0,\
                                        )
star ='RT_pav'

times = light_curves[star]['hjd'].values
phases = np.mod( times-50 ,P_lsp_rtpav) /P_lsp_rtpav
fluxes = light_curves[star]['flux (mJy)'].values #- 200 * np.sin(2*np.pi*1/P_p_rtpav * times + np.pi/3 )
sigmas = light_curves[star]['flux err'].values

plt.figure()
plt.plot(phases, fluxes, '.',label='asassn')
plt.ylabel( 'V band flux (mJy)')
plt.xlabel('phase')
plt.legend()

#%% 
"""
Solving an eclipsing binary is a very time-intensive task (both for you as well as your computer). There is no one-size-fits-all recipe to follow, but in general you might find the following workflow useful:

Create a bundle with the appropriate configuration (single star, detached binary, semi-detached, contact, etc).
Attach observational datasets
Flip constraints as necessary to parameterize the system in the way that makes sense for any information you know in advance, types of data, and scientific goals. For example: if you have an SB2 system with RVs, it might make sense to reparameterize to "fit" for asini instead of sma and incl.
Manually set known or approximate values for parameters wherever possible.
Run the appropriate estimators, checking to see if the proposed values make sense before adopting them.
Try to optimize the forward model. See which expensive effects can be disabled without affecting the synthetic model (to some precision tolerance). Make sure to revisit these assumptions as optimizers may move the system to different areas of parameter space where they are no longer valid.
Run optimizers to find (what you hope and assume to be) the global solution. Start with just a few parameters that are most sensitive to the remaining residuals and add more until the residuals are flat (no systematics). Check all read-only constrained parameters to make sure that they make sense, are consistent with known information, and are physical.
Run samplers around the global solution found by the optimizers to explore that local parameter space and the correlations between parameters. Check for convergence before interpreting the resulting posteriors.
For the sake of a simple crude example, we'll just use the synthetic light curve of a default binary with a bit of noise as our "observations". See the inverse problem example scripts for more realistic examples.

"""




# flux units are W/m2 in pheobe , but input is mJy  !! 
#b.add_dataset('lc', times = np.mod( times-50 , 757),\
#              fluxes=fluxes , sigmas = sigmas, \
#                      dataset='lc_rt_pav_asassn')

b = phoebe.default_binary()

print(b.get_value('teff@primary'), b.get_value('teff@secondary'))

#b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,101))
#b.run_compute(irrad_method='none')

#times = b.get_value('times', context='model')
#fluxes = b.get_value('fluxes', context='model') + np.random.normal(size=times.shape) * 0.01
#sigmas = np.ones_like(times) * 0.02

times = light_curves[star]['hjd'].values

#phases = np.mod( times-50 ,P_rtpav) / P_rtpav
fluxes = light_curves[star]['flux (mJy)'].values
sigmas = light_curves[star]['flux err'].values

b = phoebe.default_binary()
b.add_dataset('lc',  times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:V',dataset='rtpav_lc01')

b.set_value('q', 0.8)
b.set_value('teff', component='secondary', value=5000)

b.set_value('period@binary@component', P_lsp_rtpav )
#b.set_value('period', P_rtpav )
b.set_value('dpdt', 0.00*u.d/u.d)

print( b.get_ephemeris() )



b.add_solver('estimator.lc_geometry', lc_datasets='rtpav_lc01')
b.run_solver(kind='lc_geometry', solution='lc_geom_sol')

b.plot(show=True)
#print(b.adopt_solution('lc_geom_sol', trial_run=True))
#b.add_solver('estimator.lc_geometry', solver='my_lcgeom_solver')

#b.run_solver(solver='my_lcgeom_solver', solution='my_lcgeom_solution')
#play below 


#%% Mesh Fields
#!/usr/bin/env python
# coding: utf-8

# 'lc' Datasets and Options
# ============================
# 
# Setup
# -----------------------------



b = phoebe.default_binary()


# Dataset Parameters
# --------------------------
# 
# Let's add a lightcurve dataset to the Bundle (see also the [lc API docs](../api/phoebe.parameters.dataset.lc.md)).  Some parameters are only visible based on the values of other parameters, so we'll pass `check_visible=False` (see the [filter API docs](../api/phoebe.parameters.ParameterSet.filter.md) for more details).  These visibility rules will be explained below.



b.add_dataset('lc')
print(b.get_dataset(kind='lc', check_visible=False))



print(b.get_parameter(qualifier='times'))




print(b.get_parameter(qualifier='fluxes'))



print(b.get_parameter(qualifier='sigmas'))




print(b.get_parameter(qualifier='compute_times'))


print(b.get_parameter(qualifier='compute_phases', context='dataset'))


print(b.get_parameter(qualifier='phases_t0'))


print(b.get_parameter(qualifier='ld_mode', component='primary'))



b.set_value('ld_mode', component='primary', value='lookup')


print(b.get_parameter(qualifier='ld_func', component='primary'))



print(b.get_parameter(qualifier='ld_coeffs_source', component='primary'))



b.set_value('ld_mode', component='primary', value='manual')



print(b.get_parameter(qualifier='ld_coeffs', component='primary'))


print(b.get_parameter(qualifier='passband'))



print(b.get_parameter(qualifier='intens_weighting'))


print(b.get_parameter(qualifier='pblum_mode'))


# ### pblum_component
# 
# `pblum_component` is only available if `pblum_mode` is set to 'component-coupled'.  See the [passband luminosity tutorial](./pblum.ipynb) for more details.


b.set_value('pblum_mode', value='component-coupled')




print(b.get_parameter(qualifier='pblum_component'))


# ### pblum_dataset

# `pblum_dataset` is only available if `pblum_mode` is set to 'dataset-coupled'.  In this case we'll get a warning because there is only one dataset.  See the [passband luminosity tutorial](./pblum.ipynb) for more details.



b.set_value('pblum_mode', value='dataset-coupled')



print(b.get_parameter(qualifier='pblum_dataset'))


# ### pblum
# 
# `pblum` is only available if `pblum_mode` is set to 'decoupled' (in which case there is a `pblum` entry per-star) or 'component-coupled' (in which case there is only an entry for the star chosen by `pblum_component`).  See the [passband luminosity tutorial](./pblum.ipynb) for more details.



b.set_value('pblum_mode', value='decoupled')



print(b.get_parameter(qualifier='pblum', component='primary'))


# ### l3_mode
# 
# See the ["Third" Light tutorial](./l3.ipynb)


print(b.get_parameter(qualifier='l3_mode'))


# ### l3
# 
# `l3` is only avaible if `l3_mode` is set to 'flux'.  See the ["Third" Light tutorial](l3) for more details.



b.set_value('l3_mode', value='flux')




print(b.get_parameter(qualifier='l3'))


# ### l3_frac
# 
# `l3_frac` is only avaible if `l3_mode` is set to 'fraction'.  See the ["Third" Light tutorial](l3) for more details.

b.set_value('l3_mode', value='fraction')



print(b.get_parameter(qualifier='l3_frac'))


# Compute Options
# ------------------
# 
# Let's look at the compute options (for the default PHOEBE 2 backend) that relate to computing fluxes and the LC dataset.
# 
# Other compute options are covered elsewhere:
# * parameters related to dynamics are explained in the section on the [orb dataset](ORB.ipynb)
# * parameters related to meshing, eclipse detection, and subdivision are explained in the section on the [mesh dataset](MESH.ipynb)


print(b.get_compute())


# ### irrad_method


print(b.get_parameter(qualifier='irrad_method'))


# For more details on irradiation, see the [Irradiation tutorial](reflection_heating.ipynb)

# ### boosting_method


print(b.get_parameter(qualifier='boosting_method'))


# For more details on boosting, see the [Beaming and Boosting example script](../examples/beaming_boosting)

#atm


print(b.get_parameter(qualifier='atm', component='primary'))


# For more details on atmospheres, see the [Atmospheres & Passbands tutorial](atm_passbands.ipynb)

# Synthetics
# ------------------



b.set_value('times', phoebe.linspace(0,1,101))




b.run_compute()




print(b.filter(context='model').twigs)

print(b.get_parameter(qualifier='times', kind='lc', context='model'))

print(b.get_parameter(qualifier='fluxes', kind='lc', context='model'))


# Plotting
# ---------------
# 
# By default, LC datasets plot as flux vs time.



afig, mplfig = b.plot(show=True)


# Since these are the only two columns available in the synthetic model, the only other option is to plot in phase instead of time.



afig, mplfig = b.plot(x='phases', show=True)


# In system hierarchies where there may be multiple periods, it is also possible to determine whose period to use for phasing.


print(b.filter(qualifier='period').components)



afig, mplfig = b.plot(x='phases:binary', show=True)


# Mesh Fields
# ---------------------
# 
# By adding a mesh dataset and setting the columns parameter, light-curve (i.e. passband-dependent) per-element quantities can be exposed and plotted.
# 
# Let's add a single mesh at the first time of the light-curve and re-call run_compute


b.add_dataset('mesh', times=[0], dataset='mesh01')



print(b.get_parameter(qualifier='columns').choices)


b.set_value('columns', value=['intensities@lc01', 
                              'abs_intensities@lc01', 
                              'normal_intensities@lc01', 
                              'abs_normal_intensities@lc01', 
                              'pblum_ext@lc01', 
                              'boost_factors@lc01'])



b.run_compute()


print(b.get_model().datasets)


# These new columns are stored with the lc's dataset tag, but with the 'mesh' dataset-kind.



print(b.filter(dataset='lc01', kind='mesh', context='model').twigs)


# Any of these columns are then available to use as edge or facecolors when plotting the mesh (see the section on the [mesh dataset](MESH)).



afig, mplfig = b.filter(kind='mesh').plot(fc='intensities', ec='None', show=True)


# Now let's look at each of the available fields.

# ### pblum
# 
# For more details, see the tutorial on [Passband Luminosities](pblum)



print(b.get_parameter(qualifier='pblum_ext', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `pblum_ext` is the *extrinsic* passband luminosity of the entire star/mesh - this is a single value (unlike most of the parameters in the mesh) and does not have per-element values.

# ### abs_normal_intensities



print(b.get_parameter(qualifier='abs_normal_intensities', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `abs_normal_intensities` are the absolute normal intensities per-element.

# ### normal_intensities



print(b.get_parameter(qualifier='normal_intensities', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `normal_intensities` are the relative normal intensities per-element.

# ### abs_intensities


print(b.get_parameter(qualifier='abs_intensities', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `abs_intensities` are the projected absolute intensities (towards the observer) per-element.

# ### intensities



print(b.get_parameter(qualifier='intensities', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `intensities` are the projected relative intensities (towards the observer) per-element.

# ### boost_factors



print(b.get_parameter(qualifier='boost_factors', 
                      component='primary', 
                      dataset='lc01', 
                      kind='mesh', 
                      context='model'))


# `boost_factors` are the boosting amplitudes per-element.  See the [boosting tutorial](./beaming_boosting.ipynb) for more details.


#%%   Detached Binary: Roche vs Rotstar
#http://phoebe-project.org/docs/2.3/examples/detached_rotstar

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('mesh', compute_times=[0.75], dataset='mesh01')

b['requiv@primary@component'] = 2.

b.run_compute(irrad_method='none', distortion_method='roche', model='rochemodel')

b.run_compute(irrad_method='none', distortion_method='rotstar', model='rotstarmodel')

afig, mplfig = b.plot(model='rochemodel',show=True)

afig, mplfig = b.plot(model='rotstarmodel',show=True)




#%% single star with spot 

# http://phoebe-project.org/docs/2.3/examples/single_spots

logger = phoebe.logger()
b = phoebe.default_star()

b.add_spot(radius=30, colat=80, long=0, relteff=0.9)

print(b['spot'])

times = np.linspace(0, 10, 11)
b.set_value('period', 10)
b.add_dataset('mesh', times=times, columns=['teffs'])


b.run_compute(distortion_method='rotstar', irrad_method='none')

afig, mplfig = b.plot(x='us', y='vs', fc='teffs', 
                      animate=True, save='single_spots_1.gif', save_kwargs={'writer': 'imagemagick'})



#%% Try to build my own mesh 

b = phoebe.default_star()

afig, mplfig = b.plot(x='us', y='vs', fc='teffs', show=True)















#%% 
logger = phoebe.logger()

b = phoebe.default_binary()

b.add_spot(component='secondary', feature='spot02')

b.add_feature('spot', component='primary', feature='spot01')

b.set_value(qualifier='relteff', feature='spot01', value=0.9)
b.set_value(qualifier='radius', feature='spot01', value=30)
b.set_value(qualifier='colat', feature='spot01', value=45)
b.set_value(qualifier='long', feature='spot01', value=90)

b.add_dataset('mesh', times=[0,0.25,0.5,0.75,1.0], columns=['teffs'])

b.run_compute()

#afig, mplfig = b.filter(component='primary', time=0.75).plot(fc='teffs', show=True)

b.set_value(qualifier='syncpar', component='primary', value=1.5)
b.run_compute(irrad_method='none')

print("t0 = {}".format(b.get_value(qualifier='t0', context='system')))

afig, mplfig = b.plot(time=0, y='ws', fc='teffs', ec='None', show=True)

#%% 

star ='RT_pav'
logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('orb', 
              compute_times=phoebe.linspace(0,10,10), 
              dataset='orb01')

b.add_dataset('lc', 
              compute_times=phoebe.linspace(0,1,101),
              dataset='lc01')

b.add_dataset('mesh')

#b.set_value(qualifier='incl', kind='orbit', value=90)
#b.run_compute( context='orb',model='run_with_incl_90') 

b.set_value('compute_times', kind='mesh', value=[10])
b.set_value('include_times', kind='mesh', value=['lc01'])
b.run_compute()

b.set_value('columns', value=['teffs'])

afig, mplfig = b.plot(kind='mesh', time=0.2, fc='teffs', ec='none', show=True)


#%% play with pheobe

star ='RT_pav'
logger = phoebe.logger()

b = phoebe.default_binary()

# flux units are W/m2 in pheobe , but input is mJy  !! 
b.add_dataset('lc', times = np.mod( times-50 , 757),\
              fluxes=fluxes , sigmas = sigmas, \
                      dataset='lc_rt_pav_asassn')

print(b.filter(qualifier='times', dataset='rv01').components)

b.add_c
    
#%% 
# now we need to create a model to fit ! 
#b = phoebe.default_binary()
#b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas_lc)
b.set_value_all('ld_mode', 'manual')
b.set_value_all('ld_mode_bol', 'manual')
b.set_value_all('atm', 'blackbody')
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='default')
_ = b.plot(x='phase', show=True)


    
    
"""b.add_dataset('lc')
print(b.get_dataset(kind='lc', check_visible=False))

b.set_value('times', phoebe.linspace(0,1,101))"""
#%% 

b = phoebe.default_binary()
# set parameter values
b.set_value('q', value = 0.6)
b.set_value('incl', component='binary', value = 84.5)
b.set_value('ecc', 0.2)
b.set_value('per0', 63.7)
b.set_value('requiv', component='primary', value=1.)
b.set_value('requiv', component='secondary', value=0.6)
b.set_value('teff', component='secondary', value=5500.)


# add an lc dataset
b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,101))
b.set_value_all('ld_mode', 'manual')
b.set_value_all('ld_mode_bol', 'manual')
b.set_value_all('atm', 'blackbody')

#%%
#compute the model
b.run_compute(irrad_method='none') #what does this compute?


# extract the arrays from the model that we'll use as observables in the next step
times = b.get_value('times', context='model', dataset='lc01') #b.get_value('times', context='model', dataset='lc01')
# here we're adding noise to the fluxes as well to make the fake data more "realistic"
np.random.seed(0) # to ensure reproducibility with added noise
fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
sigmas_lc = np.ones_like(times) * 0.04

b = phoebe.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas_lc)
b.set_value_all('ld_mode', 'manual')
b.set_value_all('ld_mode_bol', 'manual')
b.set_value_all('atm', 'blackbody')
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='default')
_ = b.plot(x='phase', show=True)
#%% 
"""
primary = Star(teff=5000, mass=1.2)
primary.requiv=1.1
secondary = Star(teff=5000, mass=0.2)
orbit=Orbit(period=P_rtpav , primary=primary, secondary=secondary )
#phoebe.lc(times=np.linspace(1,1000,10),teffA=4000,teffB=200, massA=1.1)"""

#!/usr/bin/env python
# coding: utf-8

# Complete Binary Animation
# ============================
# 
# **NOTE**: animating within Jupyter notebooks can be very resource intensive.  This script will likely run much quicker as a Python script.
# 
# Setup
# -----------------------------

# Let's first make sure we have the latest version of PHOEBE 2.4 installed (uncomment this line if running in an online notebook session such as colab).


#!pip install -I "phoebe>=2.4,<2.5"


# As always, let's do imports and initialize a logger and a new bundle.



logger = phoebe.logger()

b = phoebe.default_binary()



times = np.linspace(0,1,21)



b.add_dataset('lc', times=times, dataset='lc01')


b.add_dataset('rv', times=times, dataset='rv01')



b.add_dataset('mesh', times=times, columns=['visibilities', 'intensities@lc01', 'rvs@rv01'], dataset='mesh01')


# Running Compute
# --------------------


b.run_compute(irrad_method='none')


# Plotting
# -----------
# 
# See the [Animations Tutorial](../tutorials/animations.ipynb) for more examples and details.
# 
# Here we'll create a figure with multiple subplots.  The top row will be the light curve and RV curve.  The bottom three subplots will be various representations of the mesh (intensities, rvs, and visibilities).
# 
# We'll do this by making separate calls to plot, passing the matplotlib subplot location for each axes we want to create.  We can then call `b.show(animate=True)` or `b.save('anim.gif', animate=True)`.



b['lc01@model'].plot(axpos=221)
b['rv01@model'].plot(c={'primary': 'blue', 'secondary': 'red'}, linestyle='solid', axpos=222)
b['mesh@model'].plot(fc='intensities@lc01', ec='None', axpos=425)
b['mesh@model'].plot(fc='rvs@rv01', ec='None', axpos=427)
b['mesh@model'].plot(fc='visibilities', ec='None', y='ws', axpos=224)

#fig = plt.figure(figsize=(11,4))
#afig, mplanim = b.savefig('animation_binary_complete.gif', fig=fig, tight_layouot=True, draw_sidebars=False, animate=True, save_kwargs={'writer': 'imagemagick'})


# ![animation](animation_binary_complete.gif)

# In[ ]:







#%%
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






def light_curve(dt, a, e, P, phi_x, phi_y, phi_z, R1, R2, flux_ratio, plot=False, return_images =False) :
    
    c = a*e #F1 = 0, F2 = 2c
    R1 = 2 #primary radius (au)
    R2 = 1 #secondary radius (au) to be fitted!


    """# simulation parameters 
    dt = 20 #days
    
    # orbital parameters (to do grid search over)
    a= 10 # au
    e = 0.6 # scalar ellipticity 
    c = a*e #F1 = 0, F2 = 2c
    
    P = 400 # period (days)
    R1 = 2 #primary radius (au)
    R2 = 1 #secondary radius (au) to be fitted!
    flux_ratio = 0.1 # flux ratio to be fitted!"""


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
    #j = 0 
    indx = []
    while theta_i[-1] < 2*np.pi:
        obj_fn = [abs(area_cum[j] - area_cum[i] - dA) for j in range(len( theta[:] ) )]
        i = np.argmin( obj_fn ) 
        theta_i.append( theta[i] )
        indx.append(i)
    
    if plot:
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
    if plot:
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
    rot_mat = Rz(phi_z) @ Ry(phi_y) @ Rx(phi_x) 
    pos_proj = rot_mat @  pos_i.T 
    
    if plot:
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
    
    if 0:
        plt.figure()
        plt.plot(np.linspace(0,P,len(intensity)), intensity)
        plt.xlabel('days')
        plt.ylabel('intensity')
        plt.title('SIMULATED LIGHT CURVE')
       
    if not return_images  :
        return(intensity)
    else: 
        return(images)
        

#%% visible 
dt=30
a=5
e=0
P=757
phi_x = 0#np.pi/3
phi_y = np.pi/3 #np.pi/3
phi_z = 0#np.pi/2

R1=1  # from disk fit 

R2=3
flux_ratio=0.05
plot=True


i = light_curve(dt, a, e, P, phi_x, phi_y, phi_z, R1=R1, R2=R2, flux_ratio=flux_ratio, plot=True, return_images  = False)





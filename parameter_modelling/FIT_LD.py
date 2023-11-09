#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:43:49 2023

@author: bcourtne

fit power law limb darkening model

use parameterisation outlined in table 2, Domiciano de Souza et al. 2021 A&A 654, A19

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
from matplotlib import colors
import os


def limbdark_v2_for_plot(rho, theta, alpha):
    
    nu = alpha/2 + 1
    v2 = (nu * special.gamma(nu) * 2**nu * special.jv(nu,  np.pi*rho*theta ) / (np.pi*rho*theta)**nu )**2 
    
    return(v2)

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
                
        #dec = np.deg2rad(h[0].header['DEC'])
        #ha = np.deg2rad( h[0].header['LST']/60/60 )
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


#%% fitting limb darkening model 

def ld_v2(x, params):
    #rho = B/wvl (m/m), theta is angular diamter (radians)
    u,v = x[:,0], x[:,1]
    alpha, theta  = params
    #a, phi  = params
    #theta = ud_wvl  # global parameter defined outside of function 
    rho = np.sqrt( u**2 + v**2 )
    
    nu = alpha/2 + 1
    v2 = (nu * special.gamma(nu) * 2**nu * special.jv(nu,  np.pi*rho*theta ) / (np.pi*rho*theta)**nu )**2 
    
    return(v2)



def log_likelihood(params, x, y, yerr):
    #u,v = x[:,0], x[:,1] #x is list with [u,v]
    model = ld_v2(x,params)
    sigma2 = yerr**2  #+ model**2 * np.exp(2 * log_f)
    return( -0.5 * np.sum((y - model) ** 2 / sigma2 ) )#+ np.log(sigma2)) )



def log_prior(params):
    alpha, theta = params
    #a, phi = params  
    
    # uniformative prior with basic physical limits
    alpha_uniform_cutoff = 0
    alpha_lower_limit = -0.5 # nu = alpha/2 + 1, therefore alpha<-0.5 -> nu < 0
    if (0 < theta < mas2rad(500)) & (alpha_uniform_cutoff < alpha < 1e3) :# nu = alpha/2 + 1, therefore alpha<-0.5 -> nu < 0
        return(0.0)
    else:
        return(-np.inf)
        """#return(-1e9)
        if theta< 0: 
            return(-np.inf)
        elif alpha <= alpha_uniform_cutoff:
            # then make sigmoid function that goes fast to zero at alpha_lower_limit
            # you can plot it to see plt.plot( alpha , 1/(1+np.exp(-1000*(alpha-alpha_lower_limit))) )
            return( 1/(1+np.exp(-1000*(alpha-alpha_lower_limit))) )
        """          



def log_probability(params, x, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        print(params)
        return -np.inf
    else:
        return lp + log_likelihood(params, x, y, yerr)
    
    




plot=True
fig_path = '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/LD_fit/'
ud_fits = pd.read_csv('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/UD_fit.csv',index_col=0)
param_labels=['alpha','theta'] #['a','phi'] #param_labels=['a','phi','theta'] #a, phi,theta = params

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



LD_fit_per_ins = {} # to hold fitting results per instrument photometric band

for ins in ins_vis_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to UD model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_vis_dict[ins]
    
    LD_fit_results = {}
    
    redchi2 = []
    #params = [] #best fit
    LD_result_per_wvl_dict = {xxx:{ 'mean' :[], 'median' : [], 'err' : [] } for xxx in param_labels}
    
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
        for alpha_tmp in np.linspace( -0.5,2,20):
        
            params_tmp=[alpha_tmp, theta0]
            #params_tmp=[a_tmp, phi_tmp]
            y_model_cand = ld_v2(x, params_tmp) 
            
            chi2_tmp = chi2(y_model_cand  , y, yerr)
            if chi2_tmp < best_chi2:
                best_chi2 = chi2_tmp 
                initial = params_tmp
                    
        print(f'best initial parameters = {initial} with chi2={best_chi2}')
        #a0 = 1 #squeeze/stretching (1=circle) - no units
        #phi0 = 0 #rotation (rad)

        #initial = np.array([ a0, phi0, theta0 ])
        ndim = len(initial)
        

        p0 = [initial + np.array([1, theta0/10 ]) * np.random.randn(2)  for i in range(nwalkers)]
        #p0 = [initial + np.array([0.1, np.deg2rad(10)]) * np.random.rand(ndim)  for i in range(nwalkers)]
        

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=( x, y, yerr )
        )
        sampler.run_mcmc(p0, 1000, progress=True);
        
        #samples = sampler.get_chain(flat=True)
        
        #plt.hist(np.log10(samples) ) , bins = np.logspace(-9,-7,100)) #[-1,:,0])
        
        #plt.hist( np.log10( samples ) , bins=np.linspace(-9,-5,100 ))
        
        
        # use sampler.get_autocorr_time()
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        
        
      
        if plot:
            flat_samples4plot = flat_samples.copy()
            flat_samples4plot[:,1] = rad2mas(flat_samples4plot[:,1])

            plt.figure()
            #fig=corner.corner( flat_samples ,labels=['a',r'$\phi$',r'$\theta$'],quantiles=[0.16, 0.5, 0.84],\
            #           show_titles=True, title_kwargs={"fontsize": 12})
            fig=corner.corner( flat_samples4plot ,labels=[r'$\alpha$',r'$\theta$ [mas]'],quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

            fig.gca().annotate(f'{ins} - {round(1e6*wvl,2)}um',xy=(1.0, 1.0),xycoords="figure fraction", xytext=(-20, -10), textcoords="offset points", ha="right", va="top")
            
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            plt.savefig(os.path.join(fig_path,f'LD_mcmc_corner_{ins.split()[0]}_{round(1e6*wvl,2)}um.jpeg'))
            

        
        

        for i,k in enumerate(param_labels):
            
            intermediate_results_dict[wvl][k]={}
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84],axis=0)
            q = np.diff(mcmc)
            
            intermediate_results_dict[wvl][k]['mean'] = np.mean(  flat_samples[:, i] ) 
            intermediate_results_dict[wvl][k]['median'] = mcmc[1] 
            intermediate_results_dict[wvl][k]['err'] = q 
            

            
        #best fit
        best_params_wvl = [intermediate_results_dict[wvl][k]['median'] for k in param_labels] 
        
        #
        y_model = ld_v2(x, best_params_wvl) #disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
        redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df[wvl])-ndim  ))
        
              
        intermediate_results_dict[wvl]['rho'] = x
        intermediate_results_dict[wvl]['v2_obs'] = y
        intermediate_results_dict[wvl]['v2_obs_err'] = yerr
        intermediate_results_dict[wvl]['v2_model'] = y_model
        intermediate_results_dict[wvl]['samplers'] = flat_samples
        
        #reduced chi2 
        #redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df.iloc[wvl_indx])-1))
        
        print('reduced chi2 = {}'.format(chi2(y_model, y, yerr) / (len(v2_df[wvl])-ndim )) )
    
        
    LD_fit_results['redchi2'] = redchi2
    
    LD_fit_results['intermediate_results'] = intermediate_results_dict
    
    LD_fit_per_ins[ins] = LD_fit_results


#%% Plot LD results 
fig1 = plt.figure(1,figsize=(10,12))
fig1.set_tight_layout(True)

frame1 = fig1.add_axes((.1,.6,.8,.4))
frame2 = fig1.add_axes((.1,.2,.8,.4))
#frame3 = fig1.add_axes((.1,.1,.8,.3))
frame4 = fig1.add_axes((.1,.0,.8,.2))

#param_labels=['alpha','theta']


fontsize=20
#for ins, col in zip(LD_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
for ins, col in zip(LD_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
    if 1: #ins!='Matisse (N)':
        wvl_grid = np.array( list( LD_fit_per_ins[ins]['intermediate_results'].keys() ) )
        
        redchi2 = LD_fit_per_ins[ins]['redchi2']
        frame4.semilogy(1e6*wvl_grid, redchi2, '-',lw=2, color=col)
        frame4.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
        frame4.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
        frame4.tick_params(labelsize=fontsize)
        
        for fig , k in zip( [frame1,frame2,frame3], param_labels):
            median = np.array( [LD_fit_per_ins[ins]['intermediate_results'][wvl][k]['median'] for wvl in wvl_grid] )
            err = np.array( [LD_fit_per_ins[ins]['intermediate_results'][wvl][k]['err'] for wvl in wvl_grid] )
            
            #fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
            fig.set_ylabel(k)
            if k=='alpha':
                fig.errorbar(1e6*wvl_grid, median, yerr=np.array(err).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(-2,5)
                fig.tick_params(labelsize=fontsize)
                fig.grid()
                fig.set_ylabel(r'$alpha$ [unitless]\n',fontsize=fontsize)
            if k=='theta':
                fig.errorbar(1e6*wvl_grid, rad2mas(median), yerr=np.array(rad2mas(err)).T, color = col, fmt='-o', lw = 2, label = ins)
                fig.set_ylim(0,200)
                fig.set_yscale('log')
                fig.tick_params(labelsize=fontsize)
                fig.grid()
                fig.set_ylabel(r'$\theta$ [mas]',fontsize=fontsize)

                



frame1.legend(fontsize=fontsize)
frame1.set_title('RT Pav power-law LD Fit vs Wavelength')
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/FIT_LD_logscale.pdf',bbox_inches='tight',dpi=400)




#%% save results in CSV table                                                                                                                                                             

LD_table = {'wvl_list':[],'LD_redchi2':[]}
# add in parameter mean and err                                                                                                                                                         
for _,k in enumerate(param_labels):
    LD_table[f'{k}_mean'] = []
    LD_table[f'{k}_median'] = []
    LD_table[f'{k}_err_16'] = []
    LD_table[f'{k}_err_84'] = []

for ins in LD_fit_per_ins:
    wvls_tmp = list( LD_fit_per_ins[ins]['intermediate_results'].keys() )

    LD_table['wvl_list'].append( wvls_tmp )
    LD_table['LD_redchi2'].append( LD_fit_per_ins[ins]['redchi2']  )
    for i,k in enumerate(param_labels):

        LD_table[f'{k}_mean'].append( [LD_fit_per_ins[ins]['intermediate_results'][w][k]['mean'] for w in wvls_tmp] )
        LD_table[f'{k}_median'].append( [LD_fit_per_ins[ins]['intermediate_results'][w][k]['median'] for w in wvls_tmp] )
        LD_table[f'{k}_err_16'].append( [LD_fit_per_ins[ins]['intermediate_results'][w][k]['err'][0] for w in wvls_tmp]  )
        LD_table[f'{k}_err_84'].append( [LD_fit_per_ins[ins]['intermediate_results'][w][k]['err'][1] for w in wvls_tmp]  )


for k in LD_table:
    LD_table[k] = [item for sublist in LD_table[k] for item in sublist]

LD_table = pd.DataFrame( LD_table )
LD_table = LD_table.set_index('wvl_list')
LD_table.to_csv(os.path.join(fig_path,'LD_fit.csv'))




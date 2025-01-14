
import matplotlib.pyplot as plt
import numpy as np  
from astropy.io import fits
import pmoired
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import os
import glob

from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata 


def plotV2CP( oi ,wvl_band_dict, feature, CP_ylim = 180,  logV2 = True, savefig_folder=None,savefig_name='plots') :
    """ compare observed vs modelled V2 and CP 
    for oifits loaded in a pmoired object and fitted with a parameteric model 
    wavelengths are filtered by the wvl_band_dict
    """
    
    model_col = 'orange'
    obs_col= 'grey'
    fsize=18
    fig_inx = 1 
            
    fig2 = plt.figure(2*fig_inx,figsize=(10,8))
    fig2.set_tight_layout(True)
    
    frame1=fig2.add_axes((.1,.3,.8,.6))
    frame2=fig2.add_axes((.1,.05,.8,.2))  
    
    print( f'plotting all { len( oi._merged) } merged data')
    for i in range(len( oi._merged)):
        
        #=========== for plotting 
        # filter for the wavelengths we are looking at 
        wvl_filt = (oi.data[i]['WL'] >= wvl_band_dict[feature][0]) & (oi.data[i]['WL'] <= wvl_band_dict[feature][1])

        #===========
            
        # V2
        badflag_filt = (~oi._merged[i]['OI_VIS2']['all']['FLAG'].reshape(-1) ) & (oi._model[i]['OI_VIS2']['all']['V2'].reshape(-1)>0) #& ((oi._model[0]['OI_VIS2']['all']['V2']>0).reshape(-1))
        
        wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[i]['OI_VIS2']['all']['FLAG'].shape[0] )] ).reshape(-1)
        
        flag_filt = badflag_filt & wvl_plot_filt
    

        if i == 0: # include legend label
            # data 
            frame1.errorbar(oi._merged[i]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[i]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
        
            # model
            frame1.plot(oi._model[i]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
        else: 
            # data 
            frame1.errorbar(oi._merged[i]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._merged[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt], yerr = oi._merged[i]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt],color=obs_col,alpha=0.9,fmt='.')
        
            # model
            frame1.plot(oi._model[i]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  oi._model[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt],'.', color=model_col)
            
            
        binned_chi2 = (oi._merged[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt]-oi._model[i]['OI_VIS2']['all']['V2'].reshape(-1)[flag_filt])**2 / oi._merged[i]['OI_VIS2']['all']['EV2'].reshape(-1)[flag_filt]**2
        frame2.plot( oi._merged[i]['OI_VIS2']['all']['B/wl'].reshape(-1)[flag_filt],  binned_chi2, '.', color='k' )
        
    #frame1.text(10,0.2,feature,fontsize=15)
    
    if logV2:
        frame1.set_yscale('log')
        
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks( [])
    frame1.set_ylim([0,1])
    frame1.legend(fontsize=fsize)
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    
    #plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
    if savefig_folder is not None:
        plt.savefig( savefig_folder + f'{savefig_name}_V2.png' , bbox_inches='tight', dpi=300)
        
        
    ########
    #CP
    ########

    fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
    fig3.set_tight_layout(True)
    
    frame1=fig3.add_axes((.1,.3,.8,.6))
    frame2=fig3.add_axes((.1,.05,.8,.2))  
    
    
    # data 
    for i in range(len( oi._merged)):    
        badflag_filt = (~oi._merged[i]['OI_T3']['all']['FLAG'].reshape(-1) ) 
        
        wvl_plot_filt = np.array( [wvl_filt for _ in range(oi._merged[i]['OI_T3']['all']['FLAG'].shape[0] )] ).reshape(-1)
        
        flag_filt = badflag_filt & wvl_plot_filt
        
        
        frame1.errorbar(oi._merged[i]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._merged[i]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt], yerr = oi._merged[i]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt],color=obs_col, label='obs',alpha=0.9,fmt='.')
        # model
        frame1.plot(oi._model[i]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt],  oi._model[i]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt],'.',label='model', color=model_col)
        
        binned_chi2 = (oi._merged[i]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt]-oi._model[i]['OI_T3']['all']['T3PHI'].reshape(-1)[flag_filt])**2 / oi._merged[i]['OI_T3']['all']['ET3PHI'].reshape(-1)[flag_filt]**2
        frame2.plot( oi._merged[i]['OI_T3']['all']['Bmax/wl'].reshape(-1)[flag_filt], binned_chi2, '.', color='k')
    
    frame2.axhline(1,color='grey',ls=':')
    
    #frame1.text(10,10,feature,fontsize=15)
    
    #if logV2:
    #    plt.yscale('log')
    frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame1.set_ylim([-CP_ylim, CP_ylim])
    frame2.set_yscale('log')
    frame1.legend(fontsize=fsize)
    frame1.set_xticks( [])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    
    if savefig_folder is not None:
        plt.savefig( savefig_folder + f'{savefig_name}_CP.png' , bbox_inches='tight', dpi=300)
    
    


def sort_baseline_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:] ] ))
    return(n_key)

def sort_triangle_string(B_string): #enforce correct ordering of baseline keys 
    n_key=''.join( sorted( [B_string[:2],B_string[2:4],B_string[4:]  ] ))
    return(n_key)


def enforce_ordered_baselines_keys(data, change_baseline_key_list):
    """_summary_

    Args:
        data (_type_): _description_
        change_baseline_key_list (_type_): list of keys that have baselines in them
    """
    for i in range(len(data)):
        #enforce correct ordering of baseline keys (sometimes we get 'C0D1' and the other is 'D1C0')
        for k in change_baseline_key_list:
            if (k == 'baselines') :
                tmp = [sort_baseline_string(baseline_key ) for baseline_key in data[i][k]]
                data[i][k] = tmp
            else:
                for baseline_key in data[i][k].copy().keys():
                    new_key = sort_baseline_string(baseline_key )
                    data[i][k][new_key] = data[i][k].pop(baseline_key)
           
           
             
def enforce_ordered_triangle_keys(data, change_triangle_key_list):
    """_summary_

    Args:
        data (_type_): _description_
        change_baseline_key_list (_type_): list of keys that have baselines in them
    """
    for i in range(len(data)):
        #enforce correct ordering of baseline keys (sometimes we get 'C0D1' and the other is 'D1C0')
        for k in change_triangle_key_list:
            if (k == 'triangles'):
                tmp = [sort_triangle_string(baseline_key ) for baseline_key in data[i][k]]
                data[i][k] = tmp     
            else:
                for traingle_key in data[i][k].copy().keys():
                    new_key = sort_triangle_string(traingle_key )
                    data[i][k][new_key] = data[i][k].pop( traingle_key )
             
                      
    
def simulate_obs_from_image_reco( obs_files, image_file , binning=None, insname=None):
    # insname='GRAVITY_SC_P1' for gravity data !!!! 
     # change wvl_band_dict[feature] to wvl_lims
    d_model = fits.open( image_file )
    
    img = d_model[0].data

    #assert (abs( float( d_model[0].header['CUNIT2'] ) ) - abs( float(  d_model[0].header['CUNIT1']) ) ) /  float( d_model[0].header['CDELT1'] ) < 0.001
    # we assert the image has to be square..
    #assert abs(float( d_model[0].header['CDELT2'])) == abs(float(d_model[0].header['CDELT1']))

    img_units = d_model[0].header['CUNIT1']

    img_pixscl = d_model[0].header['CDELT1']     
    if img_units == 'deg':
        img_pixscl *= 3600*1e3 # convert to mas
    if img_units == 'mas':
        pass 
    else:  
        raise TypeError('Units not recognized')

    oi = pmoired.OI(obs_files, dMJD=1e9, binning=binning, insname=insname)

    oif = pmoired.OI()

    fake_obs_list = []
    for a in oi.data: 
        
        cube = {}
        cube['scale'] = img_pixscl # mas / pixel
        x = img_pixscl * np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0])  # mas
        y = img_pixscl * np.linspace(-img.shape[0]//2, img.shape[0]//2, img.shape[0])  # mas
        cube['X'] , cube['Y'] =  np.meshgrid(x, y)
        cube['image'] = np.array([ img  for _ in a['WL']] )
        cube['WL'] = a['WL']
        
        if hasattr(a['MJD'], '__len__'):
            mjd = np.median( a['MJD'] ) 
            
        fake_obs_list.append( 
            pmoired.oifake.makeFakeVLTI(\
                t= a['telescopes'],\
                target = ( a['header']['RA']* 24 / 360 , a['header']['DEC'] ),\
                lst = [a['LST']], \
                wl = a['WL'], \
                mjd0 = a[ 'MJD'], #[mjd], #a[ 'MJD'],\
                cube = cube ) 
        )

    # makefake does some operation on MJD so still doesn't match.  
    oif.data = fake_obs_list

    # sort data by MJD - issues with Gravity when x['MJD'] is a list... TO DO 
    
    oi.data = sorted(oi.data, key=lambda x: x['MJD'][0]) # have to take first one because sometimes a list 
    
    oif.data = sorted(oif.data, key=lambda x: x['MJD'][0])
    
    ## SILLY BUG IN PMOIRED WHERE BASELINE/TRIANGLE KEYS ARE INCONSISTENT (e.g. 'D0C1' then 'C1D0')
    ## we fix this here by ordering all relevant keys
    change_baseline_key_list = ['baselines','OI_VIS2','OI_VIS']
    change_triangle_key_list = ['triangles','OI_T3']
    enforce_ordered_baselines_keys(oi.data, change_baseline_key_list)
    enforce_ordered_baselines_keys(oif.data, change_baseline_key_list)
    enforce_ordered_triangle_keys(oi.data, change_triangle_key_list)
    enforce_ordered_triangle_keys(oif.data, change_triangle_key_list)

    return( oi, oif )


def compare_V2_obs_vs_image_reco( oi, oif , return_data = False,  savefig=None, **kwargs ):
    wvl_lims=kwargs.get('wvl_lims',[-np.inf, np.inf])
    model_col = kwargs.get('model_col', 'orange')
    obs_col= kwargs.get('obs_col','grey')
    fsize=kwargs.get('fsize',18)
    logV2=kwargs.get('logV2',True)
    ylim = kwargs.get('ylim', [0,1])
    CP_ylim = kwargs.get('CPylim',180  )
    
    fig_inx = 1 
            
    fig2 = plt.figure(2*fig_inx, figsize=(10,8))
    fig2.set_tight_layout(True)
    
    frame1=fig2.add_axes((.1,.3,.8,.6))
    frame2=fig2.add_axes((.1,.05,.8,.2))  

    if logV2:
        frame1.set_yscale('log')
    else:
        frame1.set_ylim( ylim )
            
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks([])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    
    if return_data:
        return_dict = {
                'flags': {},
                'B/wl_data': {},
                'V2_data': {},
                'V2err_data': {},
                'B/wl_model': {},
                'V2_model': {},
                'flag_filt': {},  
                'chi2': {},  
                'residuals':{}
            }
    
    for i in range(len( oi.data)):
        
        fname = oi.data[i]['filename'].split('/')[-1]
        if return_data:
            for k,_ in return_dict.items():
                return_dict[k][fname]={}
        #=========== for plotting 
        # filter for the wavelengths we are looking at 
        wvl_filt = (oi.data[i]['WL'] >= wvl_lims[0]) & (oi.data[i]['WL'] <= wvl_lims[1])

        #===========
        
        assert set(oif.data[i]['baselines'] ) == set( oi.data[i]['baselines'] )
        
        for cnt, b in enumerate( oi.data[i]['OI_VIS2'].keys() ) :
            
            #assert set( oi.data[i]['OI_VIS2'][b]['B/wl'][0] ) == set( oif.data[i]['OI_VIS2'][b]['B/wl'][0] )
        
            # data 
            flags = oi.data[i]['OI_VIS2'][b]['FLAG'][0]
            B_wl_data = oi.data[i]['OI_VIS2'][b]['B/wl'][0] # usually [[1,2,3,etc]] so take first index
            V2_data = oi.data[i]['OI_VIS2'][b]['V2'][0]
            V2err_data = oi.data[i]['OI_VIS2'][b]['EV2'][0]
            # model (fake observations from image reconstruction)
            B_wl_model = oif.data[i]['OI_VIS2'][b]['B/wl'][0] 
            V2_model = oif.data[i]['OI_VIS2'][b]['V2'][0]
       
            badflag_filt = (~flags.reshape(-1) ) & (V2_data.reshape(-1)>0) #& ((oif.data[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

            flag_filt = badflag_filt & wvl_filt
            
            if not np.any(flag_filt):
                print("No valid data points after filtering!")
                continue

            # V2 
            if (i == 0) & (cnt == 0): # include legend label
                # data 
                frame1.errorbar(B_wl_data[flag_filt], V2_data[flag_filt], yerr = V2err_data[flag_filt], color=obs_col, label='obs',alpha=0.9,fmt='.')
            
                # model
                frame1.plot(B_wl_model[flag_filt],  V2_model[flag_filt],'.',label='model', color=model_col)

            else:
                #try: 
                # data 
                frame1.errorbar(B_wl_data[flag_filt], V2_data[flag_filt], yerr = V2err_data[flag_filt],color=obs_col,alpha=0.9,fmt='.')
                #print( f'{i} {oi.data[i]['filename']} {B_wl_data.shape},{flag_filt.shape}, {V2_data.shape } ')
                # model
                #print( f'{i} {B_wl_model.shape},{flag_filt.shape}, {V2_model.shape} ')
                frame1.plot(B_wl_model[flag_filt],  V2_model[flag_filt],'.', color=model_col)

            residuals = V2_data[flag_filt] - V2_model[flag_filt]
            binned_chi2 = residuals**2 / V2err_data[flag_filt]**2
            frame2.plot( B_wl_data[flag_filt],  binned_chi2, '.', color='k' )

            if return_data:
                return_dict['flags'][fname][b] = flags
                return_dict['flags'][fname][b] = flags
                return_dict['B/wl_data'][fname][b] = B_wl_data
                return_dict['V2_data'][fname][b] = V2_data
                return_dict['V2err_data'][fname][b] = V2err_data
                return_dict['B/wl_model'][fname][b] = B_wl_model
                return_dict['V2_model'][fname][b] = V2_model
                return_dict['flag_filt'][fname][b] = flag_filt
                return_dict['chi2'][fname][b] = binned_chi2
                return_dict['residuals'][fname][b] = residuals
                
    #frame1.text(10,0.2,feature,fontsize=15)
    

    frame1.legend(fontsize=fsize)

    
    #plt.savefig( save_path + f'{ins}_{feature}_pmoired_BESTFIT_V2_PLOT_{ID}.png', bbox_inches='tight', dpi=300)  
      
    if savefig is not None:
        plt.savefig( savefig, bbox_inches='tight', dpi=300)
 
    
    if return_data:   
        return( return_dict )



def compare_CP_obs_vs_image_reco( oi, oif , return_data = False, savefig=None , **kwargs):  
    
    wvl_lims=kwargs.get('wvl_lims',[-np.inf, np.inf])
    model_col = kwargs.get('model_col', 'orange')
    obs_col= kwargs.get('obs_col','grey')
    fsize=kwargs.get('fsize',18)
    CP_ylim = kwargs.get('CPylim',180  )
    
    fig_inx = 1 
    
    fig3 = plt.figure(3 * fig_inx,figsize=(10,8))
    fig3.set_tight_layout(True)
    
    frame1=fig3.add_axes((.1,.3,.8,.6))
    frame2=fig3.add_axes((.1,.05,.8,.2))  
    
        
    if return_data:
        return_dict = {
                'flags': {},
                'Bmax/wl_data': {},
                'CP_data': {},
                'CPerr_data': {},
                'Bmax/wl_model': {},
                'CP_model': {},
                'flag_filt': {},  
                'chi2': {},  
                'residuals':{}
            }
    # data 
    for i in range(len( oi.data)): 
        fname = oi.data[i]['filename'].split('/')[-1]
        if return_data:
            for k,_ in return_dict.items():
                return_dict[k][fname]={}
        #=========== for plotting 
        # filter for the wavelengths we are looking at 
        wvl_filt = (oi.data[i]['WL'] >= wvl_lims[0]) & (oi.data[i]['WL'] <= wvl_lims[1])

        assert set(oif.data[i]['triangles'] ) == set( oi.data[i]['triangles'] )
        
        for cnt, b in enumerate( oi.data[i]['OI_T3'].keys() ) :
            
            #assert set( oi.data[i]['OI_VIS2'][b]['B/wl'][0] ) == set( oif.data[i]['OI_VIS2'][b]['B/wl'][0] )
        
            # data 
            flags = oi.data[i]['OI_T3'][b]['FLAG'][0]
            B_wl_data = oi.data[i]['OI_T3'][b]['Bmax/wl'][0] # usually [[1,2,3,etc]] so take first index
            T3_data = oi.data[i]['OI_T3'][b]['T3PHI'][0]
            T3err_data = oi.data[i]['OI_T3'][b]['ET3PHI'][0]
            # model (fake observations from image reconstruction)
            B_wl_model = oif.data[i]['OI_T3'][b]['Bmax/wl'][0] 
            T3_model = oif.data[i]['OI_T3'][b]['T3PHI'][0]
       
            badflag_filt = (~flags.reshape(-1) )  #& ((oif.data[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

            flag_filt = badflag_filt & wvl_filt
            

                
            if (i == 0) & (cnt == 0): # include legend label
                # data 
                frame1.errorbar(B_wl_data[flag_filt], T3_data[flag_filt], yerr = T3err_data[flag_filt], color=obs_col, label='obs',alpha=0.9,fmt='.')
            
                # model
                frame1.plot(B_wl_model[flag_filt],  T3_model[flag_filt],'.',label='model', color=model_col)
            else: 
                # data 
                frame1.errorbar(B_wl_data[flag_filt], T3_data[flag_filt], yerr = T3err_data[flag_filt],color=obs_col,alpha=0.9,fmt='.')
            
                # model
                frame1.plot(B_wl_data[flag_filt],  T3_model[flag_filt],'.', color=model_col)
                
                
            #binned_chi2 = (T3_data[flag_filt]-T3_model[flag_filt])**2 / T3err_data[flag_filt]**2
            # USING CONVENTION OF CHI2 = (1-cos(theta))^2/sigma^2 - BEING CAREFUL WITH UNITS OF RADIANS
            # Interferometric Imaging Directly with Closure Phases and Closure Amplitudes ( Andrew A. Chael, 2018)
            residuals = 1-np.cos( np.deg2rad( T3_data[flag_filt] - T3_model[flag_filt] ) ) 
            binned_chi2 = residuals**2 / np.deg2rad(T3err_data[flag_filt])**2
            frame2.plot( B_wl_data[flag_filt],  binned_chi2, '.', color='k' )
    
            if return_data:
                return_dict['flags'][fname][b] = flags
                return_dict['Bmax/wl_data'][fname][b] = B_wl_data
                return_dict['CP_data'][fname][b] = T3_data
                return_dict['CPerr_data'][fname][b] = T3err_data
                return_dict['Bmax/wl_model'][fname][b] = B_wl_model
                return_dict['CP_model'][fname][b] = T3_model
                return_dict['flag_filt'][fname][b] = flag_filt
                return_dict['chi2'][fname][b] = binned_chi2 
                return_dict['residuals'][fname][b] = residuals
                
    frame2.axhline(1,color='grey',ls=':')
    
    #frame1.text(10,10,feature,fontsize=15)
    
    
    #if logV2:
    #    plt.yscale('log')
    frame2.set_xlabel(r'$B_{max}/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$CP$ [deg]',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame1.set_ylim([-CP_ylim, CP_ylim])
    frame2.set_yscale('log')
    frame1.legend(fontsize=fsize)
    frame1.set_xticks( [])
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )


    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300)
    
    if return_data:

        return( return_dict )

    


def compare_models( oi, oif ,measure = 'V2', **kwargs):  
    
    wvl_lims = kwargs.get('wvl_lims',[-np.inf, np.inf])
    cp_err_min = kwargs.get('cp_err_min' , 0 ) 
    cp_err_max = kwargs.get( 'cp_err_max', np.inf )
    v2_err_min = kwargs.get( 'v2_err_min', 0 ) 
    v2_err_max = kwargs.get( 'v2_err_max', np.inf )
    cp_min = kwargs.get( 'cp_min', 0 ) 
    cp_max = kwargs.get( 'cp_max', np.inf )
    v2_min = kwargs.get( 'v2_min', 0 ) 
    v2_max = kwargs.get( 'v2_max', np.inf )
    
    if measure=='CP':
        return_dict = {
                'flags': {},
                'Bmax/wl_data': {},
                'CP_data': {},
                'CPerr_data': {},
                'Bmax/wl_model': {},
                'CP_model': {},
                'flag_filt': {},  
                'chi2': {},  
                'residuals':{}
            }
        # data 
        for i in range(len( oi.data)): 
            fname = oi.data[i]['filename'].split('/')[-1]

            for k,_ in return_dict.items():
                return_dict[k][fname]={}
            #=========== for plotting 
            # filter for the wavelengths we are looking at 
            wvl_filt = (oi.data[i]['WL'] >= wvl_lims[0]) & (oi.data[i]['WL'] <= wvl_lims[1])

            assert set(oif.data[i]['triangles'] ) == set( oi.data[i]['triangles'] )
            
            for cnt, b in enumerate( oi.data[i]['OI_T3'].keys() ) :
                
                #assert set( oi.data[i]['OI_VIS2'][b]['B/wl'][0] ) == set( oif.data[i]['OI_VIS2'][b]['B/wl'][0] )
            
                # data 
                flags = oi.data[i]['OI_T3'][b]['FLAG'][0]
                B_wl_data = oi.data[i]['OI_T3'][b]['Bmax/wl'][0] # usually [[1,2,3,etc]] so take first index
                T3_data = oi.data[i]['OI_T3'][b]['T3PHI'][0]
                T3err_data = oi.data[i]['OI_T3'][b]['ET3PHI'][0]
                # setting limits 
                T3err_data[T3err_data > cp_err_max] = cp_err_max
                T3err_data[T3err_data < cp_err_min] = cp_err_min

                # model (fake observations from image reconstruction)
                B_wl_model = oif.data[i]['OI_T3'][b]['Bmax/wl'][0] 
                T3_model = oif.data[i]['OI_T3'][b]['T3PHI'][0]
        
                badflag_filt = (~flags.reshape(-1) )  #& ((oif.data[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

                #cp_min_filt = (cp_err_min <= T3err_data) * (cp_min <= T3_data)
                #cp_max_filt = (cp_err_max >= T3err_data) * (cp_max >= T3_data)

                flag_filt = badflag_filt & wvl_filt #& cp_min_filt & cp_max_filt
                
                # USING CONVENTION OF CHI2 = (1-cos(theta))^2/sigma^2 - BEING CAREFUL WITH UNITS OF RADIANS
                # Interferometric Imaging Directly with Closure Phases and Closure Amplitudes ( Andrew A. Chael, 2018)
                residuals = 1-np.cos( np.deg2rad( T3_data[flag_filt] - T3_model[flag_filt] ) ) 
                binned_chi2 = residuals**2 / np.deg2rad(T3err_data[flag_filt])**2


                return_dict['flags'][fname][b] = flags
                return_dict['Bmax/wl_data'][fname][b] = B_wl_data
                return_dict['CP_data'][fname][b] = T3_data
                return_dict['CPerr_data'][fname][b] = T3err_data
                return_dict['Bmax/wl_model'][fname][b] = B_wl_model
                return_dict['CP_model'][fname][b] = T3_model
                return_dict['flag_filt'][fname][b] = flag_filt
                return_dict['chi2'][fname][b] = binned_chi2 
                return_dict['residuals'][fname][b] = residuals
                    


    elif measure=='V2':
        return_dict = {
                'flags': {},
                'B/wl_data': {},
                'V2_data': {},
                'V2err_data': {},
                'B/wl_model': {},
                'V2_model': {},
                'flag_filt': {},  
                'chi2': {},  
                'residuals':{}
            }

        for i in range(len( oi.data)):
            
            fname = oi.data[i]['filename'].split('/')[-1]

            for k,_ in return_dict.items():
                return_dict[k][fname]={}
            #=========== for plotting 
            # filter for the wavelengths we are looking at 
            wvl_filt = (oi.data[i]['WL'] >= wvl_lims[0]) & (oi.data[i]['WL'] <= wvl_lims[1])

            #===========
            
            assert set(oif.data[i]['baselines'] ) == set( oi.data[i]['baselines'] )
            
            for cnt, b in enumerate( oi.data[i]['OI_VIS2'].keys() ) :
                
                #assert set( oi.data[i]['OI_VIS2'][b]['B/wl'][0] ) == set( oif.data[i]['OI_VIS2'][b]['B/wl'][0] )
            
                # data 
                flags = oi.data[i]['OI_VIS2'][b]['FLAG'][0]
                B_wl_data = oi.data[i]['OI_VIS2'][b]['B/wl'][0] # usually [[1,2,3,etc]] so take first index
                V2_data = oi.data[i]['OI_VIS2'][b]['V2'][0]
                V2err_data = oi.data[i]['OI_VIS2'][b]['EV2'][0]
                # setting limits 
                V2err_data[V2err_data > v2_err_max] = v2_err_max
                V2err_data[V2err_data < v2_err_min ] = v2_err_min 

                # model (fake observations from image reconstruction)
                B_wl_model = oif.data[i]['OI_VIS2'][b]['B/wl'][0] 
                V2_model = oif.data[i]['OI_VIS2'][b]['V2'][0]
        
                badflag_filt = (~flags.reshape(-1) ) & (V2_data.reshape(-1)>0) #& ((oif.data[0]['OI_VIS2']['all']['V2']>0).reshape(-1))

                v2_min_filt = (v2_err_min <= V2err_data) * (v2_min <= V2_data)
                v2_max_filt = (v2_err_max >= V2err_data) * (v2_max >= V2_data)

                flag_filt = badflag_filt & wvl_filt & v2_min_filt & v2_max_filt


                residuals = V2_data[flag_filt] - V2_model[flag_filt]
                binned_chi2 = residuals**2 / V2err_data[flag_filt]**2


                return_dict['flags'][fname][b] = flags
                return_dict['flags'][fname][b] = flags
                return_dict['B/wl_data'][fname][b] = B_wl_data
                return_dict['V2_data'][fname][b] = V2_data
                return_dict['V2err_data'][fname][b] = V2err_data
                return_dict['B/wl_model'][fname][b] = B_wl_model
                return_dict['V2_model'][fname][b] = V2_model
                return_dict['flag_filt'][fname][b] = flag_filt
                return_dict['chi2'][fname][b] = binned_chi2
                return_dict['residuals'][fname][b] = residuals
                
        

    return( return_dict )


    
def plot_image_reconstruction( file , single_plot = False , verbose=True, plot_logscale=False, savefig=None, prior = "Dirac"):
    """
    verified correct coordinates by comparing the  pmoired 
    show model of the non-degenerate parametric model 
    and the reconstructed image of the synthetic observations 
    of the model  (using coordinate_verification.py) 
    see RT Pav journel entry on 18/12/24 in notion.
    """

    h = fits.open( file )

    dirty_beam = h['IMAGE-OI DIRTY BEAM'].data

    dx = h[0].header['CDELT1'] #mas * 3600 * 1e3
    x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])

    dy = h[0].header['CDELT2'] #mas * 3600 * 1e3
    y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])

    origin = 'lower' #'upper' # default - see  

    extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]

    ii = np.fliplr( h[0].data / h[0].data.max() ) # np.fliplr(  h[0].data / h[0].data.max() )


    if single_plot:
        fig = plt.figure( figsize=(8,8) )
        if plot_logscale:
            im = plt.imshow( ii,  cmap='Reds',  extent=extent, origin=origin, norm=LogNorm(vmin=0.01, vmax=1) )
        else:
            im = plt.imshow( ii,  cmap='Reds',  extent=extent, origin=origin)
            
        plt.xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
        plt.ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
        plt.gca().tick_params(labelsize=15)

        plt.text( -x[2], y[2], 'RT Pav', color='k',fontsize=15)
        #plt.text( -x[2], y[4], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
        plt.text( -x[2], y[6], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar( im, cax=cax, orientation='vertical')
        cbar.set_label( 'Normalized flux', rotation=90,fontsize=15)
        cbar.ax.tick_params(labelsize=15)      

    else: # we plot dirty beam next to it in subplot

        ii_dirty =  np.fliplr(  dirty_beam/np.max(dirty_beam) )
        #if (prior == 'Dirac') | (prior == 'Random'): # with MiRA default priors (Dirac or Random) coordinates seem flipped in x
        #    #extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]
        #    ii_dirty =  np.fliplr(  dirty_beam/np.max(dirty_beam) )
        #else:
        #    ii_dirty =  dirty_beam/np.max(dirty_beam) 
        
        fig,ax = plt.subplots( 1,2 , figsize=(12,6) )
        if plot_logscale:
            im = ax[0].imshow( ii,  cmap='Reds',  extent=extent, origin=origin, norm=LogNorm(vmin=0.01, vmax=1) )
            ax[1].imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin , norm=LogNorm(vmin=0.01, vmax=1) )

        else:
            im = ax[0].imshow( ii,  cmap='Reds',  extent=extent, origin=origin )
            ax[1].imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin )

        ax[0].set_ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
        for axx in [ax[0],ax[1]]:
            axx.set_xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
            axx.tick_params(labelsize=15)

        ax[0].text( -x[2], y[int( 0.1*len(y)) ], 'Image Reco.', color='k',fontsize=15)
        ax[0].text( -x[2], y[int( 0.2*len(y))], 'RT Pav', color='k',fontsize=15)
        #ax[0].text( x[2], -y[30], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
        ax[0].text( -x[2], y[int( 0.3*len(y))], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)
        ax[1].text( -x[2], y[int( 0.1*len(y))], 'Dirty beam', color='k',fontsize=15)

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('top', size='5%', pad=0.05)
        cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
        cbar.ax.tick_params(labelsize=15)    

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('top', size='5%', pad=0.05)
        cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
        cbar.ax.tick_params(labelsize=15)     

    if savefig:
        plt.savefig( savefig, dpi=300, bbox_inches='tight')
        
    if verbose:
        plt.show()
    else:
        plt.close()




def fit_a_prior( prior_type, obs_files, fov, pixelsize, save_path, label="some_prior" ,**kwargs):
    
    wavemin = kwargs.get("wavemin", -np.inf ) 
    wavemax = kwargs.get("wavemax", np.inf ) 
    binning= kwargs.get("binning", 1)
    max_rel_V2_error = kwargs.get("max_rel_V2_error", 1)
    max_rel_CP_error = kwargs.get("max_rel_CP_error", 10)
    
    if prior_type == "UD":
        param_grid = kwargs.get( "param_grid", np.logspace( 0,2.5,50) )
        
        oi = pmoired.OI(obs_files, binning = binning)
        bestchi2 = np.inf

        oi.setupFit({'obs':['V2', 'T3PHI'],
                    'min relative error':{'V2':0.0},
                    'max relative error':{'V2':max_rel_V2_error, 'CP':max_rel_CP_error},
                    'wl ranges':[[wavemin, wavemax]]})

        for udtmp in param_grid:
            ud_model = {'*,ud':udtmp}

            oi.doFit(ud_model)
            if oi.bestfit['chi2'] < bestchi2:
                bestchi2 = oi.bestfit['chi2']
                bestud = oi.bestfit['best']['*,ud']

            ud_model = {'*,ud':bestud}
            oi.doFit(ud_model)
    else:
        print( "prior type does not exist.")
        
    
    best_fits = pmoiredModel_2_fits( oi, imFov = fov, imPix = pixelsize, name=f"{prior_type}_prior")
    # write the fits     
    best_fits.writeto( save_path + label + '.fits' , overwrite = True)
    
    dx = best_fits[0].header['CDELT1'] #mas * 3600 * 1e3
    x = np.linspace( -best_fits[0].data.shape[0]//2 * dx , best_fits[0].data.shape[0]//2 * dx,  best_fits[0].data.shape[0])

    dy = best_fits[0].header['CDELT2'] #mas * 3600 * 1e3
    y = np.linspace( -best_fits[0].data.shape[1]//2 * dy , best_fits[0].data.shape[1]//2 * dy,  best_fits[0].data.shape[1])

    origin = 'lower'
    
    # save the image 
    plt.figure()
    plt.imshow( best_fits[0].data , origin = origin, extent = [np.max(x), np.min(x), np.min(y), np.max(y) ] )
    plt.colorbar()
    plt.title("IMAGE RECONSTRUCTION PRIOR\n" + save_path.split('/')[-1] )
    plt.savefig( save_path + label + '.png' ) 
    plt.close()
    
    return( best_fits )


def create_parametric_prior(pmoired_model ,fov, pixelsize, save_path=None, label="some_prior" ):
    """
    outputs a fits file with parametric model prior that can be used as input to MiRA algorithm
    pmoired_model is dictionary holding parameteric model parameter in pmoired format 

    e.g: 
    UD:
    pmoired_model = {'*,ud':9.0}

    offset ellipse:
    pmoired_model = {'*,ud':9.0, '*,x':5 , '*,y':5, '*,incl':70, '*,projang':-45}
    """

    oi = pmoired.OI()
    # wavelength doesn't really matter here since we just want the image
    oi.showModel(pmoired_model, WL=np.linspace( 1, 2, 10), imFov=fov, showSED=False)
    plt.close()
    #plt.savefig('delme2.png')
    oi.bestfit = {}
    oi.bestfit['best'] = pmoired_model

    prior_fits = pmoiredModel_2_fits( oi, imFov =  fov, imPix= pixelsize, name='untitled')

    #prior_file =  'prior_test.fits'
    if save_path is not None:
        prior_fits.writeto( save_path + label + '.fits' , overwrite = True)
        
    return prior_fits 





def pmoiredModel_2_fits( oi, imFov = 200 , imPix= 2, name='untitled'):
    """
    save fits files with heade standard required by OImaging so pmoired images can be uploaded as priors

    Parameters
    ----------
    bestmodel : TYPE
    DESCRIPTION. e.g. input oi.bestfit['best']

    Returns
    -------
    fits file

    """
    _ = pmoired.oimodels.computeLambdaParams( oi.bestfit['best'] ) 

    oi.computeModelImages( imFov=imFov , imPix=imPix)

    im = oi.images['cube'] 

    dx = np.mean( np.diff( oi.images['X'], axis = 1) ) #mas 

    dy = np.mean( np.diff( oi.images['Y'], axis = 0) ) #mas 

    p = fits.PrimaryHDU( im[0] )

    # set headers 
    p.header.set('CRPIX1', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

    p.header.set('CRPIX2', oi.images['cube'][0].shape[0] / 2 ) # Reference pixel 

    p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel 

    p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel  

    #p.header.set('CDELT1', dx * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT1', dx ) # Coord. incr. per pixel 

    #p.header.set('CDELT2', dy * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT2', dy  ) # Coord. incr. per pixel 

    p.header.set('CUNIT1', 'mas'  ) # Physical units for CDELT1 and CRVAL1 

    p.header.set('CUNIT2', 'mas'  ) # Physical units for CDELT1 and CRVAL1 
    
    p.header.set('HDUNAME', name ) # Physical units for CDELT1 and CRVAL1 

    h = fits.HDUList([])
    h.append( p ) 

    """
    #example 
    oi.doFit(best, doNotFit=['*,f','c,ud'] )#,'e,projang','e,incl'])

    h = pmoiredModel_2_fits( oi, imFov = 200 , name='bens_test')

    h.writeto( '/Users/bencb/Downloads/hello_rtpav.fits' ) 

    """

    return( h )



def intensity_2_fits(projected_intensity, dx, dy, name, data_path, header_dict={}, write_file=True):

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print( 'Path created: ', data_path )
    
    p = fits.PrimaryHDU( projected_intensity )

    # set headers 
    p.header.set('CRPIX1', projected_intensity.shape[0] / 2 ) # Reference pixel 

    p.header.set('CRPIX2', projected_intensity.shape[0] / 2 ) # Reference pixel 

    p.header.set('CRVAL1', 0 ) # Coordinate at reference pixel 

    p.header.set('CRVAL2', 0 ) # Coordinate at reference pixel  

    #p.header.set('CDELT1', dx * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT1', dx ) # Coord. incr. per pixel 

    #p.header.set('CDELT2', dy * 1e-3 / 3600 * 180/np.pi ) # Coord. incr. per pixel 
    p.header.set('CDELT2', dy  ) # Coord. incr. per pixel 

    p.header.set('CUNIT1', 'mas'  ) # Physical units for CDELT1 and CRVAL1 

    p.header.set('CUNIT2', 'mas'  ) # Physical units for CDELT1 and CRVAL1 
    
    p.header.set('HDUNAME', name ) # Physical units for CDELT1 and CRVAL1 

    if header_dict:
        for k, v in header_dict.items():
            p.header.set(k, v)

    h = fits.HDUList([])
    h.append( p ) 

    if write_file:
        h.writeto( data_path + name, overwrite=True )

    return h 

def crop_image_to_extent(image, extent, original_extent):
    """
    Crops the image data to match the desired extent.

    Parameters:
        image (2D array): The original image data.
        extent (list): Desired extent [xmin, xmax, ymin, ymax].
        original_extent (list): Original extent of the image [xmin, xmax, ymin, ymax].

    Returns:
        2D array: Cropped image data.
    """
    xmin, xmax, ymin, ymax = extent
    x_min_orig, x_max_orig, y_min_orig, y_max_orig = original_extent

    # Calculate pixel indices for the desired extent
    x_min_idx = int((xmin - x_min_orig) / (x_max_orig - x_min_orig) * (image.shape[1]))
    x_max_idx = int((xmax - x_min_orig) / (x_max_orig - x_min_orig) * (image.shape[1]))
    y_min_idx = int((ymin - y_min_orig) / (y_max_orig - y_min_orig) * (image.shape[0]))
    y_max_idx = int((ymax - y_min_orig) / (y_max_orig - y_min_orig) * (image.shape[0]))

    # Ensure indices are within bounds
    x_min_idx = max(0, min(x_min_idx, image.shape[1]))
    x_max_idx = max(0, min(x_max_idx, image.shape[1]))
    y_min_idx = max(0, min(y_min_idx, image.shape[0]))
    y_max_idx = max(0, min(y_max_idx, image.shape[0]))

    # Crop the image
    cropped_image = image[y_min_idx:y_max_idx, x_min_idx:x_max_idx]

    return cropped_image


def plot_reconstructed_images_with_options(
    file_pattern, instruments, base_dir_template, 
    plot_dirty_beam=True, **kwargs
):
    """
    Plots reconstructed images for a list of instruments with additional customization options.

    Parameters:
        file_pattern (str): The file naming pattern with wildcards for variable parts.
        instruments (list): List of instrument names (e.g., "pionier", "gravity").
        base_dir_template (str): Template for base directory containing "<ins>" placeholder for instruments.
        plot_dirty_beam (bool): If True, includes the dirty beam as a second column in subplots.
        **kwargs: Additional keyword arguments:
            - same_extent (list or list of lists): Extent for plotting [xmin, xmax, ymin, ymax].
            - draw_origin (bool): If True, draws dashed black lines at the x and y origin.
            - plot_logscale (bool): If True, plots in log scale.
            - vmin (float): Minimum value for colorbar.
            - vmax (float): Maximum value for colorbar.

    Returns:
        None
    """
    # Extract kwargs
    same_extent = kwargs.get("same_extent", None)
    draw_origin = kwargs.get("draw_origin", False)
    plot_logscale = kwargs.get("plot_logscale", False)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    num_instruments = len(instruments)
    fig, axes = plt.subplots(num_instruments, 2 if plot_dirty_beam else 1, 
                             figsize=(10, 5 * num_instruments))
    axes = np.atleast_2d(axes)  # Ensure axes is 2D even for one row

    for idx, instrument in enumerate(instruments):
        # Replace <ins> with the instrument name in the base directory template
        base_dir = base_dir_template.replace("<ins>", instrument)

        # Use glob to find the first file matching the pattern
        file_path = glob.glob(os.path.join(base_dir, file_pattern.replace("<ins>", instrument)))
        if not file_path:
            print(f"No matching files found for instrument: {instrument}")
            continue

        file_path = file_path[0]  # Use the first matching file
        print(f"Using file: {file_path}")

        # Open the FITS file
        h = fits.open(file_path)
        
        # Load reconstructed image and dirty beam
        reconstructed_image = np.fliplr(h[0].data / np.max(h[0].data))
        dirty_beam = (
            np.fliplr(h['IMAGE-OI DIRTY BEAM'].data / np.max(h['IMAGE-OI DIRTY BEAM'].data))
            if 'IMAGE-OI DIRTY BEAM' in h else None
        )


        dx = h[0].header['CDELT1']
        dy = h[0].header['CDELT2']
        x = np.linspace(-h[0].data.shape[0]//2 * dx, h[0].data.shape[0]//2 * dx, h[0].data.shape[0])
        y = np.linspace(-h[0].data.shape[1]//2 * dy, h[0].data.shape[1]//2 * dy, h[0].data.shape[1])
        full_extent = [np.max(x), np.min(x), np.min(y), np.max(y)]
        # Retrieve or set extent
        if same_extent:
            if isinstance(same_extent[0], list):  # Per-instrument extent
                extent = same_extent[idx]
            else:  # Same extent for all
                extent = same_extent
        else:  # Default extent from FITS file
            extent =  full_extent


        reconstructed_image = crop_image_to_extent(reconstructed_image, extent, full_extent)
        
        # Plot reconstructed image
        norm = LogNorm(vmin=vmin, vmax=vmax) if plot_logscale else None
        if plot_logscale:
            im = axes[idx, 0].imshow(reconstructed_image, cmap='Reds', extent=extent, origin='lower', norm=norm)
        else:
            im = axes[idx, 0].imshow(reconstructed_image, cmap='Reds', extent=extent, origin='lower', vmin=vmin, vmax = vmax)

        axes[idx, 0].set_title(f"{instrument} - Reconstructed Image", fontsize=15)
        axes[idx, 0].set_xlabel('ΔRA <- E [mas]', fontsize=12)
        axes[idx, 0].set_ylabel('ΔDEC -> N [mas]', fontsize=12)
        fig.colorbar(im, ax=axes[idx, 0], orientation='vertical', fraction=0.046, pad=0.04)

        # Draw origin lines if requested
        if draw_origin:
            axes[idx, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axes[idx, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)

        # Plot dirty beam (if applicable)
        if plot_dirty_beam and dirty_beam is not None:
            dirty_beam = crop_image_to_extent(dirty_beam, extent, full_extent)
            if plot_logscale:
                im_dirty = axes[idx, 1].imshow(dirty_beam, cmap='Reds', extent=extent, origin='lower', norm=norm)
            else:
                im_dirty = axes[idx, 1].imshow(dirty_beam, cmap='Reds', extent=extent, origin='lower', vmin=vmin, vmax=vmax)
            axes[idx, 1].set_title(f"{instrument} - Dirty Beam", fontsize=15)
            axes[idx, 1].set_xlabel('ΔRA <- E [mas]', fontsize=12)
            fig.colorbar(im_dirty, ax=axes[idx, 1], orientation='vertical', fraction=0.046, pad=0.04)

            # Draw origin lines if requested
            if draw_origin:
                axes[idx, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
                axes[idx, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Thermal modes 



# Thermal modes 
def blackbody_intensity(T, wavelength):
    """
    Calculate black body intensity using Planck's law.
    """
    # Constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 3.0e8           # Speed of light (m/s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)

    return (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * k_B * T)) - 1)

def thermal_oscillation(theta, phi, t, T_eff, delta_T_eff, l, m, nu, psi_T):
    """
    Calculate the local effective temperature of a star with a thermal oscillation mode.
    """
    Y_lm = sph_harm(m, l, phi, theta)
    Y_lm_normalized = np.real(Y_lm) / np.max(np.real(Y_lm))
    time_dependent_term = np.cos(2 * np.pi * nu * t + psi_T)
    return T_eff + delta_T_eff * Y_lm_normalized * time_dependent_term

def rotate_to_observer_frame(theta, phi, theta_obs, phi_obs):
    """
    Rotate the stellar coordinates such that the observer's position aligns with the new z-axis.
    
    Parameters:
        theta, phi: Spherical coordinates of the stellar surface.
        theta_obs, phi_obs: Observer's position in spherical coordinates.
    
    Returns:
        theta_rot, phi_rot: Rotated spherical coordinates.
    """
    # Convert observer position to Cartesian coordinates
    x_obs = np.sin(theta_obs) * np.cos(phi_obs)
    y_obs = np.sin(theta_obs) * np.sin(phi_obs)
    z_obs = np.cos(theta_obs)

    # Define rotation: observer's position -> z-axis
    observer_direction = np.array([x_obs, y_obs, z_obs])
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(observer_direction, z_axis)
    rotation_angle = np.arccos(np.dot(observer_direction, z_axis))
    if np.linalg.norm(rotation_axis) > 1e-10:
        rotation_axis /= np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([1, 0, 0])  # Arbitrary axis when already aligned

    # Rotation matrix
    rotation = R.from_rotvec(rotation_angle * rotation_axis)

    # Convert stellar surface to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords = np.stack((x, y, z), axis=-1)

    # Rotate coordinates
    rotated_coords = rotation.apply(coords.reshape(-1, 3)).reshape(coords.shape)

    # Convert back to spherical coordinates
    x_rot, y_rot, z_rot = rotated_coords[..., 0], rotated_coords[..., 1], rotated_coords[..., 2]
    r_rot = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
    theta_rot = np.arccos(np.clip(z_rot / r_rot, -1, 1))
    phi_rot = np.arctan2(y_rot, x_rot)

    return theta_rot, phi_rot

def project_to_observer_plane(theta_rot, phi_rot, intensity, grid_size=500):
    """
    Project the rotated stellar surface onto the observer's 2D image plane.
    """
    # Convert spherical to Cartesian
    x = np.sin(theta_rot) * np.cos(phi_rot)
    y = np.sin(theta_rot) * np.sin(phi_rot)
    z = np.cos(theta_rot)
    
    # Only keep the visible hemisphere (z > 0)
    visible = z > 0
    x_visible = x[visible]
    y_visible = y[visible]
    intensity_visible = intensity[visible]

    # Create the observer plane grid
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    x_plane, y_plane = np.meshgrid(x_grid, y_grid)
    
    # Mask points outside the unit circle
    r_plane = np.sqrt(x_plane**2 + y_plane**2)
    mask = r_plane <= 1

    # Interpolate intensity from spherical to plane
    points = np.vstack((x_visible, y_visible)).T
    projected_intensity = np.zeros_like(x_plane)
    projected_intensity[mask] = griddata(
        points, intensity_visible, (x_plane[mask], y_plane[mask]), method='linear', fill_value=0
    )
    
    return projected_intensity








#Example Usage
# instruments = [
#     "pionier", "gravity", "matisse_L", "matisse_M", "matisse_N_8.0um",
#     "matisse_N_8.5um", "matisse_N_9.0um", "matisse_N_9.5um", "matisse_N_10.0um",
#     "matisse_N_10.5um", "matisse_N_11.0um", "matisse_N_11.5um",
#     "matisse_N_12.0um", "matisse_N_12.5um"
# ]
# base_dir_template = "/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/<ins>"
# file_pattern = "imageReco_<ins>_prior-UD_regul-hyperbolic_pixelscale-*_fov-*_wavemin-*_wavemax-*_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits"

# plot_reconstructed_images_with_options(
#     file_pattern, instruments, base_dir_template, plot_dirty_beam=True,
#     same_extent=[40, -40, -40, 40], draw_origin=True, plot_logscale=True, vmin=0.01, vmax=1
# )

# delete_patterns = [r"<ins>", r"\*", r"\.fits"]
# import re
# fname = file_pattern
# # Replace all patterns with white space
# for pattern in delete_patterns:
#     fname = re.sub(pattern, "", fname )
# for ins in instruments:
#     fname += ins+'_'

# plt.savefig("/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/aCOMBINED_PLOTS/" + f"{fname}.png")










    
# def plot_image_reconstruction_BEFORE_COORDINATE_FIX( file , single_plot = False , verbose=True, plot_logscale=False, savefig=None, prior = "Dirac"):
    
#     h = fits.open( file )

#     dirty_beam = h['IMAGE-OI DIRTY BEAM'].data

#     dx = h[0].header['CDELT1'] #mas * 3600 * 1e3
#     x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])

#     dy = h[0].header['CDELT2'] #mas * 3600 * 1e3
#     y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])

#     origin = 'lower'

#     if (prior == 'Dirac') | (prior == 'Random'): # with MiRA default priors (Dirac or Random) coordinates seem flipped in x
#         #extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]
#         origin = 'lower'
#     else:
#         origin = 'upper'

#     # if (prior == 'Dirac') | (prior == 'Random'): # with MiRA default priors (Dirac or Random) coordinates seem flipped in x
#     #     extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]
#     #     #ii = np.fliplr(  h[0].data /h[0].data.max() )
#     # else:
#     #     extent = [np.max(x), np.min(x), np.max(y), np.min(y) ]
#     #     #ii =  h[0].data /h[0].data.max() 

#     extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]

#     ii = np.fliplr(  h[0].data /h[0].data.max() )

#     #single_plot = False 
#     # the flipping of the image was cross check with pmoired generated images to make sure coordinates were consistent
#     #im = plt.imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )


#     #plt.pcolormesh(x[::-1], y,  h[0].data /h[0].data.max() , cmap='Reds')#, norm=colors.LogNorm(vmin=1e-2, vmax=1))

#     if single_plot:
#         fig = plt.figure( figsize=(8,8) )
#         if plot_logscale:
#             im = plt.imshow( ii,  cmap='Reds',  extent=extent, origin=origin, norm=LogNorm(vmin=0.01, vmax=1) )
#         else:
#             im = plt.imshow( ii,  cmap='Reds',  extent=extent, origin=origin)
            
#         plt.xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
#         plt.ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
#         plt.gca().tick_params(labelsize=15)

#         plt.text( -x[2], y[2], 'RT Pav', color='k',fontsize=15)
#         #plt.text( -x[2], y[4], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
#         plt.text( -x[2], y[6], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)

#         divider = make_axes_locatable(plt.gca())
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         cbar = fig.colorbar( im, cax=cax, orientation='vertical')
#         cbar.set_label( 'Normalized flux', rotation=90,fontsize=15)
#         cbar.ax.tick_params(labelsize=15)      

#     else: # we plot dirty beam next to it in subplot

#         ii_dirty =  np.fliplr(  dirty_beam/np.max(dirty_beam) )
#         #if (prior == 'Dirac') | (prior == 'Random'): # with MiRA default priors (Dirac or Random) coordinates seem flipped in x
#         #    #extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]
#         #    ii_dirty =  np.fliplr(  dirty_beam/np.max(dirty_beam) )
#         #else:
#         #    ii_dirty =  dirty_beam/np.max(dirty_beam) 
        
#         fig,ax = plt.subplots( 1,2 , figsize=(12,6) )
#         if plot_logscale:
#             im = ax[0].imshow( ii,  cmap='Reds',  extent=extent, origin=origin, norm=LogNorm(vmin=0.01, vmax=1) )
#             ax[1].imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin , norm=LogNorm(vmin=0.01, vmax=1) )

#         else:
#             im = ax[0].imshow( ii,  cmap='Reds',  extent=extent, origin=origin )
#             ax[1].imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin )

#         ax[0].set_ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
#         for axx in [ax[0],ax[1]]:
#             axx.set_xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
#             axx.tick_params(labelsize=15)

#         ax[0].text( -x[2], y[int( 0.1*len(y)) ], 'Image Reco.', color='k',fontsize=15)
#         ax[0].text( -x[2], y[int( 0.2*len(y))], 'RT Pav', color='k',fontsize=15)
#         #ax[0].text( x[2], -y[30], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
#         ax[0].text( -x[2], y[int( 0.3*len(y))], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)
#         ax[1].text( -x[2], y[int( 0.1*len(y))], 'Dirty beam', color='k',fontsize=15)

#         divider = make_axes_locatable(ax[0])
#         cax = divider.append_axes('top', size='5%', pad=0.05)
#         cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
#         cax.xaxis.set_ticks_position('top')
#         #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
#         cbar.ax.tick_params(labelsize=15)    

#         divider = make_axes_locatable(ax[1])
#         cax = divider.append_axes('top', size='5%', pad=0.05)
#         cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
#         cax.xaxis.set_ticks_position('top')
#         #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
#         cbar.ax.tick_params(labelsize=15)     

#     if savefig:
#         plt.savefig( savefig, dpi=300, bbox_inches='tight')
        
#     if verbose:
#         plt.show()
#     else:
#         plt.close()


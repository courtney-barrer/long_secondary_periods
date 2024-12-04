
import matplotlib.pyplot as plt
import numpy as np  
from astropy.io import fits
import pmoired
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
      
    if savefig_folder!=None:
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
    
    if savefig_folder!=None:
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
             
                      
    
def simulate_obs_from_image_reco( obs_files, image_file ):
    
     # change wvl_band_dict[feature] to wvl_lims
    d_model = fits.open( image_file )
    
    img = d_model[0].data

    assert d_model[0].header['CUNIT2'] == d_model[0].header['CUNIT1']
    # we assert the image has to be square..
    assert abs(d_model[0].header['CDELT2']) == abs(d_model[0].header['CDELT1'])

    img_units = d_model[0].header['CUNIT1']

    img_pixscl = d_model[0].header['CDELT1']     
    if img_units == 'deg':
        img_pixscl *= 3600*1e3 # convert to mas
    if img_units == 'mas':
        pass 
    else:  
        raise TypeError('Units not recognized')

    oi = pmoired.OI(obs_files, dMJD=1e9)

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
            
        fake_obs_list.append( \
            pmoired.oifake.makeFakeVLTI(\
                t= a['telescopes'],\
                target = ( a['header']['RA']* 24 / 360 , a['header']['DEC'] ),\
                lst = [a['LST']], \
                wl = a['WL'], \
                mjd0 = [mjd], #a[ 'MJD'],\
                cube = cube ) 
        )

    # makefake does some operation on MJD so still doesn't match.  
    oif.data = fake_obs_list

    # sort data by MJD - issues with Gravity when x['MJD'] is a list... TO DO 
    
    oi.data = sorted(oi.data, key=lambda x: x['MJD'][0]) # have to take first one because sometimes a list 
    
    oif.data = sorted(oif.data, key=lambda x: x['MJD'])
    
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
    
    fig_inx = 1 
            
    fig2 = plt.figure(2*fig_inx, figsize=(10,8))
    fig2.set_tight_layout(True)
    
    frame1=fig2.add_axes((.1,.3,.8,.6))
    frame2=fig2.add_axes((.1,.05,.8,.2))  
    
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
            

            # V2 
            if (i == 0) & (cnt == 0): # include legend label
                # data 
                frame1.errorbar(B_wl_data[flag_filt], V2_data[flag_filt], yerr = V2err_data[flag_filt], color=obs_col, label='obs',alpha=0.9,fmt='.')
            
                # model
                frame1.plot(B_wl_model[flag_filt],  V2_model[flag_filt],'.',label='model', color=model_col)
            else: 
                # data 
                frame1.errorbar(B_wl_data[flag_filt], V2_data[flag_filt], yerr = V2err_data[flag_filt],color=obs_col,alpha=0.9,fmt='.')
            
                # model
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
    
    if logV2:
        frame1.set_yscale('log')
        
    frame2.set_xlabel(r'$B/\lambda\ [M rad^{-1}]$',fontsize=fsize)
    frame1.set_ylabel(r'$V^2$',fontsize=fsize)
    frame2.set_ylabel(r'$\chi^2$',fontsize=fsize)
    frame2.set_yscale('log')
    frame1.set_xticks([])
    frame1.set_ylim( ylim )
    frame1.legend(fontsize=fsize)
    frame1.tick_params( labelsize=fsize )
    frame2.tick_params( labelsize=fsize )
    frame2.axhline(1,color='grey',ls=':')
    
    
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
    
    CP_ylim = kwargs.get('CPylim',180  )
    
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

    
    
    
    
def plot_image_reconstruction( file , single_plot = False , verbose=True, savefig=None):
    
    h = fits.open( file )

    dirty_beam = h['IMAGE-OI DIRTY BEAM'].data

    dx = h[0].header['CDELT1'] #mas * 3600 * 1e3
    x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])

    dy = h[0].header['CDELT2'] #mas * 3600 * 1e3
    y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])

    origin = 'lower'
    extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]

    single_plot = False 
    # the flipping of the image was cross check with pmoired generated images to make sure coordinates were consistent
    #im = plt.imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )


    #plt.pcolormesh(x[::-1], y,  h[0].data /h[0].data.max() , cmap='Reds')#, norm=colors.LogNorm(vmin=1e-2, vmax=1))

    if single_plot:
        fig = plt.figure( figsize=(8,8) )
        im = plt.imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )

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
        fig,ax = plt.subplots( 1,2 , figsize=(12,6) )
        im = ax[0].imshow( np.fliplr(  h[0].data /h[0].data.max() ),  cmap='Reds',  extent=extent, origin=origin )
        ax[1].imshow( np.fliplr(  dirty_beam/np.max(dirty_beam) ),  cmap='Reds',  extent=extent, origin=origin )

        ax[0].set_ylabel('$\Delta$ DEC -> N [mas]',fontsize=15)
        for axx in [ax[0],ax[1]]:
            axx.set_xlabel('$\Delta$ RA <- E [mas]',fontsize=15)
            axx.tick_params(labelsize=15)

        ax[0].text( x[2], -y[int( 0.1*len(y)) ], 'Image Reco.', color='k',fontsize=15)
        ax[0].text( x[2], -y[int( 0.2*len(y))], 'RT Pav', color='k',fontsize=15)
        #ax[0].text( x[2], -y[30], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
        ax[0].text( x[2], -y[int( 0.3*len(y))], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)
        ax[1].text( x[2], -y[int( 0.1*len(y))], 'Dirty beam', color='k',fontsize=15)

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
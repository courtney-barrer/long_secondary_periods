
import matplotlib.pyplot as plt
import numpy as np  
import pmoired

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
    
    
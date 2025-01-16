#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 07:56:59 2023

@author: bcourtne


fit UD diameters for pionier, gravity and matisse data 

-for each instrument 
make a wvl grid 
get V2 V2err and interpolate onto grid 
get unique baselines 
put in dataframe 

-merge all dataframes to one master 

- fit UD models per wavelength,

To Do 
===
- new scripts for Fitting Unresolved Binary (2 parameters) 


Fitting MC simulation of resolved binary (set R* at UD and fit F, P, R2) 





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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
    
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

    

def baseline2uv_matrix( h, d):
    """
    

    Parameters
    ----------
    Not used wvl : TYPE  need to divide by this later
        DESCRIPTION. wavelength (m)
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
    
    mat =  np.array([[np.sin(h), np.cos(h), 0],\
           [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],\
           [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d) ]] )
        
    return(mat)   




def B_L(wvl,T): #Spectral density as function of temperature
    #wvl is wavelength vector to calculate spectral density at
    #T is temperature of the black body 
    c = 3e8 #speed of light m/s
    h = 6.63e-34 # plank's constant J.s
    kB = 1.38e-23 #boltzman constant m2 kg /s^2/K
    Bb = 2*h*c**2/wvl**5 * 1 / (np.exp(h*c/(wvl*kB*T)) - 1)
    return(Bb)



    
def chi2(y_model,y_true,yerr):
    return(sum( (y_model-y_true)**2/yerr**2 ) )

    
def rad2mas(rad):
    return(rad*180/np.pi * 3600 * 1e3)

def mas2rad(mas):
    return(mas*np.pi/180 / 3600 / 1e3)




def fit_prep_v2(files, EXTVER=None,flip=True):    
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
        B = [] # to holdprojected baseline !
        for Bx,By in zip( h['OI_VIS2'].data['UCOORD'],h['OI_VIS2'].data['VCOORD'] ): # U=east-west , V=north-sout
            #lambda_u, lambda_v, _ = baseline2uv_matrix(ha, dec) @ np.array( [Bx,By,0] ) # lambda_u has to be multiplied by lambda to get u±!!!
            #B.append( (lambda_u, lambda_v) ) # projected baseline !
            B.append( (Bx, By) )
            
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




# Example usage (requires DataFrame inputs):
# plot_visibility_errorbars(df_vis, df_vis_err, x_axis="B/lambda", df_flags=df_flags, tick_labelsize=10, label_fontsize=14, title_fontsize=16, grid_on=True)
def plot_visibility_errorbars(df_vis, df_vis_err, x_axis="B/lambda", df_flags=None, show_colorbar=True, **kwargs):
    """
    Plot squared visibility with error bars, encoding baseline, wavelength, or B/\lambda in point colors.

    Parameters:
    df_vis: pd.DataFrame
        DataFrame of squared visibilities indexed by (Bx, By), columns are wavelengths.
    df_vis_err: pd.DataFrame
        DataFrame of squared visibility errors indexed by (Bx, By), columns are wavelengths.
    x_axis: str
        Either "baseline", "wavelength", or "B/lambda" to determine the x-axis.
    df_flags: pd.DataFrame, optional
        DataFrame of boolean flags with the same shape as df_vis, indicating valid data points.
    show_colorbar: bool
        Whether to display the colorbar for the plot.
    **kwargs: dict
        Additional keyword arguments for customizing the plot, such as:
        - tick_labelsize: int, size of tick labels
        - label_fontsize: int, size of axis labels
        - title_fontsize: int, size of title
        - grid_on: bool, whether to show grid
        - ylim: list, y-axis limits (default: [0, 1])
        - xlim: list, x-axis limits (default: None, no manual limit applied)
        - xlabel: str, custom x-axis label
        - ylabel: str, custom y-axis label
        - cbar_label: str, custom colorbar label
        - wavelength_bins: list or int, optional bins to average the observable squared visibility

    Returns:
    None
    """
    if x_axis not in ["baseline", "wavelength", "B/lambda"]:
        raise ValueError("x_axis must be either 'baseline', 'wavelength', or 'B/lambda'")

    # Compute baseline lengths
    Bx = np.array([d[0] for d in df_vis.index])
    By = np.array([d[1] for d in df_vis.index])
    baselines = np.sqrt(Bx**2 + By**2)

    # Bin wavelengths if specified
    wavelength_bins = kwargs.get("wavelength_bins")
    if wavelength_bins is not None:
        if isinstance(wavelength_bins, int):
            # Divide into N bins
            wavelengths = df_vis.columns.astype(float)
            bins = np.linspace(wavelengths.min(), wavelengths.max(), wavelength_bins + 1)
        else:
            # Use specified bins
            bins = wavelength_bins

        print(f"Generated bins: {bins}")
        print(f"Wavelength range: {df_vis.columns.min()} to {df_vis.columns.max()}")

        binned_vis = []
        binned_err = []
        binned_flags = []
        binned_wavelengths = []
        for i in range(len(bins) - 1):
            mask = (df_vis.columns.astype(float) >= bins[i]) & (df_vis.columns.astype(float) < bins[i + 1])
            print(f"Bin range: {bins[i]} to {bins[i + 1]}")
            print(f"Mask: {mask}")
            print(f"Selected wavelengths: {df_vis.columns[mask]}")

            if mask.any():
                selected_data = df_vis.loc[:, mask]
                selected_err = df_vis_err.loc[:, mask]
                selected_flags = df_flags.loc[:, mask] if df_flags is not None else None

                print(f"Data for bin {i}:{selected_data}")
                print(f"Mean visibility for bin {i}: {selected_data.mean(axis=1)}")

                binned_vis.append(selected_data.mean(axis=1))
                binned_err.append(selected_err.mean(axis=1))
                if selected_flags is not None:
                    binned_flags.append(selected_flags.any(axis=1))
                binned_wavelengths.append((bins[i] + bins[i + 1]) / 2)

        df_vis = pd.concat(binned_vis, axis=1)
        df_vis_err = pd.concat(binned_err, axis=1)
        df_vis.columns = binned_wavelengths
        df_vis_err.columns = binned_wavelengths

        if df_flags is not None:
            df_flags = pd.concat(binned_flags, axis=1)
            df_flags.columns = binned_wavelengths

        print(f"Binned DataFrame:\n{df_vis}")
        print(f"Binned Errors:\n{df_vis_err}")
        if df_flags is not None:
            print(f"Binned Flags:\n{df_flags}")

    # Prepare colormap for encoding
    if x_axis == "baseline":
        color_values = df_vis.columns.astype(float)  # wavelengths
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = cm.coolwarm
    elif x_axis == "wavelength":
        color_values = baselines
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = cm.viridis
    else:  # x_axis == "B/lambda"
        color_values = df_vis.columns.astype(float)  # wavelengths
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = cm.coolwarm

    # Filter data if flags are provided
    if df_flags is not None:
        df_vis = df_vis.where(~df_flags)
        df_vis_err = df_vis_err.where(~df_flags)

    # Apply error filtering
    max_err = kwargs.get("max_err", None)
    min_err = kwargs.get("min_err", None)
    if max_err is not None:
        df_vis_err[df_vis_err > max_err] = float(max_err)
    if min_err is not None:
        df_vis_err[df_vis_err < min_err] = float(min_err)

    # Plot
    fig, ax = plt.subplots()
    for i, wavelength in enumerate(df_vis.columns):
        if x_axis == "baseline":
            x_values = baselines
            y_values = df_vis.iloc[:, i]
            y_err = df_vis_err.iloc[:, i]
            color = cmap(norm(wavelength))
        elif x_axis == "wavelength":
            x_values = np.full_like(baselines, wavelength, dtype=float)
            y_values = df_vis.iloc[:, i]
            y_err = df_vis_err.iloc[:, i]
            color = cmap(norm(baselines))
        else:  # x_axis == "B/lambda"
            x_values = baselines / wavelength
            y_values = df_vis.iloc[:, i]
            y_err = df_vis_err.iloc[:, i]
            color = cmap(norm(wavelength))

        valid_mask = ~np.isnan(y_values)
        ax.errorbar(x_values[valid_mask], y_values[valid_mask], yerr=y_err[valid_mask], fmt='o', color=color, alpha=0.7)

    # Set labels and title
    label_fontsize = kwargs.get("label_fontsize", 12)
    title_fontsize = kwargs.get("title_fontsize", 14)

    xlabel = kwargs.get("xlabel", "Baseline Length (m)" if x_axis == "baseline" else ("Wavelength (m)" if x_axis == "wavelength" else "B/\u03bb (m^{-1})"))
    ylabel = kwargs.get("ylabel", "Squared Visibility")
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.set_title(kwargs.get("title", "Squared Visibility vs {}".format(
        "Baseline" if x_axis == "baseline" else ("Wavelength" if x_axis == "wavelength" else "B/\u03bb")
    )), fontsize=title_fontsize)

    # Customize tick label size
    tick_labelsize = kwargs.get("tick_labelsize", 10)
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Set axis limits
    if kwargs.get("xlim") is not None:
        ax.set_xlim(kwargs["xlim"])
    if kwargs.get("ylim") is not None:
        ax.set_ylim(kwargs["ylim"])
    else:
        ax.set_ylim([0, 1])

    # Add colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar_label = kwargs.get("cbar_label", "Wavelength (m)" if x_axis == "baseline" else ("Baseline Length (m)" if x_axis == "wavelength" else "Wavelength (m)"))
        cbar.set_label(cbar_label, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_labelsize)

    # Add grid if specified
    if kwargs.get("grid_on", True):
        ax.grid(True)

    plt.show()
    
    
    
path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU'

pionier_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits' ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')


gravity_files = glob.glob(path_dict[comp_loc]['data'] + 'gravity/data/*.fits')
#glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')

matisse_files_L = glob.glob(path_dict[comp_loc]['data'] + 'matisse/reduced_calibrated_data_1/all_chopped_L/*fits' ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob(path_dict[comp_loc]['data'] + "matisse/reduced_calibrated_data_1/all_merged_N_swapped_CP_sign/*fits" ) #glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]


pion_v2_df , pion_v2err_df  , pion_flag_df,  pion_obs_df = fit_prep_v2(pionier_files)

grav_p1_v2_df , grav_p1_v2err_df, grav_p1_flag_df , grav_p1_obs_df= fit_prep_v2(gravity_files, EXTVER = 11 )
grav_p2_v2_df , grav_p2_v2err_df , grav_p2_flag_df , grav_p2_obs_df = fit_prep_v2(gravity_files, EXTVER = 12 )

mati_L_v2_df , mati_L_v2err_df , mati_L_flag_df, mati_L_obs_df = fit_prep_v2(matisse_files_L )
mati_N_v2_df , mati_N_v2err_df , mati_N_flag_df, mati_N_obs_df = fit_prep_v2(matisse_files_N )


#%% V2 summary plot 


kwargs = {
    "tick_labelsize": 8,               # Font size for tick labels
    "label_fontsize": 12,             # Font size for axis labels
    "title_fontsize": 14,             # Font size for the plot title
    "grid_on": True,                  # Display grid
    "ylim": [0, 1],                   # Y-axis limits
    "xlabel": "Custom X-axis Label",  # Custom label for the X-axis
    "ylabel": "Custom Y-axis Label",  # Custom label for the Y-axis
    "cbar_label": "Custom Colorbar Label",  # Custom label for the colorbar
    "title": "Custom Title Label",  # Custom label for the colorbar
    "wavelength_bins": 5,             # Number of bins to average over wavelengths
    "max_err": 0.2,                   # Maximum error value to display
    "min_err": None                   # Minimum error value to display
}


def wavelength_filter(df, min_wl, max_wl):
    filt = df.columns[(df.columns.astype(float) > min_wl) & (df.columns.astype(float) < max_wl)]
    return filt
    

#plot_visibility_errorbars(grav_p1_v2_df, grav_p1_v2err_df,grav_p1_flag_df x_axis="baseline", show_colorbar=True)
plot_visibility_errorbars(grav_p1_v2_df[wfilt], grav_p1_v2err_df[wfilt], x_axis="B/lambda", df_flags=grav_p1_flag_df[wfilt], show_colorbar=True)

wfilt = wavelength_filter(df=mati_L_v2_df, min_wl=3e-6, max_wl=3.5e-6)
plot_visibility_errorbars(mati_L_v2_df[wfilt], mati_L_v2err_df[wfilt], x_axis="B/lambda", df_flags=mati_L_flag_df[wfilt], show_colorbar=True,**kwargs)

wfilt = wavelength_filter(df=mati_L_v2_df, min_wl=4.0e-6, max_wl=4.6e-6)
plot_visibility_errorbars(mati_L_v2_df[wfilt], mati_L_v2err_df[wfilt], x_axis="B/lambda", df_flags=mati_L_flag_df[wfilt], show_colorbar=True,**kwargs)


# Example usage (requires DataFrame inputs):
# plot_visibility_errorbars(df_vis, df_vis_err, x_axis="B/lambda", df_flags=df_flags, tick_labelsize=10, label_fontsize=14, title_fontsize=16, grid_on=True)






#%% V2 UV  summary plot 
v2_df_list = [pion_v2_df.copy() , grav_p1_v2_df.copy(), mati_L_v2_df.copy(), mati_N_v2_df.copy()]
v2err_df_list = [pion_v2err_df.copy() , grav_p1_v2err_df.copy(), mati_L_v2err_df.copy(), mati_N_v2err_df.copy()]
flag_df_list =  [pion_flag_df.copy() , grav_p1_flag_df.copy(), mati_L_flag_df.copy(), mati_N_flag_df.copy()]


fig,ax = plt.subplots(1,len(v2_df_list),figsize=(32,8),sharex=True,sharey=True)
plt.subplots_adjust(wspace=0)

ins_title = ['Pionier (H Band)','Gravity (K Band)', 'Matisse (L & M Band)','Matisse (N Band)']

                    
for i, (df, df_mask, df_err) in enumerate(zip( v2_df_list , flag_df_list,  v2err_df_list)):
#df = mati_L_v2_df.copy() #grav_p2_v2_df.copy()  #mati_L_v2_df.copy() #mati_N_v2_df.copy()
#df_mask = mat_L_flag_df.copy()
    
    # baselines between 0-1 with errorbars less than 0.5 v^2
    vis_filt =  ((df<=1) & (df>=0) & (abs( df_err )<0.5)).values
    
    # projected baselines > 1m
    B_filt = np.array( [[ (x[0]**2+x[1]**2)**0.5 > 1 for x in df_mask.index] for _ in range(df_mask.shape[1])]).T
    
    # data quality mask
    df_mask = df_mask * ~vis_filt * ~B_filt

    Bx = np.array( [bx[0] for bx in df.index] )  #projected east-west component 
    By = np.array( [by[1] for by in df.index] )  #projected north-south compone
    for wvl in df.columns: #wavelengths (m)
        filt = ~df_mask[wvl]
        fff = ax[i].scatter( Bx[filt]/wvl, By[filt]/wvl, c = df[wvl].values[filt] ,norm = colors.Normalize(vmin=0, vmax=1), cmap = plt.cm.coolwarm, alpha=0.9)#,
        #Because brightness is real, each observation provides us a second point, where: V(-u,-v) = V*(u,v) !!!
        ax[i].scatter( - Bx[filt]/wvl, -By[filt]/wvl, c = df[wvl].values[filt] ,norm = colors.Normalize(vmin=0, vmax=1), cmap = plt.cm.coolwarm, alpha=0.9)

    ax[i].set_xlabel(r'u=$B_u^{proj}/\lambda \ [rad^{-1}]$',fontsize=20)
    #ax[i].set_ylabel(r'B_x^{proj}/\lambda N->S (rad$^{-1})$',fontsize=20)
    ax[i].tick_params(labelsize=20)
    ax[i].grid()
    ax[i].set_title( ins_title[i] , fontsize=25)
ax[0].set_ylabel(r'v=$B_v^{proj}/\lambda \ [rad^{-1}]$',fontsize=20)   

divider = make_axes_locatable(ax[-1])
cax = divider.append_axes('right', size='5%', pad=0.2)
cbar = fig.colorbar(fff, cax=cax, orientation='vertical')
cbar.set_label( '       '+r'$|V|^2$', rotation=0,fontsize=25)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
#plt.savefig( '/Users/bcourtne/Documents/ANU_PHD2/RT_pav/RT_pav_uv_vs_v2.jpeg',dpi=300)


#%% CP summary plot 



#sig_id=[]
files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
"""
wvl.shape
Out[321]: (118,)

len(files)
Out[322]: 17

118*17
Out[323]: 2006
"""
def prepare_CP(files):
    
    cp_sigma = {} #all cp sigmas away from zero 
    cp_values = {} # all cp
    cp_mask = {} #bad data flag 

    for f in files :    
        h = oifits.open(f)
                
        indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
        
        current_config = ''.join(list(np.sort(h['OI_ARRAY'].data['STA_NAME'])))
        
        #effective wavelength
        wvl = h['OI_WAVELENGTH'].data['EFF_WAVE']
        
        cp = h['OI_T3'].data['T3PHI'][:,:]
        
        cp_err = h['OI_T3'].data['T3PHIERR'][:,:]
        
        cp_flag = h['OI_T3'].data['FLAG']
        
        triangles = [[indx2station[h['OI_T3'].data['STA_INDEX'][tri][tel]] for tel in range(3)] for tri in range(4)]
    
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


    return( cp_values, cp_sigma , cp_mask)





files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')

cp_values, cp_sigma , cp_mask = prepare_CP(files)

# flattening out 
cp_flat =np.array([])
cpZ_flat =np.array([])
wvl_flat =np.array([])
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

ax.set_ylabel(r'Z-score (CP=0)',fontsize=14)
ax.set_xlabel(r'wavelength [$\mu$m]',fontsize=14)
ax_histy.set_xlabel('counts',fontsize=14)
ax_histy.set_xscale('log')
ax.grid()
ax_histy.grid()


#%%
for w in cp_values:
    print(w)
    #unique_ids
    cp_tmp = np.array( cp_values[w] ).ravel()
    scatter_hist(w * np.ones(len(cp_tmp )) , cp_tmp  , ax,  ax_histy)
    



#%% L/M Band 
L_lims = [3.0e-6, 4.1e-6]
M_lims = [4.5e-6, 5e-6]
N_lims = [8e-6, 13e-6]

files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
#files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')

wvl_ranges = [L_lims,M_lims]
#wvl_ranges = [N_lims]

ylim_plot = [-50,50]
swap_wvl_order = False

cp_sigma = [] #all cp sigmas away from zero 
cp_values = [] # all cp
sig_wvl=[]
sig_id=[]
for f in files :    
    h = oifits.open(f)
            
    current_config = ''.join(list(np.sort(h['OI_ARRAY'].data['STA_NAME'])))
    
    #effective wavelength
    wvl = h['OI_WAVELENGTH'].data['EFF_WAVE']
    if swap_wvl_order:  # WHY DO WE DO THIS??
        wvl = wvl[::-1]
    #initialize wavelength filter
    filt = np.zeros(len(wvl)) == 1    
    for w_lims in wvl_ranges:
        filt = filt | ( ( wvl<w_lims[1] ) & ( wvl>w_lims[0]) )
    
    
    cp = h['OI_T3'].data['T3PHI'][:,:]
    
    cp_err = h['OI_T3'].data['T3PHIERR'][:,:]
    
    cp_flag = h['OI_T3'].data['FLAG']
    
    
    sigma_from_zero = cp / cp_err
    
    fig,ax = plt.subplots(4,1,sharex=True,figsize=(15,8))
    indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
    
    for bT in range(len(cp)):
        lab = [indx2station[h['OI_T3'].data['STA_INDEX'][bT][XXX]] for XXX in range(3)] 
        ax[bT].errorbar(wvl,cp[bT],yerr=cp_err[bT],linestyle=':',label= f'{lab[0]}-{lab[1]}-{lab[2]}');
        ax[bT].axvline(L_lims[0]); ax[bT].axvline(L_lims[1]); ax[bT].fill_betweenx([-300,300],L_lims[0],L_lims[1],alpha=0.5, color='y')
        ax[bT].axvline(M_lims[0]); ax[bT].axvline(M_lims[1]); ax[bT].fill_betweenx([-300,300],M_lims[0],M_lims[1],alpha=0.5, color='orange')
        
        ax[bT].axhline(0,color='k');ax[bT].axhline(180,color='k');ax[bT].axhline(-180,color='k')
        ax[bT].legend(loc='upper right',fontsize=12)
        ax[bT].set_ylabel('CP [deg]',fontsize=15)
        
        ax[bT].set_ylim(ylim_plot )
        
        # CP > 5 AND SIGMA > 1 AWAY FROM ZERO
        new_filt = filt #& (abs(cp[bT])>5) & (abs(sigma_from_zero[bT])>1)
        
        cp_tmp = cp[bT][new_filt]
        
        chi_tmp =  sigma_from_zero[bT][new_filt]  
        
        wvl_tmp =  wvl[new_filt]
        id_tmp = ['{}'.format(f.split('/')[-1]) + f'--{lab[0]}-{lab[1]}-{lab[2]}' for i in range(sum( new_filt ))]
                                   
        if len( chi_tmp ) > 0:
            
            cp_sigma.append( list(chi_tmp) )
            cp_values.append( cp_tmp  )
            sig_wvl.append( list(wvl_tmp ) )
            sig_id.append( id_tmp )
            
    ax[-1].set_xlabel(f'wavelength [$m$]',fontsize=15)
    ax[0].set_title('{}'.format(f.split('/')[-1]),fontsize=15)

plt.figure()    
#unique_ids = np.unique( [item for sublist in sig_id for item in sublist] ) 
for i in range(len(sig_wvl)):
    #unique_ids
    plt.plot(sig_wvl[i],cp_sigma[i],'.', label='sig_id',color='k',alpha=0.4)
plt.xlabel(r'wavelength [m]',fontsize=15)
plt.title('L and M bands')
plt.ylabel(r'CP/$\sigma_{CP}$',fontsize=15)
    

#now just need to flatten to filter for interesting ones!
plt.figure()
plt.hist( [item for sublist in cp_sigma for item in sublist] , bins = 30 )
plt.xlabel(r'CP/$\sigma_{CP}$ ',fontsize=15)
plt.title('L and M bands')
plt.ylabel('counts',fontsize=15)

plt.figure()
plt.hist( [item for sublist in cp_values for item in sublist] , bins = 30 )
plt.xlabel('CP (degrees)',fontsize=15)
plt.ylabel('counts',fontsize=15)
plt.title('L and M bands')

['2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-30T234920_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3'],


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

scatter_hist(np.array(sig_wvl[i]) * 1e6, cp_sigma[i], ax,  ax_histy)

for i in range(len(sig_wvl)):
    #unique_ids
    scatter_hist( np.array( sig_wvl[i] )* 1e6, cp_sigma[i], ax, ax_histy)
    
ax.set_ylabel(r'Z-score (CP=0)',fontsize=14)
ax.set_xlabel(r'wavelength [$\mu$m]',fontsize=14)
ax_histy.set_xlabel('counts',fontsize=14)
ax.grid()
ax_histy.grid()


#%% N Band 
#wvl_ranges = [L_lims,M_lims]
wvl_ranges = [N_lims]

ylim_plot = [-100,100]
swap_wvl_order = False

cp_sigma_N = [] #all cp sigmas away from zero 
cp_values_N = [] # all cp
sig_wvl_N=[]
sig_id_N=[]
for f in files :    
    h = oifits.open(f)
            
    current_config = ''.join(list(np.sort(h['OI_ARRAY'].data['STA_NAME'])))
    
    #effective wavelength
    wvl = h['OI_WAVELENGTH'].data['EFF_WAVE']
    if swap_wvl_order:  # WHY DO WE DO THIS??
        wvl = wvl[::-1]
    #initialize wavelength filter
    filt = np.zeros(len(wvl)) == 1    
    for w_lims in wvl_ranges:
        filt = filt | ( ( wvl<w_lims[1] ) & ( wvl>w_lims[0]) )
    
    
    cp = h['OI_T3'].data['T3PHI'][:]
    
    cp_err = h['OI_T3'].data['T3PHIERR'][:]
    
    sigma_from_zero = cp / cp_err
    
    fig,ax = plt.subplots(4,1,sharex=True,figsize=(15,8))
    indx2station = {h['OI_ARRAY'].data['STA_INDEX'][i]:h['OI_ARRAY'].data['STA_NAME'][i] for i in range(len(h['OI_ARRAY'].data['STA_NAME']))}
    
    for bT in range(len(cp)):
        lab = [indx2station[h['OI_T3'].data['STA_INDEX'][bT][XXX]] for XXX in range(3)] 
        ax[bT].errorbar(wvl,cp[bT],yerr=cp_err[bT],linestyle=':',label= f'{lab[0]}-{lab[1]}-{lab[2]}');
        ax[bT].axvline(N_lims[0]); ax[bT].axvline(N_lims[1]); ax[bT].fill_betweenx([-300,300],N_lims[0],N_lims[1],alpha=0.5, color='y')
        #ax[bT].axvline(M_lims[0]); ax[bT].axvline(M_lims[1]); ax[bT].fill_betweenx([-300,300],M_lims[0],M_lims[1],alpha=0.5, color='orange')
        
        ax[bT].axhline(0,color='k');ax[bT].axhline(180,color='k');ax[bT].axhline(-180,color='k')
        ax[bT].legend(loc='upper right',fontsize=12)
        ax[bT].set_ylabel('CP [deg]',fontsize=15)
        
        ax[bT].set_ylim(ylim_plot )
        
        # CP > 5 AND SIGMA > 1 AWAY FROM ZERO
        new_filt = filt #& (abs(cp[bT])>5) & (abs(sigma_from_zero[bT])>1)
        
        cp_tmp = cp[bT][new_filt]
        
        chi_tmp =  sigma_from_zero[bT][new_filt]  
        
        wvl_tmp =  wvl[new_filt]
        id_tmp = ['{}'.format(f.split('/')[-1]) + f'--{lab[0]}-{lab[1]}-{lab[2]}' for i in range(sum( new_filt ))]
                                   
        if len( chi_tmp ) > 0:
            
            cp_sigma_N.append( list(chi_tmp) )
            cp_values_N.append( cp_tmp  )
            sig_wvl_N.append( list(wvl_tmp ) )
            sig_id_N.append( id_tmp )
            
    ax[-1].set_xlabel(f'wavelength [$m$]',fontsize=15)
    ax[0].set_title('{}'.format(f.split('/')[-1]),fontsize=15)

plt.figure()    
#unique_ids = np.unique( [item for sublist in sig_id_N for item in sublist] ) 
for i in range(len(sig_wvl_N)):
    #unique_ids
    plt.plot(sig_wvl_N[i],cp_sigma_N[i],'.', label='sig_id_N',color='k',alpha=0.4)
plt.xlabel(r'wavelength [m]',fontsize=15)
plt.title('L and M bands')
plt.ylabel(r'CP/$\sigma_{CP}$',fontsize=15)
    

#now just need to flatten to filter for interesting ones!
plt.figure()
plt.hist( [item for sublist in cp_sigma_N for item in sublist] , bins = 30 )
plt.xlabel(r'CP/$\sigma_{CP}$ ',fontsize=15)
plt.title('L and M bands')
plt.ylabel('counts',fontsize=15)

plt.figure()
plt.hist( [item for sublist in cp_values_N for item in sublist] , bins = 30 )
plt.xlabel('CP (degrees)',fontsize=15)
plt.ylabel('counts',fontsize=15)
plt.title('L and M bands')

['2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-31T020521_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3',
       '2022-08-30T234920_VRTPav_A0G1J2J3_IR-LM_LOW_Chop_cal_oifits_0.fits--A0-J2-J3'],



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

scatter_hist(np.array(sig_wvl_N[i]) * 1e6, cp_sigma_N[i], ax,  ax_histy)

for i in range(len(sig_wvl_N)):
    #unique_ids
    scatter_hist(np.array(sig_wvl_N[i]) * 1e6, cp_sigma_N[i], ax, ax_histy)

ax.set_ylabel(r'Z-score (CP=0)',fontsize=14)
ax.set_xlabel(r'wavelength [$\mu$m]',fontsize=14)
ax_histy.set_xlabel('counts',fontsize=14)
ax.grid()
ax_histy.grid()




#%%%
def fit_prep(files, EXTVER=None,flip=True):    
    # pionier data is [wvl, B], while gravity is [B,wvl ] (so gravity we want flip =Tue)              
    
    if EXTVER==None:
        wvl_EXTNAME = 'OI_WAVELENGTH'
        v2_EXTNAME = 'OI_VIS2'
    
    else:
        wvl_EXTNAME = ('OI_WAVELENGTH',EXTVER)
        v2_EXTNAME = ('OI_VIS2',EXTVER)
        
    hdulists = [oifits.open(f) for f in files]
    
    wvls = [ h[wvl_EXTNAME].data['EFF_WAVE'] for h in hdulists]
    wvl_grid = np.median( wvls , axis=0) # grid to interpolate wvls 
    
    data_dict = {} 
    for ii, h in enumerate( hdulists ):
        
        file = files[ii].split('/')[-1]
        B = np.sqrt(h[v2_EXTNAME].data['UCOORD']**2 + h[v2_EXTNAME].data['VCOORD']**2)
        
        v2_list = []
        v2err_list = []
        dwvl = []
        obs_time = []
        if not flip:
            for b in range(len(B)):
                
                interp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2DATA'][b,:] ,fill_value =  "extrapolate" )
                
                dwvl.append( np.max( 1e9 * ( abs( h[wvl_EXTNAME].data['EFF_WAVE'] -  wvl_grid ) ) ) )
                
                obs_time.append( [h[0].header['DATE-OBS'],h[0].header['LST']] ) 
                
                
                v2_list.append( interp_fn( wvl_grid ) )
                
                interp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2ERR'][b,:] ,fill_value =  "extrapolate" )
                
                v2err_list.append( interp_fn( wvl_grid ) )
                
            print('max wvl difference in interpolatation for {} = {}nm'.format(file, np.max(dwvl)))
            
            v2_df = pd.DataFrame( v2_list , columns =wvl_grid , index = B )
            
            v2err_df = pd.DataFrame( v2err_list , columns = wvl_grid , index = B)
            
            time_df = pd.DataFrame( obs_time , columns = ['DATE-OBS','LST'] , index = B)
            
            data_dict[file] = {'v2':v2_df, 'v2err':v2err_df, 'obs':time_df}
            
            v2_df = pd.concat( [data_dict[f]['v2'] for f in data_dict] , axis=0)
            v2err_df = pd.concat( [data_dict[f]['v2err'] for f in data_dict] , axis=0)
            obs_df = pd.concat( [data_dict[f]['obs'] for f in data_dict] , axis=0)

        else: # for graivty we have to flip 
            for i in range( len(B)):
                
                interp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2DATA'].T[:,i] ,fill_value =  "extrapolate" )
                
                dwvl.append( np.max( 1e9 * ( abs( h[wvl_EXTNAME].data['EFF_WAVE'] -  wvl_grid ) ) ) )
                
                obs_time.append( [h[0].header['DATE-OBS'],h[0].header['LST']] )
                
                v2_list.append( interp_fn( wvl_grid ) )
                
                interp_fn = interp1d( h[wvl_EXTNAME].data['EFF_WAVE'], h[v2_EXTNAME].data['VIS2ERR'].T[:,i] ,fill_value =  "extrapolate" )
                
                v2err_list.append( interp_fn( wvl_grid ) )
             
            print('max wvl difference in interpolatation for {} = {}nm'.format(file, np.max(dwvl)))
            
            v2_df = pd.DataFrame( v2_list , columns =wvl_grid , index = B )
            
            v2err_df = pd.DataFrame( v2err_list , columns = wvl_grid , index = B)
            
            time_df = pd.DataFrame( obs_time , columns = ['DATE-OBS','LST'] , index = B)
            
            
            data_dict[file] = {'v2':v2_df, 'v2err':v2err_df, 'obs':time_df}
            
            v2_df = pd.concat( [data_dict[f]['v2'] for f in data_dict] , axis=0)
            v2err_df = pd.concat( [data_dict[f]['v2err'] for f in data_dict] , axis=0)
            obs_df = pd.concat( [data_dict[f]['obs'] for f in data_dict] , axis=0)
        
    return( v2_df , v2err_df , obs_df)


pionier_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/*.fits')

gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/my_reduction_v3/*.fits')
# below the gravity files in small config with new P2VM (provided by Xavier) 
#gravity_files = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/gravity/reduction_small_with_new_P2VM/*.fits')

matisse_files_L = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits')
matisse_files_N = glob.glob('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/matisse/reduced_calibrated_data_1/all_merged_N/*.fits')
#[ h[i].header['EXTNAME'] for i in range(1,8)]


pion_v2_df , pion_v2err_df ,  pion_obs_df = fit_prep(pionier_files,flip=True)

grav_p1_v2_df , grav_p1_v2err_df, grav_p1_obs_df = fit_prep(gravity_files, EXTVER = 11 ,flip=True)
grav_p2_v2_df , grav_p2_v2err_df , grav_p2_obs_df = fit_prep(gravity_files, EXTVER = 12 ,flip=True)

mati_L_v2_df , mati_L_v2err_df , mati_L_obs_df = fit_prep(matisse_files_L ,flip=True)
mati_N_v2_df , mati_N_v2err_df , mati_N_obs_df = fit_prep(matisse_files_N ,flip=True)





#%% Gravity 

"""
compare the reduced data of the SC with the FT
look at the calibrator, are you using different calibrators? 
Are you sure that one of the calibrators is not a binary? 
If the calibrator is always the same are you sure from the raw data that it is not a binary?
Clean your data, between 2 and 2.1 it looks very noisy, probably not usable

"""

# ++++++++ evidence of binarity 

date = grav_p1_obs_df['DATE-OBS']
B = np.array(list(grav_p1_v2_df.index)) #grav_p1_v2_df.index
wvl = np.array(list(grav_p1_v2_df.columns ))

b_bins = np.linspace(min(B), max(B), 30)

for b_ll, b_ul in zip(b_bins[:-1], b_bins[1:]):
    b_filt = (B>=b_ll) & (B<b_ul)
    if sum(b_filt):
        plt.figure()
        for i in range(sum(b_filt)):
            plt.semilogy(wvl, grav_p1_v2_df[b_filt].values[i], label=date[b_filt].values[i] )
        plt.title(f'(B>={round(b_ll,1)}) & (B<{round(b_ul,1)})')
        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel('wavelength [m]')
        plt.ylabel(r'$|V|^2$')
#grav_p1_obs_df




#%% compare VIS2 in FT and SCIENCE

hdulists = [oifits.open(f) for f in gravity_files]

for h in hdulists:
    wvl_sc = h['OI_WAVELENGTH',11].data['EFF_WAVE']
    wvl_ft = h['OI_WAVELENGTH',21].data['EFF_WAVE']
    v2_sc = h['OI_VIS2',11].data['VIS2DATA']
    v2_ft = h['OI_VIS2',21].data['VIS2DATA']
    #h['OI_VIS2',22].data['VIS2DATA']
    #h['OI_WAVELENGTH',22]
    
    plt.figure()
    
    plt.plot( [-1], [-1],color='k', linestyle='-',alpha=0.9,label='SC') ; plt.plot( [-1], [-1], 'x', color='k' ,label='FT')
    for b , c in zip( range(6),['b','r','g','k','y','orange'] ):
        plt.plot( wvl_sc, v2_sc.T[:,b] ,color=c, linestyle='-',alpha=0.2) ; plt.plot( wvl_ft, v2_ft.T[:,b], 'x', color=c )
    plt.title(h[0].header['DATE-OBS'])
    plt.xlim([2e-6,2.39e-6])
    plt.ylim([0,1.5])
    plt.legend()
    plt.ylabel(r'|V|$^2$',fontsize=15)
    plt.xlabel('wavelength [m]',fontsize=15)



#%% 

# filters 
grav_B_filt = grav_p1_v2_df.index.values !=0 
grav_wvl_filt = (grav_p1_v2_df.columns > 1.9e-6) & (grav_p1_v2_df.columns < 2.4e-6)

# matisse wvl limits from https://www.eso.org/sci/facilities/paranal/instruments/matisse.html
mat_L_wvl_filt = (mati_L_v2_df.columns > 3.2e-6) & (mati_L_v2_df.columns < 3.9e-6) #| (mati_L_v2_df.columns > 4.5e-6) 
mat_M_wvl_filt = (mati_L_v2_df.columns > 4.5e-6) &  (mati_L_v2_df.columns <= 5e-6)
mat_N_wvl_filt = (mati_N_v2_df.columns > 8e-6) & (mati_N_v2_df.columns <= 13e-6)#| (mati_L_v2_df.columns > 4.5e-6)

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



    
ud_fit_per_ins = {} # to hold fitting results per instrument photometric band

for ins in ins_vis_dict:
    
    print(f'\n\n\n fitting {ins} visibility data to UD model\n\n\n')
    # get the current instrument visibilities
    v2_df, v2err_df = ins_vis_dict[ins]
    
    ud_fit_results = {}
    
    redchi2 = []
    diam_mean= []
    diam_median= []
    diam_err= []
    
    intermediate_results_dict = {}
    
    for wvl_indx, wvl in enumerate(v2_df.columns ):
        
        intermediate_results_dict[wvl] = {}#{'rho':[], 'v2_obs':[], 'v2_obs_err':[],\'v2_model':[],'samplers':[] }
        
        rho = v2_df.index.values / wvl # B/wvl (angular freq , rad^-1)
        v2 = v2_df[wvl].values  # |V|^2
        v2_err = v2err_df[wvl].values # uncertainty 
        
        # filter out unphysical V2 values 
        v2_filt = (v2>0) & (v2<1.1)
        
        # short hand model notation 
        x, y, yerr = rho[v2_filt] , v2[v2_filt], v2_err[v2_filt]
    
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
        
        nwalkers = 50 #32
        ndim = 1
        # quick grid search 
        theta_grid = np.linspace(mas2rad(1),mas2rad(20),20)
        p0_mean = theta_grid[np.argmin( [np.sqrt( np.sum( abs(disk_v2(x, theta) - y)**2 ) ) for theta in theta_grid ] ) ]
        p0 = p0_mean + mas2rad(p0_mean/2) * np.random.rand(nwalkers, 1)
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=( x, y, yerr )
        )
        sampler.run_mcmc(p0, 1000, progress=True);
        
        #samples = sampler.get_chain(flat=True)
        
        #plt.hist(np.log10(samples) ) , bins = np.logspace(-9,-7,100)) #[-1,:,0])
        
        #plt.hist( np.log10( samples ) , bins=np.linspace(-9,-5,100 ))
        
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        
    
        
        """
        plt.figure()
        labels = [r'$\theta$ (mas)']
        
        fig = corner.corner(
            rad2mas(flat_samples), labels=labels
        );"""
        #plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/pionier/disk_diam_hist_example.png')
    
        
        """plt.figure() 
        plt.errorbar(v2_df.columns, v2_df.iloc[wvl_indx], yerr= v2err_df.iloc[wvl_indx], linestyle=' ')
        plt.xlabel('Baseline (m)')
        plt.ylabel(r'$V^2$')
        plt.plot(v2_df.columns,  disk_v2( rho, np.mean( rad2mas( flat_samples[:, :] ) ) *1e-3 * np.pi/180 / 3600  ) ,'.')
        """
        
        #y_model = np.median( rad2mas( flat_samples[:, :] ) ) * 1e-3 * np.pi/180 / 3600
        
        
        mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
        q = np.diff(mcmc)
        
        diam_mean.append( rad2mas( np.mean(  flat_samples[:, :] ) ))
        diam_median.append( rad2mas( mcmc[1] ))
        diam_err.append( rad2mas(q) )
        
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
        
        #best fit
        y_model = disk_v2( x, mas2rad(diam_median[-1] ) ) 
        
      
        intermediate_results_dict[wvl]['rho'] = x
        intermediate_results_dict[wvl]['v2_obs'] = y
        intermediate_results_dict[wvl]['v2_obs_err'] = yerr
        intermediate_results_dict[wvl]['v2_model'] = y_model
        intermediate_results_dict[wvl]['samplers'] = flat_samples
        
        redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df[wvl])-1))
        
        #reduced chi2 
        #redchi2.append(chi2(y_model  , y, yerr) / (len(v2_df.iloc[wvl_indx])-1))
        
        print('reduced chi2 = {}'.format(chi2(y_model, y, yerr) / (len(v2_df[wvl])-1)) )
    
    
    ud_fit_results['diam_mean'] = diam_mean
    ud_fit_results['diam_median'] = diam_median
    ud_fit_results['diam_err'] = diam_err
    ud_fit_results['redchi2'] = redchi2
    
    ud_fit_results['intermediate_results'] = intermediate_results_dict
    
    ud_fit_per_ins[ins] = ud_fit_results


#%% Plot UD results 
fig1 = plt.figure(1,figsize=(10,8))
fig1.set_tight_layout(True)

frame1=fig1.add_axes((.1,.3,.8,.6))
frame2=fig1.add_axes((.1,.05,.8,.2))   


#for ins, col in zip(ud_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
for ins, col in zip(ud_fit_per_ins, ['b','slateblue','darkslateblue','deeppink','orange','red']):
    if 1: #ins!='Matisse (N)':
        wvl_grid = np.array( list( ud_fit_per_ins[ins]['intermediate_results'].keys() ) )
        diam_median = ud_fit_per_ins[ins]['diam_median']
        diam_err = ud_fit_per_ins[ins]['diam_err']
        redchi2 = ud_fit_per_ins[ins]['redchi2']
        
        # plot it
        frame1.errorbar(1e6*wvl_grid, diam_median, yerr=np.array(diam_err).T, color = col, fmt='-o', lw = 2, label = ins)
        #frame1.yscale('log')
        plt.plot(1e6*wvl_grid, redchi2, '-',lw=2, color=col)


fontsize=20
frame1.set_title('RT Pav Uniform Disk Fit vs Wavelength')
frame1.grid()
frame1.legend(fontsize=fontsize)
frame1.set_ylabel(r'$\theta$ [mas]',fontsize=fontsize)
frame1.tick_params(labelsize=fontsize)
frame1.set_xticklabels([]) 

frame2.grid()
frame2.set_xlabel(r'wavelength [$\mu m$]',fontsize=fontsize)
frame2.set_ylabel(r'$\chi^2_\nu$',fontsize=fontsize)
frame2.tick_params(labelsize=fontsize)

#plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/RT_pav/FIT_UDs.pdf',bbox_inches='tight')

#plt.title('RT Pav\nuniform disk diameter')

def plot_sampler(ins, wvl ,fontsize=14, xlabel=r'$\theta$ [mas]'):
    
    plt.figure()
    plt.title(f'MCMC Uniform Disk fit')
    plt.hist( rad2mas(ud_fit_per_ins[ins]['intermediate_results'][wvl]['samplers']),\
             label=f'{ins} at {round(1e6*wvl,3)}um',histtype='step',color='k',lw=3 )
    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.gca().tick_params(labelsize=fontsize)
    plt.show()
    

ins = list( ud_fit_per_ins.keys() )[3] 
wvls =  list( ud_fit_per_ins[ins]['intermediate_results'].keys() ) 
plot_sampler(ins, wvls[3] , xlabel=r'$\theta$ [mas]')



#%%










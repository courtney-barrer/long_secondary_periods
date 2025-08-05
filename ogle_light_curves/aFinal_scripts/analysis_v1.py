import json
import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np

def filter_points_within_distance(x, y, a, b, distance_modulus):
    """
    Filters x, y data points based on their distance from the line y = ax + b,
    within a specified distance modulus.

    Parameters:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - a (float): Slope of the line.
    - b (float): Intercept of the line.
    - distance_modulus (float): Maximum allowable distance from the line.

    Returns:
    - x_filtered (numpy array): Filtered x values.
    - y_filtered (numpy array): Filtered y values.
    - distances (numpy array): Distances of the filtered points from the line.
    """
    x = np.array(x)
    y = np.array(y)

    # Calculate the perpendicular distance from each point to the line y = ax + b
    distances = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)

    # Filter points within the specified distance modulus
    mask = distances <= distance_modulus
    x_filtered = x[mask]
    y_filtered = y[mask]
    filtered_distances = distances[mask]

    return x_filtered, y_filtered, filtered_distances, mask



def read_photometry(star_number, base_dir="."):
    """
    Reads photometry data for a given star number from OGLE database files.

    Parameters:
        star_number (int): The star number corresponding to the database number in the ident.rtf file.
        base_dir (str): The base directory where the "phot/V/" and "phot/I/" directories are located.

    Returns:
        dict: A dictionary with keys 'V' and 'I', each containing a DataFrame of photometry data.
              Returns None for a band if the corresponding file is missing.
    """
    photometry = {}

    # Directories for V and I bands
    v_band_dir = os.path.join(base_dir, "phot/V")
    i_band_dir = os.path.join(base_dir, "phot/I")

    # Construct file names
    star_id = f"OGLE-LMC-LPV-{int(star_number):05d}"
    v_file = os.path.join(v_band_dir, f"{star_id}.dat")
    i_file = os.path.join(i_band_dir, f"{star_id}.dat")

    # Helper function to read a file
    def read_file(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path, delim_whitespace=True, header=None, names=["JD", "Magnitude", "Error"])
        return None

    # Read V and I band photometry
    photometry['V'] = read_file(v_file)
    photometry['I'] = read_file(i_file)

    return photometry

def plot_a_light_curve( star_number, band="I", base_dir = "/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/LMC_ogle/", savefig=None):
    photometry_data = read_photometry(star_number, base_dir=base_dir)
    ff = 15
    plt.figure(figsize= (8,5))
    plt.plot( photometry_data[band]['JD'], photometry_data[band]['Magnitude'], '.', label=f"OGLE STAR #{star_number}")
    plt.xlabel( "JD [days]" ,fontsize =ff)
    plt.ylabel( f"Magnitude [{band}-band]" ,fontsize =ff)
    plt.gca().tick_params(labelsize=ff)
    plt.legend()
    if savefig is not None:
        plt.tight_layout()
        #plt.savefig( savefig , dpi = 300)
    plt.show() 


###################################################
### Measured data
###################################################

# File path
file_path = "/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/ogleIII_LMC_LPV_catalog_decompressed.dat"
#path to save figures 
fig_path="/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/figures/"
if not os.path.exists(fig_path):
    os.mkdir(fig_path)


# Column names based on the description provided
columns = [
    "Star", "Field", "OGLE", "Type", "GB", "Sp", "RA", "Dec", "<I>", "<V>",
    "P1 (d)", "Iamp1", "P2 (d)", "Iamp2", "P3 (d)", "Iamp3",
    "J", "H", "Ks"
]

# Read the file into a DataFrame
data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=columns)

# Calculate Wesenheit index W_JK = Ks - 0.686 * (J - Ks)
data["W_JK"] = data["Ks"] - 0.686 * (data["J"] - data["Ks"])

# effective temperautre
R_V = 3.41
R_I = 1.85
R_J = 0.303
R_KS = 0.118

def vis_teff_updated(V, I, Fe_H=-1.0):
    """
    Estimate effective temperature using V-I color and metallicity calibration
    based on Casagrande et al. (2010).

    Parameters:
        V (float): Mean V magnitude.
        I (float): Mean I magnitude.
        Fe_H (float): Metallicity [Fe/H] (default: -1.0 for typical LMC stars).

    Returns:
        float: Estimated effective temperature (K).
    """
    # Coefficients from Casagrande et al. (2010) for V-I
    a0, a1, a2, a3, a4, a5 = 0.4033, 0.8171, -0.1987, -0.0409, 0.0319, 0.0012
    R_i = 1.85 # i band extinction
    R_v = 3.41 # v band extinction in LMC 
    A_v = 0.3 # total extinction, varies between 0.1-0.3 (source?)
    I_cor = V - A_v 
    V_cor = I - R_i/R_v * A_v 

    VI = V_cor - I_cor #V - R_i/R_v * I
    theta_eff = (
        a0
        + a1 * VI
        + a2 * VI**2
        + a3 * VI * Fe_H
        + a4 * Fe_H
        + a5 * Fe_H**2
    )
    return 5040 / theta_eff



# Example usage (vis_teff_updated bug is now fixed and looks good!) 
T_effs=[]
for V,I in zip(data["<V>"].values, data["<I>"].values):
    T_effs.append( vis_teff_updated(V,I, Fe_H=-0.19) )

## add it to data 
data["Teff_est_1"] = abs( np.array( T_effs ) ) 



###################################################
### reading in all the simulated data from our MC 
###################################################

# List of keys you want to extract from each file
selected_keys = ['Iamp1', 'Teff_est_1', 'delta_T_eff', 'W_JK', 'amp', 'f2/f1']

# Global dictionary to store lists of values per key
sim_dict = {key: [] for key in selected_keys}

# Find all matching JSON files
files = glob.glob( "/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/sim_dict_5.json" ) #"sim_dict_*.json")  # use wildcard correctly

for file_path in files:
    try:
        with open(file_path, 'r') as f:
            data_tmp = json.load(f)

        if isinstance(data_tmp, dict):
            for key in selected_keys:
                if key in data_tmp:
                    sim_dict[key].append(data_tmp[key])
                else:
                    print(f"Warning: Key '{key}' not found in '{file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# flatten from each simulation run
for k in sim_dict:
    sim_dict[k] = np.array( sim_dict[k] ).ravel()  



###################################################
### Filter for sequence D and D1/2 on measured and simulated data 
###################################################

# Extract data points near suspected lower harmonics
# extract data points near the line defined by coeffients a,b
a_1 = -4.281395023488202 #-4.3 
b_1 = 21.58247347257682 #21.8 
x_data = np.log10( data["P1 (d)"] )
y_data =  data["W_JK"] #a * x_data + b

distance_modulus = 0.13 # Maximum distance from the line

# mask for data on the sequence D1/2 line 
_,_,_, mask_1 = filter_points_within_distance(
    x_data, y_data, a_1, b_1, distance_modulus
)


# mask for data on the sequence D line
a_2 = -4.281395023488202 #-4.3 
b_2 =  22.871301797933263 #23.0
x_data = np.log10( data["P1 (d)"] )
y_data =  data["W_JK"] #a * x_data + b

_, _, _, mask_2 = filter_points_within_distance(
    x_data, y_data, a_2, b_2, distance_modulus
)


# get all the numbers corresponding to sequence D1/2 
sdhalf_stars = np.array( [d[0] for d in  data.index[mask_1] ] ) 
sd_stars = np.array( [d[0] for d in  data.index[mask_2] ] )


##### simulation filter (keep same silly naming as previous scripts )
filt_D = np.array( sim_dict['f2/f1'] ) <= 1
filt_Dhalf = np.array( sim_dict['f2/f1'] ) > 1




Pgrid = np.logspace(1, 3.5, 100)
plt.figure(figsize=(10, 6))
plt.plot( data["P1 (d)"][~(mask_1 * mask_2)], data["W_JK"][~(mask_1 * mask_2)], '.', alpha=0.01, color="k")  #, edgecolor="k" ) #, s=20)
plt.plot( data["P1 (d)"][mask_1], data["W_JK"][mask_1], '.', alpha=0.03, color="orange" )  #, edgecolor="k" ) #, s=20)
plt.plot( data["P1 (d)"][mask_2], data["W_JK"][mask_2], '.', alpha=0.02, color="red" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_1 * np.log10(Pgrid ) + b_1 , '-', alpha=1, color="orange", label = r"seq. D1/2 : W$_{JK}$" +f" = {a_1:.2f}"+r"log$_{10}$(P)+"+f"{b_1:.2f}" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_2 * np.log10( Pgrid ) + b_2 , '-', alpha=1, color="r", label = r'seq. D : W$_{JK}$' +f" = {a_2:.2f}"+r"log$_{10}$(P)+"+f"{b_2:.2f}" )  #, edgecolor="k" ) #, s=20)

plt.legend()
plt.xscale("log")
plt.xlabel("Period (days)", fontsize=14)
plt.ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
#plt.title("Wesenheit Index $W_{JK}$ vs Secondary Period $P_2$", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim( 15, 7 )
#plt.savefig(fig_path + "seqDandHalf_v2.png",bbox_inches="tight")
plt.show()


###################################################
# D vs D1/2 Amplitude ratio , measured vs simulated 
###################################################

# try take 1000 sampled from each sequence 
plt.figure()#
plt.hist( sim_dict['f2/f1'][filt_D][:5000] , bins =np.logspace(-3, 2, 30) , alpha =0.2)
plt.hist( sim_dict['f2/f1'][filt_Dhalf][:5000] , bins =np.logspace(-3, 2, 30) , alpha =0.2)
plt.xscale('log')
plt.show()

plt.figure()
fs=15 
Ns = 1000 
bins = np.logspace(-3, 3, 40)
plt.hist(  list( sim_dict['f2/f1'][filt_D][:Ns]) + list(sim_dict['f2/f1'][filt_Dhalf][:Ns]) , bins =bins, histtype='step',color='k',label='combined') 
plt.hist( sim_dict['f2/f1'][filt_D][:Ns] , bins =bins ,alpha = 0.2, label='seq. D')
plt.hist( sim_dict['f2/f1'][filt_Dhalf][:Ns], bins =bins ,alpha = 0.2, label='seq. D1/2')
plt.xlabel(r'Spectral Amplitude Ratio [$D_{1/2}/D$]',fontsize=fs)
plt.ylabel('Frequency',fontsize=fs)
plt.legend( fontsize=fs)
plt.gca().tick_params(labelsize=fs)
plt.xscale('log')
#plt.savefig(fig_path + "amplitude_ratio_simulated.jpeg",bbox_inches ='tight', dpi=300)
plt.show()

print( 'seqD f2/f1 mean amp ',np.mean( sim_dict['f2/f1'][filt_D] )  )
print( 'seqD1/2 f2/f1 mean amp ',np.mean( sim_dict['f2/f1'][filt_Dhalf] )  )

###################################################
# Population estimate , measured vs simulated 
###################################################

######### MEASURED POPULATION 

#MC of fitted model within uncertainy limits  and inlier analys slope for set modulus 

# measured from /Users/bencb/Documents/long_secondary_periods/ogle_light_curves/ogle_LPV_catalog_population_analysis_v1.py
slope_D = a_2 #-4.281395023488202
std_err_D = 0.01833574659409436

intercept_D = b_2 #22.871301797933263
intercept_err_D = 0.0024203634618315184

slope_Dhalf = a_1 #-4.281395023488202
std_err_Dhalf = 0.01833574659409436

intercept_Dhalf  = b_1 #21.58247347257682
intercept_err_Dhalf = 0.0024203634618315184


inlier_thresh = 0.12 #0.13
inlier_thresh_std = 0.01
max_aD , min_aD = slope_D + std_err_D  ,slope_D - std_err_D
max_bD , min_bD = intercept_D + intercept_err_D, intercept_D - intercept_err_D 
max_aDhalf , min_aDhalf = slope_Dhalf + std_err_Dhalf, slope_Dhalf - std_err_Dhalf
max_bDhalf , min_bDhalf = intercept_Dhalf + intercept_err_Dhalf, intercept_Dhalf - intercept_err_Dhalf

x_data = np.log10(data["P1 (d)"])
y_data = data["W_JK"]

Ns = 10000
popu = [] # hold fractional population Dhalf / (D + Dhalf)
for _ in range(Ns):
    inlier_thresh_tmp = inlier_thresh  + inlier_thresh_std * np.random.randn(  ) 
    aD = slope_D + std_err_D * np.random.randn( ) #np.random.uniform( min_aD , max_aD )
    bD = intercept_D + intercept_err_D * np.random.randn( )  #np.random.uniform( min_bD , max_bD )
    
    aDhalf = slope_Dhalf + std_err_Dhalf * np.random.randn( ) #np.random.uniform( min_aDhalf , max_aDhalf )
    bDhalf = intercept_Dhalf + intercept_err_Dhalf * np.random.randn( ) #np.random.uniform( min_bDhalf , max_bDhalf )
    
    _, _ , _, D_mask_tmp = filter_points_within_distance(x_data, y_data, aD, bD, inlier_thresh_tmp)

    _, _ , _, Dhalf_mask_tmp = filter_points_within_distance(x_data, y_data, aDhalf, bDhalf, inlier_thresh_tmp)

    D_pop = np.sum( D_mask_tmp )
    Dhalf_pop = np.sum( Dhalf_mask_tmp )
    pop_total = D_pop + Dhalf_pop

    rel_popu = Dhalf_pop / pop_total

    popu.append( rel_popu )

plt.figure()
plt.hist( popu , bins = np.linspace( 0.1, 0.2, 20), alpha = 0.5, label=r"$\frac{D1/2}{D1/2+D}$")
plt.axvline( np.median( popu ) ,color='k', ls = ':',lw=3, label=f'q50 = {np.median( popu ):.3f}')
plt.axvline( np.quantile( popu , 0.84 ) ,color='grey', ls = '-.',lw=1,label=f'q84 = {np.quantile( popu , 0.84 ):.3f}') 
plt.axvline( np.quantile( popu , 0.16 ) ,color='grey', ls = '-.',lw=1,label=f'q16 = {np.quantile( popu , 0.16 ):.3f}') 
plt.xlabel("Population Fraction",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=15)
#plt.savefig(fig_path+"measured_population_dist_bootstrapped.jpeg", dpi=300, bbox_inches='tight')
plt.show() 



######### SIMULATED POPULATION 

def bootstrap_ratio(values, threshold, n_bootstrap=1000):
    ratios = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(values, size=len(values), replace=True)
        ratio = np.sum(resample > threshold) / len(resample)
        ratios.append(ratio)
    return np.percentile(ratios, [16, 50, 84])  # median + 1σ


def estimate_ratio_with_threshold_uncertainty(xvals, T_nominal, T_frac_error=0.05, n_iter=1000):
    """
    Estimate population ratio uncertainty including uncertainty in threshold.
    
    Parameters:
        xvals : array-like
            Simulated values (length N)
        T_nominal : float
            Nominal threshold value
        T_frac_error : float
            Fractional uncertainty on threshold (e.g. 0.05 for ±5%)
        n_iter : int
            Number of MC iterations
        
    Returns:
        ratios : np.ndarray
            Array of simulated population fractions
        percentiles : tuple
            (16th, 50th, 84th) percentiles of the population ratio
    """
    N = len(xvals)
    T_vals = np.random.normal(loc=T_nominal, scale=T_frac_error * T_nominal, size=n_iter)
    ratios = np.array([np.sum(xvals > T_i) / N for T_i in T_vals])
    return ratios, np.percentile(ratios, [16, 50, 84])

popu_sim, (lo, med, hi) = estimate_ratio_with_threshold_uncertainty(xvals=sim_dict['f2/f1'], 
                                                                    T_nominal=0.99, #0.8,#1.0, 
                                                                    T_frac_error=0.1, 
                                                                    n_iter=3000)



# measured 
#print( f"Measured Population ratio = {len(sdhalf_stars) / (len(sdhalf_stars)+ len(sd_stars))}")
print(f"Population ratio = {med:.3f} (+{hi - med:.3f}/-{med - lo:.3f})")

bins = 100*np.linspace( 0.05, 0.2, 20)
plt.figure(figsize=(8,8))
plt.hist( 100*np.array(popu_sim) , bins = bins, alpha = 0.5, density=True, label=r"$\frac{D1/2}{D1/2+D}$ simulated bootstrap")
plt.hist( 100*np.array(popu) , bins = bins, alpha = 0.5,density=True, label=r"$\frac{D1/2}{D1/2+D}$ measured bootstrap")
plt.axvline( 100*np.mean( popu ) ,color='k', ls = ':',lw=3, label=f'measured mean +/- std = {100*np.mean( popu ):.2f} +/- {100*np.std( popu ):.2f}')
plt.axvline( 100*np.mean( popu_sim ) ,color='k', ls = ':',lw=3, label=f'simulated mean +/- std = {100*np.mean( popu_sim ):.2f} +/- {100*np.std( popu_sim ):.2f}')
#plt.axvline( np.quantile( popu , 0.84 ) ,color='grey', ls = '-.',lw=1,label=f'q84 = {np.quantile( popu , 0.84 ):.3f}') 
#plt.axvline( np.quantile( popu , 0.16 ) ,color='grey', ls = '-.',lw=1,label=f'q16 = {np.quantile( popu , 0.16 ):.3f}') 
plt.xlabel("Population Fraction [%]",fontsize=15)
plt.ylabel("Density",fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=15)
#plt.savefig(fig_path+"meas_vs_sim_pop_dist_thresh-1.jpeg", dpi=300, bbox_inches='tight')
plt.show() 



###################################################
### Power Law fits effective temperature vs amplitude  
###################################################


# Define power law function for fit! 
def power_law(T, beta, alpha):
    return beta * T**alpha

# Extract data
T_vals_raw = np.array(sim_dict['Teff_est_1'])
A_vals_raw = np.array(sim_dict['amp'])  # use 'mag_amp' not 'amp' now

# filter for finite, positive values
mask = np.isfinite(T_vals_raw) & np.isfinite(A_vals_raw) & (A_vals_raw > 0) & (abs(T_vals_raw) < 10000 )
T_vals= T_vals_raw[mask]
A_vals = A_vals_raw[mask]

# Fit in log-log space to improve numerical stability
log_T = np.log10(T_vals)
log_A = np.log10(A_vals)

D_set_sim = filt_D + filt_Dhalf #filt_Dhalf
D_set_meas = np.array( list(sd_stars)+list(sdhalf_stars) )  #sd_stars #sdhalf_stars  #np.array( list(sd_stars)+list(sdhalf_stars) ) 

T_sim_fit = np.array(sim_dict['Teff_est_1'])[D_set_sim ]
A_sim_fit = np.array(sim_dict['amp'])[D_set_sim ]  # use 'mag_amp' not 'amp' now


T_meas_fit = np.array(data['Teff_est_1'].loc[ D_set_meas ].values )
A_meas_fit = np.array( data['Iamp1'].loc[ D_set_meas ].values)

# Clean data
mask_sim = np.isfinite(T_sim_fit) & np.isfinite(A_sim_fit) & (A_sim_fit > 0) & (T_sim_fit > 500) & (T_sim_fit < 10e3)
mask_meas = np.isfinite(T_meas_fit) & np.isfinite(A_meas_fit) & (A_meas_fit > 0) & (T_meas_fit > 500) & (T_meas_fit < 10e3)

T_sim_fit_filt = T_sim_fit[mask_sim]
A_sim_fit_filt = A_sim_fit[mask_sim]
T_meas_fit_filt = T_meas_fit[mask_meas]
A_meas_fit_filt = A_meas_fit[mask_meas]

# Fit simulated data
log_T_sim = np.log10(T_sim_fit_filt)
log_A_sim = np.log10(A_sim_fit_filt)

(coeffs_sim, cov_sim) = np.polyfit(log_T_sim, log_A_sim, 1, cov=True)
alpha_sim, log_beta_sim = coeffs_sim
alpha_sim_err, log_beta_sim_err = np.sqrt(np.diag(cov_sim))
beta_sim = 10**log_beta_sim
beta_sim_err = np.log(10) * beta_sim * log_beta_sim_err  # propagate uncertainty

# alpha_sim, log_beta_sim = np.polyfit(log_T_sim, log_A_sim, 1, cov=True)
# beta_sim = 10**log_beta_sim

# Fit measured data
log_T_meas = np.log10(T_meas_fit)
log_A_meas = np.log10(A_meas_fit)

# Fit measured data
(coeffs_meas, cov_meas) = np.polyfit(log_T_meas, log_A_meas, 1, cov=True)
alpha_meas, log_beta_meas = coeffs_meas
alpha_meas_err, log_beta_meas_err = np.sqrt(np.diag(cov_meas))
beta_meas = 10**log_beta_meas
beta_meas_err = np.log(10) * beta_meas * log_beta_meas_err

# alpha_meas, log_beta_meas = np.polyfit(log_T_meas, log_A_meas, 1, cov=True)
# beta_meas = 10**log_beta_meas

# Generate fitted curves
T_plot = np.linspace(500, 12000, 500)
A_sim_fit_curve = beta_sim * T_plot**alpha_sim
A_meas_fit_curve = beta_meas * T_plot**alpha_meas



label_sim = (
    fr'$A = ({beta_sim:.2e} \pm {beta_sim_err:.1e}) \cdot T^{{{alpha_sim:.2f} \pm {alpha_sim_err:.2f}}}$'
)

label_meas = (
    fr'$A = ({beta_meas:.2e} \pm {beta_meas_err:.1e}) \cdot T^{{{alpha_meas:.2f} \pm {alpha_meas_err:.2f}}}$'
)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fs = 15 
# Simulated
#axes[0].set_title("Simulated")
axes[0].loglog(T_sim_fit, A_sim_fit, '.', alpha=0.05, color='k')
axes[0].loglog(T_plot, A_sim_fit_curve, '-', color='red',
             label=label_sim ) #fr'$A = {beta_sim:.2e} \cdot T^{{{alpha_sim:.2f}}}$')
axes[0].text(1500, 1, "Thermal Dipole Simulation", fontsize=15)
axes[0].set_xlim([1000, 6000])
axes[0].set_ylim([1e-3, 10])
axes[0].set_yscale('log')
axes[0].set_xlabel('Temperature [K]', fontsize=fs)
axes[0].set_ylabel('LSP Amplitude [Imag]', fontsize=fs)
axes[0].legend()
#axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Measured
#axes[1].set_title("Measured (OGLE-III)")
axes[1].text(  1500, 1, "Measured (OGLE-III)" ,fontsize=15)
axes[1].loglog(T_meas_fit, A_meas_fit, '.', alpha=0.05, color='k')
axes[1].loglog(T_plot, A_meas_fit_curve, '-', color='blue',
             label=label_meas) #fr'$A = {beta_meas:.2e} \cdot T^{{{alpha_meas:.2f}}}$')
axes[1].set_xlim([1000, 6000])
axes[1].set_ylim([1e-3, 10])
axes[1].set_yscale('log')
axes[1].set_xlabel('Temperature [K]', fontsize=fs)
#axes[1].set_ylabel('LSP Amplitude [I-band] (mag)')
axes[1].legend()

axes[0].tick_params(labelsize=fs)
axes[1].tick_params(labelsize=fs)
#axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

#plt.savefig("T_v_Amplitude_SIM_v_MEAS.jpeg",dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

print("Simulated:")
print(f"  alpha = {alpha_sim:.4f} +/- {alpha_sim_err:.4f}")
print(f"  beta  = {beta_sim:.4e} +/- {beta_sim_err:.4e}")

print("Measured:")
print(f"  alpha = {alpha_meas:.4f} +/- {alpha_meas_err:.4f}")
print(f"  beta  = {beta_meas:.4e} +/- {beta_meas_err:.4e}")


z = (alpha_sim - alpha_meas) / np.sqrt(alpha_sim_err**2 + alpha_meas_err**2)
p_value = 2 * (1 - norm.cdf(np.abs(z)))

print( f"index p-value {p_value}")

z = (beta_sim - beta_meas) / np.sqrt(beta_sim_err**2 + beta_meas_err**2)
p_value = 2 * (1 - norm.cdf(np.abs(z)))

print( f"interc p-value {p_value}")



########## ADDING PERIOD
### idea is if period is classified on D1/2, multiply by 2 and put on D, If on D then leave
# it there. Use same processing as was done in (we should move this to the actual for loop)
# then with simulated data if classified as D we keep the period, if classified as D1/2 we half it
# Then plot the 
# 

colors = {
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'yellow': '#F0E442',
    'black': '#000000',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'orange': '#E69F00',
    'purple': '#CC79A7'
}

D_set = list( sdhalf_stars ) + list( sd_stars ) 
D_norm_periods=[]
wjk = []
test_cnt = 0
for ct ,i in enumerate( D_set  ):
    #print( ct , i )
    if 1:# data['Teff_est_1'].loc[i].values[0] > 500:

        wjk.append( data['W_JK'].loc[i].values[0] )
        
        if i in list( sdhalf_stars ):
            # we multiply by 2 to normalize to sequence D ( testing hyp. is that all of these are sequence D and D1/2 is only a geometrical effect)
            D_norm_periods.append( 2 * data['P1 (d)'].loc[i].values[0] )

        else: # just normal sequence D 
            D_norm_periods.append( data['P1 (d)'].loc[i].values[0] )

    else :
        test_cnt +=1
#plt.figure(); plt.semilogx( D_norm_periods, wjk , '.', alpha =0.2);plt.show() 

# 
sim_periods = [] 
for i,fratio in enumerate( sim_dict['f2/f1'] ): 
    if fratio > 1:  # simulation classified has D1/2 - so half the period 
        sim_periods.append( 0.5 * D_norm_periods[i] )
    else:
        sim_periods.append( D_norm_periods[i] )

plt.figure() 
plt.plot(Pgrid, -4.3 * np.log10(Pgrid ) + 21.8 , '-', alpha=1, color="orange", label = "LSP lower harmonics?" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, -4.3 * np.log10( Pgrid ) + 22.7 , '-', alpha=1, color="r", label = 'LSP')  #, edgecolor="k" ) #, s=20)
# # this is wrong! iloc is counting index, df index is i+1
# plt.plot(  data['P1 (d)'].iloc[D_set], data['W_JK'].iloc[D_set],'.', alpha =0.2 )
# # This is is fine wit loc
#plt.plot(  data['P1 (d)'].loc[D_set], data['W_JK'].loc[D_set],'.', alpha =0.2 )
plt.plot(  sim_periods, wjk,'.', alpha =0.2 )
plt.xlabel( "LSP Period [days]" )
plt.ylabel( r"$W_{JK}$" )
plt.xscale('log')
plt.show() 



"""
I have 2 data sets to plot / compare which are essentially simulated and measured data from different population (or sequences) called sequence D and D1/2
- to start I need a central scatter plot of 
plt.semilogx(  sim_periods, wjk,'.', alpha =0.2 )
with imposed lines 
plt.plot(Pgrid, -4.3 * np.log10(Pgrid ) + 21.8 , '-', alpha=1, color="orange", label = "seq D1/2" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, -4.3 * np.log10( Pgrid ) + 22.7 , '-', alpha=1, color="r", label = 'seq D')  #, edgecolor="k" ) #, s=20)
I then want two shared axis histograms, one shared x above showing the histogram of the simulated sequence D and D1/2 distributions of periods. 
    np.array( sim_periods )[sim_dict['f2/f1'] > 1]
    np.array(sim_periods )[sim_dict['f2/f1'] < 1]
the other shared y on the right with a density histogram of wjk - this simulated sequence D and D1/2 distributions of wesenheit index. 
    np.array( wjk )[np.array( sim_dict['f2/f1'] ) > 1]
    np.array( wjk )[np.array( sim_dict['f2/f1'] )  < 1]
then over this I want on each histogram a step style (black line) density histogram of another variable 
for the x-axis histogram I need the measured periods
    data['P1 (d)'].loc[sd_stars]
    data['P1 (d)'].loc[sdhalf_stars]
for the y-axis histogrem I need measured wesenheit index
    data['W_JK'].loc[sd_stars]
    data['W_JK'].loc[sdhalf_stars]


"""

f2_f1 = np.array( sim_dict['f2/f1'] )

# Measured data (placeholders)
P1_D = data['P1 (d)'].loc[sd_stars]
P1_Dhalf = data['P1 (d)'].loc[sdhalf_stars]
WJK_D = data['W_JK'].loc[sd_stars]
WJK_Dhalf = data['W_JK'].loc[sdhalf_stars]

# Masks
mask_D = f2_f1 < 1
mask_Dhalf = f2_f1 > 1

# Grids for lines
Pgrid = np.logspace(2, 3.5, 500)

# Set up figure
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1: , 0:3])
ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_yhist = fig.add_subplot(gs[1:, 3], sharey=ax_main)

# Main scatter plot
ax_main.semilogx(np.array(sim_periods)[mask_D], np.array(wjk)[mask_D], '.', alpha=0.2, color='red',label='Simulated')
ax_main.semilogx(np.array(sim_periods)[mask_Dhalf], np.array(wjk)[mask_Dhalf], '.', alpha=0.2, color='orange',label='Simulated')
#ax_main.semilogx(np.array(sim_periods)[mask_D], np.array(sim_dict['W_JK'])[mask_D], '.', alpha=0.2, color='red',label='Simulated')
#ax_main.semilogx(np.array(sim_periods)[mask_Dhalf], np.array(sim_dict['W_JK'])[mask_Dhalf], '.', alpha=0.2, color='orange',label='Simulated')
#ax_main.plot(Pgrid, -4.3 * np.log10(Pgrid) + 21.8, ':', color="orange", label="seq D1/2")
#ax_main.plot(Pgrid, -4.3 * np.log10(Pgrid) + 22.7, ':', color="red", label="seq D")
ax_main.set_xlabel('Period (days)')
ax_main.set_ylabel('W_JK')
ax_main.legend()

# Top histogram (Period)
ax_xhist.hist(np.array(sim_periods)[mask_D], bins=50, density=True, alpha=0.5, color='red', label='Sim D')
ax_xhist.hist(np.array(sim_periods)[mask_Dhalf], bins=50, density=True, alpha=0.5, color='orange', label='Sim D1/2')
ax_xhist.hist(P1_D, bins=50, density=True, histtype='step', color='black',  label='Meas D')
ax_xhist.hist(P1_Dhalf, bins=50, density=True, histtype='step', color='gray',  label='Meas D1/2')
ax_xhist.legend()
ax_xhist.axis('off')

# Right histogram (W_JK)
ax_yhist.hist(np.array(wjk)[mask_D], bins=50, density=True, orientation='horizontal', alpha=0.5,color='red')
ax_yhist.hist(np.array(wjk)[mask_Dhalf], bins=50, density=True, orientation='horizontal', alpha=0.5,color='orange')
#ax_yhist.hist(np.array(sim_dict['W_JK'])[mask_D], bins=50, density=True, orientation='horizontal', alpha=0.5,color='red')
#ax_yhist.hist(np.array(sim_dict['W_JK'])[mask_Dhalf], bins=50, density=True, orientation='horizontal', alpha=0.5,color='orange')
ax_yhist.hist(WJK_D, bins=50, density=True, histtype='step', orientation='horizontal', color='black')
ax_yhist.hist(WJK_Dhalf, bins=50, density=True, histtype='step', orientation='horizontal', color='gray')
ax_yhist.axis('off')
#plt.savefig("simulated_PL_diagram_thermal_dipole.jpeg",dpi=350, bbox_inches='tight')
plt.show()


# plt.figure() 
# plt.hist(np.array(wjk)[mask_D], bins=50, density=True, orientation='horizontal', alpha=0.5)
# plt.hist(np.array(wjk)[mask_Dhalf], bins=50, density=True, orientation='horizontal', alpha=0.5)
# plt.show() 


# plt.figure() 
# plt.hist(WJK_D, bins=50, density=True, histtype='step', orientation='horizontal', color='black')
# plt.hist(WJK_Dhalf, bins=50, density=True, histtype='step', orientation='horizontal', color='gray')
# plt.show()





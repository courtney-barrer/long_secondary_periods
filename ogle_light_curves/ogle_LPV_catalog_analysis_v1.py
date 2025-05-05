import pandas as pd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 

from astropy.timeseries import LombScargle



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
        plt.savefig( savefig , dpi = 300)
    plt.show() 




def get_star_value(data, star_number, column_name):
    """
    Retrieve a specific value from the OGLE catalog for a given star number and column name.

    Parameters:
        catalog_file (str): Path to the catalog file (e.g., "ogleIII_LMC_LPV_catalog_decompressed.dat").
        star_number (int): The star number (#Star) to query.
        column_name (str): The column name to retrieve.

    Returns:
        value: The corresponding value from the catalog.
    """

    #file_path = "ogleIII_LMC_LPV_catalog_decompressed.dat"

    # Column names based on the description provided
    #columns = [
    #    "Star", "Field", "OGLE", "Type", "GB", "Sp", "RA", "Dec", "<I>", "<V>",
    #    "P1 (d)", "Iamp1", "P2 (d)", "Iamp2", "P3 (d)", "Iamp3",
    #    "J", "H", "Ks"
    #]

    # Read the file into a DataFrame
    #data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=columns)


    # Find the row corresponding to the star number
    star_numbers = np.array( [d[0] for d in data.index] )
    star_row = data.iloc[star_number==star_numbers]

    if star_row.empty:
        raise ValueError(f"Star number {star_number} not found in the catalog.")

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the catalog.")

    return star_row.iloc[0][column_name]


def analyze_photometry_period(phot_data, catalogue_data,  star_number, band, plot=False):
    """
    Perform Lomb-Scargle periodogram analysis on photometry data.

    ### TO DO: GET LSP (FUNCTION TO GET STAR DETAILS FROM NUMBER) AND 
     NORMALISE TIME SERIES BY THIS 
    Parameters:
        phot_data (dict): Dictionary containing photometry data for different stars and bands.
        band (str): The band to analyze (e.g., 'V' or 'I').

    Returns:
        float: Best period detected in the photometry data.
    """

    # extract strongest peroid
    P = float( get_star_value(catalogue_data, star_number=star_number, column_name="P1 (d)") ) 

    # Extract the time series data
    data = phot_data[star_number][band]
    time = (data['JD'].values - data['JD'].values[0]) / P
    magnitude = data['Magnitude'].values
    error = data['Error'].values

    # Calculate frequency range based on the time series
    min_time_diff = np.min(np.diff(np.sort(time)))  # Minimum time difference between observations
    max_frequency = 10 # np.log10(1 / min_time_diff)
    min_frequency = 0.01 #1e-4  # Arbitrary minimum frequency to avoid extremely long periods
    frequencies = np.linspace(min_frequency, max_frequency, 10000)

    # Perform Lomb-Scargle analysis
    ls = LombScargle(time, magnitude, error)
    power = ls.power(frequencies)

    # Find the best frequency and period
    best_frequency = frequencies[np.argmax(power)]
    best_period = 1 / best_frequency

    # Plot the periodogram
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(1 / frequencies, power, label='Lomb-Scargle Power')
        plt.axvline(best_period, color='r', linestyle='--', label=f'Best Period: {best_period:.3f} days')
        plt.xscale('log')
        plt.xlabel('Period (days)', fontsize=14)
        plt.ylabel('Power', fontsize=14)
        plt.title(f'Lomb-Scargle Periodogram ({band}-band)', fontsize=16)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    return ls, frequencies, power


    

# File path
file_path = "ogleIII_LMC_LPV_catalog_decompressed.dat"
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




###############################################################
# PLOT PL DIAGRAM 

Pgrid = np.logspace(1, 3.5, 100)
plt.figure(figsize=(10, 6))
plt.plot(data["P1 (d)"], data["W_JK"], '.', alpha=0.05, color="k" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, -4.3 * np.log10(Pgrid ) + 21.8 , '-', alpha=1, color="orange", label = "LSP lower harmonics?" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, -4.3 * np.log10( Pgrid ) + 22.7 , '-', alpha=1, color="r", label = 'LSP')  #, edgecolor="k" ) #, s=20)

plt.legend()
plt.xscale("log")
plt.xlabel("Secondary Period P2 (days)", fontsize=14)
plt.ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
#plt.title("Wesenheit Index $W_{JK}$ vs Secondary Period $P_2$", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim( 15, 7 )
plt.show()



###############################################################
# Extract data points near suspected lower harmonics
# extract data points near the line defined by coeffients a,b
a_1 = -4.3 
b_1 = 21.8 
x_data = np.log10( data["P1 (d)"] )
y_data =  data["W_JK"] #a * x_data + b

distance_modulus = 0.1 # Maximum distance from the line

# mask for data on the sequence D1/2 line 
_,_,_, mask_1 = filter_points_within_distance(
    x_data, y_data, a_1, b_1, distance_modulus
)


# mask for data on the sequence D line
a_2 = -4.3 
b_2 = 23.0
x_data = np.log10( data["P1 (d)"] )
y_data =  data["W_JK"] #a * x_data + b

distance_modulus = 0.1 # Maximum distance from the line

_, _, _, mask_2 = filter_points_within_distance(
    x_data, y_data, a_2, b_2, distance_modulus
)

# -------- visulise extracted points --------------

plt.figure(figsize=(10, 6))
plt.plot( data["P1 (d)"][~(mask_1 * mask_2)], data["W_JK"][~(mask_1 * mask_2)], '.', alpha=0.02, color="k")  #, edgecolor="k" ) #, s=20)
plt.plot( data["P1 (d)"][mask_1], data["W_JK"][mask_1], '.', alpha=0.02, color="orange" )  #, edgecolor="k" ) #, s=20)
plt.plot( data["P1 (d)"][mask_2], data["W_JK"][mask_2], '.', alpha=0.02, color="red" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_1 * np.log10(Pgrid ) + b_1 , '-', alpha=1, color="orange", label = "seq. D1/2" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_2 * np.log10( Pgrid ) + b_2 , '-', alpha=1, color="r", label = 'seq. D')  #, edgecolor="k" ) #, s=20)

plt.legend()
plt.xscale("log")
plt.xlabel("Period (days)", fontsize=14)
plt.ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
#plt.title("Wesenheit Index $W_{JK}$ vs Secondary Period $P_2$", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim( 15, 7 )
plt.savefig(fig_path + "seqDandHalf.png",bbox_inches="tight")
plt.show()

# -------- Look at P2 of filtered points with P1 on D1/2--------------


plt.figure(figsize=(10, 6))
plt.plot( data["P1 (d)"][mask_1] , data["W_JK"][mask_1], '.', alpha=0.5, color="orange", label='P1' )  #, edgecolor="k" ) #, s=20)
plt.plot( data["P2 (d)"][mask_1] , data["W_JK"][mask_1], '.', alpha=0.1, color="grey", label = "P2 | P1 "+r"$\in$ D1/2")  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_1 * np.log10(Pgrid ) + b_1 , '-', alpha=1, color="orange", label = "seq. D1/2" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_2 * np.log10( Pgrid ) + b_2 , '-', alpha=1, color="r", label = 'seq. D')  #, edgecolor="k" ) #, s=20)


plt.legend()
plt.xscale("log")
plt.xlabel("Period (days)", fontsize=14)
plt.ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
#plt.title("Wesenheit Index $W_{JK}$ vs Secondary Period $P_2$", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim( 15, 7 )
plt.savefig(fig_path + "seqD12_P2.png",bbox_inches="tight")
plt.show()


# -------- Look at P2 of filtered points with P1 on D--------------


plt.figure(figsize=(10, 6))
plt.plot( data["P1 (d)"][mask_2] , data["W_JK"][mask_2], '.', alpha=0.5, color="red", label='P1' )  #, edgecolor="k" ) #, s=20)
plt.plot( data["P2 (d)"][mask_2] , data["W_JK"][mask_2], '.', alpha=0.1, color="grey", label = "P2 | P1 "+r"$\in$ D" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_1 * np.log10(Pgrid ) + b_1 , '-', alpha=1, color="orange", label = "seq. D1/2" )  #, edgecolor="k" ) #, s=20)
plt.plot(Pgrid, a_2 * np.log10( Pgrid ) + b_2 , '-', alpha=1, color="r", label = 'seq. D')  #, edgecolor="k" ) #, s=20)


plt.legend()
plt.xscale("log")
plt.xlabel("Period (days)", fontsize=14)
plt.ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
#plt.title("Wesenheit Index $W_{JK}$ vs Secondary Period $P_2$", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim( 15, 7 )
plt.savefig(fig_path + "seqD_P2.png",bbox_inches="tight")
plt.show()



### 
# Create a figure with 3 rows and 1 column, sharing the x-axis

fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# First plot
axes[0].plot(data["P1 (d)"][~(mask_1 * mask_2)], data["W_JK"][~(mask_1 * mask_2)], '.', alpha=0.02, color="k")
axes[0].plot(data["P1 (d)"][mask_1], data["W_JK"][mask_1], '.', alpha=0.02, color="orange")
axes[0].plot(data["P1 (d)"][mask_2], data["W_JK"][mask_2], '.', alpha=0.02, color="red")
axes[0].plot(Pgrid, a_1 * np.log10(Pgrid) + b_1, '-', alpha=1, color="orange", label="seq. D1/2")
axes[0].plot(Pgrid, a_2 * np.log10(Pgrid) + b_2, '-', alpha=1, color="r", label='seq. D')
axes[0].legend()
axes[0].set_xscale("log")
axes[0].set_ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)
axes[0].set_ylim(15, 7)

# Second plot
axes[1].plot(data["P1 (d)"][mask_1], data["W_JK"][mask_1], '.', alpha=0.5, color="orange", label='P1')
axes[1].plot(data["P2 (d)"][mask_1], data["W_JK"][mask_1], '.', alpha=0.1, color="grey", label="P2 | P1 " + r"$\in$ D1/2")
axes[1].plot(Pgrid, a_1 * np.log10(Pgrid) + b_1, '-', alpha=1, color="orange", label="seq. D1/2")
axes[1].plot(Pgrid, a_2 * np.log10(Pgrid) + b_2, '-', alpha=1, color="r", label='seq. D')
axes[1].legend()
axes[1].set_xscale("log")
axes[1].set_ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)
axes[1].set_ylim(15, 7)

# Third plot
axes[2].plot(data["P1 (d)"][mask_2], data["W_JK"][mask_2], '.', alpha=0.5, color="red", label='P1')
axes[2].plot(data["P2 (d)"][mask_2], data["W_JK"][mask_2], '.', alpha=0.1, color="grey", label="P2 | P1 " + r"$\in$ D")
axes[2].plot(Pgrid, a_1 * np.log10(Pgrid) + b_1, '-', alpha=1, color="orange", label="seq. D1/2")
axes[2].plot(Pgrid, a_2 * np.log10(Pgrid) + b_2, '-', alpha=1, color="r", label='seq. D')
axes[2].legend()
axes[2].set_xscale("log")
axes[2].set_xlabel("Period (days)", fontsize=14)
axes[2].set_ylabel("Wesenheit Index $W_{JK}$", fontsize=14)
axes[2].grid(True, which="both", linestyle="--", linewidth=0.5)
axes[2].set_ylim(15, 7)

# Adjust layout
plt.tight_layout()
plt.savefig(fig_path+"seqD_harmonic_investigation.png")
plt.show()




######### GETTING LIGHT CURVES 
## Downloaded LMC AND SMC LIGHT CURVES AT
## /Users/bencb/Documents/long_secondary_periods/ogle_light_curves/
# LMC_ogle/phot/ OR SMC_ogle/phot/


# Simple Example 
base_dir = "/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/LMC_ogle/"  # Replace with the actual path to your OGLE data
star_number = "00001"  # Replace with the desired star number
photometry_data = read_photometry(star_number, base_dir=base_dir)



# get all the numbers corresponding to sequence D1/2 
sdhalf_stars = np.array( [d[0] for d in  data.index[mask_1] ] ) 
sd_stars = np.array( [d[0] for d in  data.index[mask_2] ] )


phot_dict = {}
for s in sdhalf_stars: ### Looking at stars on D1/2
    phot_dict[s] = read_photometry(s, base_dir=base_dir)


## Plot some of them 
band = "I"
N  = 64
valid = np.array( [int(k) for k in phot_dict.keys() ] )
rand_key_idx = np.random.randint( 0, len(valid),  N )
fig, ax = plt.subplots( int( np.sqrt(N )), int( np.sqrt(N )) , figsize=(12,12)  )  
for axx, i in zip( ax.reshape(-1), rand_key_idx ):
    k = valid[i]
    axx.plot( phot_dict[k][band]['JD'], phot_dict[k][band]['Magnitude'], '.', label=k)
    axx.legend(fontsize=8)
plt.show() 


# list interesting ones 
#interesting_list = [ 42510] #58908

interesting_list = [42510, 84949, 77664, 9483, 8799, 14389] #58908

save_path = base_dir + "interesting_light_curves/"

for n in interesting_list:
    plot_a_light_curve( star_number= n,
                        band="I", 
                        base_dir = "/Users/bencb/Documents/long_secondary_periods/ogle_light_curves/LMC_ogle/", 
                        savefig =save_path + f'LMC_{n}.png' )
    


#looks at the periodgram normalized to the primary period ("P1 (days)")
### we filtered for stars on D1/2 (sequence D) so normalize to strongest period (P1).  

# plt.figure()
# for k in interesting_list :
#     P = get_star_value(data, star_number=k, column_name="P1 (d)")
#     # looks at the periodgram normalized to the primary period ("P1 (days)")
#     # which should be the LSP (if we correctly filtered for sequence D1/2 or D )
#     ls, freq, power = analyze_photometry_period(phot_dict, data, k, band= "I", plot=True)



## Next step, density plot of Lomb-Scarfpld spectrum looking at 0.5, and 2 (when normalized by LSP)
# 
# interesting metric is the ratio of power at 0.5 and 2 harmonics for  D sequence 
#  vs other sequences? Is sequence D more likely to have harmonics?  

# maybe we should do this analysis for sequence D and NOT sequence D1/2?!?

# get all the numbers corresponding to sequence D1/2 
sdhalf_stars = np.array( [d[0] for d in  data.index[mask_1] ] ) 
sd_stars = np.array( [d[0] for d in  data.index[mask_2] ] )

phot_dict = {}

for lab, seq in zip(['sequence D', 'sequence D1/2'], [sd_stars,sdhalf_stars]):
    phot_dict[lab]={}
    for s in seq: ### Looking at stars on D1/2
        phot_dict[lab][s] = read_photometry(s, base_dir=base_dir)

sequence="sequence D"#  "sequence D1/2"#"sequence D"#

power_list=[]
plt.figure()
for k in list(phot_dict[sequence].keys())[:1000] :
    ls, freq, power = analyze_photometry_period(phot_dict[sequence], data, k, band= "I", plot=False)
    power_list.append( power )
    plt.loglog(1/freq, power, alpha=0.01, color='grey')

plt.loglog(1/freq, 1e15*power, color='grey', label=f"{sequence} ({band}-band) light curves")
plt.loglog(1/freq, np.mean(power_list,axis=0), color='black', label=f'{sequence} mean')
plt.axvline(1, ls='-',color='blue', label='Primary Period')
plt.axvline(0.5, ls='--',color='r', label='2nd harmonic')
plt.axvline(2, ls=':',color='black', label='half harmonic')

plt.ylim(1e-3,1e1)
plt.xlabel('Normalized Period [unitless]', fontsize=14)
plt.ylabel('Power [unitless]', fontsize=14)
plt.legend(loc='upper right')
#plt.title(f'Lomb-Scargle Periodogram ({band}-band)', fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig(fig_path+f"{sequence.replace('/','-')}_Lomb-Scargle_Periodogram.png",bbox_inches='tight')
plt.show()









##############################################################
# ESTIMATING EFFECTIVE TEMPERATURE

def vis_teff(I,V):
    ##THIS SEEMS VERY REASONABLE BUT I CANT FIND A SOLID REFERENCE
    return 9000 / (V - I + 1.5)

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
    VI = V - I
    theta_eff = (
        a0
        + a1 * VI
        + a2 * VI**2
        + a3 * VI * Fe_H
        + a4 * Fe_H
        + a5 * Fe_H**2
    )
    return 5040 / theta_eff


def correct_jk_extinction(J, Ks, A_V):
    """
    Correct J and Ks magnitudes for extinction.

    Parameters:
        J (float): Observed J magnitude.
        Ks (float): Observed Ks magnitude.
        A_V (float): Visual extinction.

    Returns:
        tuple: Extinction-corrected J and Ks magnitudes.
    """
    A_J = R_J * A_V
    A_KS = R_KS * A_V
    J_corr = J - A_J
    Ks_corr = Ks - A_KS
    return J_corr, Ks_corr

def jk_teff(J, Ks, Fe_H=-1, A_V=0):
    """
    Estimate effective temperature using J and Ks magnitudes,
    optionally corrected for extinction.

    Parameters:
        J (float): Mean J magnitude.
        Ks (float): Mean Ks magnitude.
        A_V (float): Visual extinction (default: 0, no correction).

    Returns:
        float: Estimated effective temperature (K).
    """
    if A_V > 0:
        J, Ks = correct_jk_extinction(J, Ks, A_V)
    # Coefficients for J-Ks from Casagrande et al. (2010)
    a0, a1, a2, a3, a4, a5 = 0.6528, -0.5815, 0.1755, -0.0839, 0.0274, -0.0052
    JK = J - Ks
    theta_eff = (
        a0
        + a1 * JK
        + a2 * JK**2
        + a3 * JK * (Fe_H)  # Assume [Fe/H] = -1.0
        + a4 * (-1.0)
        + a5 * (-1.0)**2
    )
    return 5040 / theta_eff

# Example usage
T_eff=[]
for I,V in zip(data["<V>"].values, data["<I>"].values):
    T_eff.append( vis_teff_updated(I,V, Fe_H=0.0) )

plt.hist(T_eff, bins=np.linspace(1000,7000,50))
plt.show()


#DONT USE THIS - NOT AS GOOD JK
# T_eff=[]
# for J,Ks in zip(data["J"].values, data["Ks"].values):
#     T_eff.append( jk_teff(J, Ks, Fe_H=-0.19, A_V=0.5) )

# plt.hist(T_eff, bins=np.linspace(1000,10000,50))
# plt.show()


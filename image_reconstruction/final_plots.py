

import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits 
import sys
import os 
import glob

from astropy.io import fits
from scipy.ndimage import zoom, gaussian_filter

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util


#############################################
######### PLOTTING RECONSTRUCTIONS FOR INSPECTION
#############################################

instruments = [
    "pionier", "gravity", "matisse_L", "matisse_M", "matisse_N_8.0um",
    "matisse_N_8.5um", "matisse_N_9.0um", "matisse_N_9.5um", "matisse_N_10.0um",
    "matisse_N_10.5um", "matisse_N_11.0um", "matisse_N_11.5um",
    "matisse_N_12.0um", "matisse_N_12.5um"
]

mu = 3000.0
tau=1.0
base_dir_template = "/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/<ins>"
file_pattern = f"imageReco_<ins>_prior-UD_regul-hyperbolic_pixelscale-*_fov-*_wavemin-*_wavemax-*_mu-{mu}_tau-{tau}_eta-1_usev2-all_uset3-phi.fits"

extent = 40 
plot_util.plot_reconstructed_images_with_options(
    file_pattern, instruments, base_dir_template, plot_dirty_beam=True,
    same_extent=[extent, -extent, -extent, extent], draw_origin=True, plot_logscale=True, vmin=0.01, vmax=1
) # swap extents for x to get the right coordinates with labels (East to the left)

# auto naming for combined plots
delete_patterns = [r"<ins>", r"\*", r"\.fits"]
import re
fname = file_pattern
# Replace all patterns with white space
for pattern in delete_patterns:
    fname = re.sub(pattern, "", fname )
# for ins in instruments:
#     fname += ins+'_'

plt.savefig("/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/aCOMBINED_PLOTS/" + f"{fname}.png")


#############################################
######### PLOTTING RECONSTRUCTIONS FOR PAPER 
#############################################

instruments = [
    "pionier", "gravity", "matisse_L", "matisse_M", "matisse_N_8.0um",
    "matisse_N_8.5um", "matisse_N_9.0um", "matisse_N_9.5um", "matisse_N_10.0um",
    "matisse_N_10.5um", "matisse_N_11.0um", "matisse_N_11.5um",
    "matisse_N_12.0um", "matisse_N_12.5um"
]

wvls = ["1.6","2.2","3.5","4.7","8.0","8.5", "9.0", "9.5", "10.0", "10.5","11.0", "11.5", "12.0", "12.5"]

instrument_dict = {k:v for k,v in zip( wvls, instruments) }

mu = [500.0, 500.0] + [3000.0 for _ in range(len( instrument_dict )-2)]
tau = [1.0 for _ in range(len( instrument_dict ))]
#mu = 3000.0
#tau=1.0

base_dir_template = "/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/<ins>"
file_dict = {}
for i, (wvl,instrument) in enumerate( instrument_dict.items() ):

    file_pattern = f"imageReco_<ins>_prior-UD_regul-hyperbolic_pixelscale-*_fov-*_wavemin-*_wavemax-*_mu-{mu[i]}_tau-{tau[i]}_eta-1_usev2-all_uset3-phi.fits"

    base_dir = base_dir_template.replace("<ins>", instrument)

    # Use glob to find the first file matching the pattern
    file_path = glob.glob(os.path.join(base_dir, file_pattern.replace("<ins>", instrument)))
    #print( file_path )

    file_dict[wvl] = file_path 



fig, ax = plt.subplots(2, len(file_dict) // 2, figsize=(20, 5), sharex=True, sharey=True)

# Further reduce spacing: Tighten vertical spacing between rows
fig.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.1, wspace=0.2, hspace=0.05)

axx = ax.reshape(-1)

# Open the FITS file
same_extent = [45, -45, -45, 45]
draw_origin = 1
plot_logscale = 1
vmin = 1e-2
vmax = 1

# Placeholder for image collection (needed for shared colorbar)
images = []

for idx, (w, f) in enumerate(file_dict.items()):
    h = fits.open(f[0])
    
    # Load reconstructed image
    reconstructed_image = np.fliplr(h[0].data / np.max(h[0].data))

    if float(w) > 3:
        reconstructed_image = np.roll( reconstructed_image, shift=-1, axis=0)

    dx = h[0].header['CDELT1']
    dy = h[0].header['CDELT2']
    x = np.linspace(-h[0].data.shape[0] // 2 * dx, h[0].data.shape[0] // 2 * dx, h[0].data.shape[0])
    y = np.linspace(-h[0].data.shape[1] // 2 * dy, h[0].data.shape[1] // 2 * dy, h[0].data.shape[1])
    full_extent = [np.max(x), np.min(x), np.min(y), np.max(y)]

    # Set extent
    extent = same_extent if same_extent else full_extent
    
    reconstructed_image = plot_util.crop_image_to_extent(reconstructed_image, extent, full_extent)

    # Plot reconstructed image
    norm = LogNorm(vmin=vmin, vmax=vmax) if plot_logscale else None
    im = axx[idx].imshow(reconstructed_image, cmap='Reds', extent=extent, origin='lower', norm=norm)

    # Collect the image for colorbar reference
    images.append(im)

    # Set x-axis and y-axis labels
    if idx % (len(file_dict) // 2) == 0:  # Only left column gets y-axis labels
        axx[idx].set_ylabel('ΔDEC -> N [mas]', fontsize=12)
    if idx >= len(file_dict) // 2:  # Only bottom row gets x-axis labels
        axx[idx].set_xlabel('ΔRA <- E [mas]', fontsize=12)

    # Place black text in the top-left corner
    axx[idx].text(
        extent[0] - 8, extent[3] - 5, f'{w}'+r'$\mu m$', 
        color='black', fontsize=12, ha='left', va='top'
    )
    axx[idx].text(
        extent[0] - 8, extent[3] - 15, 
        r'$\chi_\nu^2$={}'.format(round(h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'], 2)), 
        color='black', fontsize=12, ha='left', va='top'
    )

    # Draw origin lines if requested
    if draw_origin:
        axx[idx].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axx[idx].axvline(0, color='black', linestyle='--', linewidth=0.8)

# Add a single colorbar for all plots on the right side, height reduced by 20%
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(images[0], cax=cbar_ax, orientation='vertical')
cbar.set_label('Normalized Flux', fontsize=15)
cbar.ax.tick_params(labelsize=12)

# Save and show
plt.savefig("final_image_reco.png", dpi=300, bbox_inches='tight')
plt.show()


#############################################
######### PLOTTING INDIVIDUAL RECONSTRUCTIONS AT SHORTER WAVELENGTHS  
#############################################

# smooth it with dirty beam 


f = '/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/pionier/test1/imageReco_pionier_prior-Random_regul-compactness_pixelscale-0.69_fov-10.0_wavemin-1.5_wavemax-1.8_mu-3000.0_tau-1e-05_eta-1_usev2-all_uset3-phi.fits'
d = fits.open( f )
# /home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/pionier/imageReco_pionier_prior-specificFile_regul-hyperbolic_pixelscale-0.69_fov-10.0_wavemin-1.5_wavemax-1.8_mu-1000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits")
image_raw = d[0].data / np.max(d[0].data)
image =  np.pad(image_raw, d[0].data.shape[0]//4, mode='constant', constant_values=0)
dirty_beam = d['IMAGE-OI DIRTY BEAM'].data / np.max(d['IMAGE-OI DIRTY BEAM'].data)
levels = [np.max(dirty_beam)/2] # FWHM

# Interpolate to higher resolution
zoom_factor = 3  # Factor to increase resolution
high_res_image = zoom(image, zoom_factor, order=3)  # Cubic interpolation

#  Smooth the high-resolution image
sigma = 2  # Gaussian smoothing parameter
smoothed_image = gaussian_filter(high_res_image, sigma=sigma)


#  Offset the contour to the corner
high_res_dirt = zoom(dirty_beam, zoom_factor, order=3)
beam_shape = high_res_dirt.shape
image_shape = smoothed_image.shape
offset_x = image_shape[1]//3 #  - beam_shape[1]  # Offset to the right
offset_y = image_shape[0]//3 # - beam_shape[0]  # Offset to the bottom

x, y = np.meshgrid(np.arange(beam_shape[1]), np.arange(beam_shape[0]))
x_offset, y_offset = x + offset_x, y + offset_y

# Ensure the contour stays within the smoothed image bounds
x_offset = np.clip(x_offset, 0, image_shape[1] - 1)
y_offset = np.clip(y_offset, 0, image_shape[0] - 1)



# Plot the original, interpolated, and smoothed images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
#plt.contour(dirty_beam, colors='white', levels=levels)

#plt.colorbar()

plt.subplot(1, 3, 2)
plt.title(f"Interpolated Image ({zoom_factor}x)")
plt.imshow(high_res_image, cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Smoothed Image")
plt.imshow(smoothed_image, cmap='gray')
plt.contour(x_offset, y_offset, high_res_dirt, colors='white', levels=levels)# , label='dirty beam')
#plt.colorbar()

plt.tight_layout()
plt.savefig('delme.png')
#plt.show()
plt.close()


#############################################
######### PLOTTING DIRTY BEAMS 
#############################################
file_dict={
    "H":"image_reconstruction/image_reco/pionier/imageReco_pionier_prior-UD_regul-hyperbolic_pixelscale-0.69_fov-41.4_wavemin-1.5_wavemax-1.8_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits",
    "K":"image_reconstruction/image_reco/gravity/imageReco_gravity_prior-UD_regul-hyperbolic_pixelscale-0.95_fov-57.0_wavemin-2.1_wavemax-2.102_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits",
    "L":"image_reconstruction/image_reco/matisse_L/imageReco_matisse_L_prior-UD_regul-hyperbolic_pixelscale-1.46_fov-87.6_wavemin-3.3_wavemax-3.6_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits",
    "M":"image_reconstruction/image_reco/matisse_M/imageReco_matisse_M_prior-UD_regul-hyperbolic_pixelscale-1.46_fov-87.6_wavemin-4.6_wavemax-4.9_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits",
    "N":"image_reconstruction/image_reco/matisse_N_11.0um/imageReco_matisse_N_11.0um_prior-UD_regul-hyperbolic_pixelscale-4.73_fov-283.8_wavemin-11.0_wavemax-11.5_mu-3000.0_tau-1.0_eta-1_usev2-all_uset3-phi.fits"
    }

label_dict = {"H":"Pionier (H-band)",
    "K":"Gravity (K-band)",
    "L":"Matisse (L-Band)",
    "M":"Matisse (M-Band)",
    "N":"Matisse (N-band)"}



fig, ax = plt.subplots(1, len( file_dict) , figsize=(15,7))
plot_logscale = False
fs = 10
ax[0].set_ylabel('$\Delta$ DEC -> N [mas]',fontsize=fs)

for axx , band in zip( ax.reshape(-1) , file_dict ):
    h = fits.open( file_dict[band] )

    dirty_beam = h['IMAGE-OI DIRTY BEAM'].data

    dx = h[0].header['CDELT1'] #mas * 3600 * 1e3
    x = np.linspace( -h[0].data.shape[0]//2 * dx , h[0].data.shape[0]//2 * dx,  h[0].data.shape[0])

    dy = h[0].header['CDELT2'] #mas * 3600 * 1e3
    y = np.linspace( -h[0].data.shape[1]//2 * dy , h[0].data.shape[1]//2 * dy,  h[0].data.shape[1])

    origin = 'lower' #'upper' # default - see  

    extent = [np.max(x), np.min(x), np.min(y), np.max(y) ]


    ii_dirty =  np.fliplr(  dirty_beam/np.max(dirty_beam) )

    if plot_logscale:
        im = axx.imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin , norm=LogNorm(vmin=0.01, vmax=1) )

    else:
        im = axx.imshow( ii_dirty ,  cmap='Reds',  extent=extent, origin=origin )


    axx.set_xlabel('$\Delta$ RA <- E [mas]',fontsize=fs)
    axx.tick_params(labelsize=fs)

    #axx.text( -x[2], y[int( 0.1*len(y)) ], 'Image Reco.', color='k',fontsize=15)
    #axx.text( -x[2], y[int( 0.2*len(y))], 'RT Pav', color='k',fontsize=15)
    #ax[0].text( x[2], -y[30], r'$\Delta \lambda$ ={:.1f} - {:.1f}$\mu$m'.format( h['IMAGE-OI INPUT PARAM'].header['WAVE_MIN']*1e6  , h['IMAGE-OI INPUT PARAM'].header['WAVE_MAX']*1e6 ) ,fontsize=15, color='k')
    #axx.text( -x[2], y[int( 0.3*len(y))], r'$\chi^2$={}'.format( round( h['IMAGE-OI OUTPUT PARAM'].header['CHISQ'] , 2) ), color='k',fontsize=15)
    axx.text( -x[2], y[int( 0.1*len(y))], f'{label_dict[band]}', color='k',fontsize=fs)
    #axx.text( -x[2], y[int( 0.3*len(y))], 'Dirty beam', color='k',fontsize=fs)

    divider = make_axes_locatable(axx)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = fig.colorbar( im, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    #cbar.set_label( 'Normalized flux', rotation=0,fontsize=15)
    cbar.ax.tick_params(labelsize=fs)  

plt.savefig("dirty_beams_all_instruments.png")
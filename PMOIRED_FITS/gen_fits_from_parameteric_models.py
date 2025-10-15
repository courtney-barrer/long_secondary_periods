
import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits 
import sys
import os 
import glob 
import json
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from fpdf import FPDF
from PIL import Image
import argparse
import pmoired 

sys.path.append(os.path.abspath("/home/rtc/Documents/long_secondary_periods"))
from utilities import plot_util

path_dict = json.load(open('/home/rtc/Documents/long_secondary_periods/paths.json'))
comp_loc = 'ANU' # computer location
# 1. create parameteric image 
# 2. generate synthetic data from observations 
# 3. reconstruct synthetic data 

# 1 . 
# define parameters and create an image from a parameteric model (saved as fits file)
obs_files = glob.glob(path_dict[comp_loc]['data'] + 'pionier/data/*.fits')

save_path = "/home/rtc/Documents/long_secondary_periods/"
fname = "delme1" 

fov = 40
pixelsize = 0.27 #0.6 
ud = 3.3
model = 'dipole'


# another way to test 

oi = pmoired.OI(obs_files)
# for consistent sorting of baseline triangle stations (AB vs BA) 
oi, _ = plot_util.simulate_obs_from_image_reco_dontsort( obs_files, image_file = save_path + fname+'.fits', binning=None, insname=None)

#oif = plot_util.simulate_obs_from_image_reco_FAST( oi, image_file = img_pth , img_pixscl = 0.7 )

if 'pmoired' in model:
    pmoired_model = {"*,ud":ud,"*,f":1.0,"c,f":0.05,"c,x":3.0,"c,y":0.0,"c,ud":0.0} #{"p,f": 1, "r,x": 0, "r,y": 0, "p,ud": 5.462567068944827, "r,diamin": 10.3, "r,diamout": 50.1, "r,f": 1.0, "r,incl": 78.30669619040759, "r,projang": -36.67879133231321}
    
    # show the model (not working)     
    # imWl0 = oi.data[0]['WL'] #[1.5, 1.6, 1.7] # wavelength to plot (um)
    # imPow = 0.3 # using power law brightness scaling (1==linear, 0.5=square root, etc.)
    # oi.show(model=pmoired_model, imFov=fov, imWl0=imWl0, imMax='100', imPow=imPow, showSED=False, cmap='gist_stern')
    
    # make it a fits file 
    d_model = plot_util.create_parametric_prior(pmoired_model = pmoired_model ,fov=fov, pixelsize=pixelsize, save_path=save_path, label=fname)

    _, oif = plot_util.simulate_obs_from_image_reco_dontsort( obs_files, image_file = save_path + fname+'.fits', binning=None, insname=None)

elif 'dipole' in model:

    ### reading things in 
    save_fits_path=save_path
    wavelength=1.6e-6           # m
    Npixels=20
    UD=ud                      # mas
    psi_T_arg=0                 # will be overridden to 0 below (matching your script)
    delta_T= 212
    T_eff=3000
    incl=170                   # deg
    projang=172                 # deg
    l=1
    m=1
    plot=False                   
    save_name=fname+".fits"
    t = 0 # Default time is 0
    nu =  1 / (757 * 24 * 60 * 60)
    psi_T = 0.7 * np.pi * 2

    theta = np.linspace(0, np.pi, 50)
    phi =  np.linspace(0, 2 * np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    
    theta_o = incl #np.deg2rad(incl) #170
    phi_o =  projang #np.deg2rad(projang) #172

    # 1. Calculate local effective temperature
    T_eff_local = plot_util.thermal_oscillation(theta, phi, 0, T_eff, delta_T, l, m, nu, psi_T)

    # 2. Rotate to observer frame
    theta_rot, phi_rot = plot_util.rotate_to_observer_frame(theta, phi, theta_o, phi_o)

    # 3. Project onto observer plane
    projected_intensity = plot_util.project_to_observer_plane(
        theta_rot, phi_rot,
        plot_util.blackbody_intensity(T_eff_local, wavelength),
        grid_size=fov
    )

    #4. Pad intensity image
    pad_factor  = 4
    if 1 : #pad_factor > 1.0:
        pad_size = int(projected_intensity.shape[0] * (pad_factor - 1) // 2)
        projected_intensity = np.pad(
            projected_intensity, pad_size,
            mode='constant', constant_values=0
        )

    # 5. Save FITS file (optional)
    #if save_fits:
    # don't save the file and just pass the fits directly to avoid re-reading it in! 
    name = f'theta-{theta_o:.2f}_phi-{phi_o:.2f}_T-{delta_T:.2f}.fits'
    hh = plot_util.intensity_2_fits(
        projected_intensity,
        dx=1,
        dy=1,
        name=name,
        data_path=save_fits_path,
        #header_dict=header_dict,
        write_file=False
    )

    # 6. Simulate observed data
    #synthetic_file = f"{save_fits_path}{name}"

    # oif = simulate_obs_from_image_reco_FAST(
    #     oi, synthetic_file, img_pixscl = UD_diam / grid_size
    # )
    # dont read the file in an just pass the fits (h) directly!!!! 
    oif = plot_util.simulate_obs_from_image_reco_FAST(
        oi, hh, img_pixscl = ud / fov # UD_diam / grid_size
    )
    #plt.imshow( plot_util.thermal_dipole_image(100,50,500, grid_size=50) ) ;plt.savefig('delme.png')
    # img_pth , projected_intensity = plot_util.create_thermal_dipole_prior(
    #     save_fits_path=save_path,
    #     wavelength=1.6e-7,           # m
    #     Npixels=20,
    #     UD=ud,                      # mas
    #     psi_T_arg=0,                 # will be overridden to 0 below (matching your script)
    #     delta_T=212,
    #     T_eff=3000,
    #     incl=170,                    # deg
    #     projang=172,                 # deg
    #     l=1,
    #     m=1,
    #     plot=False,                   
    #     save_name=fname+".fits"
    # )
    #plt.figure(); plt.imshow(projected_intensity);plt.savefig('delme.png')

else:
    raise UserWarning(f"model {model} not an option")



# 2 . 
# oif is the synthetic data (in pmoired format) generated from the parametric 
# image at the UV points defined by obs_files 
## Use the modified one - I sorted previously by MJD (maybe die to some weird gravity bug- but this puts everytiong out of order ! )


#oi, oif = plot_util.simulate_obs_from_image_reco_dontsort( obs_files, image_file = save_path + fname+'.fits', binning=None, insname=None)

# check lengths make sense
print( f"obs_files len = {len(obs_files)}, oi.data len = {len( oi.data )}, oif.data len = {len( oif.data )}")

# check order of obs_list, oi, and oif match (check MJD)
for i,f in enumerate(obs_files):

    if f not in oi.data[i]['filename'] :
       raise UserWarning(f"WARNING not ordered!! Failed for idx {i}, file {f}")
    # print( oi.data[i]['filename'] )
    # print( f )
    # print("")
    # # data = fits.open( f )
    # # print(f"data idx {i} MJD = {data[0].header["MJD-OBS"]}")

    # ktmp = list(oif.data[i]['OI_T3'].keys())[0]
    # print(f"oi idx {i} MJD = {oi.data[i]['OI_T3'][ktmp]["MJD"][0]}")
    # print(f"oi idx {i} MJD = {oif.data[i]['OI_T3'][ktmp]["MJD"][0]}")

    # # mjd_oi.append(oi.data[i]['OI_T3'][ktmp]["MJD"][0] )
    # # mjd_d.append(data[0].header["MJD-OBS"] )


# Check model vs measured V2 makes sense
plot_util.compare_V2_obs_vs_image_reco( oi, oif , return_data = False,  savefig="delme.png")
# Check model vs measured CP makes sense
plot_util.compare_CP_obs_vs_image_reco( oi, oif , return_data = False, savefig="delme.png")

# check 




# make directory for the data, residual and model
# make directory for the data, residual and model
model_pth = "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/model/"
os.makedirs(model_pth, exist_ok=True)

# write them back into new FITS files (one per input)
for i, f in enumerate(obs_files):
    hdul = fits.open(f)  # read original; we'll write a new file
    #try:
    # -------------------- OI_VIS2 --------------------
    if "OI_VIS2" in hdul:
        v2_rows = oif.data[i]["OI_VIS2"]  # dict of baselines
        # fixed ordering for reproducibility
        bl_keys = v2_rows.keys()
        ucoord = []
        vcoord = []
        v2data = []
        v2err = []
        #v2flag = []

        for b in bl_keys:
            row = v2_rows[b]
            # V2DATA/VIS2ERR are (1, nchan); coords are scalar per row
            v2data.append(np.asarray(row["V2"][0], dtype=np.float64))
            v2err.append(np.asarray(row["EV2"][0], dtype=np.float64))
            ucoord.append(float(np.asarray(row["u"])[0]))
            vcoord.append(float(np.asarray(row["v"])[0]))
            #if "FLAG" in row:
            #    v2flag.append(np.asarray(row["FLAG"][0], dtype=bool))

        hdul["OI_VIS2"].data["UCOORD"] = np.asarray(ucoord, dtype=np.float64)
        hdul["OI_VIS2"].data["VCOORD"] = np.asarray(vcoord, dtype=np.float64)
        hdul["OI_VIS2"].data["VIS2DATA"] = np.asarray(v2data, dtype=np.float64)
        hdul["OI_VIS2"].data["VIS2ERR"] = np.asarray(v2err, dtype=np.float64)
        #if len(v2flag) == len(v2data):
        #    hdul["OI_VIS2"].data["FLAG"] = np.asarray(v2flag, dtype=bool)
    else: 
        raise UserWarning( f"OI_VIS2 not in {f}")
    
    # -------------------- OI_T3 (closure phase) --------------------
    if "OI_T3" in hdul:
        t3_rows = oif.data[i]["OI_T3"]  # dict of triangles
        tri_keys = t3_rows.keys()

        t3amp, t3phi = [], []
        u1_list, v1_list, u2_list, v2_list = [], [], [], []
        #t3flag = []

        for t in tri_keys:
            row = t3_rows[t]
            t3amp.append(np.asarray(row["T3AMP"][0], dtype=np.float64))  # (nchan,)
            t3phi.append(np.asarray(row["T3PHI"][0], dtype=np.float64))  # (nchan,)
            u1_list.append(float(np.asarray(row["u1"])[0]))
            v1_list.append(float(np.asarray(row["v1"])[0]))
            u2_list.append(float(np.asarray(row["u2"])[0]))
            v2_list.append(float(np.asarray(row["v2"])[0]))
            #t3flag.append(np.asarray(row["FLAG"][0], dtype=bool))

        # wrap phases to [-180,180] just in case
        #t3phi_arr = np.asarray(t3phi, dtype=np.float64)
        #t3phi_arr = (t3phi_arr + 180.0) % 360.0 - 180.0

        hdul["OI_T3"].data["T3AMP"]   = np.asarray(t3amp, dtype=np.float64)
        hdul["OI_T3"].data["T3PHI"]   = np.asarray(t3phi, dtype=np.float64) #t3phi_arr
        hdul["OI_T3"].data["U1COORD"] = np.asarray(u1_list, dtype=np.float64)
        hdul["OI_T3"].data["V1COORD"] = np.asarray(v1_list, dtype=np.float64)
        hdul["OI_T3"].data["U2COORD"] = np.asarray(u2_list, dtype=np.float64)
        hdul["OI_T3"].data["V2COORD"] = np.asarray(v2_list, dtype=np.float64)
        #if len(t3flag) == len(t3amp):
        #    hdul["OI_T3"].data["FLAG"] = np.asarray(t3flag, dtype=bool)

        # Do NOT overwrite STA_INDEX unless you rebuild station triplets.
    else: 
        raise UserWarning( f"OI_T3 not in {f}")
    # -------------------- write new file --------------------
    orig_fname = os.path.basename(f)
    out_fname = os.path.join(model_pth, orig_fname.replace(".fits", ".model.fits"))
    hdul.writeto(out_fname, overwrite=True,output_verify='ignore')
    print("[SAVE]", out_fname)
    # # below has weird behaviour 
    # # ---- repackage as PLAIN astropy HDUs (avoid pyoifits verify path) ----
    # new_hdus = [fits.PrimaryHDU(header=hdul[0].header.copy())]
    # for ext in hdul[1:]:
    #     hdr = ext.header.copy()
    #     name = ext.name if hasattr(ext, "name") else hdr.get("EXTNAME", None)
    #     if isinstance(ext, fits.BinTableHDU):
    #         new_hdus.append(fits.BinTableHDU(data=np.array(ext.data), header=hdr, name=name))
    #     elif isinstance(ext, fits.ImageHDU):
    #         new_hdus.append(fits.ImageHDU(data=None if ext.data is None else np.array(ext.data),
    #                                       header=hdr, name=name))
    #     else:
    #         new_hdus.append(fits.BinTableHDU(data=np.array(ext.data), header=hdr, name=name))
    # new = fits.HDUList(new_hdus)

    # out_fname = os.path.join(model_pth, os.path.basename(f).replace(".fits", ".model.fits"))
    # new.writeto(out_fname, overwrite=True)
    # hdul.close()
    # print("[SAVE]", out_fname)

    #finally:
    hdul.close()



# ### MY OWN VERSION 
# model_pth = "/home/rtc/Documents/long_secondary_periods/dipole_residuals_pionier/model/"
# # write them back into the original fits file 
# for i,f in enumerate( obs_files ):
        

#     # V2 
#     data = fits.open( f )
#     vcoord = [] 
#     ucoord = []
#     v2 = []
#     v2err = []
#     flag = []
#     for b in oif.data[i]['OI_VIS2']: 
#         v2.append( oif.data[i]['OI_VIS2'][b]['V2'][0].tolist() )
#         v2err.append(  oif.data[i]['OI_VIS2'][b]['EV2'][0].tolist() ) 
#         ucoord.append(  oif.data[i]['OI_VIS2'][b]['u'][0]  )
#         vcoord.append(  oif.data[i]['OI_VIS2'][b]['v'][0]  )
        
#     data['OI_VIS2'].data['UCOORD'] = np.array( ucoord )
#     data['OI_VIS2'].data['VCOORD'] = np.array( vcoord )
#     data['OI_VIS2'].data['VIS2DATA'] = np.array( v2 )
#     data['OI_VIS2'].data['VIS2ERR'] = np.array( v2err )

#     # T3 
#     """
#     In [83]: oif.data[i]['OI_T3'].keys()
#     Out[83]: dict_keys(['A0G2J2', 'A0G2J3', 'A0J2J3', 'G2J2J3'])

#     In [84]: oif.data[i]['OI_T3']['A0G2J2'].keys()
#     Out[84]: dict_keys(['MJD', 'MJD2', 'FLAG', 'T3AMP', 'ET3AMP', 'T3PHI', 'ET3PHI', 'u1', 'v1', 'u2', 'v2', 'u1/wl', 'v1/wl', 'u2/wl', 'v2/wl', 'B1', 'B2', 'B3', 'Bmin/wl', 'Bmax/wl', 'Bavg/wl', 'formula'])
#     """
    
#     t3amp = []
#     t3phi = []
#     u1 = []
#     u2 = []
#     v1 = []
#     v2 = []
    
#     flag = []
#     for t in oif.data[i]['OI_T3']:
#         t3amp.append( oif.data[i]['OI_T3'][t]['T3AMP'][0].tolist() )
#         t3phi.append( oif.data[i]['OI_T3'][t]['T3PHI'][0].tolist() )
#         u1.append( oif.data[i]['OI_T3'][t]['u1'][0].tolist() )
#         u2.append( oif.data[i]['OI_T3'][t]['u2'][0].tolist() )
#         v1.append( oif.data[i]['OI_T3'][t]['v1'][0].tolist() )
#         v2.append( oif.data[i]['OI_T3'][t]['v2'][0].tolist() )
#         #sta_idx.append( )
#         flag.append( oif.data[i]['OI_T3'][t]['FLAG'][0].tolist() )
    
#     data['OI_T3'].data['T3AMP'] = t3amp
#     data['OI_T3'].data['T3PHI'] = t3phi
#     data['OI_T3'].data['U1COORD'] = u1 
#     data['OI_T3'].data['V1COORD'] = v1 
#     data['OI_T3'].data['U2COORD'] = u2
#     data['OI_T3'].data['V2COORD'] = v2
#     data['OI_T3'].data['FLAG'] = flag

#     # origin 
#     orig_fname = f.split('/')[-1]

#     input_file = model_pth + orig_fname.split(".fits")[0] + ".model.fits"

#     data.writeto( input_file , overwrite = True ) 


#     # base = os.path.basename(orig_path)
#     # stem = base[:-5] if base.lower().endswith(".fits") else base
#     # out_path = os.path.join( subdir  ,f"{stem}.{label}.fits")


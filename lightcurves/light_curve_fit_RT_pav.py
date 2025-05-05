#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:07:07 2023

@author: bcourtne
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.time import Time
from scipy.interpolate import interp1d 

data_path = '/Users/bencb/Documents/long_secondary_periods/lightcurves/LSP_light_curves_data/'
os.chdir(data_path )

#files = os.listdir()

#stars = [xx.split('_aavso')[0] for xx in files]


#plsp = {'RZ_Ari' : 479, 'S_lep' : 880, 'GO_vel' : 501, 'RT_Cnc' : 542}


#%% plot of light curve extended to p109 

light_curve = pd.read_csv('RT_pav_asassn.csv')

light_curve.columns = ['JD', 'camera', 'filter', 'Magnitude', 'mag err', 'flux (mJy)', 'flux err']


fff = 15 
t = np.sort(light_curve['JD']) 
mag = [x for _, x in sorted(zip(light_curve['JD'], light_curve['Magnitude']))]

plt.figure(figsize=(8,5))

RTpav_interp_fn = interp1d(t,mag)
tn = np.linspace(np.min(t),np.max(t),1000)
magn = RTpav_interp_fn(tn)

tn2 = np.linspace(np.min(t),(1 + 7e-4)* np.max(t), 1000)
plt.plot(t,mag,'x', label = 'ASAS data')
plt.plot(tn2, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoidal fit')

plt.axvspan(Time('2022-03-01').jd, Time('2022-09-30').jd, alpha=0.3, color='red',label='P109')

plt.legend(loc='lower left',fontsize=13)

plt.axvline(Time('2022-03-01').jd)
plt.axvline(Time('2022-09-30').jd)

plt.gca().invert_yaxis()
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('time (JD)',fontsize=fff)
plt.ylabel('mag (V)',fontsize=fff)
plt.savefig(data_path + 'RT_pav_lightcurve.png',dpi=300)



#################################
# OTHER LIGHT CURVE PLOTTING (NEED TO INSTALL lightkurve PACKAGE)


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Sep 21 22:45:31 2021

# @author: bcourtne
# """



# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# from astropy.time import Time
# from scipy.interpolate import interp1d 
# import lightkurve as lk

# def get_LSP_phase(date_of_max, P_LSP):
#     #date_of_max = string 'YYYY-MM-DD'
#     # returns phase in april 2022
#     phase = (Time.now().jd + 6*30 - Time(date_of_max).jd) / P_LSP
#     return(phase )


# os.chdir('/Users/bcourtne/Documents/ANU_PhD2/RT_pav/LSP_vlti_proposal_p109/LSP_light_curves_data/')

# #%% Looking at The Kepler, K2, and TESS telescopes https://docs.lightkurve.org/tutorials/1-getting-started/using-light-curve-file-products.html

# search_result = lk.search_lightcurve('RT Pav')


# #%% WISE

# import numpy as np
# import matplotlib.pyplot as plt

# from astropy.io import ascii
# from astropy.table import Table

# import requests
# import os

# #author: Matthew Hill, Hsiang-Chih Hwang
# def get_by_position(ra, dec, radius=2.5):
#     ALLWISE_cat = 'allwise_p3as_mep'
#     NEOWISE_cat = 'neowiser_p1bs_psd'
#     query_url = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query'
#     payload = {
#         'catalog' : ALLWISE_cat,
#         'spatial' : 'cone',
#         'objstr' : ' '.join([str(ra), str(dec)]),
#         'radius' : str(radius),
#         'radunits' : 'arcsec',
#         'outfmt' : '1' 
#     }   
#     r = requests.get(query_url, params=payload)
#     allwise = ascii.read(r.text)
#     payload = {
#         'catalog' : NEOWISE_cat,
#         'spatial' : 'cone',
#         'objstr' : ' '.join([str(ra), str(dec)]),
#         'radius' : str(radius),
#         'radunits' : 'arcsec',
#         'outfmt' : '1',
#         'selcols' : 'ra,dec,sigra,sigdec,sigradec,glon,glat,elon,elat,w1mpro,w1sigmpro,w1snr,w1rchi2,w2mpro,w2sigmpro,w2snr,w2rchi2,rchi2,nb,na,w1sat,w2sat,satnum,cc_flags,det_bit,ph_qual,sso_flg,qual_frame,qi_fact,saa_sep,moon_masked,w1frtr,w2frtr,mjd,allwise_cntr,r_allwise,pa_allwise,n_allwise,w1mpro_allwise,w1sigmpro_allwise,w2mpro_allwise,w2sigmpro_allwise,w3mpro_allwise,w3sigmpro_allwise,w4mpro_allwise,w4sigmpro_allwise'
#     }
#     r = requests.get(query_url, params=payload)
    
#     neowise = ascii.read(r.text, guess=False, format='ipac')

#     return allwise, neowise


# def download_single_data(name, ra, dec, root_path='ipac/', radius=2.5):
#     #ra, dec: in degree
#     #name, ra, dec = row['Name'], row['RAJ2000'], row['DEJ2000']
#     #name = 'J' + ra + dec
#     if root_path[-1] != '/':
#         root_path += '/'
#     if os.path.isfile(root_path+name+'_allwise.ipac') and os.path.isfile(root_path+name+'_neowise.ipac'):
#         pass
#     else:
#         allwise, neowise = get_by_position(ra, dec, radius=radius)
#         allwise.write(root_path+name+'_allwise.ipac', format='ascii.ipac', overwrite=True)
#         neowise.write(root_path+name+'_neowise.ipac', format='ascii.ipac', overwrite=True)

# def get_data_arrays(table, t, mag, magerr):
#     """Get the time series from a potentially masked astropy table"""
#     if table.masked:
#         full_mask = table[t].mask | table[mag].mask | table[magerr].mask
#         t = table[t].data
#         mag = table[mag].data
#         magerr = table[magerr].data

#         t.mask = full_mask
#         mag.mask = full_mask
#         magerr.mask = full_mask

#         return t.compressed(), mag.compressed(), magerr.compressed()

#     else:
#         return table[t].data, table[mag].data, table[magerr].data


# def make_full_lightcurve(allwise, neowise, band):
#     """band = 'w1', 'w2', 'w3', or 'w4' """
#     """Get a combined AllWISE and NEOWISE lightcurve from their Astropy tables"""

#     if band not in ['w1', 'w2', 'w3', 'w4']:
#         raise ValueError('band can only be w1, w2, w3, or w4')

#     t, m, e = get_data_arrays(allwise, 'mjd', band+'mpro_ep', band+'sigmpro_ep')
#     if band in ['w1', 'w2']:
#         t_n, m_n, e_n = get_data_arrays(neowise, 'mjd', band+'mpro', band+'sigmpro')
#         t, m, e = np.concatenate((t, t_n)), np.concatenate((m, m_n)), np.concatenate((e, e_n))

#     t_index = t.argsort()
#     t, m, e = map(lambda e: e[t_index], [t, m, e])

#     return t, m, e


# def make_full_lightcurve_multibands(allwise, neowise, bands=['w1', 'w2']):
#     t, m, e = make_full_lightcurve(allwise, neowise, bands[0])
#     filts = [bands[0] for i in range(len(t))]
#     for band in bands[1:]:
#         t_tmp, m_tmp, e_tmp = make_full_lightcurve(allwise, neowise, band)
#         t = np.concatenate((t, t_tmp))
#         m = np.concatenate((m, m_tmp))
#         e = np.concatenate((e, e_tmp))
#         filts += [band for i in range(len(t_tmp))]
#     return t, m, e, np.array(filts)


# def plot_full_lightcurve(allwise, neowise, band='w1'):
#     plt.plot(allwise['mjd'], allwise[band+'mpro_ep'], '.', label='allwise')
#     plt.plot(neowise['mjd'], neowise[band+'mpro'], '.', label='neowise')
#     plt.legend()

#     plt.xlabel('MJD (day)')

# def plot_part_lightcurve(allwise, neowise, band='w1', time_lapse_threshold=20, mag_shift=2., ms=10.):
#     """
#     time_lapse_threshold: in days, the threshold to group segments of light curves
#     mag_shift: the shift for each segment (in mag) to avoid overlapping
#     """

#     t, mag, mag_err = make_full_lightcurve(allwise, neowise, band)

#     #grouping time
#     time_group = [[0]]
#     for i in range(len(t) - 1):
#         if t[i+1] - t[i] < time_lapse_threshold:
#             time_group[-1].append(i+1)
#         else:
#             time_group.append([i+1])
           
#     for i, g in enumerate(time_group):
#         plt.errorbar(t[g]-t[g[0]], mag[g]+i*mag_shift, mag_err[g], fmt='o', ms=ms)

#     plt.gca().invert_yaxis()
#     plt.xlabel('MJD (day)')

# def plot_individual_lightcurve(allwise, neowise, filename, band='w1', time_lapse_threshold=3):
#     """
#     time_lapse_threshold: in days, the threshold to group segments of light curves
#     """

#     t, mag, mag_err = make_full_lightcurve(allwise, neowise, band)

#     #grouping time
#     time_group = [[0]]
#     for i in range(len(t) - 1):
#         if t[i+1] - t[i] < time_lapse_threshold:
#             time_group[-1].append(i+1)
#         else:
#             time_group.append([i+1])
           
#     for i, g in enumerate(time_group):
#         plt.figure()
#         plt.errorbar(t[g]-t[g[0]], mag[g], mag_err[g], fmt='o')

#         plt.gca().invert_yaxis()
#         plt.xlabel('MJD (day)')
#         plt.savefig(filename + '_%d.pdf' %(i))


# def cntr_to_source_id(cntr):
#     cntr = str(cntr)

#     #fill leanding 0s
#     if len(cntr) < 19:
#         num_leading_zeros = 19 - len(cntr)
#         cntr = '0'*num_leading_zeros + cntr

#     pm = 'p'
#     if cntr[4] == '0':
#         pm = 'm'

#     t = chr(96+int(cntr[8:10]))
    
#     #return '%04d%s%03d_%sc%02d-%06d' %(cntr[0:4], pm, cntr[5:8], t, cntr[11:13], cntr[13:19])
#     return '%s%s%s_%cc%s-%s' %(cntr[0:4], pm, cntr[5:8], t, cntr[11:13], cntr[13:19])

# def only_good_data(allwise, neowise):
#     """
#     Select good-quality data. The criteria include:
#     - matching the all-wise ID

#     To be done:
#     - deal with multiple cntr
#     """
    
#     cntr_list = []
#     for data in neowise:
#         #print data['allwise_cntr'] 
#         if data['allwise_cntr'] not in cntr_list and data['allwise_cntr']>10.:
#             cntr_list.append(data['allwise_cntr'])

#     if len(cntr_list) >= 2:
        
#         print('multiple cntr:')
#         print(cntr_list)
#         return 0, 0
    
#     if len(cntr_list) == 0:
#         print('no cntr')
#         return 0, 0
    
#     cntr = cntr_list[0]

#     source_id = cntr_to_source_id(cntr)
    
#     allwise = allwise[
#         (allwise['source_id_mf'] == source_id) * 
#         (allwise['saa_sep'] > 0.) * 
#         (allwise['moon_masked'] == '0000') *
#         (allwise['qi_fact'] > 0.9)
#     ]
    
#     #old version
#     neowise = neowise[
#         (neowise['qual_frame'] > 0.)
#     ]


#     return allwise, neowise


# def only_good_data_v1(allwise, neowise):
#     """
#     Select good-quality data. The criteria include:
#     - matching the all-wise ID

#     To be done:
#     - deal with multiple cntr
#     """
    
#     cntr_list = []
#     for data in neowise:
#         #print data['allwise_cntr'] 
#         if data['allwise_cntr'] not in cntr_list and data['allwise_cntr']>10.:
#             cntr_list.append(data['allwise_cntr'])

#     if len(cntr_list) >= 2:
        
#         print('multiple cntr:')
#         print(cntr_list)
#         return 0, 0
    
#     if len(cntr_list) == 0:
#         print('no cntr')
#         return 0, 0
    
#     cntr = cntr_list[0]

#     source_id = cntr_to_source_id(cntr)
    
#     allwise = allwise[
#         (allwise['source_id_mf'] == source_id) * 
#         (allwise['saa_sep'] > 0.) * 
#         (allwise['moon_masked'] == '0000') *
#         (allwise['qi_fact'] > 0.9)
#     ]
    
#     #old version
#     #neowise = neowise[
#     #    (neowise['qual_frame'] > 0.)
#     #]

#     #new version
#     neowise = neowise[
#         (neowise['qual_frame'] > 0.) *
#         (neowise['qi_fact'] > 0.9) *
#         (neowise['saa_sep'] > 0) *
#         (neowise['moon_masked'] == '00')
#     ]

#     return allwise, neowise



# #  RT PAV RA, DEC J2000
# ra,dec = 279.1270174, -69.88505075

# download_single_data(name='RT_Pav', ra=ra, dec=dec, root_path='ipac/', radius=2.)

# allwise = ascii.read('ipac/RT_Pav_allwise.ipac', format='ipac')
# neowise = ascii.read('ipac/RT_Pav_neowise.ipac', format='ipac')

# plt.figure()
# plt.plot(allwise['mjd'], allwise['w1mpro_ep'], '.', label='AllWISE')
# plt.plot(neowise['mjd'], neowise['w1mpro'], '.', label='NeoWISE')
# plt.gca().invert_yaxis()
# plt.legend()
# plt.xlabel('MJD (day)')
# plt.ylabel('W1')
# plt.show()


# plt.figure()
# plt.plot(allwise['mjd'], allwise['w2mpro_ep'], '.', label='AllWISE')
# plt.plot(neowise['mjd'], neowise['w2mpro'], '.', label='NeoWISE')
# plt.gca().invert_yaxis()
# plt.legend()
# plt.xlabel('MJD (day)')
# plt.ylabel('W1')
# plt.show()

# # obs
# epochs = np.unique( 10**np.round( np.log10( neowise['mjd'] ) , 4) )
# agg_w = {}
# for w in ['w1mpro','w2mpro','w3mpro_allwise','w4mpro_allwise']:
#     agg_w[w]=[]
#     for i in range(len(epochs )):
#         filt_tmp = ((epochs[i] == 10**np.round( np.log10( neowise['mjd'] ) , 4))) 
#         agg_w[w].append( np.median( neowise[w][filt_tmp] ))

# plt.figure()
# #plt.plot(allwise['mjd'], allwise['w1mpro_ep'], '.', label='W1 AllWISE')
# plt.plot(neowise['mjd'], neowise['w1mpro'], '.', label='W1 NeoWISE',color='orange',alpha=0.3)
# plt.plot( epochs, agg_w['w1mpro'],'-',color='orange') ;
# #plt.plot(allwise['mjd'], allwise['w2mpro_ep'], '.', label='W2 AllWISE')
# plt.plot(neowise['mjd'], neowise['w2mpro'], '.', label='W2 NeoWISE',color='r',alpha=0.3)
# plt.plot(epochs, agg_w['w2mpro'],'-',color='r')
# plt.gca().invert_yaxis()
# plt.legend()
# plt.xlabel('MJD (day)')



# RTpav_interp_fn = interp1d(t,mag)
# tn = np.linspace(np.min(t),np.max(t),1000)
# magn = RTpav_interp_fn(tn)

# tn2 = np.linspace(np.min(t),(1 + 7e-4)* np.max(t), 1000)
# plt.plot(t- 2400000.5,mag,'x', label = 'ASAS data')
# plt.plot(tn2- 2400000.5, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoid fit')

# plt.axvspan(Time('2022-03-01').mjd, Time('2022-09-30').mjd, alpha=0.3, color='red',label='P109')

# plt.legend(loc='lower left',fontsize=13)

# plt.axvline(Time('2022-03-01').mjd)
# plt.axvline(Time('2022-09-30').mjd)

# plt.gca().invert_yaxis()
# plt.xlabel('time (JD)',fontsize=fff)
# plt.ylabel('Mag (V)',fontsize=fff)
# #plt.ylim([-2,2])
# plt.show()



# #or 
# plt.figure()
# plt.plot( pd.Series( epochs ).rolling(2).mean(), pd.Series(agg_w['w2mpro']).rolling(2).mean() ,'.')
# plt.plot( epochs, agg_w['w1mpro']) ; plt.plot(epochs, agg_w['w2mpro'])


# #or wrap 
# plt.figure()
# plt.plot( np.mod(  epochs, 757) , agg_w['w1mpro'],'.')
# #%% 


# files = os.listdir() # [x if '_aavso' in x else None for x in os.listdir()]
# stars = [xx.split('_aavso')[0] for xx in files]


# plsp = {'RZ_Ari' : 479, 'S_lep' : 880, 'GO_vel' : 501, 'RT_Cnc' : 542}


# #set up plot

# fig, ax = plt.subplots(len(files),3,sharex=False, sharey=False,figsize=(20,20))


# for s_indx, file in enumerate(files):
    
#     if '_aavso' in file:
#         #light_curve = pd.read_csv('/Users/bcourtne/Documents/ANU-PhD/LSP_vlti_proposal_p109/LSP_light_curves_data/RS-CrB_aavsodata.txt')
#         light_curve = pd.read_csv(file)
        
        
#         #filtering, detecting and folding LSP light curve (currently not filtering other pulsations)
        
#         band = 'Vis.'
#         i2keep = [] #indices to keep 
#         for i in light_curve.index:
            
            
#             try:
#                 float(light_curve['Magnitude'].loc[i])    
                
#                 if light_curve.loc[i]['Band']==band:
                    
#                     i2keep.append(i)
                    
#             except: 
                
#                 None
            
#         #what is mean sampling in light curve 
#         np.mean(np.diff(light_curve.loc[i2keep]['JD']))  #8.02 days
        
#         #interpolate onto even grid 
#         lc_interp_fn = interp1d(light_curve.loc[i2keep]['JD'].values, light_curve.loc[i2keep]['Magnitude'].values.astype(float))
        
#         ts_grid = np.linspace(min(light_curve.loc[i2keep]['JD']),max(light_curve.loc[i2keep]['JD']), 4000)
#         Fs = 1/np.diff(ts_grid)[0] #1/days
#         lc_interp = lc_interp_fn(ts_grid)
        
#         #mask for interpolated data if too far away from a real sample ( 30 day threshold )
#         orig_ts_grid = light_curve.loc[i2keep]['JD'].values
#         mask = np.array([np.nan if (np.min(abs(ttt - orig_ts_grid)) > 30)  else 1 for ttt in ts_grid])
        
    
#         ts_grid = Time(ts_grid,format='jd')
        
#         #lets look at the ts from 1992 (ts_grid.iso[2500:])
        
#         ts_window = (ts_grid.value > Time('1992-04-12').jd) #& (ts_grid.value < Time('2017-04-12').jd)
        
#         #plt.figure()
#         ax[s_indx,0].plot( ts_grid[ts_window ].value, lc_interp[ts_window ])
#         ax[s_indx,0].set_ylabel(stars[s_indx], fontsize=20)
#         # take fft
#         lc_fft = np.fft.fft(lc_interp[ts_window])
#         lc_freq = np.fft.fftfreq( len(lc_interp[ts_window]) , d = 1/Fs )
        
#         # plot the fourier transform (freq units are 1/days)
#         #plt.figure()
#         ax[s_indx,1].loglog(lc_freq[:len(lc_fft)//2], abs(lc_fft)[:len(lc_fft)//2])
#         ax[s_indx,1].set_xlabel('1/days')
#         ax[s_indx,1].set_ylabel('amplitude')
#         ax[s_indx,1].axvline()
        
#         # detect peak 
        
#         peak_search_window = (lc_freq > 1/3000) & (lc_freq < 1/300)  #look for peak between 300 - 3000 days
        
#         #i_peak = np.argmax(lc_freq[peak_search_window])
#         i_peak = np.argmax(abs(lc_fft[peak_search_window]))
#         #COULD ALSO DO SNR threshold 
#         LSP_period = 1/lc_freq[peak_search_window][i_peak]  #days
        
#         ax[s_indx,1].axvline(lc_freq[peak_search_window][i_peak],linestyle='--',alpha=1)
        
#         if stars[s_indx] in plsp.keys():
#             ax[s_indx,1].axvline( 1/int(plsp[stars[s_indx]] ) , color='red') 
        
#         ax[s_indx,1].text(1e-3,1e2,LSP_period )
#         ax[s_indx,1].text(1e-3,1e3,plsp ) 
        
#         #but now do folded light curve 
        
#         #p_sample = 2 * int(round(LSP_period * Fs))
#         if stars[s_indx] in plsp.keys():
#             p_sample = 2 * int(plsp[stars[s_indx]] * Fs)
#         else: 
#             p_sample = 2 * int(round(LSP_period * Fs))
            
#         LSP_blocs = []
#         #plt.figure()
#         phase_tmp = np.linspace(0,2,p_sample)
#         for i in range(len(lc_interp[ts_window])//p_sample):
            
#             #
#             bloc_tmp = mask[ts_window][i*p_sample : (i+1)*p_sample] * lc_interp[ts_window][i*p_sample : (i+1)*p_sample]
#             bloc_norm = ( bloc_tmp - np.nanmean(bloc_tmp) ) / np.nanstd(bloc_tmp)
            
#             LSP_blocs.append( bloc_norm )
#             ax[s_indx,2].plot(phase_tmp , bloc_norm,'-',alpha=0.1,color='k')
            
#         ax[s_indx,2].plot(phase_tmp , np.nanmean(np.array(LSP_blocs),axis=0),color='k',linestyle='--')
        
        
    
# #%% from ASASSN

# fig, ax = plt.subplots(2,3,sharex=False, sharey=False,figsize=(20,20))

# s_indx =0

# files = [x if 'asassn' else None in x for x in os.listdir()]
# stars = [xx.split('_asassn')[0] for xx in files]


# plsp = {'RZ_Ari' : 479, 'S_lep' : 880, 'GO_vel' : 501, 'RT_Cnc' : 542, 'RT_pav':757}



# light_curve = pd.read_csv('RT_pav_asassn.csv')

# light_curve.columns = ['JD', 'camera', 'filter', 'Magnitude', 'mag err', 'flux (mJy)', 'flux err']

# i2keep = light_curve.index

# #interpolate onto even grid 
# lc_interp_fn = interp1d(light_curve.loc[i2keep]['JD'].values, light_curve.loc[i2keep]['Magnitude'].values.astype(float))

# ts_grid = np.linspace(min(light_curve.loc[i2keep]['JD']),max(light_curve.loc[i2keep]['JD']), 4000)
# Fs = 1/np.diff(ts_grid)[0] #1/days
# lc_interp = lc_interp_fn(ts_grid)

# #mask for interpolated data if too far away from a real sample ( 30 day threshold )
# orig_ts_grid = light_curve.loc[i2keep]['JD'].values
# mask = np.array([np.nan if (np.min(abs(ttt - orig_ts_grid)) > 30)  else 1 for ttt in ts_grid])


# ts_grid = Time(ts_grid,format='jd')



# #lets look at the ts from 1992 (ts_grid.iso[2500:])
 
# ts_window = (ts_grid.value > Time('2000-04-12').jd) #& (ts_grid.value < Time('2017-04-12').jd)

# #plt.figure()
# ax[s_indx,0].plot( ts_grid[ts_window ].value, lc_interp[ts_window ])
# ax[s_indx,0].set_ylabel(stars[s_indx], fontsize=20)
# # take fft
# lc_fft = np.fft.fft(lc_interp[ts_window])
# lc_freq = np.fft.fftfreq( len(lc_interp[ts_window]) , d = 1/Fs )

# # plot the fourier transform (freq units are 1/days)
# #plt.figure()
# ax[s_indx,1].loglog(lc_freq[:len(lc_fft)//2], abs(lc_fft)[:len(lc_fft)//2])
# ax[s_indx,1].set_xlabel('1/days')
# ax[s_indx,1].set_ylabel('amplitude')
# ax[s_indx,1].axvline()

# # detect peak 

# peak_search_window = (lc_freq > 1/3000) & (lc_freq < 1/300)  #look for peak between 300 - 3000 days

# #i_peak = np.argmax(lc_freq[peak_search_window])
# i_peak = np.argmax(abs(lc_fft[peak_search_window]))
# #COULD ALSO DO SNR threshold 
# LSP_period = 1/lc_freq[peak_search_window][i_peak]  #days

# ax[s_indx,1].axvline(lc_freq[peak_search_window][i_peak],linestyle='--',alpha=1)

# if stars[s_indx] in plsp.keys():
#     ax[s_indx,1].axvline( 1/int(plsp[stars[s_indx]] ) , color='red') 

# ax[s_indx,1].text(1e-3,1e2,LSP_period )
# ax[s_indx,1].text(1e-3,1e3,plsp ) 

# #but now do folded light curve 

# #p_sample = 2 * int(round(LSP_period * Fs))
# if stars[s_indx] in plsp.keys():
#     p_sample = int(plsp[stars[s_indx]] * Fs)
# else: 
#     p_sample = int(round(LSP_period * Fs))
    
# LSP_blocs = []
# #plt.figure()
# phase_tmp = np.linspace(0,2,p_sample)
# for i in range(len(lc_interp[ts_window])//p_sample):
    
#     #
#     bloc_tmp = mask[ts_window][i*p_sample : (i+1)*p_sample] * lc_interp[ts_window][i*p_sample : (i+1)*p_sample]
#     bloc_norm = ( bloc_tmp - np.nanmean(bloc_tmp) ) / np.nanstd(bloc_tmp)
    
#     LSP_blocs.append( bloc_norm )
#     ax[s_indx,2].plot(phase_tmp , bloc_norm,'-',alpha=0.1,color='k')
    
# ax[s_indx,2].plot(phase_tmp , np.nanmean(np.array(LSP_blocs),axis=0),color='k',linestyle='--')

   



# #%% plot of light curve extended to p109 

# fff = 15 
# t = np.sort(light_curve['JD']) 
# mag = [x for _, x in sorted(zip(light_curve['JD'], light_curve['Magnitude']))]

# plt.figure()

# RTpav_interp_fn = interp1d(t,mag)
# tn = np.linspace(np.min(t),np.max(t),1000)
# magn = RTpav_interp_fn(tn)

# tn2 = np.linspace(np.min(t),(1 + 7e-4)* np.max(t), 1000)
# plt.plot(t,mag,'x', label = 'ASAS data')
# plt.plot(tn2, 0.7 * np.sin(-2 * np.pi/757 * tn2 + 1.8*np.pi/3) + 9, color='k', label = 'LSP sinusoid fit')

# plt.axvspan(Time('2022-03-01').jd, Time('2022-09-30').jd, alpha=0.3, color='red',label='P109')

# plt.legend(loc='lower left',fontsize=13)

# plt.axvline(Time('2022-03-01').jd)
# plt.axvline(Time('2022-09-30').jd)

# plt.gca().invert_yaxis()
# plt.xlabel('time (JD)',fontsize=fff)
# plt.ylabel('Mag (V)',fontsize=fff)
# #plt.savefig('/Users/bcourtne/Documents/ANU-PhD/LSP_vlti_proposal_p109/RT_pav_lightcurve.png')

# #win = 120
# #plt.plot(pd.Series(tn).rolling(win).mean(),pd.Series(magn).rolling(win).mean())

# #plt.fill_betweenx(magn, tn2, where = (tn > Time('2022-03-01').jd) & (tn< x2=Time('2022-09-01').jd) )








# light_curve['Magnitude']
# plt.plot(light_curve['JD']-2.45e6, light_curve['Magnitude'],'.')
# plt.plot( (light_curve['JD']-2.45e6).rolling(10).mean(), light_curve['Magnitude'].rolling(10).mean())
# plt.gca().invert_yaxis()








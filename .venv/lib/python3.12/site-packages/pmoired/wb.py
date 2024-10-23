# -- wide binaries

import numpy as np
import matplotlib.pyplot as plt

def wideOI(oi, rmin=None, rmax=None, obs=None):
    """
    find wide band binaries as chromatic variations of visibility

    obs = ['OI_VIS2', 'OI_VIS']

    """
    if obs is None:
        obs = [o for o in ['OI_VIS2', 'OI_VIS'] if o in oi]
    # -- angular resolution
    if 'OI_VIS2' in obs:
        step = 180*3600*1000e-6/np.pi/max([np.max(oi['OI_VIS2'][k]['B/wl']) for k in oi['OI_VIS2']])
    elif 'OI_VIS' in obs:
        step = 180*3600*1000e-6/np.pi/max([np.max(oi['OI_VIS'][k]['B/wl']) for k in oi['OI_VIS']])
    # -- spectral resolution
    R = np.mean(oi['WL']/oi['dWL'])

    if rmin is None:
        rmin = 1.0*step
    if rmax is None:
        rmax = R*step/2

    R = np.linspace(rmin, rmax, int((rmax-rmin)/step))
    res = {}

    plt.close(0)
    plt.figure(0)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, aspect='equal')

    for o in obs:
        if o=='OI_VIS2':
            m = 'V2'
        elif o=='OI_VIS':
            m = '|V|'
        for k in oi[o].keys():
            for i,mjd in enumerate(oi[o][k]['MJD']):
                tmp = []
                mask = ~oi[o][k]['FLAG']
                mask *= (np.abs(oi[o][k]['MJD']-mjd)<1e-6)[:,None]
                # -- compute power of FT of visibility signal
                for r in R:
                    # -- wave
                    w = np.cos(2*np.pi*oi[o][k]['B/wl'][mask]*r*np.pi/180/3600/1000/1e-6) + \
                       1j*np.sin(2*np.pi*oi[o][k]['B/wl'][mask]*r*np.pi/180/3600/1000/1e-6)
                    # -- signal: 2*(V-mean)/mean
                    s = 2*(oi[o][k][m][mask]-np.mean(oi[o][k][m][mask]))/np.mean(oi[o][k][m][mask])
                    # -- scalar correlation
                    tmp.append(np.abs(np.mean(s*w)))
                tmp = np.array(tmp)
                # -- find maximum in correlation (parabolic fit)
                i = np.argmax(tmp)
                c = np.polyfit(R[max(i-2, 0):min(i+2, len(R)-1)],
                               tmp[max(i-2, 0):min(i+2, len(R)-1)], 2)
                # -- PA of the baseline
                PA = np.mean(oi[o][k]['PA'][mask])
                # -- save result
                key = o+';'+k+';%.3f'%PA
                res[key] = -c[1]/(2*c[0])
                ax1.plot(R, tmp, label=key)

                # -- show projected position
                x = np.array([-1, 1, -1, 1])*np.sqrt(rmax**2-res[key]**2)
                y = np.array([1, 1, -1, -1])*res[key]
                x, y = np.cos(PA*np.pi/180)*x + np.sin(PA*np.pi/180)*y, \
                       -np.sin(PA*np.pi/180)*x + np.cos(PA*np.pi/180)*y
                plt.plot(x[:2], y[:2], '-', color='k', alpha=min(max(tmp), 1))
                plt.plot(x[2:], y[2:], '-', color='k', alpha=min(max(tmp), 1))

    ax2.invert_xaxis()
    ax1.legend(fontsize=5)
    ax2.set_xlabel('x = RA separation (mas)')
    ax2.set_ylabel('y = Dec separation (mas)')
    plt.tight_layout()
    return

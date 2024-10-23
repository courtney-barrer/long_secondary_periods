from multiprocessing import Pool
import pmoired
import numpy as np
from matplotlib import pyplot as plt

def _chi2map2D(oidata, PARAM):
    return [np.mean(pmoired.oimodels.residualsOI(oidata, p)**2) for p in PARAM]

def chi2map2D(oi, expl, model=None, constrain=None, verbose=True):
    """
    oi: pmoired OI object
    expl: {'grid':{'param1':(lower bound, upper bound, step), 'param2':(lower bound, upper bound, step)}}
    model: dict model (will try best fit by default)
    constrain: see gridFit
    """
    oi._merged = pmoired.oifits.mergeOI(oi.data, collapse=True, verbose=False)

    if model is None:
        model = oi.bestfit['best'].copy()
    # -- only for "grid"
    if not (type(expl) is dict and  'grid' in expl):
        raise Exception("'expl' should be {'grid':{}}")
    kx = sorted(expl['grid'].keys())[0]
    ky = sorted(expl['grid'].keys())[1]
    nx = int((expl['grid'][kx][1]-expl['grid'][kx][0])/expl['grid'][kx][2] + 1)
    ny = int((expl['grid'][ky][1]-expl['grid'][ky][0])/expl['grid'][ky][2] + 1)
    N = nx*ny
    if verbose:
        print(N, 'grid points')
    X = np.linspace(expl['grid'][kx][0], expl['grid'][kx][1], nx)
    Y = np.linspace(expl['grid'][ky][0], expl['grid'][ky][1], ny)
    X, Y = np.meshgrid(X, Y)
    X, Y = X.flatten(), Y.flatten()
    if constrain is None:
        constrain = []
    PARAM = []
    for i in range(len(X)):
        model[kx] = X[i]
        model[ky] = Y[i]
        res = [0]
        for p in constrain:
            form = p[0]
            val = str(p[2])
            for i in range(3):
                for k in model.keys():
                    if k in form:
                        form = form.replace(k, '('+str(model[k])+')')
                    if k in val:
                        val = val.replace(k, '('+str(model[k])+')')
            # -- residual
            resi = '0'
            if len(p)==3:
                resi = '('+form+'-'+str(val)+')'
            elif len(p)==4:
                resi = '('+form+'-'+str(val)+')/abs('+str(p[3])+')'
            if p[1]=='<' or p[1]=='<=' or p[1]=='>' or p[1]=='>=':
                resi = '%s if 0'%resi+p[1]+'%s else 0'%resi
            try:
                res.append(eval(resi))
            except:
                print('WARNING: could not compute constraint "'+resi+'"')
        if all(np.array(res)==0):
            PARAM.append(model.copy())
    if len(PARAM)<N and verbose:
        print('%d/%d'%(N-len(PARAM), N), 'grid point%s not within constraints'%('' if (N-len(PARAM))==1 else 's'))

    if True:
        pool = Pool()
        _chi2 = []
        N = 8
        for i in range(N):
            _chi2.append(pool.apply_async(_chi2map2D, (oi._merged, PARAM[i::N],)))
        pool.close()
        pool.join()
        _chi2 = [r.get(timeout=1) for r in _chi2]
        chi2 = np.zeros(len(PARAM))
        for i in range(N):
            chi2[i::N] = np.array(_chi2[i])
    else:
        # -- single thread
        chi2 = _chi2map2D(oi._merged, PARAM)

    X, Y = [p[kx] for p in PARAM], [p[ky] for p in PARAM]
    fig = 123
    plt.close(fig)
    plt.figure(fig, figsize=(pmoired.FIG_MAX_HEIGHT, pmoired.FIG_MAX_HEIGHT))
    #plt.subplot(111, aspect='equal')
    X, Y = [p[kx] for p in PARAM], [p[ky] for p in PARAM]
    plt.scatter(X, Y, c=np.log10(chi2), s=10, cmap='gist_stern')
    plt.colorbar(label=r'log$_{10}$($\chi2$)')
    return X, Y, chi2

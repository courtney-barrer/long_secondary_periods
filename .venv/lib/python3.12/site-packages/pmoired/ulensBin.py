import os
import pickle
import time
import numpy as np
from matplotlib import pyplot
import multiprocessing

this_dir, this_filename = os.path.split(__file__)

directory = 'binlens_images'
directory = os.path.join(this_dir, directory)
exe = 'binlen_images_oneepoch.out'
txt = 'parameters.txt'
#txt = os.path.join(directory, txt)
assert os.path.exists(directory), 'directory "'+directory+'" not found'
if not os.path.exists(os.path.join(directory, exe)):
    if os.path.exists(os.path.join(directory, 'compile.sh')):
        print(this_filename, '> compiling', exe, end='...')
        os.system('cd '+directory+'; ./compile.sh')
        assert os.path.exists(os.path.join(directory, exe)), 'compilation failed'
        print('OK')
    else:
        assert False, 'cannot find compile script'
assert os.path.exists(os.path.join(directory, exe)), 'executable "'+exe+'" not found in "'+directory+'"'

debug = False
def clean():
    global directory, exe
    files = os.listdir(directory)
    for f in files:
        if f.startswith('img.') and (f.endswith('.pickle') or
                                     f.endswith('.txt')):
            os.system('rm '+os.path.join(directory, f))
        if f.startswith('_tmp'):
            os.system('rm -rf '+os.path.join(directory, f))
        if f in [txt]:
            os.system('rm '+os.path.join(directory, f))
    return

def computeSparseParam(MJD, param, parallel=True):
    images = computeImages(MJD, param, parallel=parallel)
    fluxes = []
    mjd = []
    for k in images:
        mjd.append(50000.5+float(os.path.basename(images[k]).split('img.')[1].split('.pickle')[0]))
        with open(images[k], 'rb') as f:
            tmp = pickle.load(f)
        fluxes.append(np.sum(tmp['I']))

    fluxes = np.array(fluxes)
    fluxes = fluxes[np.argsort(mjd)]
    mjd = sorted(mjd)
    s = 'np.interp($MJD, %s, %s)'%('['+','.join(['%.2f'%x for x in mjd])+']',
                                   '['+','.join(['%.3f'%x for x in fluxes])+']')
    return {'sparse':images, 'spectrum(mjd)':s}

def computeImages(MJD, param, parallel=False, debug=debug, directory=directory):
    global txt, exe
    if parallel:
        preptime = 0
        t0 = time.time()
        nproc = min(multiprocessing.cpu_count(), len(MJD))
        dirs = []
        for i in range(nproc):
            dirs.append(os.path.join(directory, '_tmp%03d'%i))
            os.mkdir(dirs[-1])
            os.system('cp '+os.path.join(directory, exe)+' '+os.path.join(dirs[-1], exe))

        Pool = multiprocessing.Pool()
        res = []
        for i,mjd in enumerate(MJD):
            res.append(Pool.apply_async(computeImages, ([mjd], param, ),
                {'parallel':False, 'debug':False,
                'directory':dirs[i%len(dirs)]}))
        preptime += time.time()-t0
        Pool.close()
        Pool.join()
        t1 = time.time()
        tmp = [r.get(timeout=1) for r in res]
        res = {}
        for t in tmp:
            res.update(t)
        for d in dirs:
            os.system('mv '+os.path.join(d, '*.pickle')+' '+directory)
            os.system('rm -rf '+d)
        for k in res:
            for d in dirs:
                if os.path.basename(d) in res[k]:
                    res[k] = res[k].replace(os.path.basename(d), '')
                    res[k] = res[k].replace('//', '/')
        preptime += time.time()-t1
        print('preptime: %.2fs'%preptime)
        return res

    cmd = 'cd '+directory+'; ./'+exe
    if len(MJD)==1:
        print(directory, MJD)
    t0 = time.time()
    cols = ['t0', 'u0', 'tE', 'b', 'q', 'theta', 'rhos']
    if not 't0' in param and 'mjd0' in param:
        param['t0'] = param['mjd0']-50000.5
    res = {}
    if debug:
        print('D> -- computeImage starts')
    for mjd in MJD:
        t = round(mjd -50000.5, 2)
        img = 'img.%.2f.txt'%t
        if debug:
            print('D> computing for epoch', img)
        t1 = time.time()
        with open(os.path.join(directory, txt), 'w') as f:
            tmp = ' '.join(['%f'%param[k] for k in cols])
            tmp += ' %.2f'%t
            f.write(tmp+'\n')
            if debug:
                print('D>', txt, '=', '"'+tmp+'"')
        if debug:
            print('D> executing "'+cmd+'"')
        os.system(cmd)
        if debug:
            print('D> image computed in %.1fs'%(time.time()-t1))
        t1 = time.time()
        imageTxt2pickle(os.path.join(directory, img),
                        rhos=param['rhos'], debug=debug)
        if debug:
            print('D> conversion in %.1fs'%(time.time()-t1))
        os.system('rm '+os.path.join(directory, img))
        res[round(mjd, 2)] = os.path.join(directory, img.replace('.txt', '.pickle'))
    if debug:
        print('D> -- computeImage done in %.1fs'%(time.time()-t0))
    return res

def imageTxt2pickle(filename, debug=debug, rhos=None):
    tinit = time.time()
    data = []
    with open(filename) as f:
        for l in f.readlines():
            data.append((np.double(l.split()[0]), np.double(l.split()[1])))
    X, Y = np.array([d[0] for d in data]), np.array([d[1] for d in data])
    x0 = np.mean(X)
    y0 = np.mean(Y)
    dx = 0.005 # original file

    dx = 0.005*4 # binned

    print('X:', min(X), max(X))
    print('Y:', min(Y), max(Y))


    nx, ny = int(X.ptp()/dx)+1, int(Y.ptp()/dx)+1

    x = np.linspace(X.min(), X.max(), nx)
    y = np.linspace(Y.min(), Y.max(), ny)

    _X, _Y = np.meshgrid(x, y)
    _I = np.zeros(_X.shape)
    t = time.time()
    _i = np.int_([(d[1]-y.min())/dx for d in data])
    _j = np.int_([(d[0]-x.min())/dx for d in data])
    for k,d in enumerate(data):
        _I[_i[k], _j[k]] += 1

    # -- generate sparse image
    sparse = {'x':[], 'y':[], 'I':[]}
    for i in range(ny):
        for j in range(nx):
            if _I[i,j]>0:
                sparse['x'].append(_X[i,j])
                sparse['y'].append(_Y[i,j])
                sparse['I'].append(_I[i,j])
    for k in sparse:
        sparse[k] = np.array(sparse[k])

    if not rhos is None:
        # -- normalise fluxes to flux of source
        # -- not sure how to justify the "0.002"!
        fscale = (0.002/rhos)**2/np.pi
        sparse['I'] *= fscale
    fileb = filename.replace('.txt', '.pickle')
    if debug:
        print('D> saving in', fileb)
    with open(fileb, 'wb') as f:
        pickle.dump(sparse, f)
    return fileb

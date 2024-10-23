import Vmodels # f2py --opt='-O3' -c Vmodels.f95 -m Vmodels

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import time

# testing numerical integration of images?
testImage = False

# -- uv plane (in m)
nuv = 31
b = np.linspace(10, 330, nuv)
pa = np.linspace(0, 6*np.pi, nuv)
u = b*np.cos(pa)
v = b*np.sin(pa)

# -- wavelength vector (in um)
nwl = 1001
wl = np.linspace(2., 2.5, nwl)

print('running for %d u,v points, and %d wavelength'%(nuv, nwl))

# -- radial profile (r in mas)
diamin, diamout = 2.34, 3.14
uniform = False

if uniform:
	Fr = lambda r: r**0 # uniform
else:
	Fr = lambda r: 1-np.abs(2*(r-0.5*(diamin/2+diamout/2))/np.ptp(r))**1
nr = 103
#r = np.linspace(diamin/2, diamout/2, nr)
r = np.logspace(np.log10(diamin/2), np.log10(diamout/2), nr)
I = Fr(r)

# -- image
x = np.linspace(-0.6*diamout, 0.6*diamout, 2*(nr//2)+1)
y = np.linspace(-0.6*diamout, 0.6*diamout, 2*(nr//2)+1)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2+Y**2)
Image = np.zeros(R.shape)
Image[(R<=diamout/2)*(R>=diamin/2)] = Fr(R[(R<=diamout/2)*(R>=diamin/2)])

# == using Fortran numerical integratiom
t = time.time()
Vf = Vmodels.hankelvis(u=u, v=v, wl=wl, iprofile=I, r=r, x0=0, y0=0, order=0)
print('Fortran (Hankel):', time.time()-t)

# == using Numpy numerical integratiom
t = time.time()
Bwl = np.sqrt(u**2+v**2)[:,None]/wl[None,:]
x = 2*np.pi**2*Bwl/180/3600/1000/1e-6
Vp = np.trapz(I[None,None,:]*r[None,None,:]*scipy.special.j0(r[None,None,:]*x[:,:,None]), 
		r[None,None,:],axis=2)
Vp /= np.trapz(I*r, r[None,None,:])
#Vp = np.sum(I[None,None,:]*r[None,None,:]*scipy.special.jn(order,r[None,None,:]*x[:,:,None]), axis=2)
#Vp /= np.sum(I*r)
print('Numpy (Hankel):', time.time()-t)

# == using image
if testImage:
	t = time.time()
	Vi = Vmodels.im2vis(u=u, v=v, wl=wl, x=X.flatten(), y=Y.flatten(), image=Image.flatten())
	print('Fortran (from image):', time.time()-t)

if uniform:
	# -- analytical
	Va = 2*scipy.special.j1(x*diamout/2)/(x*diamout/2)*diamout**2 -\
	     2*scipy.special.j1(x*diamin/2)/(x*diamin/2)*diamin**2
	Va /= (diamout**2-diamin**2)
# =======================================
plt.close(0)
plt.figure(0, figsize=(5,7))

plt.subplot(321)
plt.plot(r, I, '.k')
plt.title('intensity profile')
plt.xlabel('radial distance (mas)')

plt.subplot(322, aspect='equal')
plt.pcolormesh(X, Y, Image, cmap='inferno')
plt.xlabel('offset on sky (mas)')
plt.ylabel('offset on sky (mas)')

ax1 = plt.subplot(312)
if uniform:
	plt.plot(Bwl.flatten(), np.abs(Va.flatten()), '.k', label='analytical', alpha=0.5)

plt.plot(Bwl.flatten(), np.abs(Vp.flatten()), 'xb', label='Numpy Hankel', alpha=0.5)
plt.plot(Bwl.flatten(), np.abs(Vf.flatten()), '+m', label='Fortran Hankel', alpha=0.5)
if testImage:
	plt.plot(Bwl.flatten(), np.abs(Vi.flatten()), 'vg', label='Fortran from image', alpha=0.1)
plt.xlabel('B/$\lambda$ (m/$\mu$m)')
plt.ylabel('visibility amplitude')

plt.legend()
plt.subplot(313, sharex=ax1)
if uniform:
	plt.plot(Bwl.flatten(), (np.abs(Vp.flatten())-np.abs(Va.flatten()))/np.abs(Va.flatten()), 'xb', label='Np_Hankel - A')
	plt.plot(Bwl.flatten(), (np.abs(Vf.flatten())-np.abs(Va.flatten()))/np.abs(Va.flatten()), '+m', label='F95_Hankel - A')
	# -- residuals are very large...
	#plt.plot(Bwl.flatten(), (np.abs(Vi.flatten())-np.abs(Va.flatten()))/np.abs(Va.flatten()), 'vg', label='F95_i - A')
	plt.xlabel('B/$\lambda$ (m/$\mu$m)')

else:
	plt.plot(Bwl.flatten(), (np.abs(Vf.flatten())-np.abs(Vp.flatten()))/np.abs(Vp.flatten()), '+m', label='F95_Hankel - Np_Hankel')
	if testImage:
		plt.plot(Bwl.flatten(), (np.abs(Vi.flatten())-np.abs(Vp.flatten()))/np.abs(Vp.flatten()), 'vg', label='F95_image - Np_Hankel')
	plt.xlabel('B/$\lambda$ (m/$\mu$m)')

plt.legend()
plt.tight_layout()

plt.show()




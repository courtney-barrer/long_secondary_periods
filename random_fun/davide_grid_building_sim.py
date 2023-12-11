import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
from astropy import units as u

# INPUT: M1, q, f   --  OUTPUT: a, Rd
Period = 757   # in days
M1 = 4.3
q = 100
f = 0.3


Omega = 2 * np.pi / (Period*86400)   # in rad/s
Egg = ( 0.49*q**(2/3) ) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
M_sum = M1 + M1/q
a = ((const.G.cgs.value * M_sum * const.M_sun.cgs.value)/(Omega**2))**(1/3) / const.au.cgs.value
a_mas = a / 0.509  # in mas
Rd = a * f * Egg  #in AU
Rd_mas = Rd / 0.509  # in mas
M2 = M1/q


print(
    f'a: {a:.3f} AU ({a_mas:.3f} mas)\n',
    f'R dust: {Rd:.3f} AU ({Rd_mas:.3f} mas)\n',
    f'filling factor: {f:.3f} \n',
    f'M sum: {M_sum:.3f} Solar masses\n',
    f'M1: {M1:.3f} Solar masses\n',
    f'M2: {M2:.3f} Solar masses\n',
    f'q: {q:.5f} \n'
)


# INPUT: q, a, Rd  --  OUTPUT: f, M1
Period = 757   # in days
q = 1
a = 2.1
Rd = 323 * const.R_sun.cgs.value * u.cm.to(u.au)


Omega = 2 * np.pi / (Period*86400)   # in rad/s
Egg = ( 0.49*q**(2/3) ) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
M_sum = (Omega**2 * (a*const.au.cgs.value)**3 ) / const.G.cgs.value / const.M_sun.cgs.value  #in Solar masses
a_mas = a / 0.509  # in mas
f = Rd / a / Egg
Rd_mas = Rd / 0.509  # in mas
M1 = M_sum / (1+1/q)
M2 = M1/q


print(
    '--------------------\n',
    f'a: {a:.3f} AU ({a_mas:.3f} mas)\n',
    f'R dust: {Rd:.3f} AU ({Rd_mas:.3f} mas)\n',
    f'filling factor: {f:.3f} \n',
    f'M sum: {M_sum:.3f} Solar masses\n',
    f'M1: {M1:.3f} Solar masses\n',
    f'M2: {M2:.3f} Solar masses\n',
    f'q: {q:.5f} \n'
)
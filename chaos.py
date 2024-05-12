import numpy as np
from scipy.integrate import odeint
import dna
def gen_chaos_seq(m,n,x0, y0, z0,a,b,c):
    global N
    N=m*n*4
    x= np.array((m,n*4))
    y= np.array((m,n*4))
    z= np.array((m,n*4))
    t = np.linspace(0, dna.tmax, N)
    f = odeint(dna.lorenz, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    x=x[:(N)]
    y=y[:(N)]
    z=z[:(N)]
    return x,y,z
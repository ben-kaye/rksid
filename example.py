import rksid
import numpy as np
import scipy.signal as sig

RNG = np.random.default_rng()


N = 200
u = 3. + RNG.normal(0., 2., (N, 1))

G_true = sig.TransferFunction(
    [1, 3, 3], [1, 2, 1], dt=0.1
)
y =  sig.dlsim(G_true,u)[1] + RNG.normal(0,0.3,(N,1))

regr = rksid.ImpulseIdentifier()
G_est = regr(u,y)

pass

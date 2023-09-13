import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import exp


# randchoice([(a[1],p[1]),...,(a[n],p[n])]) 
# chooses a[i] with prob. proportional to p[i]
def randchoice(x, rng):
    return min([(rng.exponential()/p, a) for a, p in x])[1]


# c_Delta in the paper
def coeff(d):
    return d*(1+exp(-d))/(1-exp(-d))/2


# Compute r
def get_r(l, d):
    denom = (1 + exp(-d))**2 * (2/(1+exp(-2*d)) - d*l - 1)
    if abs(denom) < 1e-12:
        return 1.0
    num = 4 - 4 * (d*l+1) * exp(-d)
    return num / denom


# delta_0
def get_d0(l):
    dmin = 0.0
    dmax = 30.0
    for _ in range(120):
        d = (dmin + dmax) * 0.5
        if exp(d) > d*l + 1:
            dmax = d
        else:
            dmin = d
    return (dmin + dmax) * 0.5


def compute_FT(l, maxt):
    d0 = get_d0(l)
    ds = [d0 * (2.0 ** -t) for t in range(maxt)]
    rs = [get_r(l, d) for d in ds]
    
    Fs = [1.0] * maxt
    F = 1.0
    for t in range(maxt-1, -1, -1):
        Fs[t] = F
        F *= rs[t]
    
    return (ds, rs, Fs)


# Generate T
def genT(Fs, rng):
    r = rng.random()
    t = 0
    while r > Fs[t]:
        t += 1
    return t

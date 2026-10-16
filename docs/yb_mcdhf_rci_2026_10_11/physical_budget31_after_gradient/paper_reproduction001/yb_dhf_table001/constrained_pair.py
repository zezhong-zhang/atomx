"""Bounded original-pencil Newton correction with occupied constraints.

The bordered equations differentiate Hx-eSx-SCmu=0, xTSx=1,
and CTSx=0. Unlike a rounded B.T H B pencil, residuals consume the
original physical H/S. This is a local reproduction driver, not a new
public Atomx solver or a change to its defaults.
"""
import numpy as np
from scipy import linalg


def refine(h, s, occupied, energy, vector):
    n, m = occupied.shape
    if n > 512 or m > 16:
        raise MemoryError('bounded physical-pencil correction only')
    x = vector.copy()
    c = occupied
    wide = np.longdouble
    hw, sw, cw = [a.astype(wide) for a in (h,s,c)]
    gram = np.asarray(cw.T@sw@cw,dtype=float)
    def normalize(x):
        overlap = np.asarray(cw.T@sw@x.astype(wide),dtype=float)
        x = x-c@linalg.solve(gram,overlap,assume_a='pos')
        return x/np.sqrt(float(x.astype(wide)@sw@x.astype(wide)))
    x = normalize(x)
    mu = linalg.solve(gram,np.asarray(cw.T@hw@x.astype(wide),dtype=float),assume_a='pos')
    for _ in range(2):
        xw = x.astype(wide)
        r = np.asarray(hw@xw-wide(energy)*(sw@xw)-sw@cw@mu.astype(wide),dtype=float)
        sx, sc = s@x, s@c
        bordered = np.block([[h-energy*s,-sx[:,None],-sc],
            [-sx[None,:],np.zeros((1,1+m))],
            [-sc.T,np.zeros((m,1+m))]])
        step = linalg.solve(bordered,np.r_[-r,np.zeros(1+m)],assume_a='sym')
        x = normalize(x+step[:n])
        energy += step[n]
        mu += step[n+1:]
    return float(energy),x


def independent_controls():
    # Explicit scalar constrained blocks give known roots without a
    # transformed-pencil solver or the Newton equations as an oracle.
    rng = np.random.default_rng(52013)
    maximum = 0.0
    for seed in range(3):
        d = np.array([.1,1.,3.,10.])
        s = np.diag(d*d)
        q,_ = np.linalg.qr(rng.normal(size=(4,4)))
        basis = q/d[:,None]
        # Occupied vector is not a full-H eigenvector; the Lagrange
        # multiplier must remove the genuine signed H/core coupling.
        transformed = np.array([[2.,.13,-.07,.09],[.13,-.6,0.,0.],
                                [-.07,0.,.2,0.],[.09,0.,0.,.8]])
        h = (d[:,None]*q)@transformed@(q.T*d[None,:])
        c = basis[:,:1]
        initial = basis[:,1]+(seed+1)*1e-5*basis[:,2]
        e,x = refine(h,s,c,-.6+1e-7,initial)
        error=max(abs(e+.6),abs(abs(x@s@basis[:,1])-1),abs(float(c.T@s@x)))
        assert error < 1e-12
        maximum=max(maximum,error)
    return dict(analytic_cases=3,maximum_error=maximum,status='PASS')

"""Probe H: TRK sum rule over B-spline pseudostates (hydrogen 1s, length gauge).

Nonrelativistically: sum_n (E_n - E_1s) |<n|r|1s>|^2 = 3/2 (one electron,
3 Cartesian components). In the Dirac pseudostate basis: sum over the full
physical spectrum of both allowed kappa channels (+1, -2), using the
velocity-form identity <m|r|i> = <m|alpha_r + ...|i>/(E_m-E_i). We use
length form with energies from the same basis: the sum includes positive
AND negative energy states; the NR limit isolates positive-energy part.
"""
import numpy as np
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.evaluate import eval_P, eval_Q

Z = 1
t, ns = make_physics_knots(ks=9, Z=Z, nuclear='point')
data = prepare_bspline_data(t, ns, 9, kinetic_balance=True)
gr, grw = data["gr"], data["grw"]

H, S = build_dirac_matrices(ns, -1, Z, data)
res1s = solve_dirac(H, S, ns, -1, nsp=data["nsp"], nsq=data["nsq"])
P1, Q1, E1 = res1s.get_orbital(0)
fP, fQ = eval_P(P1, data).ravel(), eval_Q(Q1, data).ravel()
w = grw.ravel(); rr = gr.ravel()

total = 0.0
for kappa in (+1, -2):  # p1/2 and p3/2
    Hk, Sk = build_dirac_matrices(ns, kappa, Z, data)
    resk = solve_dirac(Hk, Sk, ns, kappa, nsp=data["nsp"], nsq=data["nsq"])
    evals = resk.eigenvalues          # physical branch (above sea edge)
    vecs = resk.coefficients          # (ms, n_states)
    nsp_, nsq_ = resk.nsp, resk.nsq
    for m in range(len(evals)):
        vec = vecs[:, m]
        Pm = np.zeros(len(vec)//2); Qm = np.zeros(len(vec)//2)
        Pm[:nsp_] = vec[:nsp_]; Qm[:nsq_] = vec[len(vec)//2:len(vec)//2+nsq_]
        gP, gQ = eval_P(Pm, data).ravel(), eval_Q(Qm, data).ravel()
        mel = np.sum(w * rr * (gP*fP + gQ*fQ))
        total += (evals[m] - E1) * mel**2
print(f"TRK sum (both p channels, length form): {total:.9f}  (NR exact 1.5)")
print(f"E1s = {E1:.12f}")
# angular weight note: each kappa channel carries full reduced |<f||r||i>|^2 /(2j_i+1) weight;
# the raw radial sum above = sum over m of (E_m-E_1)*|reduced radial mel|^2; for 1s->np both
# channels each contribute the same radial integral; TRK 3/2 = sum_channels ... check ratio:

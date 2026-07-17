"""Hydrogenic |E(Z; n)| vs Z and the relativistic departure (Fig. 1.1).

Panel (a): |E_{ns}| for n = 1, 2, 3 from HydrogenicSchrodinger across
Z = 1..110 on log-log axes (exact Z^2/2n^2 scaling), with HydrogenicDirac
1s1/2 points showing the relativistic departure at high Z.
Panel (b): the departure E_D/E_S - 1 for 1s against the leading-order
prediction (Z*alpha)^2/4 — the coming attraction for Chapter 3.

Also prints the ratio table quoted in the chapter.

Run:  python docs/book/vol1/ch01_scales_units_conventions/calc/hydrogenic_z_scaling.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atomx import Atom, HydrogenicDirac, HydrogenicSchrodinger
from atomx.constants import ALPHA

STYLE = Path(__file__).resolve().parents[3] / "bookstyle.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

Z_all = np.arange(1, 111)
E_S = {n: [] for n in (1, 2, 3)}
E_D1s = []
for Z in Z_all:
    rs = Atom(Z=int(Z), config="1s1").solve(HydrogenicSchrodinger())
    E_S[1].append(-rs.orbital_energies["1s"])
    rs2 = Atom(Z=int(Z), config="2s1").solve(HydrogenicSchrodinger())
    E_S[2].append(-rs2.orbital_energies["2s"])
    rs3 = Atom(Z=int(Z), config="3s1").solve(HydrogenicSchrodinger())
    E_S[3].append(-rs3.orbital_energies["3s"])
    rd = Atom(Z=int(Z), config="1s1").solve(HydrogenicDirac())
    E_D1s.append(-rd.orbital_energies["1s1/2"])

E_D1s = np.array(E_D1s)
for n in (1, 2, 3):
    E_S[n] = np.array(E_S[n])

departure = E_D1s / E_S[1] - 1.0
leading = (Z_all * ALPHA) ** 2 / 4.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

mark = Z_all % 6 == 1  # thin the markers
for n, color in zip((1, 2, 3), ("C0", "C1", "C2")):
    ax1.loglog(Z_all, E_S[n], "-", color=color, lw=1.2,
               label=rf"Schrödinger $ns$, $n={n}$: $Z^2/2n^2$")
ax1.loglog(Z_all[mark], E_D1s[mark], "k^", ms=4.5, mfc="none",
           label=r"Dirac $1s_{1/2}$")
ax1.set_xlabel(r"nuclear charge $Z$ (dimensionless)")
ax1.set_ylabel(r"binding energy $|\varepsilon|$ ($E_\mathrm{h}$)")
ax1.legend(fontsize=8, loc="upper left")
ax1.set_title("(a) hydrogenic scaling", fontsize=10)

ax2.loglog(Z_all, departure, "k^", ms=3.5, mfc="none",
           label=r"$\varepsilon_D/\varepsilon_S - 1$ (computed)")
ax2.loglog(Z_all, leading, "r-", lw=1.2,
           label=r"$(Z\alpha)^2/4$ (leading order)")
ax2.set_xlabel(r"nuclear charge $Z$ (dimensionless)")
ax2.set_ylabel(r"relativistic departure of $\varepsilon_{1s}$ (relative)")
ax2.legend(fontsize=8, loc="upper left")
ax2.set_title(r"(b) the $(Z\alpha)^2$ ladder rung", fontsize=10)

fig.tight_layout()
out = FIGDIR / "hydrogenic_z_scaling.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")

print("\nZ    eps_S(1s)/Eh     eps_D(1s1/2)/Eh        eps_D/eps_S-1   (Z*alpha)^2/4")
for Z in (1, 20, 26, 60, 92, 110):
    i = Z - 1
    print(f"{Z:<4d} {-E_S[1][i]:<16.6f} {-E_D1s[i]:<22.13f} "
          f"{departure[i]:<15.6e} {leading[i]:<15.6e}")

"""Standalone publication exports from the saved measured CSV panels."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent

def rows(name):
    with (HERE/name).open() as f:
        return list(csv.DictReader(f))

pi,fl,y,sc=map(rows,['pi.csv','fluorescence_totals.csv','yield.csv','scattering.csv'])
fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for s in ['1s','2s','2p']:
    zs=sorted({int(r['Z']) for r in pi})
    means=[];low=[];high=[]
    for z in zs:
        v=[float(r['atomx_over_fac']) for r in pi if r['shell']==s and int(r['Z'])==z]
        means.append(np.mean(v));low.append(min(v));high.append(max(v))
    line,=axs[0,0].plot(zs,means,'o-',label=s)
    axs[0,0].fill_between(zs,low,high,color=line.get_color(),alpha=.15)
axs[0,0].set(title='Native HF PI / FAC E1',xlabel='Z',ylabel='Ratio (three energies above K edge)')
axs[0,0].axhline(1,color='grey',ls=':');axs[0,0].legend()
zs=[int(r['Z']) for r in y]
for key,label in [('atomx_radiative_per_s','atomx HF + E1'),('fac_radiative_per_s','FAC TRTable'),('xatom_radiative_per_s','XATOM HFS')]:
    axs[0,1].semilogy(zs,[float(r[key]) for r in y],'o-',label=label)
axs[0,1].set(title='K radiative width',xlabel='Z',ylabel='Rate / s');axs[0,1].legend()
for key,label in [('atomx_omega_K','atomx LS + E1'),('fac_omega_K','FAC'),('xatom_omega_K','XATOM')]:
    axs[0,2].semilogy(zs,[float(r[key]) for r in y],'o-',label=label)
axs[0,2].set(title='K fluorescence yield',xlabel='Z',ylabel='omega K');axs[0,2].legend()
for key,label in [('fac_auger_per_s','FAC'),('xatom_auger_per_s','XATOM')]:
    axs[1,0].plot(zs,[float(r['atomx_auger_per_s'])/float(r[key]) for r in y],'o-',label=label)
axs[1,0].axhline(1,color='grey',ls=':');axs[1,0].set(title='Native LS Auger width / reference',xlabel='Z',ylabel='Ratio');axs[1,0].legend()
for el in ['Ne','Ar','Fe']:
    data=[r for r in sc if r['element']==el]
    energies=[int(r['photon_eV'])/1000 for r in data]
    axs[1,1].plot(energies,[float(r['atomx_rayleigh_barn'])/float(r['fac_rayleigh_barn']) for r in data],'o-',label=el)
    axs[1,2].plot(energies,[float(r['atomx_compton_KN_barn'])/float(r['fac_compton_KN_barn']) for r in data],'o-',label=el)
for ax,title in [(axs[1,1],'Rayleigh / FAC spinor postprocessing'),(axs[1,2],'KN × WH S / FAC spinor postprocessing')]:
    ax.axhline(1,color='grey',ls=':');ax.set(title=title,xlabel='Photon energy / keV',ylabel='Integrated cross-section ratio');ax.legend()
for ax in axs.flat:
    ax.grid(alpha=.15)
fig.suptitle('C, Ne, Ar, Fe, Kr, Xe: model differences are retained (not fitted)')
fig.savefig(HERE/'systematic_comparison.png',dpi=180)
fig.savefig(HERE/'systematic_comparison.pdf')
plt.close(fig)

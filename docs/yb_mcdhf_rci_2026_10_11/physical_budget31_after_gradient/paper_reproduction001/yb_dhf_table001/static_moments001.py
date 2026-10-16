"""Native static E2 moments with independent spline and analytic angular anchors."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0, {27})
    out = HERE / 'static_moments001'
    out.mkdir(exist_ok=False)
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
        HERE/'contract003.json', HERE/'actual003/result.json'] +
        [HERE/f'actual003/{name}_ns{ns}.{suffix}' for ns in (180,220)
         for name,suffix in [('core','h5'),('blocks','npz')]]}
    contract = dict(hypothesis='The same frozen-core Yb+ DC states also reproduce the scale of the paper DHF static quadrupole row; independent polynomial quadrature and analytic one-d-electron angular factors check the actual property convention.',
        fixed='Source359 fingerprint0d73a18232a5ecb1b5dcb5871d71fd6fd7c56df6ab286658d90422d5da59b99a; full saved ns180/220 states, same normalized P/Q, A171 empirical Fermi nucleus and DC reference. No SCF/Fock/CI solve or borrowed corrections.',
        operator='Theta(J)=-<J,M=J|sum_i r_i^2 C20(i)|J,M=J> in e a0^2; static omega=0 and closed-core trace zero.',
        references=dict(target='https://arxiv.org/html/2504.01438v1 Table1 DHF Theta:2.500/3.690',
            convention='https://tf.boulder.nist.gov/general/pdf/2107.pdf Eqs1,2a,2b: Theta(d3/2)=<r2>/5; Theta(d5/2)=2<r2>/7; distinct69e-EOL orbitals/TableVIII are not this frozen68e model'),
        acceptance=dict(angular_absolute=1e-14, full_closed_core_trace=1e-13,
            independent_radial_absolute=1e-10, independent_norm=1e-10,
            two_basis_moment_drift_au=1e-7),
        resources=dict(threads=1, cpu=27, decoded_NPZ_bytes_max=256*1024**2,
            independent_quadrature='ten Gauss-Legendre nodes per distinct knot interval, exact for degree<=19; RKB P/Q squares times r2 have degree<=18',
            calls='six native static radial evaluations and six independent spline integrals; no atomic solves'),
        inputs=inputs, stopping='Preserve first failed control, no tolerance/grid/model changes',
        limitations='DHF only; not correlated moments, field-shift/hyperfine constants, MCDHF stationarity or full paper reproduction.')
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    import numpy as np
    from scipy.interpolate import BSpline

    from atomx.angular.wigner import wigner_3j
    from atomx.methods.dirac_fock import SCFResult
    from atomx.properties.transitions.angular import angular_eL
    from atomx.properties.transitions.multipole import radial_eL

    report = dict(status='RUNNING', arms=[], controls=[])
    def gate(name, value, cap):
        passed = bool(abs(value) <= cap)
        report['controls'].append(dict(name=name,measured=value,cap=cap,passed=passed))
        if not passed:
            raise AssertionError(name+': '+str(value))
    try:
        acceptance = contract['acceptance']
        expected = {-1:0.0, 2:1/5, -3:2/7}
        for ns in (180,220):
            core = SCFResult.load(HERE/f'actual003/core_ns{ns}.h5')
            d = core._bsp_data
            with np.load(HERE/f'actual003/blocks_ns{ns}.npz',allow_pickle=False) as saved:
                gate(f'{ns}:decoded_bytes',sum(saved[k].nbytes for k in saved),
                     contract['resources']['decoded_NPZ_bytes_max'])
                keys, ps, qs = [saved[k] for k in ('all_orbital_keys','all_P','all_Q')]
            knots = np.unique(d['t'])
            gx, gw = np.polynomial.legendre.leggauss(10)
            left, right = knots[:-1,None],knots[1:,None]
            x = (left+right)/2 + (right-left)*gx/2
            weight = (right-left)*gw/2
            rows = []
            for key in ((6,-1),(5,2),(5,-3)):
                index = list(map(tuple,keys)).index(key)
                p,q = ps[index],qs[index]
                k = key[1]
                tj = 2*abs(k)-1
                angular = -wigner_3j(tj,4,tj,-tj,0,tj)*angular_eL(k,k,2)
                gate(f'{ns}:angular:{key}',angular-expected[k],acceptance['angular_absolute'])
                radial = radial_eL(p,q,p,q,d,L=2,omega=0)
                pp = BSpline(d['tp'],p[:d['nsp']],d['ksp']-1,extrapolate=False)(x)
                qq = BSpline(d['t'],q[:d['nsq']],d['ksq']-1,extrapolate=False)(x)
                norm = float(np.sum(weight*(pp*pp+qq*qq)))
                independent = float(np.sum(weight*(pp*pp+qq*qq)*x*x))
                gate(f'{ns}:radial:{key}',radial-independent,acceptance['independent_radial_absolute'])
                gate(f'{ns}:norm:{key}',norm-1,acceptance['independent_norm'])
                rows.append(dict(key=key,mean_r2_au=radial,
                    independent_mean_r2_au=independent, norm=norm,
                    angular_factor=angular, Theta_au=angular*radial,
                    paper_DHF_Theta_au={-1:None,2:2.500,-3:3.690}[k]))
            traces = []
            for sh in core.subshells:
                tj = 2*abs(sh.kappa)-1
                trace = sum((-1.0)**((tj-m)//2)*wigner_3j(tj,4,tj,-m,0,m)
                    * angular_eL(sh.kappa,sh.kappa,2) for m in range(-tj,tj+1,2))
                traces.append(trace)
            gate(f'{ns}:all19_closed_subshell_traces',max(abs(x) for x in traces),
                 acceptance['full_closed_core_trace'])
            report['arms'].append(dict(ns=ns,rows=rows,closed_core_angular_traces=traces))
        drift=[b['Theta_au']-a['Theta_au'] for a,b in zip(report['arms'][0]['rows'],
                    report['arms'][1]['rows'],strict=True)]
        report['two_basis_drift_au']=drift
        gate('two_basis_moment_drift',max(abs(x) for x in drift),
             acceptance['two_basis_moment_drift_au'])
        gate('inputs_unchanged',int(not all(sha(ROOT/p)==h for p,h in inputs.items())),0)
        report['status']='PASS_STATIC_DHF_MOMENTS_ONLY'
    except Exception as error:
        report.update(status='FAILED',exception=repr(error))
        raise
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

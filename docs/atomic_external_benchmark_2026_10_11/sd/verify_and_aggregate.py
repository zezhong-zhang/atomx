"""Independently replay complete saved outputs and source/input identity chains."""
from __future__ import annotations
import hashlib
import json
import datetime
from pathlib import Path
import numpy as np
OWNER=Path(__file__).resolve().parent
ROOT=OWNER.parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def archive_gate(out,kind):
    d=json.loads((out/f'{kind}_sha256.json').read_text())
    changes=[]
    for rel,sha in d.items():
        p=out/('source' if kind=='source' else 'inputs')/rel
        if digest(p)!=sha or digest(ROOT/rel)!=sha:changes.append(rel)
    return dict(files=len(d),changed=changes,passed=not changes)


def backends_gate(out):
    reports=[]
    for p in sorted(out.glob('backend*/manifest.json')):
        d=json.loads(p.read_text());bad=[n for n,h in d['files'].items() if digest(p.parent/n)!=h]
        reports.append(dict(directory=p.parent.name,files=len(d['files']),changed=bad,passed=not bad))
    return reports


def array_identity(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),elements=int(a.size),
                full_F_order_sha256=hashlib.sha256(a.tobytes(order='F')).hexdigest())


def finite():
    p=OWNER/'finite_action001';r=json.loads((p/'result.corrected.json').read_text())
    with np.load(p/'warmup_original.npz',allow_pickle=False) as f:reference={k:f[k].copy() for k in f}
    values=[];maxerr=0.0;maxrel=0.0
    for file in sorted(p.glob('round_*.npz')):
        with np.load(file,allow_pickle=False) as f:
            if set(f)!=set(reference):raise AssertionError('finite full output member set differs')
            rows={};error=0.0;relative=0.0
            for k,b in reference.items():
                a=f[k]
                if a.shape!=b.shape or not np.all(np.isfinite(a)):raise AssertionError('finite full output coordinates differ')
                difference=float(np.max(abs(a-b)))
                error=max(error,difference)
                relative=max(relative,difference/max(float(np.max(abs(b))),1e-300))
                rows[k]=array_identity(a)
            if error>1e-11 or relative>1e-10:raise AssertionError('finite whole action tolerance failed')
        maxerr=max(maxerr,error);maxrel=max(maxrel,relative)
        values.append(dict(file=file.name,sha256=digest(file),maximum_signed_error_Ha=error,maximum_relative_error=relative,arrays=rows))
    if len(values)!=25:raise AssertionError('five full finite paired rounds incomplete')
    source=archive_gate(p,'source');backends=backends_gate(p)
    if not source['passed'] or not all(b['passed'] for b in backends):raise AssertionError('finite actual source/backend changed')
    proof=dict(result_sha256=digest(p/'result.corrected.json'),raw_result_sha256=digest(p/'result.json'),
        correction_sha256=digest(p/'measurement_scope_correction.json'),source=source,backends=backends,
        whole_output_files=len(values),whole_arrays=len(values)*len(reference),
        elements_per_complete_five_column_output=sum(v.size for v in reference.values()),
        maximum_signed_error_Ha=maxerr,maximum_relative_error=maxrel,outputs=values,
        physical_input_sha256=digest(p/'physical_inputs.npz'),numerator_input_sha256=digest(p/'numerator_inputs.npz'),
        current_core_residual_controls=r['core_control'])
    write(p/'full_output_verification.json',proof)
    return {k:r[k] for k in ['status','maximum_signed_error_Ha','maximum_relative_error','timing_guard_pass','scaling','setup_seconds','peak_rss_bytes','scope','pci_oracle_lineage']},proof


def yb():
    reference_path=ROOT/'docs/atomic_pci_sd_native_2026_10_09/valence_term2_full003/term2_6_-1.npz'
    with np.load(reference_path,allow_pickle=False) as f:
        references=(f['original'].copy(),f['original_prime'].copy())
        serial=(f['streamed'].copy(),f['streamed_prime'].copy())
    all_rows=[];proofs=[];preparation=[]
    for round_number in range(3):
        p=OWNER/f'yb_round{round_number:03d}'
        if not (p/'result.json').exists():return None,None
        result=json.loads((p/'result.json').read_text())
        if result['status']!='complete signed control passed' or not result['precision_pass'] or not result['all_workers_bitwise_equal']:raise AssertionError('complete Yb round proof failed')
        source=archive_gate(p,'source');inputs=archive_gate(p,'input');backends=backends_gate(p)
        if not source['passed'] or not inputs['passed'] or not all(x['passed'] for x in backends):raise AssertionError('actual complete Yb source/input/backend changed')
        observations=json.loads((p/'observations.json').read_text());rows=[r for r in observations if not r['warmup']]
        if sorted(r['workers'] for r in rows)!=[1,2,4,8]:raise AssertionError('Yb full four-worker round incomplete')
        outputs=[]
        for r in observations:
            file=p/f"map_{r['ordinal']}_{r['workers']}.npz"
            with np.load(file,allow_pickle=False) as f:
                arrays=(f['signed'],f['response']);errors=[];relative=[]
                for a,b,c in zip(arrays,references,serial):
                    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.array_equal(a,c):raise AssertionError('Yb complete signed/response worker array gate failed')
                    difference=float(np.max(abs(a-b)))
                    errors.append(difference)
                    relative.append(difference/max(float(np.max(abs(b))),1e-300))
                if max(errors)>1e-11 or max(relative)>1e-10:raise AssertionError('Yb original diagram tolerance failed')
                outputs.append(dict(file=file.name,sha256=digest(file),arrays=[array_identity(a) for a in arrays],maximum_signed_error_Ha=max(errors),maximum_relative_error=max(relative)))
        for r in rows:
            all_rows.append(dict(r,round=round_number))
        runtime={r['path']:digest(r['path'])==r['sha256'] for r in result['runtime']}
        if not all(runtime.values()):raise AssertionError('mapped runtime changed after complete Yb run')
        proof=dict(round=round_number,source=source,inputs=inputs,backends=backends,runtime_unchanged=runtime,
            outputs=outputs,result_sha256=digest(p/'result.json'),all_original_arrays_preserved=True)
        write(p/'full_output_verification.json',proof);proofs.append(proof)
        preparation.append(dict(round=round_number,setup_seconds=result['setup_seconds'],peak_rss_bytes=result['peak_rss_bytes'],total_lock_wait_seconds=sum(r['lock_wait_seconds'] for r in observations)))
    valid=all(r['timing_guard_pass'] for r in all_rows)
    scaling=[]
    for workers in (1,2,4,8):
        rows=[r for r in all_rows if r['workers']==workers]
        ratios=[next(x['seconds'] for x in all_rows if x['round']==r['round'] and x['workers']==1)/r['seconds'] for r in rows]
        median=float(np.median(ratios))
        scaling.append(dict(workers=workers,seconds=[r['seconds'] for r in rows],median_seconds=float(np.median([r['seconds'] for r in rows])),
            observed_speedups=ratios,speedup=median if valid else None,efficiency=median/workers if valid else None))
    return dict(status='three complete independent capped Yb rounds passed',maximum_signed_error_Ha=max(r['maximum_signed_error_Ha'] for r in all_rows),
        maximum_relative_error=max(r['maximum_relative_error'] for r in all_rows),
        all_workers_bitwise_equal=True,timing_guard_pass=valid,scaling=scaling,preparation=preparation,
        full_response=True,full_core_electrons=68,full_virtual_orbitals=414,shape=result['shape'],
        scope=result['scope'],fresh_original_pci_Yb_timing=False,full_action_speedup_claim=False,full_solve_speedup_claim=False,
        original_reference_sha256=digest(reference_path)),proofs


def thread_evidence():
    path=OWNER/'thread_samples.jsonl'
    if not path.exists():return None
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    last={};stable=[]
    for row in rows:
        key=(row['pid'],row['round']);previous=last.get(key)
        if previous is not None and previous['completed_observations']==row['completed_observations'] and previous['next_requested_workers']==row['next_requested_workers'] and row['interval_seconds']>1:
            stable.append(row)
        last[key]=row
    records=[]
    kernel=(ROOT/'src/atomx/methods/_pci_valence_term2_streamed.f90').read_text()
    for round_number in (0,1,2):
        p=OWNER/f'yb_round{round_number:03d}'
        if not (p/'result.json').exists():continue
        runtime=json.loads((p/'result.json').read_text())['core_provenance']['blas_runtime']
        for workers in (1,2,4,8):
            relevant=[r for r in stable if r['round']==round_number and r['next_requested_workers']==workers
                      and 1 <= r['completed_observations'] <= 4]
            maximum=max((r['active_count'] for r in relevant),default=None)
            name=f'benchmark_term2_{workers}'
            file=p/'backend'/f'{name}.f90'
            expected=kernel.replace('CASE_THREADS',str(workers)).replace('valence_term2_streamed',name)
            actual=file.read_text().split(f'subroutine atomx_benchmark_term2_{workers}')[0]
            if actual!=expected+'\n':raise AssertionError('compiled requested-worker source differs')
            records.append(dict(round=round_number,compiled_outer_requested_workers=workers,
                compiled_body_exact_public_template=True,compiled_source_sha256=digest(file),
                actual_bound_inner_BLAS_threads=runtime['threads'],stable_cpu_sample_intervals=len(relevant),
                maximum_observed_active_CPU_threads=maximum,observed_matching_requested=maximum==workers if maximum is not None else None,
                scope='Read-only proc thread CPU deltas above25% of a2-second interval; transition intervals, cold preparation and warmup excluded. Whole-call lock wait/output compression may lower counts. No missing samples are invented.'))
    proof=dict(samples_sha256=digest(path),sampler_sha256=digest(OWNER/'thread_sampler.py'),total_intervals=len(rows),stable_intervals=len(stable),
        transition_intervals_rejected=len(rows)-len(stable),records=records,
        round0_p1_sample_not_collected=True,note='Round0 serial ran before sampler start. Its explicit compiled num_threads(1) and bound BLAS=1 are verified, while OS sample is null. Later complete rounds sample p1 directly.')
    write(OWNER/'thread_verification.json',proof)
    return proof


def main():
    finite_result,finite_proof=finite();yb_result,yb_proof=yb();threads=thread_evidence()
    result=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),finite_whole_action=finite_result,
        yb_full_term2=yb_result,status='complete' if yb_result else 'finite complete; Yb rounds still measuring',
        independent_standalone_pci_comparison=False,experimental_accuracy_gain_claim=False,full_SD_solve_efficiency_claim=False)
    write(OWNER/'result.json',result)
    proof=dict(updated_at=result['updated_at'],finite_full_output_verification=finite_proof,yb_full_output_verification=yb_proof,actual_thread_evidence=threads,
        contract_sha256=digest(OWNER.parent/'contract.json'),private_verifier_sha256=digest(Path(__file__)),
        failed_first_setup_retained=True,proactively_stopped_pilot_retained=True,
        original_resource_and_precision_gates_unchanged=True)
    write(OWNER/'verification.json',proof)
    print(json.dumps({k:result[k] for k in ['updated_at','status']}))

if __name__=='__main__':main()

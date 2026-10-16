"""Independent terminal guards and epoch-preserving official performance table."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

OWNER=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    original=OWNER/'initial_epoch.result.json'
    if not original.exists():
        shutil.copyfile(OWNER/'result.json',original)
    old=read(original);fresh=read(OWNER/'fresh_cache002/result.json')
    manifest=read(OWNER/'fresh_cache002/source_before.json')
    with tarfile.open(OWNER/'actual_sources_build.fresh_cache002.tar.gz','w:gz') as archive:
        for name,expected in manifest.items():
            assert sha(name)==expected,name
            archive.add(name,arcname=name.lstrip('/'),recursive=False)
    old_sources=read(OWNER/'source_before.json')
    common=set(old_sources)&set(manifest)
    source_checks={'common_files':len(common),'common_changed':[n for n in common if old_sources[n]!=manifest[n]],
                   'new_private_helpers':sorted(set(manifest)-set(old_sources))}
    source_checks['passed']=not source_checks['common_changed'] and read(OWNER/'source_guard.json')['passed'] and read(OWNER/'fresh_cache002/source_guard.json')['passed']
    dump(OWNER/'source_epochs_guard.json',source_checks)
    mapped={};conflicts=[]
    for path in sorted(OWNER.rglob('mapped_runtime.json')):
        for name,digest in read(path).items():
            if name in mapped and mapped[name]!=digest:
                conflicts.append({'path':str(path),'file':name,'first':mapped[name],'later':digest})
            mapped[name]=digest
    terminal={name:sha(name) if Path(name).is_file() else None for name in mapped}
    changed={name:{'initial':digest,'terminal':terminal[name]} for name,digest in mapped.items() if terminal[name]!=digest}
    runtime_guard={'passed':not conflicts and not changed,'files':len(mapped),
                   'conflicting_initial_digests':conflicts,'changed_or_missing':changed}
    dump(OWNER/'runtime_before.json',mapped);dump(OWNER/'runtime_after.json',terminal)
    dump(OWNER/'runtime_guard.json',runtime_guard)
    guards=source_checks['passed'] and runtime_guard['passed']
    rows=[r for r in old['cpu_rows'] if r['workload'].startswith('FAC')]+fresh['cpu_rows']
    for row in rows:
        row['epoch']='initial_epoch' if row['workload'].startswith('FAC') else 'fresh_cache002'
        row['controlled']=row['controlled'] and guards
        row['resource_efficiency']=row['speedup']/row['actual_threads_peak'] if row['controlled'] else None
        if not row['controlled']:
            for field in ('speedup','speedup_min','speedup_max','efficiency','efficiency_min','efficiency_max'):
                row[field]=None
    rows.append({'workload':'Atomx full four-ion frozen CI/E1 CLI workflow','workers':8,'rounds':0,
                 'seconds_median':None,'seconds_min':None,'seconds_max':None,'speedup':None,
                 'speedup_min':None,'speedup_max':None,'efficiency':None,'efficiency_min':None,
                 'efficiency_max':None,'resource_efficiency':None,'actual_threads_peak':15,
                 'controlled':False,'observed_times_seconds':[], 'observed_paired_speedups':[],
                 'scope':'Rejected warmup children at actual15threads > unchanged8totalprocesscap; no complete-work cost or speedup admitted.',
                 'epoch':'initial_epoch','rejection_path':'warmup/Atomx/p8'})
    spread=read(OWNER/'model_spread.preliminary.json')
    spread['scope']='Independent optimized local radial models; complete same-J/parity eigenvalue-rank multisets, no avoided-crossing overlap identity, numerical-equivalence or experimental validation. Each configuration groundzero is common; FAC E1 S=rme^2/2.'
    spread['source_guard_passed']=source_checks['passed'];spread['runtime_guard_passed']=runtime_guard['passed']
    for case in spread['cases']:
        case['max_excitation_relative']=case['maximum_excitation_energy_relative_difference']
        case['max_A_relative']=case['maximum_line_A_relative_difference']
        case['raw_inputs_sha256']={str(p.resolve().relative_to(OWNER)):sha(p) for folder in (case['FAC_folder'],case['Atomx_folder']) for p in Path(folder).iterdir() if p.is_file()}
    dump(OWNER/'model_spread.json',spread)
    stages=[]
    for backend,result in (('FAC',old),('Atomx',fresh)):
        selected=[g for g in result['groups'] if g['backend']==backend and g['workers']==1]
        for group in selected:
            sums={}
            for child in group['cases']:
                science=read(OWNER/child['folder']/'science.json')
                if backend=='FAC':
                    times={}
                    for line in science['stage_timing_lines']:
                        fields=line.split();times[fields[-1]]=float(fields[1])
                    child_stages={'init_including_loader':times['initialized'],
                                  'physical_settings_grid_and_ground_scf':times['ground_scf']-times['initialized'],
                                  'virtual_integrals_CI_and_H_export':times['CI_up3d']-times['ground_scf'],
                                  'level_table_export':times['export_levels']-times['CI_up3d'],
                                  'photon_operators_and_line_table_export':times['export_lines']-times['export_levels'],
                                  'remaining_full_child_lifetime':child['seconds']-times['complete']}
                else:
                    child_stages=science['stages'].copy()
                    child_stages.pop('science_seconds_excluding_import_export')
                    child_stages['remaining_full_child_lifetime']=child['seconds']-sum(child_stages.values())
                for key,value in child_stages.items():sums[key]=sums.get(key,0.)+value
            stages.append({'backend':backend,'round':group['round'],'workers':1,'four_ion_stage_seconds':sums})
    medians={backend:{key:statistics.median(g['four_ion_stage_seconds'][key] for g in stages if g['backend']==backend) for key in next(g['four_ion_stage_seconds'] for g in stages if g['backend']==backend)} for backend in ('FAC','Atomx')}
    dump(OWNER/'stages.json',{'scope':'Three serial rounds, native clocks forFAC (printed5-6significantdigits) and private perf_counter forAtomx; stage borders differ and each median is independently taken. FAC post-exit NPZ normalization excluded; Atomx NPZ diagnostic export included. No silent subtraction.','rounds':stages,'medians':medians})
    result={'scope':old['scope'],'levels_per_Z':17,'lines_per_Z':10,'Z_values':[18,26,54,82],
            'cpu_rows':rows,'groups':[g for g in old['groups'] if g['backend']=='FAC']+fresh['groups'],
            'source_guard_passed':source_checks['passed'],'runtime_guard_passed':runtime_guard['passed'],
            'same_backend_complete_array_correctness':all(g['full_array_correctness_ok'] for g in old['groups'] if g['backend']=='FAC') and all(g['full_array_correctness_ok'] for g in fresh['groups']),
            'model_spread_path':'model_spread.json','model_spread_sha256':sha(OWNER/'model_spread.json'),
            'stage_costs_path':'stages.json','failed_epochs':['initial_epoch.result.json'],
            'cold_warmups_separate':True,'private_cache_remedy':'Only compiler NUMBA_CACHE_DIR isolatedperion; sources/physics/caps unchanged. Fresh cache first3coldion builds separated in40.55576s warmup; Pb preflightcold12.87638s separate. OldglobalcachePb ReferenceError retained.',
            'timing_asymmetry':'FAC stockSFAC includesnativebinaryH/level/lineexports then privateNPZ normalizationoutsideclock; Atomx private fullpublicworkflow includesH/eigenvectors/signedamplitude capture+NPZ/scienceJSON exportsinsideclock. Absolutecrosscodecosts are instrumentedinterface observations atdifferentlocalmodels, not equalwork/equalaccuracyproductionrank.',
            'throughput_measured':False,'highest_level_reached':False}
    dump(OWNER/'result.json',result)
    print(json.dumps({'source_guard':source_checks['passed'],'runtime_guard':runtime_guard,'rows':len(rows),'model_spread_sha256':result['model_spread_sha256']}))


if __name__=='__main__':main()

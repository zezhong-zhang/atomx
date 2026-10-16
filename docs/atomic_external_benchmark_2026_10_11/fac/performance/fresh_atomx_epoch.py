"""Repeat complete Atomx strong scaling after bounded per-ion compiler-cache isolation."""
import importlib.util
import json
from pathlib import Path

OWNER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fac_performance_runner',OWNER/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
BASE = OWNER/'fresh_cache002'
original_environment = r.environment
CURRENT_Z = None


def scoped_environment(workers):
    env = original_environment(workers)
    env['NUMBA_CACHE_DIR'] = str(BASE/'cache'/f'Z{CURRENT_Z:03d}')
    return env


def case(workers,z,path):
    global CURRENT_Z
    CURRENT_Z=z
    (BASE/'cache'/f'Z{z:03d}').mkdir(parents=True,exist_ok=True)
    return r.timed_case('Atomx',workers,z,path/f'Z{z:03d}')


def group(workers,path):
    import os
    load_before=os.getloadavg()[0]
    cases=[case(workers,z,path) for z in r.ZVALUES]
    load_after=os.getloadavg()[0]
    result={'backend':'Atomx','workers':workers,'cases':cases,
            'seconds':sum(v['seconds'] for v in cases),
            'load_before':load_before,'load_after':load_after,
            'load_ok':load_before+workers<=len(os.sched_getaffinity(0)) and abs(load_after-load_before)<=4.,
            'resource_ok':all(v['resource_ok'] for v in cases),
            'science_ok':all(v['science_ok'] for v in cases)}
    r.dump(path/'group.json',result)
    return result


def main():
    r.environment=scoped_environment
    before=r.source_snapshot();r.dump(BASE/'source_before.json',before)
    warmups=[];groups=[]
    with r.timing_lock():
        for workers in (1,2,4):
            value=group(workers,BASE/'warmup'/f'p{workers}')
            warmups.append(value);r.dump(BASE/'warmup.json',warmups)
            print(json.dumps({'phase':'warmup','workers':workers,'seconds':value['seconds'],
                              'resource_ok':value['resource_ok'],'science_ok':value['science_ok']}),flush=True)
            if not value['science_ok'] or not value['resource_ok']:
                raise RuntimeError('Fresh complete per-ion-cache warmup failed; no measured repetitions')
        for round_index,order in enumerate(((1,2,4),(4,2,1),(1,2,4))):
            for workers in order:
                value=group(workers,BASE/'measured'/f'round{round_index:03d}'/f'p{workers}')
                value['round']=round_index;groups.append(value)
                r.dump(BASE/'groups.raw.json',groups)
                print(json.dumps({'phase':'measured','round':round_index,'workers':workers,
                                  'seconds':value['seconds'],'load_ok':value['load_ok'],
                                  'resource_ok':value['resource_ok'],'science_ok':value['science_ok']}),flush=True)
                if not value['science_ok'] or not value['resource_ok']:
                    raise RuntimeError('Fresh complete epoch failed; retained and stop without more repetitions')
        for value in groups:
            r.verify_against_reference(value,groups[0])
        r.dump(BASE/'groups.verified.json',groups)
    after=r.source_snapshot();r.dump(BASE/'source_after.json',after)
    changed={k:{'before':v,'after':after.get(k)} for k,v in before.items() if after.get(k)!=v}
    guard={'passed':not changed and set(after)==set(before),'changed':changed,
           'missing':sorted(set(before)-set(after)),'added':sorted(set(after)-set(before)),
           'files':len(before)}
    r.dump(BASE/'source_guard.json',guard)
    result={'scope':'Fresh per-ion Numba compiler cache isolates stale dynamic-dispatcher cache index; same actual Atomx source/input/model/caps. Original global-cache Pb failure preserved.',
            'cpu_rows':r.summarize(groups),'groups':groups,'warmups':warmups,'source_guard_passed':guard['passed']}
    r.dump(BASE/'result.json',result)


if __name__=='__main__':
    main()

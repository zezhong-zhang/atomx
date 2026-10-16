"""Close saved-algebra JSON and bound archives with stdlib only."""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    def read(path):
        return json.loads(Path(path).read_text())
    r, e, c = read(HERE/'actual/result.json'), read(HERE/'root_execution.json'), read(HERE/'contract.json')
    controls = []

    def check(name, passed):
        controls.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise ValueError(name)

    check('exact closed successful sole saved-algebra consumer', e['status']=='foreground_closed' and e['exit_code']==0
          and r['status']=='completed' and r['all_declared_controls_passed'] is True)
    check('all149 original offline gates passed and zero physical calls', len(r['controls'])==149
          and all(v['passed'] for v in r['controls']) and r['scientific_calls']==0
          and r['array_reads_attempted']==r['array_reads_completed']==2)
    inputs = {str(HERE/filename):sha(HERE/filename) for filename in
              ('actual/result.json','root_execution.json','execution_contract.json','contract.json','run.py')}
    check('exact root result, source and contract identity', e['actual_files_sha256']['actual/result.json']==sha(HERE/'actual/result.json')
          and r['driver_sha256']==e['driver_sha256']==sha(HERE/'run.py')
          and r['contract_sha256']==e['contract_sha256']==sha(HERE/'contract.json')
          and r['execution_sha256']==e['execution_contract_sha256']==sha(HERE/'execution_contract.json'))
    check('literal80and208 point model/metric lineage', [v['global_accepted_step'] for v in r['points']]==[80,208]
          and all(v['metric_sha256']==c['identity']['metric_sha256'] and v['geometry_sha256']==c['identity']['geometry_sha256']
                  and v['source_map_sha256']==c['identity']['source_map_sha256'] and v['objective_sha256']==c['identity']['objective_sha256']
                  and v['stationarity_passed'] is False for v in r['points']))
    for point in r['points']:
        path=HERE/'actual'/(point['point']+'_algebra.npz')
        if sha(path)!=point['saved_algebra_sha256']:
            raise ValueError('Changed saved algebra archive')
        inputs[str(path)]=sha(path)
    summary=[]
    for point in r['points']:
        rs=point['orbital_residual']**2
        group_shares={group:math.fsum(v['pg_norm_squared'] for v in point['per_orbital_columns'] if v['group']==group)/rs
                      for group in point['group_order']}
        summary.append({'global_step':point['global_accepted_step'],'residual':point['orbital_residual'],
                        'indicator_Ha':point['energy_indicator_Ha'],'coordinate_residual_squared_shares':group_shares,
                        'dominant_columns':sorted(point['per_orbital_columns'],key=lambda v:v['pg_norm_squared'],reverse=True)[:6],
                        'dominant_kappa':sorted(point['per_kappa'],key=lambda v:v['pg_norm_squared'],reverse=True),
                        'raw_mask_TMT_driver_norms':point['TMT_group_drive_norms'],
                        'complete_signed_cross_work_matrix':point['TMT_group_signed_work_matrix'],
                        'complete_residual_cross_matrix':point['TMT_group_residual_cross_matrix']})
    check('finite saved column/group scalar bookkeeping', all(math.isfinite(v) for v in
          [p['residual'] for p in summary]+[v for p in summary for v in p['coordinate_residual_squared_shares'].values()]))
    check('stdlib completion has no numerical or physical import', not any(n=='numpy' or n.startswith(('numpy.','atomx.'))
          or n=='atomx' for n in sys.modules))
    before,after=r['points']
    initial={tuple(v['key']):v for v in before['per_orbital_columns']}
    reductions=[{'label':v['label'],'delta_pg_squared':initial[tuple(v['key'])]['pg_norm_squared']-v['pg_norm_squared']}
                for v in after['per_orbital_columns']]
    total_reduction=before['orbital_residual']**2-after['orbital_residual']**2
    for row in reductions:
        row['share_of_net_residual_squared_reduction']=row['delta_pg_squared']/total_reduction
    out={'all_completion_checks_passed':True,'checks':len(controls),'controls':controls,'inputs_sha256':inputs,
         'completion_driver_sha256':sha(__file__),'actual_original_offline_controls':149,
         'scientific_calls':0,'array_numerical_replays_in_completion':0,'point_summaries':summary,
         'residual_squared_reduction':total_reduction,'residual_squared_reduction_fraction':total_reduction/before['orbital_residual']**2,
         'column_reductions':sorted(reductions,key=lambda v:v['delta_pg_squared'],reverse=True),
         'original_metric_and_stationarity_flags_unchanged':True,'core_dominance_supported':False,
         'physical_or_error_bound_claim':False,'scope':'Close sole149gate saved-coordinate replay; JSON scalar analysis is not a second full-array or physical replay.'}
    (HERE/'completion.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks':len(controls),'completion_sha256':sha(HERE/'completion.json'),'scientific_calls':0}))


if __name__=='__main__':
    main()

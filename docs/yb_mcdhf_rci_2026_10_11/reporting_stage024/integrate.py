"""Append closed paired search evidence and correct the current native spectrum panel."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from atomx_report_utils import bound, load, once, sha

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = TOP.parents[1]
REGISTRY = REPO/'docs/atomx_hourly_progress_2026_10_10/progress.json'


def main():
    contract = load(HERE/'contract.json')
    assert sha(Path(__file__))==contract['driver_sha256']
    assert sha(REGISTRY)==contract['registry_before_sha256']
    checks=[]

    def check(name, condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:
            raise ValueError(name)

    for rel, digest in contract['inputs'].items():
        check('input/'+rel,sha(REPO/rel)==digest)
    previous=load(TOP/'results_stage023.json')
    pair=TOP/'correlated29_history_comparison_preparation'
    common=load(TOP/'correlated29_optimization_run005_preparation/actual/checkpoints/accepted_0128.json')
    branches={}
    for name in ('history10','history40'):
        d=pair/name
        actual, root=load(d/'actual/result.json'),load(d/'root_execution.json')
        check(name+'/closed',root['status']=='foreground_closed' and root['exit_code']==0
              and actual['status']=='bounded_public_optimizer_returned' and actual['error'] is None)
        check(name+'/all2219controls',len(actual['controls'])==2219 and all(c['passed'] for c in actual['controls']))
        check(name+'/all33points',len(actual['trajectory'])==33 and
              [v['iteration'] for v in actual['trajectory']]==list(range(33)))
        for step in actual['trajectory']:
            p=REPO/step['manifest_path']
            check(name+'/point'+str(step['iteration']),sha(p)==step['manifest_sha256'])
            m=load(p)
            check(name+'/gate'+str(step['iteration']),m['checkpoint_qualified']
                  and m['source_map_sha256']==common['source_map_sha256']
                  and m['metric_sha256']==common['metric_sha256'] and m['objective_sha256']==common['objective_sha256']
                  and (step['iteration']==0 or m['exact_acceptance']['qualified']))
        terminal=load(REPO/actual['trajectory'][-1]['manifest_path'])
        check(name+'/original_failure',not terminal['stationarity_passed'] and not terminal['experimental_certification'])
        branches[name]={'actual':bound(d/'actual/result.json'),'root_receipt':bound(d/'root_execution.json'),
            'terminal':bound(REPO/actual['trajectory'][-1]['manifest_path']),
            'global_accepted_step_on_branch':240,'fork_baseline_global_step':208,'new_accepted_steps':32,
            'completed_calls':{k:actual['completed_calls'][k] for k in ('H','CI','gradient','optimizer','validation_Dirac')},
            'controls':2219,'qualified_points':33,'stationarity':terminal['diagnostics'],'stationarity_passed':False,
            'nominal_four_gap_RMS_cm1':terminal['nominal_four_gap_RMS_cm1'],'targets':terminal['targets'],
            'multiplets':terminal['multiplets'],'elapsed_uncontrolled_seconds':root['elapsed_uncontrolled_seconds'],
            'speedup_certified':False}
    pair_close=load(pair/'completion.json')
    check('pair_completion',pair_close['all_completion_metadata_checks_passed'] and pair_close['same_baseline_identity_check_passed'] and pair_close['no_physical_imports'])
    adopted=TOP/'correlated29_radial_reuse_qualification003_preparation/shared_implementation'
    a=load(adopted/'completion.json')
    check('shared_adopted123',a['status']=='shared_applied_affected_regression_passed'
          and a['root_test_exit_code']==0 and a['complete_source_map_matches_private_candidate']
          and a['source_file_count']==356)
    route=TOP/'physical_next_step208'
    check('source_integer_route_only',load(route/'verification.json')['all_checks_passed']
          and load(route/'summary.json')['new_scientific_calls']==0)
    delivered=load(TOP/'delivery_stage023/delivery_proof.json')
    check('actual_card_then_pair',delivered['passed'] and delivered['source_stage']['number']==23
          and delivered['actual_readback_calls_by_root']==1
          and set(delivered['separate_root_bounded_iteration']['branches'])=={'history10','history40'})
    result=copy.deepcopy(previous)
    result['created_at']=datetime.now(timezone.utc).isoformat()
    for rel,digest in contract['inputs'].items():
        result['inputs'][rel]={'path':rel,'sha256':digest}
    result['historical_stage023_primary_view']={k:previous[k] for k in ('card_summary','selection_reason','next_execution_status')}
    result['matched_history32_comparison']={'branches':branches,'completion':bound(pair/'completion.json'),
        'comparison':{k:pair_close['comparison'][k] for k in ('same_literal208_baseline_qualified','common_accepted_horizon','first_observed_numerical_divergence','history_capacity_interpretation','producer_work','single_winner_selected','NIST_used_to_select_memory')},'pair_receipt':bound(pair/'root_pair_execution.json'),
        'sole_search_parameter':'history_size10vs40; both empty on resume','physical_model_changed':False,
        'stationarity_metric_changed':False,'default_history_changed':False,'NIST_selected_winner':False,
        'no_unique_global240_endpoint':True,'memory40_not_fully_populated_within32':True}
    result['shared_numerical_adoption']={'completion':bound(adopted/'completion.json'),
        'README':bound(adopted/'README.md'),'affected_regression_counts':a['test_counts'],
        'whole356_matches_qualified_source':True,'source_map_sorted_json_sha256':a['shared_source_map_sorted_json_sha256'],
        'timing_certified':False,'native_f043_runs_source_unchanged':True}
    result['physical_next_step208']={'summary':bound(route/'summary.json'),'verification':bound(route/'verification.json'),
        'integer_estimates':bound(route/'integer_space_estimates.json'),'README':bound(route/'README.md'),
        'new_scientific_calls':0,'next_candidate':'Complete5f dual-jj nested fixed208 RCI scale discrimination; prospective, not executed.',
        'main_missing_physics_identified':False,'same208_stationarity_failed':True}
    result['latest_manual_delivery']={'proof':bound(TOP/'delivery_stage023/delivery_proof.json'),
        'message_id':delivered['message_id'],'actual_message_link':delivered['actual_message_link'],
        'scope':'Actualstage023 then two bounded native32-step pilots. Service additionally four small registered comparison probes; no service native-Yb optimizer.'}
    result['selection_reason']=('同208两路32步：history10残差3.56e−4/指标3.69e−7Ha，history40残差6.49e−4/指标3.31e−7Ha；'
        '两驻值门仍FAIL。40较低EOL但残差较大，43vs47H/CI是工作量区别，不以gap或单指标宣称赢家。'
        '有限搜索参数改变仅使四线RMS2054.32→约2051.4cm⁻¹；下一以完整5f双jj固定208 nested-RCI'
        '筛选缺角通道量级，再单独4d/多核孔与同点修正，保留未驻值边界。')
    result['next_execution_status']=('完整5f双jj同点RCI源-only准备中，1595完整目标CSF/31轨道，新空间epoch；'
        '根审资源/签名/nested-H后才实算。已采用两数值复用改进，123回归PASS；同H/CI/g受控暖态配对另备，未有加速比。')
    result['card_summary']=('Atomx原生MCDHF→RCI：同208/29轨道1046目标CSF的history10/40各32步已前台闭合，'
        '两路各2219控制、33检查点和32接受门PASS，原驻值与实验门仍FAIL。10/40末态残差3.56e−4/6.49e−4，'
        'RMS2051.37/2051.47cm⁻¹；47/43H-CI、各33g。未更改默认或按实验选优。'
        '下一完整5f双通道固定点量级筛选；共享两处数值改进已采用并过123回归，受控性能待测。'
        '网页v7仍保留共同208及19模型76行/209点，双分支尚未发布。最高水平未达。')
    baseline_rows=[]
    for t in common['targets']:
        uid=t['UID']
        val=t['excitation_cm1']
        ref=t['NIST_nominal_cm1']
        error=val-ref
        baseline_rows.append({'label':t['label'],'twoJ':uid[0],'parity':uid[1],'energy_Ha':t['energy_Ha'],
            'computed_cm1':val,'experimental_cm1':ref,'error_cm1':error,'relative_error':error/ref if ref else None,
            'gate_pass':abs(error/ref)<=.01 if ref else True,'uncertainty_cm1':None})
    table=['|谱线|同208基线|history10末态|history40末态|NIST名义值|','|---|---:|---:|---:|---:|']
    rows_by={n:{t['label']:t for t in v['targets']} for n,v in branches.items()}
    for row in baseline_rows:
        if row['experimental_cm1']:
            label=row['label']
            table.append('|'+label+'|'+f"{row['computed_cm1']:.2f}|{rows_by['history10'][label]['excitation_cm1']:.2f}|{rows_by['history40'][label]['excitation_cm1']:.2f}|{row['experimental_cm1']:.2f}|" )
    panel={'method':'MCDHF -> RCI','computed_run':'correlated29同208共同基线；history10/40各32步分叉末态另列，未选择唯一末态',
        'producer':'Atomx native public mixed-CI MCDHF f043 immutable source','CSFs':1046,'radial_orbitals':29,'electrons':69,
        'stationarity':copy.deepcopy(previous['stationarity']),'rows':baseline_rows,'production_certified':False,
        'experimental_gate_passed':False,'highest_level_reached':False,'fitted_corrections':False,
        'inputs':{str(i):bound(REPO/rel) for i,rel in enumerate(contract['inputs'])},
        'paired_endpoint_rows':branches,'card_summary':'上述驻值量属于共同208基线；两分支仍FAIL，单位cm⁻¹。\n'+'\n'.join(table)+'\n'+result['selection_reason'],
        'historical_panel_preserved':bound(TOP/'reporting_stage006/current_card_evidence.json')}
    check('panel_original208_metrics',panel['stationarity']==previous['stationarity'])
    registry=load(REGISTRY)
    registry['updated_at']=result['created_at']
    registry['physics_budget_result']=str((TOP/'results_stage024.json').relative_to(REPO))
    registry['yb_mcdhf_result']=str((HERE/'current_card_evidence.json').relative_to(REPO))
    registry['tasks'][0].update(status='同点两路32步比较完成；驻值仍FAIL',detail=result['card_summary'])
    registry['tasks'][1].update(status='完整缺失角通道量级判别',detail=result['selection_reason'])
    registry['tasks'][5].update(status='共享改进123回归PASS；受控暖态比较准备',detail=result['next_execution_status'])
    registry['physics_next_setups'][1]=result['selection_reason']
    registry['physics_next_setups'][5]=result['next_execution_status']
    once(HERE/'registry_before.json',load(REGISTRY))
    once(HERE/'current_card_evidence.json',panel)
    result['current_native_card_evidence']=bound(HERE/'current_card_evidence.json')
    once(TOP/'results_stage024.json',result)
    REGISTRY.write_text(json.dumps(registry,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    once(HERE/'integration_proof.json',{'all_passed':True,'checks':checks,'checks_count':len(checks),
         'new_science_calls':0,'API_calls':0,'cards_sent':0,'driver':bound(Path(__file__)),'contract':bound(HERE/'contract.json'),
         'result':bound(TOP/'results_stage024.json'),'registry':bound(REGISTRY),'panel':bound(HERE/'current_card_evidence.json')})
    print(json.dumps({'checks':len(checks),'all_passed':True,'result':bound(TOP/'results_stage024.json'),
                      'registry':bound(REGISTRY),'panel':bound(HERE/'current_card_evidence.json')}))


if __name__=='__main__':
    main()

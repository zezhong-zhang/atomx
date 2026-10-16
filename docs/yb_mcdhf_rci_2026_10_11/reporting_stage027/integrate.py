"""Current report: independent finite-array qualification; derivative attempt failure."""
import copy,hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent
T=B.parent
R=T.parents[1]
H=R/'docs/atomx_hourly_progress_2026_10_10'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(p):return {'path':str(p.relative_to(R)),'sha256':sha(p)}
def once(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    c=read(B/'contract.json');registry=H/'progress.json';checks=[]
    def check(n,v):
        checks.append({'name':n,'passed':bool(v)});assert v,n
    check('driver',sha(__file__)==c['driver_sha256'])
    check('live_registry_before',sha(registry)==c['registry_before_sha256'])
    for p,d in c['inputs'].items():check('frozen/'+p,sha(R/p)==d)
    previous=read(T/'results_stage026.json');record=read(registry);old=read(R/record['yb_mcdhf_result'])
    audit=T/'complete5f_fixed208_import_repair_preparation/completion_counter_audit/saved_array_qualification_preparation/output_serialization_repair'
    saved=read(audit/'actual/verification.json');receipt=read(audit/'root_execution.json')
    check('independent_389_actual_PASS',receipt['exit_code']==0 and saved['saved_array_qualification_passed'] and len(saved['checks'])==389 and all(x['passed'] for x in saved['checks']))
    check('saved18_no_new_producers',saved['NPZ_loads']==18 and saved['eigensolves_SVD_Atomx_H_CI_g_Fock_Dirac_optimizer_calls']==0)
    check('original_failures_preserved',not saved['original_expected14gate_passed'] and not saved['physics_complete'] and read(audit.parent/'root_execution.json')['exit_code']==1)
    fd=T/'stationarity_next_step208';failed=read(fd/'actual/result.json')
    check('FD_instrumentation_only_failure',read(fd/'root_execution.json')['exit_code']==1 and failed['error'].startswith('FrozenInstanceError:') and all(failed['completed'][k]==0 for k in ('H','CI','gradient','scipy_eigh','polar','optimizer')) and len(failed['controls'])==58 and all(x['passed'] for x in failed['controls']))
    summary=('Atomx原生MCDHF→RCI主线继续，原驻值与实验门仍FAIL。固定208核场5f完整1046/1595目标RCI保存数组已独立389项PASS：H/全符号谱/Gram/嵌套Ritz/29波函数/118完整虚层/UID差能；0新求解。原14vs7调用计数失败、审计序列化失败均保留，独立资格不改写原整轮。新增核内局域f使四线移动−6.40/−3.18/+8.18/+9.10cm⁻¹，约千cm⁻¹实验差距仍在，不能排除f径向不足。实际208导数首轮被不可变tracker观测钩子阻断：7初始h/S、58先行门PASS、0H/CI/g，导数未测；修复观测后另准入。屏蔽f执行器由原生closed68 J/K与32独立符号矩阵元锚开始，尚未数值执行。暖态负载FAIL无加速认证；网页v8已发布21模型84行。最高水平未达。')
    result=copy.deepcopy(previous);panel=copy.deepcopy(old);now=datetime.now(timezone.utc)
    result.update(created_at=now.isoformat(),card_summary=summary,selection_reason='先检验真实208约束导数与屏蔽f径向种子；已资格的小核场增量不能证明物理完备或排除f。依据原生物理量级推进，不按NIST选解。',next_execution_status='导数观测钩子修复后独立准入；屏蔽f先32独立矩阵元锚、再118谱与完整1595RCI。小时服务与root科学运行分别记账。')
    qualification={'verification':bound(audit/'actual/verification.json'),'receipt':bound(audit/'root_execution.json'),'checks_PASS':389,'unique_NPZ_loads':18,'new_physical_producer_calls':0,'original_failed_execution_preserved':True,'finite_saved_array_scope_only':True,'expanded_MCDHF_stationarity_evaluated':False,'radial_adequacy_certified':False}
    result['complete5f_saved_array_qualification']=qualification
    result['complete5f_fixed208_failed_diagnostic']['independent_saved_array_qualification_completed']=True
    result['complete5f_fixed208_failed_diagnostic']['independent_qualification']=qualification
    result['literal208_derivative_failed_instrumentation']={'result':bound(fd/'actual/result.json'),'failure':bound(fd/'actual/failure.json'),'receipt':bound(fd/'root_execution.json'),'observed_preliminary_gates_PASS':58,'completed':failed['completed'],'directional_derivative_evaluated':False,'original_failed_attempt_preserved':True,'status':'instrumentation_failure_before_H_CI_gradient'}
    for p,d in c['inputs'].items():result['inputs'][p]={'path':p,'sha256':d};panel['inputs'][p]={'path':p,'sha256':d}
    base_summary=panel['card_summary'].split('\n5f中间诊断：')[0]
    panel['card_summary']=base_summary+'\n固定208核场5f：389项独立保存数组门PASS，四线位移约−6.40/−3.18/+8.18/+9.10cm⁻¹；原计数FAIL保留，未新增MCDHF驻值。导数尝试因观测钩子失败，0H/CI/g；待修复，尚无导数结论。'
    panel['complete5f_saved_array_qualification']=qualification
    panel['complete5f_failed_diagnostic']=result['complete5f_fixed208_failed_diagnostic']
    check('baseline_physics_unchanged',panel['rows']==old['rows'] and panel['stationarity']==old['stationarity'] and not panel['production_certified'])
    once(B/'current_card_evidence.json',panel)
    result['inputs']['historical_current_mcdhf_stage026']=previous['inputs']['current_mcdhf']
    result['inputs']['current_mcdhf']=bound(B/'current_card_evidence.json');result['current_native_card_evidence']=bound(B/'current_card_evidence.json')
    for k,v in result['inputs'].items():check('result_identity/'+k,sha(R/v['path'])==v['sha256'])
    once(T/'results_stage027.json',result)
    record.update(physics_budget_result=str((T/'results_stage027.json').relative_to(R)),yb_mcdhf_result=str((B/'current_card_evidence.json').relative_to(R)),updated_at=result['created_at'])
    record['tasks'][0].update(status='驻值FAIL；核场5f独立389门PASS',detail=summary)
    record['tasks'][1].update(status='屏蔽f与实际约束导数优先',detail=result['selection_reason'])
    record['tasks'][5].update(status='导数观测失败待修复；暖态负载FAIL',detail=result['next_execution_status'])
    record['physics_next_setups'][1]=result['selection_reason'];record['physics_next_setups'][5]=result['next_execution_status']
    once(B/'registry_before.json',read(registry));(B/'planner_scope').mkdir();once(B/'planner_scope/progress.json',record)
    sys.path.insert(0,str(H));spec=importlib.util.spec_from_file_location('stage027_planner',H/'parallel_planner.py');planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
    d={'status':'implementation_frontier','jobs':[],'frontier':[],'evidence':[]};planner.annotate_physics_priority(R,B/'planner_scope',d)
    check('public_proposed_planner_binding',d['direction']==summary and d['evidence'][-1]['sha256']==sha(T/'results_stage027.json'))
    import sync_card
    card,citations=sync_card.render(record,now);check('actual_public_card_render',card['schema']=='2.0' and bool(citations));once(B/'preview_card.json',card);once(B/'preview_citations.json',citations)
    check('registry_atomic_before',sha(registry)==c['registry_before_sha256'])
    temporary=registry.with_suffix('.stage027.tmp');temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(registry)
    d={'status':'implementation_frontier','jobs':[],'frontier':[],'evidence':[]};planner.annotate_physics_priority(R,H,d);check('public_live_planner_binding',d['direction']==summary)
    once(B/'integration_proof.json',{'all_passed':True,'checks':checks,'checks_count':len(checks),'driver':bound(Path(__file__)),'contract':bound(B/'contract.json'),'result':bound(T/'results_stage027.json'),'panel':bound(B/'current_card_evidence.json'),'registry':bound(registry),'new_science_API_cards':0,'original_failures_and_model_limits_preserved':True})
    print(json.dumps({'PASS':len(checks),'result':bound(T/'results_stage027.json'),'registry':bound(registry),'card_elements':len(card['body']['elements']),'citations':len(citations)}))
if __name__=='__main__':main()

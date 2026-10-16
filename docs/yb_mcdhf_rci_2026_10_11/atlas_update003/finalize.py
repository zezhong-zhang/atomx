"""Saved metadata/hash/doclink final handoff; zero science or publication."""
import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=load(HERE/'contract.json');site=Path(c['site']);completion=load(HERE/'completion.json');data=load(HERE/'data_verification.json');browser=load(HERE/'browser_result.json');manifest=load(HERE/'site_manifest.json');checks=[]
def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
for name,digest in completion['inputs_sha256'].items():check('completion_exact:'+name,sha(HERE/name)==digest)
for rel,item in manifest['changed_or_added_files'].items():check('new_site_exact:'+rel,sha(site/rel)==item['sha256'])
check('hosting_manifest_exact',sha(site/'.openai/hosting.json')==manifest['files']['.openai/hosting.json']['sha256'])
check('exact_page_in_all_proofs',completion['page_sha256']==data['page_sha256']==browser['page_sha256']==sha(site/'dist/index.html'))
check('exact_overlay_in_all_proofs',completion['overlay_sha256']==data['overlay_sha256']==browser['overlay_sha256']==sha(site/c['new_overlay']))
check('data6677_passed',data['all_passed'] and data['checks_count']==6677 and all(x['passed'] for x in data['checks']))
check('browser67_passed',browser['all_passed'] and browser['checks_count']==67 and all(x['passed'] for x in browser['checks']) and not browser['javascript_errors'])
check('all_scopes14_56',completion['model_count']==data['models']==browser['models']==14 and completion['row_count']==data['rows']==browser['rows']==56)
check('original48_retained',data['prior_rows']==browser['old_rows_preserved']==completion['original_rows_retained']==48)
check('15sitechanges_zero_removed',len(manifest['changed_or_added_files'])==15 and not manifest['removed_files'])
check('site_total_bytes',data['uncompressed_static_bytes']==manifest['total_static_bytes']==completion['static_bytes']==52892373)
check('no_science_no_publication',completion['new_scientific_calls']==data['scientific_calls']==browser['science_calls']==0 and not completion['publication_performed_here'] and not data['publication_performed'] and not browser['publication_performed'])
check('original_gates_FAIL',not completion['original_stationarity_gates_passed'] and not completion['rank_expanded_stationarity_evaluated'])
check('teardown',load(HERE/'teardown.json')['exit_code']==0 and not load(HERE/'teardown.json')['server_and_browser_running'])
browsercontracts=['browser_contract.json','browser_contract002.json','browser_contract003.json','browser_contract004.json','browser_contract005.json']
scripts=['browser_test_initial.js','browser_test_failure002.js','browser_test_failure003.js','browser_test_failure004.js','browser_test.js']
for i,(name,script) in enumerate(zip(browsercontracts,scripts,strict=True)):
    x=load(HERE/name);check(name+'/driver_epoch',x['driver_sha256']==sha(HERE/script))
    if i:check(name+'/predecessor_epoch',x['predecessor_contract_sha256']==sha(HERE/browsercontracts[i-1]))
for name in completion['failed_browser_QA_evidence']:check('preserved_failure:'+name,(HERE/name).is_file())
check('failed004_61checks_not_called_complete',len(load(HERE/'browser_failure004.json')['checks'])==61 and all(x['passed'] for x in load(HERE/'browser_failure004.json')['checks']))
for rel in re.findall(r'\]\(([^)]+)\)',(HERE/'README.md').read_text()):
    if rel not in {'handoff.json','verification.json'}:check('README_link:'+rel,(HERE/rel).exists())
check('README_facts',all(t in (HERE/'README.md').read_text() for t in ['14 models / 56 rows','6677 checks pass','67 checks pass','1240 signed CSFs','zero new H/CI/gradient/optimizer']))
paths=[*completion['inputs_sha256'],'completion.json','README.md','docimpact.json',*browsercontracts,*scripts,*completion['failed_browser_QA_evidence'],'finalize.py']
inputs={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in paths}
report={'all_passed':all(x['passed'] for x in checks),'checks_count':len(checks),'checks':checks,'inputs_sha256':inputs,'page_sha256':completion['page_sha256'],'overlay_sha256':completion['overlay_sha256'],'scope':'Final saved metadata/hash/doclink qualification; prior data/browser proofs consumed rather than repeated. No scientific calls.','scientific_calls':0,'publication_performed':False}
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['all_passed']:print(json.dumps([x for x in checks if not x['passed']]));raise SystemExit(1)
handoff={'status':'root_review_ready_local_v4_no_publication_performed','site_checkout':str(site),'initial_HEAD':c['initial_HEAD'],'audience':'Preserve existing owner custom/private revision1 users1 editors0 external0; rootget_site verified. No access change here.','changed_or_added_files':list(manifest['changed_or_added_files']),'site_manifest_sha256':sha(HERE/'site_manifest.json'),'page_sha256':completion['page_sha256'],'overlay_sha256':completion['overlay_sha256'],'contract_sha256':sha(HERE/'contract.json'),'completion_sha256':sha(HERE/'completion.json'),'data_verification_sha256':sha(HERE/'data_verification.json'),'browser_verification_sha256':sha(HERE/'browser_result.json'),'verification_sha256':sha(HERE/'verification.json'),'README_sha256':sha(HERE/'README.md'),'docimpact_sha256':sha(HERE/'docimpact.json'),'new_models':2,'new_rows':8,'total_models':14,'total_rows':56,'data_checks':6677,'browser_checks':67,'final_hash_doc_checks':len(checks),'source_map_sha256':c['fixed_conditions']['source_epoch'],'original_stationarity_passed':False,'rank_expanded_stationarity_evaluated':False,'all_old_48rows_and_6019gzip_exact':True,'static_bytes':52892373,'scientific_calls':0,'publication_performed':False,'owned_preview_stopped':True,'scope_fixed_no_virtual_additions':True}
(HERE/'handoff.json').write_text(json.dumps(handoff,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'all_passed':True,'final_checks':len(checks),'handoff':sha(HERE/'handoff.json'),'verification':sha(HERE/'verification.json'),'data':handoff['data_verification_sha256'],'browser':handoff['browser_verification_sha256'],'completion':handoff['completion_sha256']}))

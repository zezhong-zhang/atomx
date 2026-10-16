"""Capture existing02HKT hourly send once; no send/new timer/science."""
import difflib,hashlib,importlib.util,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parent
R=B.parents[2]
S=Path('/home/zzhang/.local/state/atomx-lark-hourly-aisi-codex')
REG=R/'docs/atomx_hourly_progress_2026_10_10/progress.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    c=read(B/'contract.json');status=read(S/'last_status.json')
    assert status['cycle']==c['cycle'] and status['passed'] and status['iteration_dispatch_ok']
    assert status['registry_sha256']==sha(REG)==c['registry_SHA']
    key=status['idempotency_key'];request=read(S/'iteration_requests'/(key+'.json'));service=read(S/'iteration_status.json')
    assert request['key']==service['key']==key and request['message_id']==service['message_id']==status['message_id']
    write(B/'once_marker.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'cycle':status['cycle'],'message_id':status['message_id']})
    for original,saved in {'latest_card.json':'card.json','last_status.json':'send_status.json','receipt_'+key+'.json':'receipt.json','iteration_requests/'+key+'.json':'iteration_request.json','iteration_status.json':'service_iteration_status.json'}.items():(B/saved).write_bytes((S/original).read_bytes())
    (B/'registry_stage028.json').write_bytes(REG.read_bytes())
    command=['lark-cli','im','+messages-mget','--message-ids',status['message_id'],'--no-reactions','--json'];t=time.monotonic()
    with (B/'readback.json').open('xb') as out,(B/'readback_stderr.log').open('xb') as err:p=subprocess.run(command,stdout=out,stderr=err,timeout=60)
    write(B/'readback_execution.json',{'actual_readback_calls':1,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t,'command':command,'readback_sha256':sha(B/'readback.json')})
    assert p.returncode==0
    back,card,receipt=map(read,(B/'readback.json',B/'card.json',B/'receipt.json'))
    m=back['data']['messages'][0];assert back['ok'] and back['identity']=='bot' and back['data']['total']==1
    assert m['message_id']==receipt['data']['message_id']==status['message_id'] and m['chat_id']==receipt['data']['chat_id']==c['chat']
    assert m['sender']['open_bot_id']==c['bot'] and m['sender']['sender_type']=='app' and m['msg_type']=='interactive' and not m['deleted'] and not m['updated']
    utility_path=B.parent/'delivery_stage021/verify_saved_delivery.py';assert sha(utility_path)==c['text_utility_SHA']
    spec=importlib.util.spec_from_file_location('frozen_card_text',utility_path);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    expected='<card title="'+card['header']['title']['content']+'">\n'+'\n'.join(e['content'] for e in card['body']['elements'])+'\n</card>'
    a,b=u.visible(expected),u.visible(m['content']);diff=[{'operation':op,'payload':a[x:y],'readback':b[v:w]} for op,x,y,v,w in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if op!='equal']
    table_rows=[line.strip() for e in card['body']['elements'] for line in e['content'].splitlines() if line.strip().startswith('|') and '---' not in line]
    source=B.parent/'results_stage028.json';assert sha(source)==c['result_SHA'] and any(v['path']==str(source.relative_to(R)) and v['sha256']==sha(source) for v in status['evidence'])
    assert card['schema']=='2.0' and len(card['body']['elements'])==15
    gate=diff==c['allowed_exact_table_format_operations'] and all(u.visible(row) in b for row in table_rows)
    proof={'passed':gate,'created_utc':datetime.now(timezone.utc).isoformat(),'message_id':m['message_id'],'chat_id':m['chat_id'],'bot_open_id':m['sender']['open_bot_id'],'position':int(m['message_position']),'actual_message_link':m['message_app_link'],'cycle':status['cycle'],'source_stage':28,'source_stage_sha256':sha(source),'registry_sha256':sha(REG),'actual_card':{'schema':'2.0','elements':15,'raw_visible_string_equal':a==b,'exact_observed_formatting_operations':diff,'all_table_rows_match':all(u.visible(row) in b for row in table_rows),'nonformatting_text_exact_by_bound_operation_list':gate},'actual_send':{'passed':True,'started_at':status['started_at'],'finished_at':status['finished_at']},'service_iteration':{'status':service['status'],'automatic_jobs_in_top_level':len(service.get('jobs',[])),'decision_status':service.get('decision',{}).get('status'),'highest_level_reached':service.get('decision',{}).get('highest_level_reached')},'actual_readback_calls_by_root':1,'new_send_calls':0,'new_scientific_calls':0,'request_key':key,'files_sha256':{p.name:sha(p) for p in B.glob('*.json')},'verifier_sha256':sha(__file__)}
    write(B/'delivery_proof.json',proof)
    print(json.dumps({'passed':gate,'message_id':m['message_id'],'position':proof['position'],'service_status':service['status'],'proof_sha256':sha(B/'delivery_proof.json'),'observed_format_operations':len(diff),'new_send_science':0}))
    return 0 if gate else 1
if __name__=='__main__':raise SystemExit(main())

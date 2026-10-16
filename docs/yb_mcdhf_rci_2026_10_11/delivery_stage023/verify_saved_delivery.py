"""Saved actual send/readback and optional closed paired iteration; no API/science."""
import difflib
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]
U = B.parent/'delivery_stage021/verify_saved_delivery.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    assert sha(U) == 'f8bf88f8caae2285620273f449c6f3b3d705364cc8115b2699284dd7c0bba093'
    spec = importlib.util.spec_from_file_location('archived_delivery_text_utils', U)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    status, receipt, card, back, executed, request, service = [read(B/p) for p in (
        'send_status.json', 'receipt.json', 'card.json', 'readback.json', 'readback_execution.json',
        'iteration_request.json', 'service_iteration_status.json')]
    m = back['data']['messages'][0]
    assert back['ok'] and back['identity']=='bot' and back['data']['total']==1
    assert m['message_id']==receipt['data']['message_id']==status['message_id']=='om_x100b638fe573bca0c4b8f84994a0f31'
    assert m['chat_id']==receipt['data']['chat_id']==utility.CHAT
    assert m['sender']['open_bot_id']==utility.BOT and m['sender']['sender_type']=='app'
    assert m['msg_type']=='interactive' and not m['deleted'] and not m['updated']
    assert executed['actual_readback_calls']==1 and executed['exit_code']==0
    assert executed['readback_sha256']==sha(B/'readback.json')
    assert status['cycle']=='mcdhf-stage023' and status['passed'] and status['iteration_dispatch_ok']
    assert status['registry_sha256']==sha(B/'registry_stage023.json')=='f0feab13bb351a47715bfa2a16fcd7a1e54d126dabfbc7602fc2b46ba1ad2ddb'
    assert card['schema']=='2.0' and len(card['body']['elements'])==15
    assert all(e['tag']=='markdown' for e in card['body']['elements'])
    expected='<card title="'+card['header']['title']['content']+'">\n'+'\n'.join(e['content'] for e in card['body']['elements'])+'\n</card>'
    assert utility.visible(expected)==utility.visible(m['content'])
    differences=[{'operation':op,'payload':expected[a:b],'readback':m['content'][c:d]} for op,a,b,c,d in
        difflib.SequenceMatcher(None,expected,m['content'],autojunk=False).get_opcodes() if op!='equal']
    stage=B.parent/'results_stage023.json'
    assert sha(stage)=='969c11b63755d275ed5824ef30396aaf97649d1482345cabf07a74f5b63785fb'
    assert any(e['path']==str(stage.relative_to(R)) and e['sha256']==sha(stage) for e in status['evidence'])
    assert request['status']==service['status']=='completed'
    assert request['decision']['status']==service['decision']['status']=='implementation_frontier'
    assert request['decision']['job'] is None and service['decision']['job'] is None
    assert [j['id'] for j in request['decision']['jobs']]==[j['id'] for j in service['decision']['jobs']]
    assert len(request['jobs'])==len(service['jobs'])==4
    assert all(j['status']=='completed' and j['request_key']==status['idempotency_key'] for j in request['jobs'])
    assert request['key']==service['key']==status['idempotency_key']
    assert request['message_id']==service['message_id']==m['message_id']
    proof={'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'message_id':m['message_id'],
        'position':int(m['message_position']),'actual_message_link':m['message_app_link'],
        'source_stage':{'number':23,'path':str(stage.relative_to(R)),'sha256':sha(stage),'global_accepted_step':208},
        'registry_source':{'path':str((B/'registry_stage023.json').relative_to(R)),'sha256':sha(B/'registry_stage023.json')},
        'actual_card':{'schema':'2.0','elements':15,'full_visible_content_matches':True,'readback_characters':len(m['content']),
            'readback_is_raw_DSL':False,'rendering_differences':differences},
        'actual_send':{'started_at':status['started_at'],'finished_at':status['finished_at'],'passed':True},
        'service_iteration':{'status':'completed','decision_status':'implementation_frontier','job':None,
            'jobs':[{k:j[k] for k in ('job_id','status','output','result_sha256','process_started_at','process_finished_at')} for j in request['jobs']],
            'automatic_reference_probe_jobs':4,'automatic_native_Yb_optimizer_jobs':0,
            'scope':'Shared-source epoch triggered four small registered reference/property/SD/GPU probes; separate from root two native Yb pilots.'},
        'actual_readback_calls_by_root':1,'new_API_calls_by_this_audit':0,'new_scientific_calls_by_this_audit':0,
        'verifier_sha256':sha(__file__),'shared_utilities_sha256':sha(U),'files_sha256':{
            p.name:sha(p) for p in sorted(B.glob('*.json')) if p.name not in ('delivery_proof.json','delivery_readback_proof.json')},
        'scope':'Receipt and rendered-text binding; native pair independently root-launched, no service auto-launch or physical/experimental promotion.'}
    pair=B.parent/'correlated29_history_comparison_preparation'
    closed=pair/'root_pair_execution.json'
    filename='delivery_readback_proof.json'
    if closed.exists():
        p=read(closed)
        assert p['status']=='foreground_pair_closed'
        science={}
        for name in ('history10','history40'):
            d=pair/name
            root=read(d/'root_execution.json')
            assert root['status']=='foreground_closed' and root['exit_code']==0
            assert datetime.fromisoformat(root['started_utc'])>datetime.fromisoformat(status['finished_at'])
            result=read(d/'actual/result.json')
            assert result['status']=='bounded_public_optimizer_returned' and result['error'] is None
            assert all(c['passed'] for c in result['controls']) and len(result['trajectory'])==33
            science[name]={'started_after_this_card':True,'elapsed_uncontrolled_seconds':root['elapsed_uncontrolled_seconds'],
                'completed_calls':{k:result['completed_calls'][k] for k in ('H','CI','gradient','optimizer','validation_Dirac')},
                'qualified_points':33,'new_accepted_steps':32,'stationarity_passed':result['stationarity_passed'],
                'source_map_sha256':result['source_map_sha256'],'root_receipt_sha256':sha(d/'root_execution.json'),
                'actual_result_sha256':sha(d/'actual/result.json'),'all_controls_passed':True,'timing_claim_valid':False}
        proof['separate_root_bounded_iteration']={'kind':'Two same208 history10/40 native MCDHF pilots','branches':science,
            'pair_receipt_sha256':sha(closed),'service_auto_launched_these_jobs':False}
        filename='delivery_proof.json'
    target=B/filename
    assert not target.exists()
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'proof':filename,'sha256':sha(target),'position':proof['position'],'source_stage':23,'new_API_calls':0,'new_scientific_calls':0}))


if __name__=='__main__':
    main()

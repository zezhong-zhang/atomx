"""Read-only verification of current source, guide links and prior topic seals."""
import hashlib,json,re
from pathlib import Path
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    source=json.loads((HERE/'final_source_sha256.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in source.items())
    paths=['README.md','AGENTS.md','docs/ATOMIC_LINEARIZED_SD.md','docs/ATOMIC_FSCC.md','docs/README.md','docs/ARCHITECTURE_CAPABILITIES.md','docs/ARCHITECTURE_DESIGN.md','docs/AGENT_REFERENCE.md','docs/AGENT_INSIGHTS.md','docs/TESTING.md','docs/VERIFICATION.md','docs/MIGRATION.md','docs/AGENT_KNOWLEDGE_WORKFLOW.md',str((HERE/'README.md').relative_to(ROOT))]
    retained={'AGENTS.md','docs/ARCHITECTURE_DESIGN.md','docs/AGENT_KNOWLEDGE_WORKFLOW.md'};broken=[];links=0;review=[]
    for name in paths:
        p=ROOT/name
        review.append(dict(path=name,sha256=sha(p),disposition='retain' if name in retained else 'add' if name=='docs/ATOMIC_LINEARIZED_SD.md' else 'update',reason='Existing principles/dependency/evidence workflow remains applicable' if name in retained else 'Scoped native linearized SD and exact scalar planning contract/evidence/navigation'))
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=raw.strip().split(' "')[0].strip('<>')
            if target.startswith(('http:','https:','mailto:','#','app:')):continue
            target=unquote(target.split('#')[0])
            if not target:continue
            links+=1
            if not (p.parent/target).exists():broken.append((name,target))
    root=(ROOT/'AGENTS.md').read_bytes();assert len(root.splitlines())<=150 and len(root)<=12288
    (HERE/'documentation_impact_review.json').write_text(json.dumps(dict(review=review,links_checked=links,broken=broken,passed=not broken,root_lines=len(root.splitlines()),root_bytes=len(root)),indent=2)+'\n');assert not broken,broken
    manifests=[];failed=[]
    for manifest in sorted((ROOT/'docs').rglob('artifact_manifest.json')):
        if manifest.parent==HERE:continue
        data=json.loads(manifest.read_text())
        entries=([(r['path'],r) for r in data['files']] if isinstance(data.get('files'),list) else [(k,dict(sha256=v) if isinstance(v,str) else v) for k,v in data.items() if isinstance(v,dict) and 'sha256' in v or isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v)])
        count=0
        for name,record in entries:
            p=ROOT/name if name.startswith('docs/') else manifest.parent/name
            if not p.resolve().is_relative_to(manifest.parent.resolve()):continue
            count+=1
            if not p.is_file() or ('bytes' in record and p.stat().st_size!=record['bytes']) or sha(p)!=record['sha256']:failed.append(str(p))
        manifests.append(dict(manifest=str(manifest.relative_to(ROOT)),verified_files=count))
    record=dict(records=manifests,verified=sum(x['verified_files'] for x in manifests),failed=failed,passed=not failed)
    (HERE/'old_archive_integrity.json').write_text(json.dumps(record,indent=2)+'\n');assert not failed,failed
    print(json.dumps(dict(modules=len(source),guide_links=links,old_files=record['verified'],passed=True)))
if __name__=='__main__':main()

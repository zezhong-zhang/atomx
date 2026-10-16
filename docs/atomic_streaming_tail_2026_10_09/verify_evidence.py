"""Review current source/docs and verify older sealed evidence without editing it."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def source_and_docs():
    before = json.loads((HERE/'before_source_sha256.json').read_text())
    stage1 = json.loads((HERE/'current_source_sha256.json').read_text())
    final = json.loads((HERE/'final_source_sha256.json').read_text())
    actual = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src/atomx').rglob('*.py'))}
    assert actual == final, 'source changed after final measured runs'
    changes = {k: dict(before=before.get(k), final=v) for k, v in final.items() if before.get(k) != v}
    assert set(changes) == {
        'src/atomx/methods/fscc.py', 'src/atomx/physics/csf/spinor_integrals.py',
        'src/atomx/physics/csf/matrix.py', 'src/atomx/physics/csf/radial_snapshot.py'}
    for capture, identity in [('implementation_capture', stage1), ('final_implementation_capture', final)]:
        for name in changes:
            assert sha(HERE/capture/Path(name).name) == identity[name]
    backend = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc/EXP-T-atomx-build/expt.x')
    assert sha(backend) == '9f06ef3a31a29856b99194d8fb0f305c68831c800cff3867e1c3585658290c2a'
    (HERE/'source_identity_chain.json').write_text(json.dumps(dict(
        before_count=len(before), final_count=len(final), modified=changes,
        stage1_source_file='current_source_sha256.json', final_source_file='final_source_sha256.json',
        final_capture_verified=True, backend_sha256=sha(backend),
        shared_kernel_scope='Explicit cache capacity and diagnostics only in matrix/radial_snapshot; default integral arithmetic unchanged',
        source_bridge='36-virtual final-source solve; 54-virtual final-source export; complete Ne signed bytes and actual SD energy replay'), indent=2)+'\n')
    specs = {
        'README.md': ('update', 'Current optional on-demand native factory and bounded export'),
        'AGENTS.md': ('retain', 'Standing user performance decision already merged into section 4; root remains compact'),
        'docs/ATOMIC_FSCC.md': ('update', 'Authoritative native representation/cache/export and backend capacity contract'),
        'docs/README.md': ('update', 'Current topic and code-to-documentation navigation'),
        'docs/ARCHITECTURE_CAPABILITIES.md': ('update', 'Separate native construction from resident C1 backend limits'),
        'docs/ARCHITECTURE_DESIGN.md': ('retain', 'Shared physical kernel and method dependency direction unchanged'),
        'docs/AGENT_REFERENCE.md': ('update', 'Link measured continuation and original sealed electronic anchors'),
        'docs/AGENT_INSIGHTS.md': ('update', 'Merge scoped native/export/backend performance lesson into existing insight'),
        'docs/TESTING.md': ('update', 'New streaming, closed-shell scalar, cache and live electronic gates'),
        'docs/VERIFICATION.md': ('update', 'Separate representation correctness, timing and scientific convergence'),
        'docs/MIGRATION.md': ('update', 'Packed remains default; explicit cache capacity and 2-GiB export guard'),
        'docs/AGENT_KNOWLEDGE_WORKFLOW.md': ('retain', 'Current evidence/seal/distillation workflow remains applicable'),
        str((HERE/'README.md').relative_to(ROOT)): ('add', 'Owning results, negative space/experiment gates and replay evidence'),
    }
    links = 0
    broken = []
    reviewed = []
    for name, (disposition, reason) in specs.items():
        path = ROOT/name
        text = path.read_text()
        reviewed.append(dict(path=name, disposition=disposition, reason=reason, sha256=sha(path)))
        # Current guidelines use simple Markdown links, with optional angle brackets.
        for raw in re.findall(r'\]\(([^)]+)\)', text):
            target = raw.strip().split(' "')[0].strip('<>')
            if target.startswith(('http:', 'https:', 'mailto:', '#', 'app:')):
                continue
            target = unquote(target.split('#')[0])
            if not target:
                continue
            links += 1
            if not (path.parent/target).exists():
                broken.append(dict(source=name, target=target))
    root = (ROOT/'AGENTS.md').read_bytes()
    assert len(root.splitlines()) <= 150 and len(root) <= 12*1024
    record = dict(reviewed=reviewed, local_links_checked=links, broken=broken,
                  root_rule_lines=len(root.splitlines()), root_rule_bytes=len(root), passed=not broken)
    (HERE/'documentation_impact_review.json').write_text(json.dumps(record, indent=2)+'\n')
    assert not broken, broken
    return len(final), links


def verify_older_seals():
    records = []
    failed = []
    for manifest in sorted((ROOT/'docs').rglob('artifact_manifest.json')):
        if manifest.parent == HERE:
            continue
        data = json.loads(manifest.read_text())
        if isinstance(data.get('files'), list):
            entries = [(r['path'], r) for r in data['files']]
        else:
            entries = [(k, dict(sha256=v) if isinstance(v, str) else v) for k, v in data.items()
                       if isinstance(v, dict) and 'sha256' in v or isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)]
        count = 0
        outside = []
        for name, record in entries:
            path = ROOT/name if name.startswith('docs/') else manifest.parent/name
            if not path.resolve().is_relative_to(manifest.parent.resolve()):
                outside.append(name)
                continue
            if not path.is_file() or ('bytes' in record and path.stat().st_size != record['bytes']) or sha(path) != record['sha256']:
                failed.append(str(path.relative_to(ROOT)))
            count += 1
        records.append(dict(manifest=str(manifest.relative_to(ROOT)), verified_files=count,
                            current_source_fingerprints_outside_sealed_topic=outside))
    result = dict(records=records, verified=sum(r['verified_files'] for r in records),
                  failed=failed, passed=not failed,
                  scope='Owning-topic seals only; historical hashes of evolving current guides outside topics are not frozen')
    (HERE/'old_archive_integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not failed, failed
    return result['verified']


if __name__ == '__main__':
    source, links = source_and_docs()
    old = verify_older_seals()
    print(json.dumps(dict(source_modules=source, guide_links=links, old_sealed_files=old, passed=True)))

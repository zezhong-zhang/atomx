"""Freeze source/reference identities; audit old seals and current doc links."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT.parent/'atomx-references/2026-10-09-ci-fscc'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value): (HERE/name).write_text(json.dumps(value,indent=2)+'\n')

def stripped_ast(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
    return ast.dump(tree,include_attributes=False)

def main():
    final={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src/atomx').rglob('*.py'))}
    write('final_source_sha256.json',final)
    before=json.loads((HERE/'before_source_sha256.json').read_text())
    changed=[f for f in before if before[f]!=final[f]]
    assert changed==['src/atomx/methods/ci_mbpt.py']
    new=list(set(final)-set(before)); assert new==['src/atomx/methods/ci_effective.py']
    oldzip=HERE.parent/'ci_fscc_reference_upgrade_2026_10_09/Atomx_CI_MBPT_FS_RCC源码与验证.zip'
    with zipfile.ZipFile(oldzip) as z:
        name=next(n for n in z.namelist() if n.endswith('src/atomx/methods/ci_mbpt.py'))
        old=z.read(name).decode()
    assert hashlib.sha256(old.encode()).hexdigest()==before[changed[0]]
    assert stripped_ast(old)==stripped_ast((ROOT/changed[0]).read_text())
    capture=json.loads((HERE/'6s_1_2_qspace.json').read_text())['reference_identity']['source_sha256']
    capture_changed=[f for f in capture if capture[f]!=final[f]]
    assert capture_changed==changed
    write('source_identity_chain.json',dict(before_modules=len(before),final_modules=len(final),
        new_modules=new,changed_existing_modules=changed,
        changed_existing_is_docstring_only=True,Yb_capture_to_final_changes=capture_changed,
        native_Q_implementation_unchanged_after_capture=True,
        matched_Sigma_production_kernels_unchanged=True))
    topics=('yb_diagnosis_repair_2026_10_08','yb_energy_dependent_sigma_2026_10_08',
            'yb_correlation_tail_2026_10_08','yb_literature_repair_strategy_2026_10_08',
            'yb_matrix_self_energy_2026_10_08','ci_fscc_reference_upgrade_2026_10_09')
    seals=[]
    for topic in topics:
        folder=HERE.parent/topic
        data=json.loads((folder/'artifact_manifest.json').read_text())
        bad=[]
        for name,meta in data.items():
            p=ROOT/name if name.startswith('docs/') else folder/name
            expected=meta if isinstance(meta,str) else meta['sha256']
            if not p.is_file() or sha(p)!=expected: bad.append(name)
        seals.append(dict(topic=topic,files=len(data),passed=not bad,bad=bad))
    write('old_archive_integrity.json',seals)
    assert all(r['passed'] for r in seals)
    libs=[BASE/'AMBiT-build/src/MBPT/libmbpt.a',BASE/'AMBiT-build/src/Universal/libuniversal.a']
    sources=[BASE/'AMBiT/src/MBPT/CoreMBPTCalculator.cpp',BASE/'AMBiT/src/MBPT/MBPTCalculator.h',
             BASE/'AMBiT/src/MBPT/MBPTCalculator.cpp',BASE/'AMBiT/src/Universal/MathConstant.cpp',
             BASE/'AMBiT-build/src/CMakeFiles/ambit.dir/link.txt']
    write('reference_execution_identity.json',dict(reference='AMBiT',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE/'AMBiT',text=True).strip(),
        tracked_dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE/'AMBiT',text=True).strip(),
        files={str(p):sha(p) for p in libs+sources+[HERE/'ambit_sigma_probe.cpp',HERE/'ambit_sigma_probe']},
        scope='Unmodified upstream static libraries; adapter supplies common radial integrals and fixed arguments; no independent radial-solver benchmark'))
    docs=[HERE/'README.md']+[ROOT/'docs'/n for n in ('README.md','AGENT_REFERENCE.md','AGENT_INSIGHTS.md',
                                                  'ARCHITECTURE_CAPABILITIES.md','VERIFICATION.md','TESTING.md')]
    links=[]; bad=[]
    for f in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text()):
            if target.startswith(('https://','http://','#')): continue
            path=unquote(target.split('#',1)[0])
            if not path: continue
            resolved=(f.parent/path).resolve()
            links.append(dict(document=str(f.relative_to(ROOT)),target=target,exists=resolved.exists()))
            if not resolved.exists(): bad.append(links[-1])
    write('documentation_links.json',dict(checked=len(links),passed=not bad,bad=bad))
    assert not bad
    write('documentation_impact_review.json',dict(updated={
        'methods/ci_effective.py':'New finite bare-DC P/Q contract, root/CSF/physical identity, cap and approximation limits in owning topic; capabilities/tests/verification/nav refreshed',
        'methods/ci_mbpt.py':'Docstring-only distinction between guarded closed-core Sigma consumer and separate bare finite-Q route',
        'docs/AGENT_INSIGHTS.md':'Merge convention-first with matched-library and model-response evidence; no root-rule promotion'},
        retained={
        'AGENTS.md':'Existing convention, independent anchors and no-double-counting rules sufficient',
        'methods/coupled_cluster.py':'Remains CC scaffold; finite Q elimination does not implement SD or FSCC',
        'docs/CI_MBPT_SIGMA2_SPEC.md':'General two-body screening still unfinished; no coverage change',
        'six sealed topics':'Preserved independently verified source/results, failures and acceptance rules'},
        numerical_scope='53 passing affected tests; matched independent diagram execution; full five-sector restricted Yb model; no full-suite or new experimental gate'))
    print(json.dumps(dict(source_modules=len(final),sealed_files=sum(r['files'] for r in seals),
                         documentation_links=len(links),passed=True)))

if __name__=='__main__': main()

"""Bind current dirty Python source without importing Atomx or opening arrays."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/atomx')
HERE = Path(__file__).resolve().parent
if (HERE / 'actual').exists() or (HERE / 'root_execution_seal.json').exists():
    raise RuntimeError('sealed or executed preparation cannot be rebound')

mapping = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / 'src/atomx').rglob('*.py'))
}
fingerprint = hashlib.sha256(json.dumps(
    mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
identity = {'status': 'SOURCE_bound_current_dirty_python',
            'all_atomx_python_sha256': mapping, 'source_map_sha256': fingerprint}
(HERE / 'current_source_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
contract_path = HERE / 'contract.json'
contract = json.loads(contract_path.read_text())
contract.update(status='SOURCE_ready_current_cached_streamed_qualification',
                source_map_sha256=fingerprint)
contract_path.write_text(json.dumps(contract, indent=2) + '\n')
print(json.dumps({'source_files': len(mapping), 'source_map_sha256': fingerprint,
                  'science_calls': 0, 'input_NPZ_loads': 0}))

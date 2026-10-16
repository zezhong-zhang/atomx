"""Explicit old test content identity; no historical scientific gate changes."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
ORIGINAL_PATH = 'tests/test_mixed_ci_finite_increment.py'
ORIGINAL_SHA256 = '045ee541fc648ea9bbc51a476d04fe2a3bf31b0158bfcc02c9daa8357ab2caa8'
ARCHIVE_PATH = ('docs/yb_mcdhf_rci_2026_10_11/mixed_ci_finite_increment_implementation/'
                'test_source_archive/' + ORIGINAL_SHA256 + '/test_mixed_ci_finite_increment.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_binding_path(repo_relative_path, expected_sha256):
    """Resolve only the declared old test epoch; every hash remains exact.

    This returns a file for content verification. It does not substitute an
    unverified test, modify a contract, execute a driver, or change __file__.
    Materialize exact bytes at the original relative path in an isolated
    epoch root if an original executable's path assumptions are required.
    """
    if repo_relative_path == ORIGINAL_PATH and expected_sha256 == ORIGINAL_SHA256:
        resolved = REPO / ARCHIVE_PATH
    else:
        relative = Path(repo_relative_path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('historical resolver requires a repo-relative path')
        resolved = REPO / relative
    if sha(resolved) != expected_sha256:
        raise ValueError('exact historical input binding changed: ' + repo_relative_path)
    return resolved


def main():
    path = historical_binding_path(ORIGINAL_PATH, ORIGINAL_SHA256)
    print(json.dumps(dict(original_logical_path=ORIGINAL_PATH, physical_archive_path=ARCHIVE_PATH,
        content_sha256=sha(path), exact_original_content=True, Atomx_NumPy_imports=0,
        H_CI_gradient_radial_calls=0, scope='declared test-content resolution only')))


if __name__ == '__main__':
    main()

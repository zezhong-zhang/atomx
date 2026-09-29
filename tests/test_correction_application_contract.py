"""Correction preflight and result validation before energy mutation."""
from types import SimpleNamespace
import pytest

from atomx.corrections.protocol import CorrectionResult
from atomx.corrections.stack import CorrectionStack, finalize_with_corrections


class Term:
    description = 'test correction'

    def __init__(self, name, *, supported=True, shift=.1, returned_name=None):
        self.name, self.supported, self.shift = name, supported, shift
        self.returned_name = returned_name or name
        self.calls = 0

    def supports(self, result):
        return self.supported

    def evaluate(self, result):
        self.calls += 1
        return CorrectionResult(self.returned_name, self.shift)


def stack(*terms, strict=False):
    s = CorrectionStack(se_backend=None, include_vp=False, include_breit=False, strict=strict)
    for term in terms:
        s.add(term)
    return s


def test_strict_preflight_prevents_partial_evaluation():
    good, missing = Term('good'), Term('missing', supported=False)
    s = stack(good, missing, strict=True)
    with pytest.raises(ValueError, match='unsupported requested corrections: missing'):
        s.apply(SimpleNamespace())
    assert good.calls == missing.calls == 0
    applied = s.apply(SimpleNamespace(), strict=False)
    assert 'missing' in applied['_skipped'].provenance
    assert good.calls == 1


@pytest.mark.parametrize('names', [('same', 'same'), ('good', '_skipped')])
def test_ambiguous_contributions_cannot_be_overwritten(names):
    terms = [Term(n) for n in names]
    with pytest.raises(ValueError, match='duplicate|underscore'):
        stack(*terms).apply(SimpleNamespace())
    assert not any(t.calls for t in terms)


@pytest.mark.parametrize('bad', [Term('bad', shift=float('nan')),
                                Term('bad', shift=float('inf')),
                                Term('bad', returned_name='different')])
def test_invalid_correction_does_not_partially_change_energy(bad):
    result = SimpleNamespace(total_energy=-1., energy_components={})
    with pytest.raises(ValueError, match='non-finite|inconsistent'):
        finalize_with_corrections(result, stack(Term('good'), bad))
    assert result.total_energy == -1.
    assert result.energy_components == {}
    assert not hasattr(result, '_correction_results')


def test_finalization_retains_evaluations_and_skip_reasons():
    result = SimpleNamespace(total_energy=-1., energy_components={})
    good, missing = Term('good'), Term('missing', supported=False)
    finalize_with_corrections(result, stack(good, missing))
    assert result.total_energy == pytest.approx(-.9)
    assert result.correction_application.base_energy == -1.
    assert result.correction_application.total_energy == pytest.approx(-.9)
    assert result._correction_results['good'].energy_shift == .1
    assert 'missing' in result._correction_results['_skipped'].provenance
    assert good.calls == 1


def test_finalization_rejects_double_counting_before_evaluation():
    term = Term('shift')
    result = SimpleNamespace(total_energy=-1., energy_components={})
    finalize_with_corrections(result, stack(term))
    with pytest.raises(ValueError, match='already finalized'):
        finalize_with_corrections(result, stack(term))
    assert term.calls == 1
    assert result.total_energy == pytest.approx(-.9)
    assert result.correction_application.base_energy == -1.
    assert result.correction_application.total_energy == pytest.approx(-.9)


@pytest.mark.parametrize('name,operator,stage', [
    ('breit', 'breit', 'ci_hamiltonian'),
    ('self_energy[mohr]', 'self_energy', 'post_ci_expectation'),
    ('nms', 'nms', 'post_ci_expectation'),
])
def test_operator_overlap_rejected_before_any_evaluation(name, operator, stage):
    first, duplicate = Term('unrelated'), Term(name)
    result = SimpleNamespace(total_energy=-1., energy_components={},
                             operator_stages=((operator, stage),))
    with pytest.raises(ValueError, match=f'already included.*{stage}'):
        finalize_with_corrections(result, stack(first, duplicate))
    assert first.calls == duplicate.calls == 0
    assert result.total_energy == -1.
    assert result.operator_stages == ((operator, stage),)


def test_operator_stages_persist_and_guard_reapplication():
    from atomx.io.results import _encode_metadata, _restore_metadata
    import json
    result = SimpleNamespace(total_energy=-1., energy_components={})
    finalize_with_corrections(result, stack(Term('breit', shift=0.)))
    restored = SimpleNamespace()
    _restore_metadata(restored, json.loads(_encode_metadata(result)), 2)
    assert restored.operator_stages == (('breit', 'post_scf'),)
    term = Term('breit')
    with pytest.raises(ValueError, match='already included'):
        stack(term).validate_addition(restored)
    assert term.calls == 0


def test_read_only_reevaluation_does_not_add_correction_again():
    result = SimpleNamespace(total_energy=-1., energy_components={})
    term = Term('breit')
    s = stack(term)
    finalize_with_corrections(result, s)
    assert s.apply(result)['breit'].energy_shift == .1
    assert result.total_energy == pytest.approx(-.9)


def test_two_self_energy_models_cannot_both_be_added():
    first, second = Term('self_energy[mohr]'), Term('self_energy[welton]')
    result = SimpleNamespace(total_energy=-1., energy_components={})
    with pytest.raises(ValueError, match='overlapping operator requests: self_energy'):
        finalize_with_corrections(result, stack(first, second))
    assert first.calls == second.calls == 0
    assert result.total_energy == -1.

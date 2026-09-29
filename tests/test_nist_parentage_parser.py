from atomx.reference_data.nist import _clean, normalise_configuration


def test_clean_preserves_parent_j_and_remains_idempotent():
    raw='3p<sup>5</sup>.(2P*&lt;3/2&gt;).4s<sup>2</sup>.5s'
    expected='3p5.(2P*<3/2>).4s2.5s'
    assert _clean(raw)==expected
    assert _clean(_clean(raw))==expected
    assert normalise_configuration(raw)==normalise_configuration(expected)
    assert _clean('<a href="test">label</a>')=='label'


def test_hyperfine_annotation_preserves_occupied_subshell():
    raw = '5s<1/2,F=4>.5g'
    assert _clean(raw) == raw
    assert normalise_configuration(raw) == '5s1 5g1'
    assert normalise_configuration('5s&lt;1/2,F=5&gt;.5g') == '5s1 5g1'

import pytest
from atomx.reference_data.cowan import parse_rcn_energies, rcn2_centroid_energies
from atomx.reference_data.cowan import summarize_centroid_comparison


def test_centroid_metrics_use_same_excited_states_and_separate_corrf():
    def row(name, nist, atomx, bare, exported, corrf=0):
        return dict(case='test', configuration=name, corrf=corrf,
                    nist_excitation_cm1=nist, atomx_cm1=atomx,
                    hfr_before_correlation_cm1=bare, cowan_exported_cm1=exported)
    report = summarize_centroid_comparison([
        row('ground', 0, 0, 0, 0),
        row('a', 100, 90, 80, 105),
        row('b', 200, 220, 160, 195),
        row('missing', 300, None, 300, 300),
        row('invalid', 400, float('nan'), 400, 400),
        row('unmatched', None, 1, 1, 1),
        row('other potential', 100, None, 95, 99, corrf=1),
    ])
    assert report[0]['eligible_count'] == 4
    assert report[0]['common_count'] == 2
    assert report[0]['metrics']['atomx_cm1']['mae_cm1'] == 15
    assert report[0]['metrics']['hfr_before_correlation_cm1']['mean_error_cm1'] == -30
    assert report[0]['metrics']['cowan_exported_cm1']['rms_cm1'] == 5
    assert len(report[0]['excluded_configurations']) == 2
    assert report[1]['common_count'] == 0
    assert report[1]['metrics']['atomx_cm1']['rms_cm1'] is None


def test_correlation_ledger_retains_bare_and_exported_energies():
    row = parse_rcn_energies(' etot= -10.0 -10.1 -10.2 -10.3 -10.4 -10.5 -10.6')[0]
    assert row['relativistic'] == -10.1
    assert row['exported'] == -10.6
    with pytest.raises(ValueError, match='incomplete'):
        parse_rcn_energies(' etot= -10.0 -10.1')


def test_unscaled_centroids_reject_hidden_empirical_scaling():
    row = '0C000 42.2002018 0.0 hr fact= 1.00 1.00 1.00 1.00 1.00\n'
    assert rcn2_centroid_energies(row, ['C000'])['C000'] == pytest.approx(42200.2018)
    with pytest.raises(ValueError, match='radial scales'):
        rcn2_centroid_energies(row.replace('1.00', '0.85', 1), ['C000'])


def test_radial_reader_enforces_layout_and_preserves_shell_order(tmp_path):
    import numpy as np
    from scipy.io import FortranFile
    from atomx.reference_data.cowan import read_rcn_radial_records
    header=bytearray(1306)
    def put(offset, values, dtype):
        data=np.asarray(values,dtype=dtype).tobytes()
        header[offset:offset+len(data)]=data
    put(0,[3,4],'<i4');put(80,[0,0],'<i4');put(160,[1.,1.],'<f8')
    put(320,[2],'<i4');header[724:742]=b'C001'.ljust(18)
    put(746,[12],'<i4');put(758,[0],'<i4');put(774,[3],'<i4');put(1302,[2],'<i4')
    arrays=np.zeros((4,1801),dtype='<f8')
    arrays[0,:3]=[0.,1.,2.];arrays[2,:3]=[0.,1.,0.];arrays[3,:3]=[0.,.5,.2]
    path=tmp_path/'tape2n'
    raw=np.frombuffer(bytes(header)+arrays.tobytes()+b'\0'*4,dtype=np.uint8)
    with FortranFile(path,'w') as f: f.write_record(raw)
    row=read_rcn_radial_records(path)[0]
    assert row['configuration']=='C001' and row['z']==12
    np.testing.assert_array_equal(row['n'],[3,4])
    np.testing.assert_array_equal(row['orbitals'],arrays[2:,:3])
    with FortranFile(path,'w') as f: f.write_record(raw[:-1])
    with pytest.raises(ValueError,match='length'):
        read_rcn_radial_records(path)


def test_overlap_audit_does_not_confuse_radial_and_angular_overlap():
    import numpy as np
    from atomx.reference_data.cowan import radial_overlap_summary
    r=np.linspace(0,5,101);p=r*np.exp(-r)
    record=dict(radius=r,orbitals=np.array([p,p,p]),n=np.array([1,2,2]),l=np.array([0,0,1]))
    result=radial_overlap_summary(record)
    assert len(result['same_l_pairs'])==1
    assert result['same_l_pairs'][0]['overlap']==pytest.approx(1.)

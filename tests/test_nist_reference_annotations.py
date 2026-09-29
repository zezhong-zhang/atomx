import pytest
from atomx.reference_data.annotations import AnnotationLinks,parse_annotation


def test_annotation_link_requires_exact_nist_source_and_comment_identity():
    parser=AnnotationLinks()
    parser.feed('''<a onclick="popded('https://physics.nist.gov/cgi-bin/ASBib1/get_ASBib_ref.cgi?db=el&amp;db_id=&amp;comment_code=c112')">c112</a>
    <a href="https://evil.test/cgi-bin/ASBib1/get_ASBib_ref.cgi?db_id=&amp;comment_code=c113">c113</a>
    <a href="https://physics.nist.gov/cgi-bin/ASBib1/get_ASBib_ref.cgi?db_id=&amp;comment_code=c114">c115</a>''')
    assert list(parser.links)==['c112']
    assert '&amp;' not in parser.links['c112'][0]


def test_comment_identity_and_safe_text():
    body=b'<body><h3>NIST Atomic Spectra Database Wavelength Comment c112</h3><p>Source attribution</p><script>secret</script></body>'
    result=parse_annotation(body,'https://physics.nist.gov/','c112')
    assert 'Source attribution' in result['text'] and 'secret' not in result['text']
    with pytest.raises(ValueError,match='requested wavelength comment'):
        parse_annotation(body,'https://physics.nist.gov/','c113')
    with pytest.raises(ValueError,match='requested wavelength comment'):
        parse_annotation(b'<body>Access error</body>','https://physics.nist.gov/','c112')


def test_repeated_comment_keeps_distinct_original_query_contexts():
    parser=AnnotationLinks()
    base='https://physics.nist.gov/cgi-bin/ASBib1/get_ASBib_ref.cgi?db_id=&amp;comment_code=c153&amp;type='
    parser.feed(f'<a href="{base}">c153</a><a href="{base}E2">c153</a><a href="{base}">c153</a>')
    assert len(parser.links['c153'])==2
    assert parser.links['c153'][0].endswith('type=')
    assert parser.links['c153'][1].endswith('type=E2')

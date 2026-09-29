import pytest
from atomx.reference_data.bibliography import parse_citation


def test_citation_preserves_bibliography_but_excludes_scripts_and_unsafe_links():
    body=b'''<html><body><h3>NIST Energy Levels and Wavelengths Bibliographic Reference # 7185</h3>
    <a id="at" href="https://example.test/article">Atomic <i>levels</i></a>
    <a id="aa">J.&nbsp;Author</a><a id="aj">Journal</a> 14, 1-20 (1985)
    <script>secret script text</script><a href="javascript:alert(1)">close</a>
    <p>Context note for Ca I</p></body></html>'''
    result=parse_citation(body,'https://physics.nist.gov/source',7185)
    assert result['title']=='Atomic levels'
    assert result['authors']==['J. Author']
    assert result['journal']=='Journal'
    assert 'Context note for Ca I' in result['text']
    assert 'secret script text' not in result['text']
    assert result['links']==['https://example.test/article']
    with pytest.raises(ValueError,match='requested bibliography'):
        parse_citation(body,'https://physics.nist.gov/source',999)


def test_challenge_page_is_not_a_citation():
    with pytest.raises(ValueError,match='requested bibliography'):
        parse_citation(b'<body>Please enable JavaScript</body>','https://physics.nist.gov',1)


@pytest.mark.parametrize('code,kind,category',[
    ('T1115CA','lines','T'),('L7185','lines','L'),('L7185','levels','E'),
    ('L7185','ionization','I'),("T6066LS'",'lines','T')])
def test_reference_category_is_not_transition_multipolarity(code,kind,category):
    from urllib.parse import urlparse,parse_qs
    from atomx.reference_data.bibliography import bibliography_url
    query=parse_qs(urlparse(bibliography_url(code,'Ca',0,kind)).query)
    assert query['type']==[category]
    assert query['spectr_charge']==['1']
    if code=='T1115CA':assert query['comment_code']==['CA']

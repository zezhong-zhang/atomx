from atomx.reference_data.spectrum_notes import extract_spectrum_notes




def test_extract_editorial_context_and_safe_reference_links():
    html='''<table><tr><td class='brt1'>Lines:</td><td class='blf'>Intensity at 1.9&nbsp;eV.
    <a href="javascript:void(0)" onclick="popded('https://physics.nist.gov/ref?a=1&amp;b=2')">Source</a>
    <p>Uncertainty in cm<sup>-1</sup>.</td></tr></table>
    <pre><td class='blf'>Not a header note</td></pre>'''
    notes=extract_spectrum_notes(html,'lines')
    assert len(notes)==1
    assert notes[0]['section']=='Lines'
    assert notes[0]['note_text']=='Intensity at 1.9 eV. Source Uncertainty in cm-1.'
    assert notes[0]['links']==['https://physics.nist.gov/ref?a=1&b=2']
    assert '<sup>-1</sup>' in notes[0]['raw_html']


def test_level_note_and_empty_note_are_distinct():
    assert extract_spectrum_notes("<td class='blf2'>&nbsp;</td>",'levels')==[]
    notes=extract_spectrum_notes("<td class='blf2'>Laboratory levels only.</td>",'levels')
    assert notes[0]['section']=='Energy Levels'

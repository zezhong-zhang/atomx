"""Extract NIST's spectrum-level editorial notes from cached HTML headers."""
from html.parser import HTMLParser
import re
from urllib.parse import urlparse


class _Cell(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text=[]
        self.links=[]

    def handle_starttag(self,tag,attrs):
        if tag.lower() in ('p','br','div','li'):self.text.append(' ')
        attributes=dict(attrs)
        candidates=[attributes.get('href','')]
        candidates+=re.findall(r"['\"](https?://[^'\"]+)['\"]",attributes.get('onclick',''))
        for url in candidates:
            if urlparse(url).scheme in ('http','https') and url not in self.links:
                self.links.append(url)

    def handle_data(self,text):self.text.append(text)


def extract_spectrum_notes(html,kind):
    """Return known NIST note cells, retaining original HTML and safe links.

    Only editorial header cells (blf/blf2) are selected. Line/level tables and
    generic footer text are excluded. Consumers must render text, not raw HTML.
    """
    if kind not in ('lines','levels'):raise ValueError('unknown NIST record kind')
    header=re.split(r'<pre\b',html,maxsplit=1,flags=re.I)[0]
    records=[]
    labels=[]
    for match in re.finditer(r'<td\b[^>]*class\s*=\s*[\'"]([^\'"]+)[\'"][^>]*>(.*?)</td>',header,re.S|re.I):
        classes=match[1].split()
        cell=_Cell();cell.feed(match[2]);cell.close()
        text=re.sub(r'\s+',' ',''.join(cell.text)).strip()
        if 'brt1' in classes:labels.append(text.rstrip(':').strip())
        if not {'blf','blf2'}.intersection(classes) or not text:continue
        section='Energy Levels' if kind=='levels' else labels[-1] if labels else 'Unclassified header note'
        records.append(dict(section=section,note_text=text,links=cell.links,raw_html=match[2]))
    return records

"""Safe extraction of public NIST ASBib citation responses."""
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlparse, urlencode

REFERENCE_TOKEN=re.compile(r"^([LT])(\d+)([A-Za-z][A-Za-z0-9']*)?$")


def bibliography_url(code, symbol, charge, kind):
    """Use ASBib's data category, never an E1/M1 transition multipolarity."""
    match=REFERENCE_TOKEN.fullmatch(code)
    if match is None or kind not in ('lines','levels','ionization'):
        raise ValueError('unsupported NIST reference identity')
    category='T' if match[1]=='T' else {'lines':'L','levels':'E','ionization':'I'}[kind]
    return 'https://physics.nist.gov/cgi-bin/ASBib1/get_ASBib_ref.cgi?'+urlencode(dict(
        db='el' if match[1]=='L' else 'tp',db_id=match[2],comment_code=match[3] or '',
        element=symbol,spectr_charge=charge+1,ref=match[2],type=category))


class CitationParser(HTMLParser):
    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url=url; self.hidden=0; self.body=False; self.text=[]; self.links=[]
        self.active=None; self.parts=[]; self.fields={'authors':[]}

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='body':self.body=True
        if tag in ('script','style'):self.hidden+=1
        if self.hidden or not self.body:return
        if tag in ('br','p','dd','dt','td','h3'):self.text.append('\n')
        if tag=='a':
            self.active=attrs.get('id'); self.parts=[]
            url=urljoin(self.url,attrs.get('href',''))
            if attrs.get('href') and urlparse(url).scheme in ('http','https'):
                self.links.append(url)

    def handle_endtag(self,tag):
        if tag in ('script','style'):self.hidden=max(0,self.hidden-1)
        if tag=='a' and self.active:
            value=' '.join(''.join(self.parts).split())
            if self.active=='aa':self.fields['authors'].append(value)
            elif self.active=='at':self.fields['title']=value
            elif self.active=='aj':self.fields['journal']=value
            self.active=None

    def handle_data(self,text):
        if not self.hidden and self.body:
            self.text.append(text)
            if self.active:self.parts.append(text)


def parse_citation(body, url, reference_id):
    # ASBib declares Latin-1; retain the original bytes in the source cache.
    match=re.search(br'charset\s*=\s*["\']?([\w-]+)',body,re.I)
    encoding=match[1].decode('ascii') if match else 'utf-8'
    parser=CitationParser(url);parser.feed(body.decode(encoding));parser.close()
    text='\n'.join(' '.join(line.split()) for line in ''.join(parser.text).splitlines() if line.strip())
    if not re.search(r'Bibliographic Reference\s*#\s*'+re.escape(str(reference_id))+r'\b',text):
        raise ValueError('NIST response does not identify the requested bibliography record')
    return dict(parser.fields, text=text, links=list(dict.fromkeys(parser.links)),encoding=encoding)

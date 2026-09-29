"""Standalone NIST wavelength comments, with exact source-link validation."""
from html.parser import HTMLParser
import re
from urllib.parse import urlparse,parse_qs
from atomx.reference_data.bibliography import CitationParser

CODE = re.compile(r'c\d+')


class AnnotationLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active=None;self.parts=[];self.links={}

    def handle_starttag(self,tag,attrs):
        if tag!='a':return
        attrs=dict(attrs);self.parts=[]
        urls=[attrs.get('href','')]+re.findall(r"['\"](https://[^'\"]+)['\"]",attrs.get('onclick',''))
        self.active=urls

    def handle_data(self,text):
        if self.active is not None:self.parts.append(text)

    def handle_endtag(self,tag):
        if tag!='a' or self.active is None:return
        code=''.join(self.parts).strip()
        if CODE.fullmatch(code):
            for url in self.active:
                p=urlparse(url);q=parse_qs(p.query,keep_blank_values=True)
                if (p.scheme=='https' and p.netloc=='physics.nist.gov'
                    and p.path=='/cgi-bin/ASBib1/get_ASBib_ref.cgi'
                    and q.get('comment_code')==[code] and q.get('db_id')==['']):
                    variants=self.links.setdefault(code,[])
                    if url not in variants:variants.append(url)
        self.active=None


def parse_annotation(body,url,code):
    if not CODE.fullmatch(code):raise ValueError('unsupported annotation code')
    match=re.search(br'charset\s*=\s*["\']?([\w-]+)',body,re.I)
    encoding=match[1].decode('ascii') if match else 'utf-8'
    parser=CitationParser(url);parser.feed(body.decode(encoding));parser.close()
    text='\n'.join(' '.join(line.split()) for line in ''.join(parser.text).splitlines() if line.strip())
    if not re.search(r'NIST Atomic Spectra Database Wavelength Comment\s+'+re.escape(code)+r'\b',text):
        raise ValueError('NIST response does not identify requested wavelength comment')
    return dict(text=text,links=parser.links,encoding=encoding,annotation_code=code)

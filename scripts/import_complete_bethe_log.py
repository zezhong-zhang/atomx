"""Reproduce the complete n<=200 Bethe-log table from the retained source PDF text."""
import argparse
import hashlib
import re
import runpy
import subprocess
from pathlib import Path

SOURCE_SHA256='88bc1448d1b5fbca1a1948ce99b89f4e471375a948af13a40bea138751e85114'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--text',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('src/atomx/data/nist_bethe_log.py'))
    args=parser.parse_args()
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Unexpected reference PDF hash')
    extracted = subprocess.check_output(['pdftotext', '-layout', str(args.pdf), '-'], text=True)
    if extracted != args.text.read_text():
        raise ValueError('Text does not match a fresh extraction of the verified PDF')
    text=extracted.replace('−','-')
    pattern=re.compile(r'(?<!\d)(\d{1,3})\s+(\d{1,3})\s+(-?0\.\d{3})\s+(\d{3})\s+(\d{3})\s*×\s*10(-?\d+)')
    matches=list(pattern.finditer(text))
    data={(int(m[1]),int(m[2])):float(m[3]+m[4]+m[5]+'e'+m[6]) for m in matches}
    expected={(n,l) for n in range(1,201) for l in range(n)}
    if len(matches)!=20100 or set(data)!=expected:
        raise ValueError('Reference extraction is incomplete or duplicated')
    old=runpy.run_path(str(args.output))['BETHE_LOG_NIST']
    assert all(data[n,l]==value for n,row in old.items() for l,value in enumerate(row)), 'Existing values would change'
    header='''"""Hydrogenic Bethe logarithms from NIST Technical Note 1467.

Jentschura and Mohr, Bethe Logarithms for Rydberg States: Numerical
Values for n <= 200 (2005). Source: https://qedtheory.org/wp-content/uploads/2021/01/values.pdf
PDF SHA-256: 88bc1448d1b5fbca1a1948ce99b89f4e471375a948af13a40bea138751e85114

All 20,100 published (n,l) values are retained. Reproduce with
scripts/import_complete_bethe_log.py using pdftotext -layout output.
"""

BETHE_LOG_NIST: dict[int, tuple[float, ...]] = {
'''
    lines=[header]
    for n in range(1,201):
        lines.append(f'    {n}: (\n')
        for first in range(0,n,8):
            lines.append('        '+', '.join(repr(data[n,l]) for l in range(first,min(n,first+8)))+',\n')
        lines.append('    ),\n')
    lines.append('}\n')
    args.output.write_text(''.join(lines))
    print(f'Imported {len(data)} values; all {sum(map(len,old.values()))} existing entries unchanged')


if __name__=='__main__':
    main()

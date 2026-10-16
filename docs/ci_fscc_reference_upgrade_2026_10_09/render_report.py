"""Render the complete Chinese report without changing physics dependencies."""
from pathlib import Path
import importlib.metadata
import json
import runpy
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parent
CSS=runpy.run_path(str(ROOT.parent/'rare_earth_yb_spectrum_2026_10_08/render_report.py'))['CSS']
CSS=CSS.replace('Atomx · Yb⁺ 谱线计算与实验对照','Atomx · CI / MBPT / FS-RCC 原生升级')
CSS+='\ntable { break-inside: avoid; } h1 { font-size:18pt; }\n'
if __name__=='__main__':
    body=markdown.markdown((ROOT/'README.md').read_text(),extensions=['tables','fenced_code'])
    document=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Atomx CI-MBPT-FS-RCC 源码与升级</title><style>{CSS}</style><body>{body}</body></html>'
    (ROOT/'Atomx_CI_MBPT_FS_RCC升级报告.html').write_text(document)
    HTML(string=document,base_url=str(ROOT)).write_pdf(ROOT/'Atomx_CI_MBPT_FS_RCC升级报告.pdf')
    (ROOT/'render_dependencies.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in ('markdown','weasyprint')},indent=2))

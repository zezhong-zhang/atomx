"""Render the scientific report separately from the numerical environment."""
import importlib.metadata
from pathlib import Path
import runpy
import json
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parent
CSS=runpy.run_path(str(ROOT.parent/'rare_earth_yb_spectrum_2026_10_08/render_report.py'))['CSS']
CSS=CSS.replace('谱线计算与实验对照','能量自洽与尾部控制')
CSS += '\nimg {max-width:100%;} h2 {break-after:avoid;} body {line-height:1.5;} p {margin:2mm 0;} table {font-size:8.6pt;}\n'
content=markdown.markdown((ROOT/'README.md').read_text(),extensions=['tables','fenced_code'])
document=f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>Yb⁺ 能量自洽与尾部控制</title><style>{CSS}</style><body>{content}</body></html>'
(ROOT/'Yb能量自洽与尾部控制.html').write_text(document)
HTML(string=document,base_url=str(ROOT)).write_pdf(ROOT/'Yb能量自洽与尾部控制.pdf')
(ROOT/'render_dependencies.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in ['markdown','weasyprint']},indent=2)+'\n')

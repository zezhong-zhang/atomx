"""Render only; use separate uv environment so physics dependencies are unchanged."""
from pathlib import Path
import importlib.metadata
import json
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parent
CSS="""
@page { size:A4; margin:16mm 16mm 17mm;
 @bottom-center { content:'Atomx · Yb⁺ 谱线计算与实验对照    ' counter(page) ' / ' counter(pages); font:8pt 'Noto Sans CJK SC'; color:#627686; }
}
body { font:9.7pt 'Noto Sans CJK SC',sans-serif; line-height:1.58; color:#203342; }
h1 { font-size:21pt; line-height:1.4; color:#174d69; margin-bottom:7mm; }
h2 { font-size:14.5pt; color:#174d69; border-bottom:1px solid #cadbe4; padding-bottom:1.5mm; margin-top:6mm; }
h3 { font-size:11.4pt; color:#315e77; margin-top:5mm; }
h1,h2,h3 { break-after:avoid; }
p { margin:2.5mm 0; orphans:3; widows:3; }
a { color:#1f6487; text-decoration:none; overflow-wrap:anywhere; }
table { border-collapse:collapse; width:100%; font-size:7.8pt; margin:4mm 0; line-height:1.5; }
th,td { border:1px solid #cad9e2; padding:1.4mm; overflow-wrap:anywhere; }
th { background:#e9f0f4; text-align:left; }
thead { display:table-header-group; } tr { break-inside:avoid; }
img { display:block; width:100%; height:auto; margin:4mm 0; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f1f5f7; padding:3mm; font-size:7.5pt; line-height:1.45; }
code { font-family:'DejaVu Sans Mono','Noto Sans CJK SC',monospace; font-size:.87em; }
li { margin:1.6mm 0; }
@media screen { body { max-width:1000px; margin:40px auto; padding:0 24px; } }
"""


def main():
    content=markdown.markdown((ROOT/"README.md").read_text(),extensions=["tables","fenced_code"])
    document=f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>Atomx Yb⁺ 谱线计算与实验对照</title><style>{CSS}</style><body>{content}</body></html>'
    (ROOT/"Yb谱线计算与实验对照.html").write_text(document,encoding="utf-8")
    HTML(string=document,base_url=str(ROOT)).write_pdf(ROOT/"Yb谱线计算与实验对照.pdf")
    (ROOT/"render_dependencies.json").write_text(json.dumps({x:importlib.metadata.version(x) for x in ["markdown","weasyprint"]},indent=2)+"\n")


if __name__=="__main__":main()

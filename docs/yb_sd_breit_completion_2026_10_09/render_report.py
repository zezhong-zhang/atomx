"""Render the measured residual figure and standalone Chinese report."""
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    import markdown
    from importlib.metadata import version
    from weasyprint import HTML
    rows = json.loads((HERE / 'results.json').read_text())['rows']
    labels = ('5d_3/2', '5d_5/2', '6p_1/2', '6p_3/2')
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    for offset, model, title, color in [(-.24, 'one_hole_DC', 'One-hole DC', '#385f86'),
                                        (0., 'DC', 'Complete declared NR-SD / DC', '#e0a447'),
                                        (.24, 'DCB', 'Complete declared NR-SD / DCB', '#008c88')]:
        ax.bar(x + offset, [rows[k]['residuals_cm1'][model] for k in labels], .23,
               label=title, color=color)
    ax.axhline(0, color='#374151', lw=.8)
    ax.set_xticks(x, ['5d 3/2', '5d 5/2', '6p 1/2', '6p 3/2'])
    ax.set_ylabel('Theory - experiment (cm$^{-1}$)')
    ax.set_title('Yb II: frozen orbitals, complementary Q, own common ground')
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(HERE / 'sd_breit_residuals.png', dpi=180)
    fig.savefig(HERE / 'sd_breit_residuals.svg')
    plt.close(fig)
    css = runpy.run_path(str(HERE.parent / 'rare_earth_yb_spectrum_2026_10_08/render_report.py'))['CSS']
    css = css.replace('Atomx · Yb⁺ 谱线计算与实验对照', 'Atomx · 有限 CCSD 与 SD/Breit 修复')
    css += '\nimg {max-width:100%;} h1 {font-size:18pt;} table {break-inside:avoid;}\n'
    body = markdown.markdown((HERE / 'README.md').read_text(), extensions=['tables', 'fenced_code'])
    doc = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
           f'<title>Atomx 有限 CCSD 与 SD/Breit 修复报告</title><style>{css}</style><body>{body}</body></html>')
    (HERE / 'Atomx_有限CCSD与SD_Breit修复报告.html').write_text(doc)
    HTML(string=doc, base_url=str(HERE)).write_pdf(HERE / 'Atomx_有限CCSD与SD_Breit修复报告.pdf')
    (HERE / 'render_dependencies.json').write_text(json.dumps(
        {n: version(n) for n in ('markdown', 'weasyprint', 'matplotlib')}, indent=2) + '\n')


if __name__ == '__main__':
    main()

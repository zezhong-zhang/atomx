"""Standalone scientific plot and Chinese PDF; rendering deps are isolated."""
from pathlib import Path
import json
import runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent

def figure():
    rows=json.loads((HERE/'qspace_analysis.json').read_text())['rows']
    labels=('5d_3/2','5d_5/2','6p_1/2','6p_3/2')
    x=np.arange(4)
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    for offset,name,column,color in ((-.24,'Bare P','bare_transition_cm1','#385f86'),
                                    (0,'Restricted one-hole Born','Born_one_hole_transition_cm1','#e0a447'),
                                    (.24,'Full finite Q','full_Q_transition_cm1','#008c88')):
        values=[rows[k][column]-rows[k]['experiment_cm1'] for k in labels]
        ax.bar(x+offset,values,.23,label=name,color=color)
    ax.axhline(0,color='#374151',lw=.8)
    ax.set_xticks(x,['5d 3/2','5d 5/2','6p 1/2','6p 3/2'])
    ax.set_ylabel('Theory - experiment (cm$^{-1}$)')
    ax.set_title('Yb II: shared orbitals, nucleus and ground-state reference')
    ax.legend(loc='lower left',frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.savefig(HERE/'qspace_residuals.png',dpi=180)
    fig.savefig(HERE/'qspace_residuals.svg')
    plt.close(fig)

def main():
    import markdown
    from weasyprint import HTML
    from importlib.metadata import version
    figure()
    css=runpy.run_path(str(HERE.parent/'rare_earth_yb_spectrum_2026_10_08/render_report.py'))['CSS']
    css=css.replace('Atomx · Yb⁺ 谱线计算与实验对照','Atomx · 同积分核对与有限 P/Q 传播')
    css+='\nimg {max-width:100%;} h1 {font-size:18pt;} table {break-inside:avoid;}\n'
    body=markdown.markdown((HERE/'README.md').read_text(),extensions=['tables','fenced_code'])
    doc=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Atomx 同积分验证与核心空穴传播</title><style>{css}</style><body>{body}</body></html>'
    (HERE/'Atomx_同积分核对与核心空穴传播.html').write_text(doc)
    HTML(string=doc,base_url=str(HERE)).write_pdf(HERE/'Atomx_同积分核对与核心空穴传播.pdf')
    (HERE/'render_dependencies.json').write_text(json.dumps({n:version(n) for n in ('markdown','weasyprint','matplotlib')},indent=2)+'\n')

if __name__=='__main__': main()

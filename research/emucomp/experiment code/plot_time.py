import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns

plt.style.use(['science', 'no-latex', 'high-vis', 'grid'])

parent_datadir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\experiment code\save'
output_figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\experiment code\outfigs'
flist = glob.glob(parent_datadir + r'\*.json')
d = []
for fname in flist:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

df = pd.DataFrame(d)
df1 = df
df1['npts'] = df1.n * df1.nx * (1 - df1.failfraction.mean())

# create valid markers from mpl.markers
valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if
                  item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

# valid_markers = mpl.markers.MarkerStyle.filled_markers
markers = np.random.choice(valid_markers, df.method.nunique(), replace=False)

ylabels = {'rmse': 'RMSE',
           'mae': 'MAE',
           'medae': 'median absolute error',
           'me': 'mean error',
           'crps': 'CRPS',
           'coverage': r'90\% coverage',
           'avgintwidth': r'90\% interval width',
           'intscore': r'interval score',
           'emutime': r'construction time'
           }

for y, ylabel in ylabels.items():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    p = sns.lineplot(x='npts', y=df1[y],
                     hue='method',
                     style='method',
                     markersize=12,
                     lw=4,
                     alpha=0.65,
                     estimator='median',
                     ci=None,
                     # logx=True,
                     # s=80,
                     # alpha=0.5,
                     # lw=4,
                     data=df1)
    # p.axhline(3600, color='k', linestyle='--')
    p.set(xscale="log")
    if y != 'coverage':
        p.set(yscale="log")
    plt.ylabel(ylabel, labelpad=40, fontsize=20)
    plt.xlabel('N', labelpad=20, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_figdir + r'\{:s}.png'.format(y))
    plt.close()



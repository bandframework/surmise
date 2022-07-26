import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns

plt.style.use(['science', 'no-latex', 'high-vis', 'grid'])

parent_datadir = r'C:\Users\cmyh\Documents\git\surmise\research\emucomp\experiment code\emulator_timing_results\3\data'
output_figdir = r'C:\Users\cmyh\Documents\git\surmise\research\emucomp\experiment code\emulator_timing_results\3\plot'
flist = glob.glob(parent_datadir + r'\*.json')
d = []
for fname in flist:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

root_df = pd.DataFrame(d)
root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())

markers = ['D', 'v', 'X', 'o']
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
    p = sns.lineplot(x='npts', y=root_df[y],
                     hue='method',
                     style='method',
                     markersize=12,
                     lw=4,
                     alpha=0.65,
                     estimator='mean',
                     ci=None,
                     # logx=True,
                     # s=80,
                     # alpha=0.5,
                     # lw=4,
                     data=root_df)
    # p.axhline(3600, color='k', linestyle='--')
    p.set(xscale="log")
    if y != 'coverage':
        p.set(yscale="log")
    plt.ylabel(ylabel, labelpad=40, fontsize=20)
    plt.xlabel('N', labelpad=20, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_figdir + r'\{:s}.png'.format(y))
    plt.close()



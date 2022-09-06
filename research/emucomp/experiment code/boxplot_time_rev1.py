import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns
import itertools

plt.style.use(['science', 'bright', 'grid'])

if False:
    parent_datadir = r'./research/emucomp/experiment code/save'
    output_figdir = r'./research/emucomp/experiment code/outfigs'
    flist = glob.glob(parent_datadir + r'\*.json')
    d = []
    for fname in flist:
        with open(fname, 'r') as f:
            x = json.load(f)
            d.append(json.loads(x))


    df = pd.DataFrame(d)
    df.to_json(r'./research/emucomp/experiment code/compiled_df2.json')

df = pd.read_json(r'./research/emucomp/experiment code/compiled_df.json')
df1 = df
df1['npts'] = df1.n * df1.nx * (1 - df1.failfraction.mean())

# create valid markers from mpl.markers
valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if
                  item[1] != 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

# valid_markers = mpl.markers.MarkerStyle.filled_markers
markers = np.random.choice(valid_markers, df.method.nunique(), replace=False)

timedf = df1[(df1['function']=='OTLcircuit') & (df1['randomfailures']=='False') & \
             (df1['failfraction']==0.25)]
timedf = timedf[(timedf.method == 'PCGPwM') | (timedf.method == 'colGP') | \
                (timedf.method == 'GPEmGibbs') | (timedf.method == 'GPy')]
timedf['N'] = timedf.n * timedf.nx
timedf['logtime'] = np.log(timedf.emutime)
timedf = timedf[timedf.emutime <= 3600]

fig, ax = plt.subplots(figsize=(8, 6))
p = sns.boxplot(x='N', y='emutime',
                hue='method', hue_order=['GPy', 'colGP', 'GPEmGibbs', 'PCGPwM'],
                whis=3, fliersize=0,
                data=timedf)
p.axhline(3600, color='k', linestyle='--', linewidth=2)
p.set_ylim(0.02, 7500)
p.set_xticklabels(['$50m$', '$100m$', '$250m$', '$1000m$', '$2500m$'])
p.tick_params(which='minor', axis='x', bottom=False, top=False)
p.tick_params(which='minor', axis='y', left=False, right=False)
p.tick_params(which='major', axis='both', labelsize=16)
p.set(yscale='log')

handles, _ = p.get_legend_handles_labels()
p.legend(handles, ['omit', 'colGP', 'EMGP', 'PCGPwM'],
         title=None, ncol=4, loc='lower center', frameon=False,
         fontsize=14)

plt.ylabel('Construction time ($s$)', fontsize=24)
plt.xlabel('$N$', fontsize=24)
plt.tight_layout()
plt.savefig('./research/emucomp/experiment code/revfigs/timeplot.png', dpi=300)

#
# timeregdf = []
# for method in timedf.method.unique():
#     subdf = timedf[timedf.method == method]
#     subdf['logN'] = np.log(subdf.N)
#     nsamp = len(subdf)
#     NXY = nsamp * (subdf.logtime * subdf.logN).sum()
#     SXSY = subdf.logtime.sum() + subdf.logN.sum()
#     NX2 = nsamp * (subdf.logN ** 2).sum()
#     SX2 = (subdf.logN.sum())**2
#     slope = (NXY - SXSY) / (NX2 - SX2)
#     intercept = subdf.logtime.mean() - slope * subdf.logN.mean()
#     timeregdf.append([method, slope, intercept])
# p = sns.lmplot(x='N', y='logtime',
#                 hue='method', # style='method',
#                 x_ci=None, # s=80,
#                 # alpha=0.3,
#                 data=timedf,
#                 )
#
#
# ylabels = {'rmse': 'RMSE',
#            'mae': 'MAE',
#            'medae': 'median absolute error',
#            'me': 'mean error',
#            'crps': 'CRPS',
#            'coverage': r'90\% coverage',
#            'avgintwidth': r'90\% interval width',
#            'intscore': r'interval score',
#            'emutime': r'construction time'
#            }
#
# for y, ylabel in ylabels.items():
#     fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
#     p = sns.lineplot(x='npts', y=df1[y],
#                      hue='method',
#                      style='method',
#                      markersize=12,
#                      lw=4,
#                      alpha=0.65,
#                      estimator='median',
#                      ci=None,
#                      # logx=True,
#                      # s=80,
#                      # alpha=0.5,
#                      # lw=4,
#                      data=df1)
#     # p.axhline(3600, color='k', linestyle='--')
#     p.set(xscale="log")
#     if y != 'coverage':
#         p.set(yscale="log")
#     plt.ylabel(ylabel, labelpad=40, fontsize=20)
#     plt.xlabel('N', labelpad=20, fontsize=20)
#     plt.tight_layout()
#     plt.savefig(output_figdir + r'\{:s}.png'.format(y))
#     plt.close()
#
#

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import pandas as pd
import seaborn as sns
import json
import glob
plt.style.use(['science', 'high-vis', 'grid'])

output_figdir = r'./research/emucomp/experiment code/revfigs/panelplots'


root_df = pd.read_json(r'./research/emucomp/experiment code/compiled_df.json')
root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())
# root_df.randomfailures = root_df.randomfailures.astype(str)
root_df[['rmse', 'mae', 'medae', 'me', 'crps']] = root_df[['rmse', 'mae', 'medae', 'me', 'crps']].astype(float)

fail_configs = [
    (True, 0.01),
    (True, 0.05),
    (True, 0.25),
    (False, 0.01),
    (False, 0.05),
    (False, 0.25),
]

markers = ['D', 'v', 'X', 's', 'o', '^', 'P']
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

funcs = pd.unique(root_df.function)
funcs[1], funcs[3] = funcs[3], funcs[1]

for fail_random, fail_level in fail_configs:
    df = root_df[(root_df.randomfailures == fail_random) &
                 (root_df.failfraction == fail_level)]
    df.method = df.method.str.replace(r'_', r'-')

    for y, ylabel in ylabels.items():
        std = df[y].std()
        df[y][df[y] > 10**6] = np.nan
        fig, ax = plt.subplots(nrows=2, ncols=2,
                               figsize=(8, 6),
                               sharex='all',
                               sharey='all')

        for i, func in enumerate(funcs):
            subdf = df[df.function == func]
            r, c = divmod(i, 2)
            sns.lineplot(x='npts', y=subdf[y],
                         hue='method',
                         style='method',
                         markers=markers,
                         markersize=12,
                         lw=4,
                         alpha=0.65,
                         estimator='median',
                         ci=None,
                         # err_kws={'alpha': 0.25},
                         ax=ax[r][c],
                         data=subdf
                         )
            ax[r][c].set_xlabel('')
            ax[r][c].set_ylabel('')
            ax[r][c].set_yticks([])
            ax[r][c].set_title(func)

        handles, labels = ax[r][c].get_legend_handles_labels()
        for axis in ax.flatten():
            axis.set_xscale('log')
            if y not in ['coverage']:
                axis.set_yscale('log', nonpositive='clip')
            else:
                axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axis.set_ylim((-0.1, 1.1))
            try:
                axis.get_legend().remove()
            except:
                pass

        fig.add_subplot(111, frameon=False)
        fig.legend(handles, labels, loc='lower center', ncol=5)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(ylabel, labelpad=40, fontsize=20)
        plt.xlabel('data size', labelpad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(output_figdir + r'\{:s}_{:s}.png'.format(y, str(int(fail_level*100)) + '_random' + str(fail_random)))
        # plt.show()
        plt.close()
        # break
        # plt.show()
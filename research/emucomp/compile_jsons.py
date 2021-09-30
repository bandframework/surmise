import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.25, style='ticks', palette='colorblind')
# plt.style.use('science')
figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs\emucomparisons_20210923'
directory = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emucomparison\emucomparisons_20210923'

file_list = glob.glob(directory+r'\*\*.json')

d = []
for fname in file_list:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

df = pd.DataFrame(d)
df['npts'] = df.n * df.nx
df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']] = \
    df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']].apply(pd.to_numeric, args=('coerce',))
df[['rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']] += 10**(-8)
funcs = pd.unique(df.function)

##
markers = ['D', 'v', 'X', 's']  #, 'X', 'p', 'D', '*', '+']

ylabels = {'rmse': 'RMSE',
           'mae': 'MAE',
           'medae': 'median absolute error',
           'me': 'mean error',
           'crps': 'CRPS',
           'coverage': r'90% coverage',
           'avgintwidth': r'90% interval width',
           'intscore': r'interval score',
           'emutime': r'time'
}

randomlabels = {True: 'random missingness',
                False: 'structured missingness'}

for y in ['rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore', 'emutime']:
    for random in [True, False]:
        for level in ['high', 'low']:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharey='all', sharex='all')
            for i, func in enumerate(funcs):
                subdf = df[(df.function == func) & (df.randomfailures == str(random)) & (df.failureslevel == level)]
                r, c = divmod(i, 2)
                npoints = subdf.npts * 0.75 if level == 'low' else subdf.npts * 0.25
                sns.lineplot(x=npoints, y=y,
                             hue='method',
                             markers=markers,
                             markersize=10,
                             lw=3,
                             alpha=0.8,
                             estimator='mean',
                             ci=None,
                             err_kws={'alpha': 0.0},
                             data=subdf,
                             ax=ax[r][c])
                if c > 0:
                    ax[r][c].set_ylabel('')
                else:
                    ax[r][c].set_ylabel(ylabels[y])
                if r > 0:
                    ax[r][c].set_xlabel('data size')
                else:
                    ax[r][c].set_xlabel('')
                ax[r][c].set_title(func)
                if i < len(funcs) - 1:
                    try:
                        ax[r][c].get_legend().remove()
                    except:
                        pass

            for axi in ax.flatten():
                axi.set_xscale('log')
                if y != 'coverage':
                    axi.set_yscale('log')
            plt.suptitle(level + ' ' + randomlabels[random], x=0, y=0.99, size='small', ha='left')
            plt.tight_layout()
            fig.subplots_adjust(top=0.9, left=0.1, right=0.95, bottom=0.18)
            ax.flatten()[-1].legend(loc='upper right', bbox_to_anchor=(0.99, -0.3), ncol=4)
            plt.savefig(figdir + '\\' + y + 'random' + str(random) + level + r'.png', dpi=150)
            fig.close()
            # print(figdir + '\\' + y + 'random' + str(random) + level + r'.png')

import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-colorblind')

figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs'
directory = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emucomparison\20210820040351'

file_list = glob.glob(directory+r'\*.json')

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
markers = ['D', 'v', 'X']  #, 'X', 'p', 'D', '*', '+']

ylabels = {'rmse': 'RMSE',
           'mae': 'MAE',
           'medae': 'median absolute error',
           'me': 'mean error',
           'crps': 'CRPS',
           'coverage': r'90\% pred. interval coverage',
           'avgintwidth': r'90\% pred. interval width',
           'intscore': r'interval score',
           'emutime': r'time'
}

for y in ['rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore', 'emutime']:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharey='all', sharex='all')
    for i, func in enumerate(funcs):
        r, c = divmod(i, 2)
        sns.lineplot(x='npts', y=y,
                     hue='method', style='method',
                     markers=markers,
                     err_kws={'alpha': 0.0},
                     data=df[df.function == func],
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
            ax[r][c].get_legend().remove()

    for axi in ax.flatten():
        axi.set_xscale('log')
        if y != 'coverage':
            axi.set_yscale('log')

    plt.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    ax.flatten()[-1].legend(loc='upper right', bbox_to_anchor=(0.5, -0.20), ncol=3)
    plt.savefig(figdir + '\\' + y + r'.png', dpi=150)

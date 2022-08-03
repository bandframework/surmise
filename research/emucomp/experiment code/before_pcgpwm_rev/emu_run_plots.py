import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
plt.style.use('seaborn-colorblind')
fail_configs = [(True, 'low'), (True, 'high'), (False, 'low'), (False, 'high'), (None, 'none')]

# Plotting adjustments
markers = ['D', 'v', 'X', 's']


def run_plots(plot_dir, listJSONs):
    d = []
    for fname in listJSONs:
        with open(fname, 'r') as f:
            x = json.load(f)
            d.append(json.loads(x))

    df = pd.DataFrame(d)

    df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']] = \
        df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']].apply(
            pd.to_numeric, args=('coerce',))
    df['npts'] = df.n * df.nx
    df.bigM = df.bigM.astype(str)

    funcs = pd.unique(df.function)

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

    for func in funcs:
        for fail_random, fail_level in fail_configs:
            subdf = df[(df.function == func) & (df.randomfailures == str(fail_random)) & (df.failureslevel == str(fail_level))]
            for y, ylabel in ylabels.items():
                plt.figure()
                if len(pd.unique(subdf.bigM)) > 1:
                    hue = subdf.bigM
                else:
                    hue = None
                sns.lineplot(x=subdf.npts, y=subdf[y],
                             hue=hue,
                             markers=markers,
                             markersize=10,
                             lw=3,
                             alpha=0.8,
                             estimator='mean',
                             ci=None,
                             err_kws={'alpha': 0.0},)
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel(ylabel)
                plt.xlabel('data size')
                plt.tight_layout()
                plt.savefig(plot_dir + '\\' + y + func + 'random' + str(fail_random) + fail_level + r'.png')
                plt.close()

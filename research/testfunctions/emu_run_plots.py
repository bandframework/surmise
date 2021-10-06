import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns

plt.style.use('seaborn-whitegrid')

fail_configs = [(True, 'low'), (True, 'high')]# , (False, 'low'), (False, 'high'), (None, 'none')]


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
                    sns.lineplot(x=subdf.npts, y=subdf[y], hue=subdf.bigM)
                    plt.legend()
                else:
                    sns.lineplot(x=subdf.npts, y=subdf[y])
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel(ylabel)
                plt.xlabel('data size')
                plt.tight_layout()
                plt.savefig(plot_dir + '\\' + y + func + 'random' + str(fail_random) + fail_level + r'.png')
                plt.close()
#
# import glob
# filelist = glob.glob(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emulator_PCGPwM_results\1\data' + r'\*.json')
# run_plots(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emulator_PCGPwM_results\1\plot', filelist)
#

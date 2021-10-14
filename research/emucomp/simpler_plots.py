import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns

figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs\20210917145801'
directory = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emucomparison\20210917145801'


file_list = glob.glob(directory+r'\*.json')

d = []
for fname in file_list:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

df = pd.DataFrame(d)

df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']] = \
    df[['avgvarconstant', 'rmse', 'mae', 'medae', 'me', 'crps', 'coverage', 'avgintwidth', 'intscore']].apply(pd.to_numeric, args=('coerce',))
df['npts'] = df.n * df.nx
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


for y, ylabel in ylabels.items():
    plt.figure()
    sns.lineplot(x=df.npts, y=df[y], hue=df.bigM, ci=None)
    # plt.scatter(df.npts, df[y])
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel('data size')
    plt.tight_layout()

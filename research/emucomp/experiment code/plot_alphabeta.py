import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis', 'grid'])

output_figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs'
datafile = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\results_alphabeta\errors_20211018181009_randomTrue.json'
with open(datafile, 'r') as f:
    df_json = json.load(f)
df = pd.read_json(df_json)


markers = ['D', 'v']

funcs = pd.unique(df.function)
funcs[1], funcs[3] = funcs[3], funcs[1]

fig, ax = plt.subplots(nrows=2, ncols=2,
                       figsize=(8, 6),
                       sharex='all',
                       sharey='all')

for i, func in enumerate(funcs):
    subdf = df[df.function == func]
    r, c = divmod(i, 2)
    sns.lineplot(x='dampalpha', y=subdf['rmse'],
                 hue='varc_status',
                 style='varc_status',
                 markers=markers,
                 markersize=12,
                 lw=4,
                 alpha=0.65,
                 estimator='mean',
                 ci=None,
                 ax=ax[r][c],
                 data=subdf
                 )
    ax[r][c].set_xlabel('')
    ax[r][c].set_ylabel('')
    ax[r][c].set_yticks([])
    ax[r][c].set_title(func)

handles, labels = ax[r][c].get_legend_handles_labels()
for axis in ax.flatten():
    # axis.set_xscale('log')
    axis.set_yscale('log', nonpositive='clip')
    axis.get_legend().remove()

fig.add_subplot(111, frameon=False)
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.xticks([])
plt.yticks([])
plt.ylabel('RMSE', labelpad=40, fontsize=20)
plt.xlabel('data size', labelpad=20, fontsize=20)
plt.tight_layout()
# plt.savefig(output_figdir + r'\alphabeta.png')
# plt.close()
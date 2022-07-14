import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_fails(x, theta, model):
    f = model(x, theta)
    fisnan = np.isnan(f).sum(0)
    # fisnan = np.isinf(f).sum(0)
    df = pd.DataFrame(theta)
    df['fails'] = fisnan.astype(float)
    df['failsbin'] = fisnan > 0
    # sns.pairplot(df, vars=df.columns[:theta.shape[1]], hue='fails', palette='coolwarm')
    colors = ["#4374B3", "#FF0B04"]
    sns.set_palette(sns.color_palette(colors))
    del df['fails']
    meltdf = pd.melt(df, df.columns[-1], df.columns[:-1])
    g = sns.FacetGrid(meltdf, col="variable", hue='failsbin', legend_out=True)
    g.map(sns.kdeplot, "value", shade=True)

    for i, ax in enumerate(g.axes[0]):
        ax.set_xlabel(r'$\theta_{:d}$'.format(i + 1))
        ax.set_title('')

    g.add_legend()
    g._legend.set_title('Fails')
    new_labels = ['0', '> 0']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


def plot_marginal(x, theta, model, modelname, emus=()):
    sns.set_palette('icefire')
    markers = ['o', 'v', 's', 'X', 'p', 'D', '*', '+']
    if theta.shape[0] > 100:
        theta = theta[:100, :]

    alpha = emus[-1]._info['dampalpha']

    f = model(x, theta)

    fig, axes = plt.subplots(1, theta.shape[1], figsize=(4 * theta.shape[1], 4))
    for k, ax in enumerate(axes):
        sns.scatterplot(x=theta[:, k], y=f[10], color='r', label='true', marker='D', ax=ax, s=20, zorder=100)
        for i, emu in enumerate(emus):
            emumean = emu.predict(x, theta).mean()[10]
            emulabel = emu.method.__name__.split('.')[-1]
            if 'logvarc' in emu._info:
                emulabel = r'$\beta$=' + str(np.round(np.mean(emu._info['logvarc']), 0))
            sns.lineplot(theta[:, k], emumean, label=emulabel, marker=markers[i], sort=True, ax=ax, zorder=1)
        ax.set_xlabel(r'$\theta_{:d}$'.format(k + 1))
        ax.set_ylabel('function prediction')

    plt.suptitle(r'{:s}, $\alpha$ = {:.3f}'.format(modelname, alpha))
    plt.legend()
    plt.tight_layout()

    dirname = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs'
    fname = r'\{:s}_alph{:.3f}.png'.format(modelname, alpha)
    plt.savefig(dirname + fname, dpi=75)

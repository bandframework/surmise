import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def mse(emu, x, theta, model):
    return np.nanmean((emu.predict(x, theta).mean() - model(x, theta)) ** 2)


def mae(emu, x, theta, model):
    return np.nanmean(np.abs(emu.predict(x, theta).mean() - model(x, theta)))


def medae(emu, x, theta, model):
    return np.nanmedian(np.abs(emu.predict(x, theta).mean() - model(x, theta)))


def me(emu, x, theta, model):
    return np.nanmean(emu.predict(x, theta).mean() - model(x, theta))


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
        ax.set_xlabel(r'$\theta_{:d}$'.format(i+1))
        ax.set_title('')

    g.add_legend()
    g._legend.set_title('Fails')
    new_labels = ['0', '> 0']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


def plot_marginal(x, theta, model, emus=()):
    sns.set_palette('icefire')
    markers = ['o', 'v', 's', 'X', 'p', 'D', '*', '+']
    f = model(x, theta)

    fig, axes = plt.subplots(1, theta.shape[1])
    for k, ax in enumerate(axes):
        sns.scatterplot(x=theta[:, k], y=f[5], color='r', label='true', marker='D', ax=ax, s=20, zorder=100)
        for i, emu in enumerate(emus):
            emumean = emu.predict(x, theta).mean()[5]
            emulabel = emu.method.__name__.split('.')[-1]
            sns.lineplot(theta[:, k], emumean, label=emulabel, marker=markers[i], sort=True, ax=ax, zorder=1)
        ax.set_xlabel(r'$\theta_{:d}$'.format(k+1))
        ax.set_ylabel('function prediction')

    plt.legend()
    plt.show()


def errors(emu, x, theta, model):
    results = {}
    results['method'] = emu.method.__name__.split('.')[-1]
    if 'logvarc' in emu._info.keys():
        results['avgvarconstant'] = '{:.3f}'.format(np.mean(np.exp(emu._info['logvarc'])))
    else:
        results['avgvarconstant'] = np.nan
    results['mse'] = '{:.3f}'.format(mse(emu, x, theta, model))
    results['mae'] = '{:.3f}'.format(mae(emu, x, theta, model))
    results['medae'] = '{:.3f}'.format(medae(emu, x, theta, model))
    results['me'] = '{:.3f}'.format(me(emu, x, theta, model))
    return results

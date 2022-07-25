import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns

plt.style.use(['science', 'no-latex', 'high-vis', 'grid'])

parent_datadir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\experiment code\emulator_timing_results\3\data'
flist = glob.glob(parent_datadir + r'\*.json')
d = []
for fname in flist:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

root_df = pd.DataFrame(d)
root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())

markers = ['D', 'v', 'X', 'o']
p = sns.lmplot(x='npts', y='emutime',
                    hue='method',
                    # style='method',
                    markers=markers,
                    # logx=True,
                    # s=80,
                    # alpha=0.5,
                    # lw=4,
                    data=root_df)
# p.axhline(3600, color='k', linestyle='--')
p.set(xscale="log", yscale="log")

import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns
plt.style.use(['science', 'high-vis', 'grid'])


parent_datadir = r'C:\Users\cmyh\Documents\git\surmise\research\emucomp\experiment code\emulator_timing_results\1\data'
flist = glob.glob(parent_datadir + r'\*.json')
d = []
for fname in flist:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

root_df = pd.DataFrame(d)
root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())

markers = ['D', 'v', 'X']
p = sns.scatterplot(x='npts', y='emutime',
             hue='method',
             style='method',
             markers=markers,
             s=80,
             # lw=4,
             data=root_df)
p.set_xscale('log')
p.set_yscale('log')

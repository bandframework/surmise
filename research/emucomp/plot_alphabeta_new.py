import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis', 'grid'])

datafile = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\results\errors_20210817100630_randomTrue.json'
with open(datafile) as f:
    df_raw = json.load(f)

df = pd.read_json(df_raw)
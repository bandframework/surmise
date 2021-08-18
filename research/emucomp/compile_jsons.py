import pandas as pd
import json
import glob

directory = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\emucomparison'

file_list = glob.glob(directory+r'\*.json')

d = []
for fname in file_list:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

df = pd.DataFrame(d)


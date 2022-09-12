import pandas as pd
import numpy as np

reps = [0, 1, 2, 3, 4]
n = [50, 100, 250, 500, 1000]
function = ['borehole']
failrandom = ['False']
failfrac = [0.25]
method = ['PCGPwM', 'EMGP', 'colGP', 'GPy']

base = np.array(np.meshgrid(n, function, failrandom, failfrac, method, reps)).T.reshape(-1, 6)

np.savetxt(r'./research/emucomp/experiment code/params_time.txt', base, fmt='%s', delimiter='\t')


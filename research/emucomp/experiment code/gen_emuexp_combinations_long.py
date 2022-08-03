import pandas as pd
import numpy as np

runParams = pd.read_table('./research/emucomp/experiment code/params_surmiserev1/params.txt', header=None)
reps = np.random.randint(1, 50000, size=20)

n = [2500]
function = ['borehole', 'piston', 'otlcircuit', 'wingweight']
failrandom = ['True', 'False']
failfrac = [0.01, 0.05, 0.25]
method = ['PCGPwM', 'PCGP_KNN', 'PCGP_benchmark']

base = np.array(np.meshgrid(n, function, failrandom, failfrac, method, reps)).T.reshape(-1, 6)
knn = base[base[:, -2]=='PCGP_KNN']
bench = base[base[:, -2]=='PCGP_benchmark']
pcgpwm = base[base[:, -2]=='PCGPwM']

subbench = bench[bench[:, 1]=='piston']
subpcgpwm = pcgpwm[(pcgpwm[:, 1]=='borehole') | (pcgpwm[:, 1]=='piston')]

subbase = np.row_stack((knn, subbench, subpcgpwm))

np.savetxt('params_long.txt', subbase, fmt='%s', delimiter='\t')

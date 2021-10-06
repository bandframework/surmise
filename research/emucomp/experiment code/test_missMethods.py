# import numpy as np
# from rpy2.robjects.packages import importr
# from rpy2.robjects import FloatVector
# from rpy2.robjects import r
#
# importr('missMethods')
#
# M = np.random.uniform(0, 1, size=(20, 15))
# rM = r.matrix(FloatVector(M), nrow=20, ncol=15)
# print(r['ncol'](rM))
# rM = r['delete_MAR_censoring'](rM, 0.25)
#
# import matplotlib.pyplot as plt
# plt.imshow(M, aspect='auto')


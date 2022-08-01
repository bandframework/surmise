import numpy as np
import scipy.stats as sps
from testfunc_wrapper import TestFunc
import matplotlib.pyplot as plt

# Number of input locations
nx = 15

funcs = ['piston', 'borehole', 'otlcircuit', 'wingweight']
emulator_methods = ['PCGPwM']# , 'GPy'] #, 'PCGP_KNN', 'PCGP_BR'] #

for func in ['wingweight', 'borehole']:#funcs:
    func_caller = TestFunc(func).info
    function_name = func_caller['function']
    xdim = func_caller['xdim']
    thetadim = func_caller['thetadim']

    np.random.seed()
    x = sps.uniform.rvs(0, 1, (nx, xdim))
    np.random.seed()

    thetas = []
    for ntheta in [50, 100, 250, 1000, 2500]:
        thetas.append(np.random.uniform(0, 1, (ntheta, thetadim))) # lhs(thetadim, ntheta))

    for p in [0.01]:#, 0.05, 0.25]:
        res = []
        for i, ntheta in enumerate([50, 100, 250, 1000, 2500]):
            theta = thetas[i]
            model = func_caller['failmodel']
            f = model(x, theta, p)
            model2 = func_caller['failmodel_random']
            f2 = model2(x, theta, p)

            print((func, int(ntheta),
                        np.isnan(f).mean(),
                        np.isnan(f).sum(1).var() / ntheta,
                        np.isnan(f).sum(0).var() / nx, '\t',
                        np.isnan(f2).mean(),
                        np.isnan(f2).sum(1).var() / ntheta,
                        np.isnan(f2).sum(0).var() / nx
                        ))
            plt.figure()
            plt.imshow(np.isnan(f).T, aspect='auto', cmap='gray', interpolation='none')
            plt.colorbar()
            plt.figure()
            plt.imshow(np.isnan(f2).T, aspect='auto', cmap='gray', interpolation='none')
            plt.colorbar()

        plt.show()
        #
        # print(p)
        # for row in res:
        #     print(['{:.3f}'.format(x) for x in row])


#
# raise
# f1 = f.copy()
#
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# plt.style.use('science')
#
# plt.figure()
# plt.imshow(~np.isnan(f1.T), cmap='gray', aspect='auto', interpolation='none')
# plt.colorbar()
# plt.title('Missing values (shown as dark patches)')
#
# f1mask = ~np.isnan(f1)
#
# emu1 = single_test('PCGPwM', x, theta, f1, model, testtheta,
#             func, ntheta, 'none', 'none', 10, 0, '')
#
# diff = np.sqrt((emu1.predict(x, testtheta).mean() - model(x, testtheta))**2)
# plt.figure()
# plt.scatter(emu1.predict(x, testtheta).mean(), model(x, testtheta))
#
# curr_cm = matplotlib.cm.get_cmap('autumn')
# curr_cm.set_bad(color='black')
# plt.figure(figsize=(8,8))
# plt.imshow(diff.T, cmap=curr_cm, aspect='auto', norm=LogNorm(), interpolation='none')
# plt.colorbar()
# plt.title('RMSE (full)')
# plt.show()
# #
# # plt.figure(figsize=(3,8))
# plt.imshow(np.atleast_2d(diff[12]).T, cmap='autumn', aspect='auto', norm=LogNorm(), interpolation='none')
# plt.xticks([])
# plt.xlabel(12)
# plt.colorbar()
# plt.title('RMSE (Column 12)')
# plt.show()
# emu2 = single_test('GPy', x, theta, f1, model, testtheta,
#              'otlcircuit', 500, 'none', 'none', 10, 0, '')

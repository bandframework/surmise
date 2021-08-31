import numpy as np
from testdiagnostics import errors
from testplots import plot_marginal, plot_fails


def print_struc_fails(meta, model, c=None):
    if c is None:
        f = model(x, theta)
    else:
        f = model(x, theta, c)

    print(meta['function'])
    print(r'no. failures: {:d}/{:d}, {:.3f}'.format(np.isnan(f).sum(), f.size, np.isnan(f).sum()/f.size))
    print(r'no. anyfail rows: {:d}/{:d}'.format(np.isnan(f).any(0).sum(), f.shape[1]))
    print(r'no. allfail rows: {:d}/{:d}'.format(np.isnan(f).all(0).sum(), f.shape[1]))


def print_random_fails(meta, model, p=None):
    if p is None:
        f = model(x, theta)
    else:
        f = model(x, theta, p)

    print(meta['function'])
    print(r'no. failures: {:d}/{:d}, {:.3f}'.format(np.isnan(f).sum(), f.size, np.isnan(f).sum() / f.size))
    print(r'no. anyfail rows: {:d}/{:d}'.format(np.isnan(f).any(0).sum(), f.shape[1]))
    print(r'no. allfail rows: {:d}/{:d}'.format(np.isnan(f).all(0).sum(), f.shape[1]))


if __name__ == '__main__':
    i = 3
    if i == 0:
        import boreholetestfunctions as func
        from boreholetestfunctions import borehole_failmodel as failmodel
        from boreholetestfunctions import borehole_failmodel_random as failmodel_random
        from boreholetestfunctions import borehole_model as nofailmodel
        from boreholetestfunctions import borehole_true as truemodel
    elif i == 1:
        import TestingfunctionPiston as func
        from TestingfunctionPiston import Piston_failmodel as failmodel
        from TestingfunctionPiston import Piston_failmodel_random as failmodel_random
        from TestingfunctionPiston import Piston_model as nofailmodel
        from TestingfunctionPiston import Piston_true as truemodel
    elif i == 2:
        import TestingfunctionOTLcircuit as func
        from TestingfunctionOTLcircuit import OTLcircuit_failmodel as failmodel
        from TestingfunctionOTLcircuit import OTLcircuit_failmodel_random as failmodel_random
        from TestingfunctionOTLcircuit import OTLcircuit_model as nofailmodel
        from TestingfunctionOTLcircuit import OTLcircuit_true as truemodel
    elif i == 3:
        import TestingfunctionWingweight as func
        from TestingfunctionWingweight import Wingweight_failmodel as failmodel
        from TestingfunctionWingweight import Wingweight_failmodel_random as failmodel_random
        from TestingfunctionWingweight import Wingweight_model as nofailmodel
        from TestingfunctionWingweight import Wingweight_true as truemodel

    ntheta = 100
    nx = 15

    func_meta = func.query_func_meta()

    x = np.random.uniform(0, 1, (nx, func_meta['xdim']))
    theta = np.random.uniform(0, 1, (ntheta, func_meta['thetadim']))

    import matrix_completion
    f = failmodel_random(x, theta, 'low')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(f)
    f_scaled = scaler.transform(f)
    ffix_scaled = matrix_completion.svt_solve(f_scaled, ~np.isnan(f).astype(int))
    ffix = scaler.inverse_transform(ffix_scaled)

    #
    # print_struc_fails(func_meta, failmodel, (0.25, 0.6))  #, 1.8)
    # # print_struc_fails(func_meta, failmodel, 0.7)
    #
    # print_random_fails(func_meta, failmodel_random, 0.75)
    # print_random_fails(func_meta, failmodel_random, 0.25)
    #
    print('\nStructured failures:')
    print_struc_fails(func_meta, failmodel, 'low')
    print_struc_fails(func_meta, failmodel, 'high')

    print('\nRandom failures:')
    print_random_fails(func_meta, failmodel_random, 'low')
    print_random_fails(func_meta, failmodel_random, 'high')

    # plot_marginal(x, theta, nofailmodel)
    # plot_fails(x, theta, nofailmodel)

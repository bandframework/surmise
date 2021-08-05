import TestingfunctionPiston
from TestingfunctionOTLcircuit import OTLcircuit_true, OTLcircuit_failmodel, OTLcircuit_model
from TestingfunctionPiston import Piston_model, Piston_failmodel, Piston_true
from boreholetestfunctions import borehole_model
import TestingfunctionWingweight
from TestingfunctionWingweight import Wingweight_model, Wingweight_failmodel
from testdiagnostics import plot_marginal, plot_fails, errors
import numpy as np


ntheta = 100
nx = 20

func_meta = TestingfunctionWingweight.query_func_meta()

theta = np.random.uniform(0, 1, (ntheta, func_meta['thetadim']))
x = np.random.uniform(0, 1, (nx, func_meta['xdim']))
f = Wingweight_model(x, theta)

failf = Wingweight_failmodel(x, theta)

plot_marginal(x, theta, Wingweight_model)
plot_fails(x, theta, Wingweight_failmodel)

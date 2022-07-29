import argparse
import numpy as np
from scipy import stats as sps
from testfunc_wrapper import TestFunc
from emu_single_test import single_test
parser = argparse.ArgumentParser(description='Takes argument to mainemutest().')

parser.add_argument('--n', type=int, help='number of parameters')
parser.add_argument('--function', help='name of test function')
parser.add_argument('--failrandom', type=bool, help='True if failures are random (False if structured)')
parser.add_argument('--failfraction', type=float, help='fraction of failures')
parser.add_argument('--method', help='name of emulator method')
parser.add_argument('--rep', type=int, help='id of replication')

args = parser.parse_args()

nx = 15

func_caller = TestFunc(args.function).info
function_name = func_caller['function']
xdim = func_caller['xdim']
thetadim = func_caller['thetadim']

thetasampler = sps.qmc.LatinHypercube(d=thetadim)

np.random.seed(args.rep)
x = sps.uniform.rvs(0, 1, (nx, xdim))
theta = thetasampler.random(args.n)
testtheta = np.random.uniform(0, 1, (1000, thetadim))

if args.failfraction == 0:
    model = func_caller['nofailmodel']
    f = model(x, theta)
elif args.failrandom is True:
    model = func_caller['failmodel_random']
    f = model(x, theta, args.failfraction)
elif args.failrandom is False:
    model = func_caller['failmodel']
    f = model(x, theta, args.failfraction)
else:
    raise ValueError('function caller definition went wrong.')
np.random.seed(None)

result_fname = single_test(emuname=args.method, x=x, theta=theta, f=f, model=model,
                           testtheta=testtheta, modelname=function_name,
                           ntheta=args.n, fail_random=args.failrandom, fail_frac=args.failfraction,
                           j=args.rep, directory='./save', caller=func_caller)


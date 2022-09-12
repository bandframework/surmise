import TestingfunctionBorehole
import TestingfunctionPiston
import TestingfunctionWingweight
import TestingfunctionOTLcircuit


class TestFunc(object):
    def __init__(self, func):
        if func == 'borehole':
            meta = TestingfunctionBorehole.query_func_meta()
            failmodel = TestingfunctionBorehole.borehole_failmodel
            failmodel_random = TestingfunctionBorehole.borehole_failmodel_random
            failmodel_MAR = TestingfunctionBorehole.borehole_failmodel_MAR
            nofailmodel = TestingfunctionBorehole.borehole_model
        elif func == 'otlcircuit':
            meta = TestingfunctionOTLcircuit.query_func_meta()
            failmodel = TestingfunctionOTLcircuit.OTLcircuit_failmodel
            failmodel_random = TestingfunctionOTLcircuit.OTLcircuit_failmodel_random
            failmodel_MAR = TestingfunctionOTLcircuit.OTLcircuit_failmodel_MAR
            nofailmodel = TestingfunctionOTLcircuit.OTLcircuit_model
        elif func == 'wingweight':
            meta = TestingfunctionWingweight.query_func_meta()
            failmodel = TestingfunctionWingweight.Wingweight_failmodel
            failmodel_random = TestingfunctionWingweight.Wingweight_failmodel_random
            failmodel_MAR = TestingfunctionWingweight.Wingweight_failmodel_MAR
            nofailmodel = TestingfunctionWingweight.Wingweight_model
        elif func == 'piston':
            meta = TestingfunctionPiston.query_func_meta()
            failmodel = TestingfunctionPiston.Piston_failmodel
            failmodel_random = TestingfunctionPiston.Piston_failmodel_random
            failmodel_MAR = TestingfunctionPiston.Piston_failmodel_MAR
            nofailmodel = TestingfunctionPiston.Piston_model
        else:
            raise ValueError('Choose between (\'borehole\', \'otlcircuit\', \'wingweight\', \'piston\')')

        self.info = {'function':         meta['function'],
                     'xdim':             meta['xdim'],
                     'thetadim':         meta['thetadim'],
                     'nofailmodel':      nofailmodel,
                     'failmodel':        failmodel,
                     'failmodel_random': failmodel_random,
                     'failmodel_MAR':    failmodel_MAR
                     }
        return

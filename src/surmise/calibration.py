"""
This module contains a class that implements the main calibration method.
"""
import numpy as np
from .helper import cast_f64_dtype, save_file, load_file
import importlib
import copy
import warnings


class calibrator(object):

    def __init__(self,
                 emu=None,
                 y=None,
                 x=None,
                 thetaprior=None,
                 yvar=None,
                 method='directbayes',
                 args={}):
        '''
        A class to represent a calibrator. Fits a calibrator model provided
        in ``calibrationmethods/[method].py`` where [method] is the user
        option with default listed above.

        .. tip::
           To use a new calibrator, just drop a new file to the
           ``calibrationmethods/`` directory with the required formatting.

        :Example:

            .. code-block:: python

               calibrator(emu=emu, y=y, x=x, thetaprior=thetaprior,
                          method='directbayes', args=args)

        Parameters
        ----------
        emu : surmise.emulation.emulator, optional
            An emulator class instance as defined in surmise.emulation.
            The default is None.

        y : numpy.ndarray, optional
            Array of observed values at x. The default is None.

        x : numpy.ndarray, optional
            An array of x values that match the definition of "emu.x".
            Currently, existing methods supports only the case when x is a
            subset of "emu.x". The default is None.

        thetaprior : class, optional
            class instance with two built-in functions. The default is None.

            .. important::
                If a calibration method requires sampling, then
                the prior distribution of the parameters should be included
                into the calibrator. In this case, thetaprior class
                should include two methods:

                    - ``lpdf(theta)``
                        Returns the log of the pdf of a given theta with size
                        ``(len(theta), 1)``
                    - ``rnd(n)``
                        Generates n random variable from a prior distribution.

            :Example:

                .. code-block:: python

                    class prior_example:
                        def lpdf(theta):
                            return sps.uniform.logpdf(
                                theta[:, 0], 0, 1).reshape((len(theta), 1))

                        def rnd(n):
                            return np.vstack((sps.uniform.rvs(0, 1, size=n)))

        yvar : numpy.ndarray, optional
            The vector of observation variances at y. The default is None.

        method : str, optional
            A string that points to the file located in ``calibrationmethods/``
            you would like to use. The default is 'directbayes'.

        args : dict, optional
            Optional dictionary containing options you would like to pass to
            [method].fit(x, theta, f, args)
            or
            [method].predict(x, theta args) The default is {}.

        Raises
        ------
        ValueError
            If the dimension of the data do not match with the fitted emulator.

        Returns
        -------
        None.

        '''
        # Calibrators that could be loaded, but that are research-grade only and
        # should not be offered through the public interface.
        #
        # TODO: This should be removed as part of refactoring the calibrator
        # portion of the public interface.
        RESEARCH_CALS = ["mlbayeswoodbury", "simulationpost"]
        if method.lower() in RESEARCH_CALS:
            if ("expertMode" not in args.keys()) or (not args["expertMode"]):
                msg = "{} is included for unofficial research purposes only"
                raise ValueError(msg.format(method))
            else:
                # With the current implementation, expertMode=True could be
                # added to the calibration arguments with the intent of using a
                # research-grade calibrator but unintentionally allowing the use
                # of a research-grade sampler (or vice versa).  Refactorings
                # will hopefully avoid this ambiguity.
                #
                # Emit warning to extend a helping hand to the experts.
                msg = f"Using unofficial research {method} calibrator"
                warnings.warn(msg)

        # cast to numpy.float64, currently only for theta and f.
        if y is not None:
            y = cast_f64_dtype(y)
        if yvar is not None:
            yvar = cast_f64_dtype(yvar)

        # default to showing all warnings
        if ('warnings' in args.keys()) and ~args['warnings']:
            warnings.simplefilter('ignore')
        else:
            warnings.resetwarnings()

        self.args = args
        if y is None:
            raise ValueError('You have not provided any y.')
        if y.ndim > 1.5:
            y = np.squeeze(y)
        if y.shape[0] < 5:
            raise ValueError('5 is the minimum number of observations at this '
                             'time.')
        self.y = y
        if emu is None:
            raise ValueError('You have not provided any emulator.')
        self.emu = emu

        try:
            thetatestsamp = thetaprior.rnd(100)
        except Exception:
            raise ValueError('thetaprior.rnd(100) failed.')

        if thetatestsamp.shape[0] != 100:
            raise ValueError('thetaprior.rnd(100) failed to give 100 values.')

        try:
            thetatestlpdf = thetaprior.lpdf(thetatestsamp)
        except Exception:
            raise ValueError('thetaprior.lpdf(thetatestsamp) failed.')

        if thetatestlpdf.shape[0] != 100:
            raise ValueError('thetaprior.lpdf(thetaprior.rnd(100)) failed to '
                             'give 100 values.')
        # if thetatestlpdf.ndim != 1:
        #    raise ValueError('thetaprior.lpdf(thetaprior.rnd(100)) has '
        #                     'dimension higher than 1.')

        self.info = {}
        self.info['thetaprior'] = copy.deepcopy(thetaprior)

        if x is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError('If x is provided, shape[0] must align with '
                                 'the length of y.')
        self.x = copy.deepcopy(x)
        predtry = emu.predict(copy.copy(self.x), thetatestsamp)
        if y.shape[0] != predtry().shape[0]:
            if x is None:
                raise ValueError('y and emu.predict(theta) must have the same '
                                 'shape')
            else:
                raise ValueError('y and emu.predict(x,theta) must have the '
                                 'same shape')
        else:
            prednotfinite = np.logical_not(np.isfinite(predtry()))
            if np.any(prednotfinite):
                warnings.warn('Some non-finite values from emulation '
                              'received.')
                fracfail = np.mean(prednotfinite, 1)
                if np.sum(fracfail <= 10**(-3)) < 5:
                    raise ValueError('Your emulator failed enough places to '
                                     'give up.')
                else:
                    warnings.warn('Current protocol is to remove observations'
                                  ' that have nonfinite values.')
                    whichrm = np.where(fracfail > 10**(-3))[0]
                    warnings.warn('Removing values at %s.'
                                  % np.array2string(whichrm))
                    whichkeep = np.where(fracfail <= 10**(-3))[0]
                    if x is not None:
                        self.x = self.x[whichkeep, :]
                    self.y = self.y[whichkeep]
            else:
                whichkeep = None
        if yvar is not None:
            if yvar.shape[0] != y.shape[0] and yvar.shape[0] > 1.5:
                raise ValueError('yvar must be the same size as y or '
                                 'of size 1.')
            if np.min(yvar) < 0:
                raise ValueError('yvar has at least one negative value.')
            if np.min(yvar) < 10 ** (-6) or np.max(yvar) > 10 ** (6):
                raise ValueError('Rescale your problem so that the yvar'
                                 ' is between 10 ^ -6 and 10 ^ 6.')
            self.info['yvar'] = copy.deepcopy(yvar)
            if whichkeep is not None:
                self.info['yvar'] = self.info['yvar'][whichkeep]

        try:
            self.method = importlib.import_module('surmise.calibrationmethods.'
                                                  + method)
        except Exception:
            raise ValueError('Module not found!')

        self.fit()

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                         if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('__')]
        object_method = [x for x in object_method if not x.startswith('emu')]
        strrepr = ('A calibration object where the code in located in the file'
                   ' calibration. The main method are cal.' +
                   ', cal.'. join(object_method) + '. Default of cal(x) is '
                   'cal.predict(x). Run help(cal) for the document string.')
        return strrepr

    def __call__(self, x=None):
        return self.predict(x)

    def fit(self, args=None):
        """
        Calls "calibrationmethods.[method].fit" where "[method]" is the user
        option.

        Parameters
        ----------
        args : dict
            A dictionary containing options you would like to pass
        """

        if args is not None:
            argstemp = {**self.args, **copy.deepcopy(args)}
        else:
            argstemp = copy.copy(self.args)

        self.method.fit(self.info, self.emu, self.x, self.y, **argstemp)
        if hasattr(self, 'theta'):
            del self.theta
        # : theta attribute of calibrator
        self.theta = thetadist(self)
        return

    def predict(self, x=None, args=None):
        '''
        Returns predictions at x.

        :Example:

            .. code-block:: python

              calibrator.predict(x=x, args=args)

        Parameters
        ----------
        x : numpy.ndarray, optional
            An array of inputs to the model where to predict. The default is
            None.
        args : dict, optional
            A dictionary containing options. The default is None.

        Returns
        -------
        surmise.calibration.prediction
            An instance of calibration class prediction.

        '''

        if args is None:
            args = self.args
        if x is None:
            x = self.x
        info = {}
        if 'predict' in dir(self.method):
            self.method.predict(info, self.info, self.emu, x, args)
        else:
            nsamp = 1000
            emupred = self.emu.predict(x, self.theta.rnd(nsamp))
            info['mean'] = np.mean(emupred.mean(), 1)
            info['var'] = np.var(emupred.mean(), 1)
            info['rnd'] = (emupred.mean()).T

        predobj = prediction(info, self)
        predobj.empirical_coverage()
        return predobj

    def save_to(self, filename):
        """
        Simple serialization and save function for calibrator object.

        :Example:

        .. code-block:: python

            cal = calibrator(...)

            cal.save_to('cal_example.pkl')

            loaded_cal = calibrator.load_from('cal_example.pkl')

        """
        save_file(self, filename)
        return

    @staticmethod
    def load_from(filename):
        cal = load_file(filename, 'surmise.calibration.calibrator')
        return cal

    @staticmethod
    def load_prediction(filename):
        pred = load_file(filename, 'surmise.calibration.prediction')
        return pred


class prediction(object):
    '''
    A class to represent a calibration prediction.
    predict.info will give the dictionary from the method.

    :Example:

        .. code-block:: python

            prediction.lpdf()

            prediction.mean()

            prediction.var()

            prediction.rnd()
    '''

    def __init__(self, info, cal):
        self.info = info
        self.cal = cal

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                         if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('cal')]
        strrepr = ('A calibration prediction object predict where the code in'
                   ' located in the file calibration. The main method are'
                   ' predict.' +
                   ', predict.'.join(object_method) + '. Default of predict() '
                   'is predict.mean() and ' +
                   'predict(s) will run predict.rnd(s). '
                   'Run help(predict) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)

    def __methodnotfoundstr(self, pfstr, opstr):
        warnings.warn(pfstr + opstr + ' functionality not in method... \n' +
                      ' Key labeled ' + opstr + ' not ' +
                      'provided in ' + pfstr + '.info... \n' +
                      ' Key labeled rnd not ' +
                      'provided in ' + pfstr + '.info...')
        return 'Could not reconcile a good way to compute this value'\
               ' in current method.'

    def mean(self, args=None):
        """
        Returns the mean at all x in when building the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'mean'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictmean(self.info,
                                                             args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.mean(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args=None):
        """
        Returns the variance at all x in when building the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'var'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictvar(self.info,
                                                            args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.var(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def rnd(self, s=100, args=None):
        """
        Returns s random draws at all x in when building the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'rnd'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictrnd(self.info,
                                                            args))
        elif 'rnd' in self.info.keys():
            return self.info['rnd'][np.random.choice(self.info['rnd'].shape[0],
                                                     size=s), :]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def lpdf(self, y=None, args=None):
        """
        Returns a log pdf given theta.
        """
        raise ValueError('lpdf functionality not in method')

    def save_to(self, filename):
        """
        Simple serialization and save function for calibrator prediction object.

        :Example:

        .. code-block:: python

            cal = calibrator(...)

            calpred = cal.predict(...)

            calpred.save_to('calpred_example.pkl')

            loaded_calpred = calibrator.load_prediction('calpred_example.pkl')

        """
        save_file(self, filename)
        return

    def empirical_coverage(self, p=np.array((0.68, 0.9, 0.95, 0.99))):
        """
        Computes empirical coverage given predictions using samples collected from calibration.
        """
        y = self.cal.info['y']
        ypred = self.info['rnd']

        ylowers = np.quantile(ypred, q=(1-p)/2, axis=0)
        yuppers = np.quantile(ypred, q=(1+p)/2, axis=0)

        coverage = None
        try:
            coverage = np.mean(np.logical_and(y <= yuppers, y >= ylowers), axis=1)
        except ValueError:
            warnings.warn('Use the same \'x\' for prediction to compute empirical coverages.')
        self.info['coverage'] = (p, coverage)
        return


class thetadist(object):
    """
    A class to represent a theta predictive distribution.
    """

    def __init__(self, cal):
        self.cal = cal

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                         if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('cal')]
        strrepr = ('A theta distribution object where the code in located in'
                   ' the file calibration. The main method are cal.theta' +
                   ', cal.theta.'.join(object_method) + '. Default of '
                   'predict() is' +
                   ' cal.theta.mean() and ' +
                   'cal.theta(s) will cal.theta.rnd(s).'
                   ' Run help(cal.theta) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)

    def __methodnotfoundstr(self, pfstr, opstr):
        warnings.warn(pfstr + opstr + 'functionality not in method... \n' +
                      ' Key labeled ' + (pfstr+opstr) + ' not ' +
                      'provided in cal.info... \n' +
                      ' Key labeled ' + pfstr + 'rnd not ' +
                      'provided in cal.info...')
        return 'Could not reconcile a good way to compute this value in'\
               ' current method.'

    def mean(self, args=None):
        """
        Returns mean of each element of theta found during calibration.
        """

        pfstr = 'theta'  # prefix string
        opstr = 'mean'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.thetamean(self.cal.info,
                                                           args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            return np.mean(self.cal.info[(pfstr+'rnd')], 0)
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args=None):
        """
        Returns predictive variance of each element of theta found
        during calibration.
        """

        pfstr = 'theta'  # prefix string
        opstr = 'var'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.thetavar(self.cal.info,
                                                          args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            return np.var(self.cal.info[(pfstr+'rnd')], 0)
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def rnd(self, s=1000, args=None):
        """
        Returns s predictive draws for theta found during calibration.
        """

        pfstr = 'theta'  # prefix string
        opstr = 'rnd'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.copy(self.cal.method.thetarnd(self.cal.info, s,
                                                      args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return self.cal.info['thetarnd'][
                        np.random.choice(self.cal.info['thetarnd'].shape[0],
                                         size=s), :]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def lpdf(self, theta=None, args=None):
        """
        Returns a log pdf given theta.
        """

        pfstr = 'theta'  # prefix string
        opstr = 'lpdf'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.copy(self.cal.method.thetalpdf(self.cal.info, theta,
                                                       args))
        else:
            raise ValueError('lpdf functionality not in method')

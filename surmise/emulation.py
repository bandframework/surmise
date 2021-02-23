"""
This module contains a class that implements the main emulation method.
"""
import numpy as np
import importlib
import copy
import warnings


class emulator(object):

    def __init__(self,
                 x=None,
                 theta=None,
                 f=None,
                 method='PCGP',
                 passthroughfunc=None,
                 args={},
                 options={}):
        '''
        A class used to represent an emulator or surrogate model. Fits an
        emulator or surrogate model provided in
        ``emulationmethods/[method].py`` where [method] is the user option
        with default listed above.

        .. tip::
            To use a new emulator, just drop a new file to the
            ``emulationmethods/`` directory with the required formatting.

        :Example:
            .. code-block:: python

               emulator(x=x, theta=theta, f=f, method='PCGP', args=args)

        Parameters
        ----------
        x : numpy.ndarray, optional
            An array of inputs. Each row should correspond to a row in f.
            The default is None.
            We will attempt to resolve size differences.

        theta : numpy.ndarray, optional
            An array of parameters. Each row in theta should correspond to a
            column in f. The default is None.
            We will attempt to resolve size differences.

        f : numpy.ndarray, optional
            An array of responses with 'nan' representing responses not yet
            available. The default is None.
            Each column in f should correspond to a row in x.
            Each row should correspond to a row in f.
            We will attempt to resolve if these are flipped.

        method : str, optional
            A string that points to the file located in ``emulationmethods/``.
            The default is ``PCGP``.

        passthroughfunc : function, optional
            DESCRIPTION. The default is None.

        args : dict, optional
            Optional dictionary containing options you would like to pass to
            [method].fit(x, theta, f, args)
            or
            [method].predict(x, theta args) The default is {}.

        options : dict, optional
            Dictionary containing options you would like
            emulation to have. This does not get passed to the method.
            The default is {}.

        Returns
        -------
        None.

        '''

        if ('warnings' in args.keys()) and args['warnings']:
            warnings.resetwarnings()
        else:
            warnings.simplefilter('ignore')

        self.__ptf = passthroughfunc
        if self.__ptf is not None:
            return
        self._args = copy.deepcopy(args)

        if f is not None:
            if f.ndim < 1 or f.ndim > 2:
                raise ValueError('f must have either 1 or 2 dimensions.')
            if (x is None) and (theta is None):
                raise ValueError('You have not provided any theta or x, no'
                                 ' emulator inference possible.')
            if x is not None:
                if x.ndim < 0.5 or x.ndim > 2.5:
                    raise ValueError('x must have either 1 or 2 dimensions.')
            if theta is not None:
                if theta.ndim < 0.5 or theta.ndim > 2.5:
                    raise ValueError('theta must have either 1 or 2'
                                     ' dimensions.')
        else:
            raise ValueError('You have not provided f, cannot include theta'
                             ' or x.')

        if x is not None and (f.shape[0] != x.shape[0]):
            if theta is not None:
                if (f.ndim == 2 and f.shape[1] == x.shape[0] and
                        f.shape[0] == theta.shape[0]):
                    warnings.warn('Transposing f to try to get agreement')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the'
                                     ' number of rows in x.')
            else:
                if f.ndim == 2 and f.shape[1] == x.shape[0]:
                    warnings.warn('Transposing f to try to get agreement')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the'
                                     ' number of rows in x.')

        if theta is not None and (f.shape[1] != theta.shape[0]):
            if x is not None:
                if not (f.ndim == 2 and f.shape[0] == theta.shape[0] and
                        f.shape[1] == x.shape[0]):
                    raise ValueError('The number of columns in f must match'
                                     ' the number of rows in theta.')
            else:
                if f.ndim == 2 and f.shape[0] == theta.shape[0]:
                    warnings.warn('Transposing f to try to get agreement')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                elif f.ndim == 1 and f.shape[0] == theta.shape[0]:
                    warnings.warn('Transposing f to try to get agreement')
                    self.__f = np.reshape(copy.copy(f), (1, -1))
                    f = np.reshape(copy.copy(f), (1, -1))
                else:
                    raise ValueError('The number of columns in f must match'
                                     ' the number of rows in theta.')

        if x is not None:
            self.__x = copy.copy(x)
        else:
            self.__x = None

        if theta is not None:
            self.__theta = copy.copy(theta)
        else:
            self.__theta = None
            raise ValueError('This feature has not developed yet.')

        self.__suppx = None
        self.__supptheta = None
        self.__f = copy.copy(f)

        try:
            self.method = \
                importlib.import_module('surmise.emulationmethods.' + method)
        except Exception:
            raise ValueError('Module not loaded correctly.')
        if "fit" not in dir(self.method):
            raise ValueError('Function fit not found in module!')
        if "predict" not in dir(self.method):
            raise ValueError('Function predict not found in module!')
        if "supplementtheta" not in dir(self.method):
            warnings.warn('Function supplementtheta not found in module!')

        self.__options = {}
        self.__optionsset(options)
        self._info = {}
        self._info = {'method': method}

        if (self.__f is not None) and (self.__options['autofit']):
            self.fit()

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                         if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('__')]
        strrepr = ('An emulation object where the code in located in the file '
                   + 'emulation. The main method are emu.' +
                   ', emu.'. join(object_method) + '. Default of emu(x,theta)'
                   ' is emu.predict(x,theta). '
                   'Run help(emu) for the document string.')
        return strrepr

    def __call__(self, x=None, theta=None, args=None):
        return self.predict(x, theta, args)

    def fit(self, args=None):
        """
        Fits an emulator or surrogate and places that in emu._info

        Calls
        emu._info = [method].fit(emu.__theta, emu.__f, emu.__x, args = args)

        Parameters
        ----------
        args : dict
            Optional dictionary containing options you would like to pass to
            fit function. It will add/modify those in self._args.
        """

        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)}
        else:
            argstemp = copy.copy(self._args)
        x, theta, f = self.__preprocess()
        self.method.fit(self._info, x, theta, f, args=argstemp)

    def predict(self, x=None, theta=None, args={}):
        '''
        Fits an emulator or surrogate.

        :Example:
            .. code-block:: python

              emulator.predict(x=x, theta=theta, args=args)

        Parameters
        ----------
        x : numpy.ndarray, optional
            An array of inputs. Each row in x should correspond to a row in f.
            The default is None.

        theta : numpy.ndarray, optional
            An array of parameters. Each row should correspond to a
            column in f. The default is None.

        args : dict, optional
            A dictionary containing args. The default is {}.

        Raises
        ------
        ValueError
            If the dimensions of inputs do not match with the fitted
            emulator.

        Returns
        -------
        surmise.emulation.prediction
            An instance of emulation class prediction

        '''

        if self.__ptf is not None:
            info = {}
            if theta is not None:
                info['mean'] = self.__ptf(x, theta)
            else:
                info['mean'] = self.__ptf(x, self.__theta)
            info['var'] = 0 * info['mean']
            info['covxhalf'] = 0 * np.stack((info['mean'], info['mean']), 2)
            return prediction(info, self)
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)}
        else:
            argstemp = copy.copy(self._args)
        if x is None:
            x = copy.copy(self.__x)
        else:
            x = copy.copy(x)
            if x.ndim == 1:
                if self.__x.ndim == 2 and x.shape[0] == self.__x.shape[1]:
                    x = np.reshape(x, (1, -1))
                elif self.__x.ndim == 2:
                    raise ValueError('Your x shape seems to not agree with the'
                                     ' emulator build.')
            elif x.ndim == 2:
                if self.__x.ndim == 1:
                    raise ValueError('Your x shape seems to not agree with the'
                                     ' emulator build.')
                elif x.shape[1] != self.__x.shape[1] and \
                        x.shape[0] == self.__x.shape[1]:
                    x = x.T
                elif x.shape[1] != self.__x.shape[1] and \
                        x.shape[0] != self.__x.shape[1]:
                    raise ValueError('Your x shape seems to not agree with the'
                                     ' emulator build.')
        if theta is None:
            theta = copy.copy(self.__theta)
        else:
            theta = copy.copy(theta)
            if theta.ndim == 2 and self.__theta.ndim == 1:
                raise ValueError('Your theta shape seems to not agree with the'
                                 ' emulator build.')
            # note: dont understand why we have this statement
            elif theta.ndim == 1 and self.__theta.ndim == 2 and \
                    theta.shape[0] == self.__theta.shape[1]:
                theta = np.reshape(theta, (1, -1))
            elif theta.ndim == 1 and self.__theta.ndim == 2:
                raise ValueError('Your theta shape seems to not agree with the'
                                 ' emulator build.')
            elif theta.shape[1] != self.__theta.shape[1] and \
                    theta.shape[0] == self.__theta.shape[1]:
                theta = theta.T
            elif theta.shape[1] != self.__theta.shape[1] and \
                    theta.shape[0] != self.__theta.shape[1]:
                raise ValueError('Your theta shape seems to not agree with the'
                                 ' emulator build.')

        info = {}
        self.method.predict(info, self._info, x, theta, args=argstemp)
        return prediction(info, self)

    def supplement(self,
                   size,
                   x=None,
                   xchoices=None,
                   theta=None,
                   thetachoices=None,
                   choicescost=None,
                   cal=None,
                   args=None,
                   overwrite=False,
                   removereps=None):
        '''
        Chooses new theta or x to be investigated.

        .. important::
            A user must provide either x or theta (or cal).

        :Example:
            .. code-block:: python

              emulator.supplement(size=size, theta=theta)

        Parameters
        ----------
        size : int
            The number of new supplements to return.

            .. note::
                - If only theta is supplied, returns at most size of those.
                - If only x is supplied, returns at most size of those.
                - If both x and theta are supplied, then size will be less
                  than the product of the number of returned theta and the
                  number of x.

        x : numpy.ndarray, optional
            An array of parameters where to predict.
            The default is None.
        xchoices : numpy.ndarray, optional
            An  array of inputs to select from.
            If not provided, a subset of x is used. The default is None.

            .. warning:: self.__suppx has not developed yet.

        theta : numpy.ndarray, optional
             An array of parameters where to predict. The default is None.
        thetachoices : numpy.ndarray, optional
            An  array of parameters to select from.
            If not provided, a subset of x is used. The default is None.
        choicescost : numpy.ndarray, optional
            An array of positive cost of each element in choice.
            The default is None.
        cal : surmise.calibration.calibrator, optional
            A calibrator object that contains information about calibration.
            The default is None.
        args : dict, optional
            A dictionary containing options to pass to the method.
            The default is None. If not provided, defaults to the one used to
            build the emulator.
        overwrite : boolean, optional
            True if an existing supplement is replaced. If False, and one
            exists, returns without doing anything. The default is False.
        removereps : boolean, optional
            True if any replications existing supplement is removed.
            The default is None. If not provided, defaults to the one used to
            build the emulator.

        Raises
        ------
        ValueError
            If the dimensions do not match the fitted emulator

        Returns
        -------
        numpy.ndarray
            self.__thetasupp (or self.__x)
        numpy.ndarray
           suppinfo

        '''

        if args is not None:
            argstemp = {**self._args, **args}
        else:
            argstemp = self._args

        if removereps is None:
            if x is not None:
                removereps = not self.__options['xreps']
            if theta is not None:
                removereps = not self.__options['thetareps']

        if size < 1:
            if size == 0:
                if self.__supptheta is None:
                    raise ValueError('No self.__supptheta exists.')
                else:
                    print('Returning self.__supptheta.')
                    return copy.deepcopy(self.__supptheta)
            else:
                raise ValueError('Size should be a positive integer.')

        if cal is not None:
            try:
                if theta is None:
                    theta = cal.theta(1000)
                    if self.__theta.shape[1] != theta.shape[1]:
                        raise ValueError('cal.theta(n) produces the wrong '
                                         'shape.')
            except Exception:
                raise ValueError('cal.theta(2000) failed.')

        if (theta is None) and (x is None):
            raise ValueError('Provide either x or (theta or cal).')
        else:
            if (x is not None):
                if (theta is not None):
                    raise ValueError('Provide either x or (theta or cal).')
                else:
                    raise ValueError('Supplement x has not supported yet.')
            elif (theta is not None):
                if self.__theta.shape[1] == theta.shape[1]:
                    if thetachoices is None:
                        if theta.shape[0] > 30 * size:
                            thetachoices =\
                                theta[np.random.choice(theta.shape[0],
                                                       30 * size,
                                                       replace=False), :]
                        else:
                            thetachoices = copy.copy(theta)
                    else:
                        if thetachoices.shape[1] != theta.shape[1]:
                            raise ValueError('Dimensions of choices and '
                                             'predictions are not aligning.')
                    if choicescost is None:
                        choicescost = np.ones(thetachoices.shape[0])
                    else:
                        if thetachoices.shape[0] != choicescost.shape[0]:
                            raise ValueError('choicecost is not the right '
                                             'shape.')
                else:
                    raise ValueError('theta has the wrong shape, it does not '
                                     'match emu._emulator__theta.')

        try:
            supptheta, suppinfo = \
                self.method.supplementtheta(self._info,
                                            copy.copy(size),
                                            copy.copy(theta),
                                            copy.copy(thetachoices),
                                            copy.copy(choicescost),
                                            copy.copy(cal),
                                            argstemp)
        except Exception:
            raise ValueError('supplementtheta does not exist.')

        if supptheta is not None:
            if removereps:
                nctheta, ctheta, rtheta = _matrixmatching(self.__theta,
                                                          supptheta)
                if nctheta.shape[0] < 0.5:
                    supptheta = None
                    raise ValueError('supptheta is a complete replication of '
                          'self.__theta.')
                else:
                    if nctheta.shape[0] < supptheta.shape[0]:
                        print('Removing replications from supptheta.')
                        supptheta = supptheta[nctheta, :]

            if (self.__supptheta is not None) and (not overwrite):
                raise ValueError('Either evaluate self.__supptheta or '
                                 'select overwrite = True.')
            else:
                self.__supptheta = copy.copy(supptheta)
                return copy.copy(self.__supptheta), suppinfo
        else:
            raise ValueError('method.supplementtheta provides None.')

    def update(self, x=None, theta=None, f=None, args=None, options=None):
        '''
        Updates and refits the emulator.

        :Example:
            .. code-block:: python

                emulator.update(x=x)  # Replace self.__x with x

                emulator.update(theta=theta)  # Replace self.__theta with theta

                emulator.update(f=f)  # Replace self.__f with f if
                                      # self.__supptheta is None. Otherwise,
                                      # update self.__theta
                                      # with (self.__theta, self.__supptheta)
                                      # and self.__f with (self.__f, f)

                emulator.update(x=x, f=f)  # Update self.__x with (self.__x, x)
                                           # and self.__f with (self.__f, f)

                emulator.update(theta=theta, f=f)  # Update self.__theta with
                                                 # (self.__theta, theta) and
                                                 # self.__f with (self.__f, f)

        .. warning:: self.__suppx has not developed yet.

        Parameters
        ----------
        x : numpy.ndarray, optional
            xs you would like to append. The default is None.

        theta : numpy.ndarray, optional
            thetas you would like to append. Defaults to emu.__supptheta.
            The default is None.

        f : numpy.ndarray, optional
            An array of responses. The default is None.

        args : dict, optional
            A dictionary containing options you would like to pass to
            [method].update(f, theta, x, args).
            Defaults to the one used to build the emulator.

        options : dict, optional
            A dictionary containing options to build the emulator.
            Modify with update when you want to change it.
            The default is None.

        Raises
        ------
        ValueError
            If the dimensions of inputs do not match with the existing
            emulator.

        Returns
        -------
        None.

        '''

        if options is not None:
            self.__optionsset(copy.copy(options))
        if args is not None:
            self._args = {**self._args, **copy.deepcopy(args)}

        if (theta is not None) and (x is not None):
            # provide either x or theta for now
            raise ValueError('Adding new theta and x at once is currently not '
                             'supported. Supply either theta or x.')
        elif (f is not None):
            if (theta is None) and (x is None):
                if (f.shape[0] == self.__f.shape[0]):
                    # no of rows of f (and x) is still the same
                    if self.__supptheta is not None:
                        if f.shape[1] == self.__supptheta.shape[0]:
                            if self.__options['thetareps']:
                                # update with self.__supptheta and f
                                self.__theta = np.vstack((self.__theta,
                                                          self.__supptheta))
                                self.__f = np.hstack((self.__f, f))
                                self.__supptheta = None
                            else:
                                # identify matches
                                nc, c, r = _matrixmatching(self.__theta,
                                                           self.__supptheta)
                                # update __f with f for the matches
                                self.__f[:, r] = f[:, c]
                                if nc.shape[0] > 0.5:
                                    f = f[:, nc]
                                    supptheta = self.__supptheta[nc, :]
                                    self.__f = np.hstack((self.__f, f))
                                    self.__theta = np.vstack((self.__theta,
                                                              supptheta))
                                    self.__supptheta = None
                        else:
                            raise ValueError('Could not resolve absense of '
                                             ' theta, please provide theta.')
                    elif (f.shape[1] == self.__theta.shape[0]):
                        # just updating f
                        self.__f = f
                    else:
                        raise ValueError('Could not resolve absense of theta, '
                                         'please provide theta')
                else:
                    # no of rows of f (and x) is not the same
                    raise ValueError('Could not resolve absense of x. Provide '
                                     'x.')
            elif (theta is not None):
                if (f.shape[0] == self.__f.shape[0]) and \
                    (f.shape[1] == theta.shape[0]) and \
                        (theta.shape[1] == self.__theta.shape[1]):
                    if self.__options['thetareps']:
                        # if replicated thetas are allowed
                        self.__theta = np.vstack((self.__theta, theta))
                        self.__f = np.hstack((self.__f, f))
                    else:
                        # if replicated thetas are not allowed
                        nc, c, r = _matrixmatching(self.__theta, theta)
                        self.__f[:, r] = f[:, c]
                        if nc.shape[0] > 0.5:
                            # append the unmatched thetas
                            f = f[:, nc]
                            theta = theta[nc, :]
                            self.__f = np.hstack((self.__f, f))
                            self.__theta = np.vstack((self.__theta, theta))
                else:
                    raise ValueError('Check the dimensions of theta and f. '
                                     'Possible solutions: '
                                     '1) Use emu.update(theta=theta) to '
                                     'update the emulator first. '
                                     '2) Provide x for alignment.')

            elif (x is not None):
                if (f.shape[1] == self.__f.shape[1]) and \
                    (f.shape[0] == x.shape[0]) and \
                        (x.shape[1] == self.__x.shape[1]):
                    if options['xreps']:
                        # if replicated xs are allowed
                        self.__x = np.vstack((self.__x, x))
                        self.__f = np.vstack((self.__f, f))
                    else:
                        # if replicated xs are not allowed
                        nc, c, r = _matrixmatching(self.__x, x)
                        self.__f[r, :] = f[c, :]
                        if nc.shape[0] > 0.5:
                            # append the unmatched xs
                            f = f[nc, :]
                            x = x[nc, :]
                            self.__f = np.vstack((self.__f, f))
                            self.__x = np.vstack((self.__x, x))
                else:
                    raise ValueError('Check the dimensions of x and f. '
                                     'Possible solutions: '
                                     '1) Use emu.update(x=x) to '
                                     'update the emulator first. '
                                     '2) Provide theta for alignment.')
        elif (x is not None):  # theta None, f None
            # Update self.__x with x
            if x.shape[0] != self.__f.shape[0]:
                raise ValueError('Number of rows of x is changed but new f is '
                                 'not provided.')
            else:
                self.__x = x
        elif (theta is not None):  # x None, f None
            # Update self.__theta with theta
            if theta.shape[0] != self.__f.shape[1]:
                raise ValueError('Number of rows of theta is changed but new '
                                 'f is not provided.')
            else:
                self.__theta = theta

        if self.__options['autofit']:
            self.fit()
        return

    def remove(self, x=None, theta=None, cal=None, options=None):
        '''
        Removes either x or theta, and the corresponding f values from
        the fitted emulator, and refits the emulator.

        :Example:

            .. code-block:: python

                emlator.remove(theta=theta)

        Parameters
        ----------
        x : numpy.ndarray, optional
            x to remove from self.__x. The default is None.
        theta : numpy.ndarray, optional
            theta to remove from self.__theta. The default is None.
        cal : surmise.calibration.calibrator, optional
            A calibrator class instance as defined in surmise.calibration.
            The default is None.
        options : dict, optional
            A dictionary containing options to build the emulator.
            The default is None.

        Returns
        -------
        None.

        '''

        if cal is not None:
            totalseen = np.where(np.mean(np.logical_not(np.isfinite(self.__f)),
                                         0) < self.__options['thetarmnan'])[0]
            lpdf_ex = cal.theta.lpdf(self.__theta[totalseen, :])
            thetasort = np.argsort(lpdf_ex)
            m_cutoff = max(lpdf_ex.shape[0]-10*self.__theta.shape[1], 0)
            numcutoff = np.minimum(-500, lpdf_ex[thetasort[m_cutoff]])
            if any(lpdf_ex < numcutoff):
                rmtheta = totalseen[np.where(lpdf_ex < numcutoff)[0]]
                theta = self.__theta[rmtheta, :]
                print('removing %d thetas' % rmtheta.shape[0])
        if (theta is not None):
            nc, c, r = _matrixmatching(theta, self.__theta)
            self.__theta = self.__theta[nc, :]
            self.__f = self.__f[:, nc]
            if self.__options['autofit']:
                self.fit()
        return

    def __optionsset(self, options=None):
        options = copy.deepcopy(options)
        # options will always be lowercase
        options = {k.lower(): v for k, v in options.items()}

        if 'thetareps' in options.keys():
            if type(options['thetareps']) is bool:
                self.__options['thetareps'] = options['thetareps']
            else:
                raise ValueError('option thetareps must be true or false')

        if 'xreps' in options.keys():
            if type(options['xreps']) is bool:
                self.__options['xreps'] = options['xreps']
            else:
                raise ValueError('option xreps must be true or false')

        if 'thetarmnan' in options.keys():
            if type(options['thetarmnan']) is bool:
                if options['thetarmnan']:
                    self.__options['thetarmnan'] = 0
                else:
                    self.__options['thetarmnan'] = 1 + (10 ** (-12))
            elif type(options['thetarmnan']) is str:
                if isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'any':
                    self.__options['thetarmnan'] = 0
                elif isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'some':
                    self.__options['thetarmnan'] = 0.2
                elif isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'most':
                    self.__options['thetarmnan'] = 0.5
                elif isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'alot':
                    self.__options['thetarmnan'] = 0.8
                elif isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'all':
                    self.__options['thetarmnan'] = 1 - (10 ** (-8))
                elif isinstance(options['thetarmnan'], str) and \
                        options['thetarmnan'] == 'never':
                    self.__options['thetarmnan'] = 1 + (10 ** (-8))
                else:
                    raise ValueError('option thetarmnan must be True, False,'
                                     ' ''any'', ''some'', ''most'', ''alot'','
                                     ' ''all'', ''never'' or an scaler bigger'
                                     'than zero and less than one.')
            elif np.isfinite(options['thetarmnan']) and \
                    options['thetarmnan'] >= 0 and \
                    options['thetarmnan'] <= 1:
                self.__options['thetarmnan'] = options['thetarmnan']
            else:
                raise ValueError('option thetarmnan must be True, False,'
                                 ' ''any'', ''some'', ''most'', ''alot'','
                                 ' ''all'', ''never'' or an scaler bigger'
                                 'than zero and less than one.')
        if 'xrmnan' in options.keys():
            if type(options['xrmnan']) is bool:
                if options['xrmnan']:
                    self.__options['xrmnan'] = 0
                else:
                    self.__options['xrmnan'] = 1 + (10 ** (-12))
            elif type(options['xrmnan']) is str:
                if isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'any':
                    self.__options['xrmnan'] = 0
                elif isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'some':
                    self.__options['xrmnan'] = 0.2
                elif isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'most':
                    self.__options['xrmnan'] = 0.5
                elif isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'alot':
                    self.__options['xrmnan'] = 0.8
                elif isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'all':
                    self.__options['xrmnan'] = 1 - (10 ** (-8))
                elif isinstance(options['xrmnan'], str) and \
                        options['xrmnan'] == 'never':
                    self.__options['xrmnan'] = 1 + (10 ** (-8))
                else:
                    raise ValueError('option xrmnan must be True, False,'
                                     ' ''any'', ''some'', ''most'', ''alot'','
                                     ' ''all'', ''never'' or an scaler bigger'
                                     'than zero and less than one.')
            elif np.isfinite(options['xrmnan']) and options['xrmnan'] >= 0 \
                    and options['xrmnan'] <= 1:
                self.__options['xrmnan'] = options['xrmnan']
            else:
                raise ValueError('option xrmnan must be True, False,'
                                 ' ''any'', ''some'', ''most'', ''alot'','
                                 ' ''all'', ''never'' or an scaler bigger'
                                 'than zero and less than one.')

        if 'rmthetafirst' in options.keys():
            if type(options['rmthetafirst']) is bool:
                self.__options['rmthetafirst'] = options['rmthetafirst']
            else:
                raise ValueError('option rmthetafirst must be True or False.')

        if 'autofit' in options.keys():
            if type(options['autofit']) is bool:
                self.__options['minsampsize'] = options['autofit']
            else:
                raise ValueError('option autofit must be of type bool.')

        if 'thetareps' not in self.__options.keys():
            self.__options['thetareps'] = False
        if 'xreps' not in self.__options.keys():
            self.__options['xreps'] = False
        if 'thetarmnan' not in self.__options.keys():
            self.__options['thetarmnan'] = 0.8
        if 'xrmnan' not in self.__options.keys():
            self.__options['xrmnan'] = 0.8
        if 'autofit' not in self.__options.keys():
            self.__options['autofit'] = True
        if 'rmthetafirst' not in self.__options.keys():
            self.__options['rmthetafirst'] = True

    def __preprocess(self):
        x = copy.copy(self.__x)
        theta = copy.copy(self.__theta)
        f = copy.copy(self.__f)
        options = self.__options
        isinff = np.isinf(f)
        if np.any(isinff):
            print('All infs were converted to nans.')
            f[isinff] = float("NaN")
        isnanf = np.isnan(f)

        # first, check missing thetas
        if self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < options['thetarmnan'])[0]
            f = f[:, j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j, :]
        # then, check missing xs
        j = np.where(np.mean(isnanf, 1) < options['xrmnan'])[0]
        f = f[j, :]
        if x is not None:
            if x.ndim == 1:
                x = x[j]
            else:
                x = x[j, :]

        if not self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < options['thetarmnan'])[0]
            f = f[:, j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j, :]
        return x, theta, f


class prediction(object):
    '''
    A class to represent an emulation prediction. predict._info returns the
    dictionary from the method.

        :Example:

            .. code-block:: python

                prediction.mean()

                prediction.var()

                prediction.covx()

                prediction.rnd()
    '''

    def __init__(self, _info, emu):
        self._info = _info
        self.emu = emu

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                         if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('emu')]
        strrepr = ('A emulation prediction object predict where the code in'
                   ' located in the file '
                   + ' emulation.  The main method are predict.' +
                   ', predict.'.join(object_method) + '.  Default of predict()'
                   ' is predict.mean() and ' +
                   'predict(s) will run pred.rnd(s).'
                   ' Run help(predict) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)

    def __methodnotfoundstr(self, pfstr, opstr):
        msg = (pfstr + opstr + ' functionality not in method... \n' +
               ' Key labeled ' + opstr + ' not ' +
               'provided in ' + pfstr + '._info... \n' +
               ' Key labeled rnd not ' +
               'provided in ' + pfstr + '._info...')

        return msg

    def mean(self, args=None):
        '''
        Returns the mean at theta and x in when building the prediction.
        '''

        pfstr = 'predict'  # prefix string
        opstr = 'mean'  # operation string
        if (self.emu._emulator__ptf is None) and \
                ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictmean(self._info, args))
        elif opstr in self._info.keys():
            return self._info[opstr]
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.mean(self._info['rnd'], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def mean_gradtheta(self, args=None):
        """
        Returns the gradient of the mean at theta and x with respect to theta
        when building the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'mean_gradtheta'  # operation string
        if opstr in self._info.keys():
            return self._info[opstr]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args=None):
        """
        Returns the pointwise variance at theta and x when building
        the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'var'  # operation string
        if (self.emu._emulator__ptf is None) and \
                ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictvar(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.var(self._info['rnd'], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def covx(self, args=None):
        """
        Returns the covariance matrix at theta and x when building
        the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'covx'  # operation string
        if (self.emu._emulator__ptf is None) and \
                ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'covxhalf' in self._info.keys():
            if self._info['covxhalf'].ndim == 2:
                return self._info['covxhalf'] @ self._info['covxhalf'].T
            else:
                am = self._info['covxhalf'].shape
                covx = np.ones((am[0], am[1], am[0]))
                for k in range(0, self._info['covxhalf'].shape[1]):
                    A = self._info['covxhalf'][:, k, :]
                    covx[:, k, :] = A @ A.T
            self._info['covx'] = covx
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def covxhalf(self, args=None):
        """
        Returns the sqrt of the covariance matrix at theta and x when building
        the prediction.
        That is, if this returns A = predict.covhalf(.)[k],
        then A.T @ A = predict.cov(.)[k]
        """

        pfstr = 'predict'  # prefix string
        opstr = 'covxhalf'  # operation string
        if (self.emu._emulator__ptf is None) and \
                ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'covx' in self._info.keys():
            covxhalf = np.ones(self._info['covx'].shape)
            if self._info['covx'].ndim == 2:
                W, V = np.linalg.eigh(self._info['covx'])
                covxhalf = (V @ (np.sqrt(np.abs(W)) * V.T))
            else:
                for k in range(0, self._info['covx'].shape[0]):
                    W, V = np.linalg.eigh(self._info['covx'][k])
                    covxhalf[k, :, :] = (V @ (np.sqrt(np.abs(W)) * V.T))
            self._info['covxhalf'] = covxhalf
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def covxhalf_gradtheta(self, args=None):
        """
        Returns the gradient of the covxhalf matrix at theta and x when
        building the prediction.
        """

        pfstr = 'predict'  # prefix string
        opstr = 'covxhalf_gradtheta'  # operation string
        if opstr in self._info.keys():
            return self._info[opstr]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def rnd(self, s=100, args=None):
        """
        Returns a rnd draws of size s at theta and x
        """
        raise ValueError('rnd functionality not in method')

    def lpdf(self, f=None, args=None):
        """
        Returns a log pdf at theta and x
        """
        raise ValueError('lpdf functionality not in method')


def _matrixmatching(mat1, mat2):
    """
    This is an internal function to do matching between two vectors
    """
    # This is an internal function to do matching between two vectors
    # it just came up alot
    # It returns the where each row of mat2 is first found in mat1
    # If a row of mat2 is never found in mat1, then 'nan' is in that location

    if (mat1.shape[0] > (10 ** (4))) or (mat2.shape[0] > (10 ** (4))):
        raise ValueError('too many matchings attempted.'
                         'Don''t make the method work so hard!')
    if mat1.ndim != mat2.ndim:
        raise ValueError('Somehow sent non-matching information to'
                         ' _matrixmatching')
    if mat1.ndim == 1:
        matchingmatrix = np.isclose(mat1[:, None].astype('float'),
                                    mat2.astype('float'))
    else:
        matchingmatrix = np.isclose(mat1[:, 0][:, None].astype('float'),
                                    mat2[:, 0].astype('float'))
        for k in range(1, mat2.shape[1]):
            try:
                matchingmatrix *= \
                    np.isclose(mat1[:, k][:, None].astype('float'),
                               mat2[:, k].astype('float'))
            except Exception:
                matchingmatrix *= np.equal(mat1[:, k], mat2[:, k])
    r, c = np.where(matchingmatrix.cumsum(axis=0).cumsum(axis=0) == 1)

    nc = np.array(list(set(range(0, mat2.shape[0])) - set(c))).astype('int')
    return nc, c, r

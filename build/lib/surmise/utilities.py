import importlib


class sampler(object):

    def __init__(self, logpostfunc, options={}):
        '''
        A class used to represent a sampler.

        .. tip::
            To use a new sampler, just drop a new file to the
            ``utilitiesmethods/`` directory with the required formatting.

        Parameters
        ----------
        logpostfunc : function
             A function call describing the log of the posterior distribution.
        options : dict, optional
            Dictionary containing options to be passed to the sampler.
            The default is {}.

        '''

        try:
            method = options['sampler']
        except Exception:
            method = 'metropolis_hastings'

        self.logpostfunc = logpostfunc
        self.options = options
        self.sampler_info = {}
        self.draw_samples(method)

    def draw_samples(self, method):
        '''
        Calls "utilitiesmethods.[method].sampler" where [method] is the
        user option.

        Parameters
        ----------
        method : str
            name of the sampler.

        Returns
        -------
        None.

        '''
        self.method = importlib.import_module('surmise.utilitiesmethods.'
                                              + method)
        # update sampler_info with the output of the sampler
        self.sampler_info = self.method.sampler(self.logpostfunc, self.options)

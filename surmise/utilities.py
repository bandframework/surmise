import importlib


class sampler(object):

    def __init__(self,
                 logpost_func,
                 draw_func,
                 sampler='metropolis_hastings',
                 **sampler_options):
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

        self.logpost_func = logpost_func
        self.draw_func = draw_func
        self.options = sampler_options
        self.sampler_info = {}
        self.draw_samples(sampler)

    def draw_samples(self, sampler):
        '''
        Calls "utilitiesmethods.[method].sampler" where [method] is the
        user option.

        Parameters
        ----------
        sampler_method : str
            name of the sampler.

        Returns
        -------
        None.

        '''
        self.method = importlib.import_module('surmise.utilitiesmethods.'
                                              + sampler)

        # update sampler_info with the output of the sampler
        self.sampler_info = self.method.sampler(self.logpost_func,
                                                self.draw_func,
                                                **self.options)

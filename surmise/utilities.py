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
            The sampling methods under surmise can be returned
            within Python via: surmise.__utilitiesmethods__.

        Parameters
        ----------
        logpostfunc : function
             A function call describing the log of the posterior distribution.
        draw_func : function
            A function returning a random sample from a prior distribution.
        sampler : str, optional
            A string indicating the sampling method to be used.
            It points to the script located in ``utilitiesmethods/``.
            The default is 'metropolis_hastings'.
        sampler_options : dict, optional
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

        sampler_info is a dictionary keeping the outputs from the sampler.
        sampler_info['theta'] is required and keeps the posterior draws to be
        used in the calibration. If additional outputs from a sampler are needed
        to be passed to the calibrator, those can also be kept in sampler_info.

        Parameters
        ----------
        sampler_method : str
            Name of the sampler.

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

        if 'theta' not in self.sampler_info.keys():
            raise ValueError('A sample from a posterior distribution is required.')

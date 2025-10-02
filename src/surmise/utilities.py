import warnings
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
        # Samplers that could be loaded, but that are research-grade only and
        # should not be offered through the public interface.
        #
        # TODO: This should be removed as part of refactoring the sampler
        # portion of the public interface (Issue #159).
        KEY = "expertMode"
        RESEARCH_SAMPLERS = ["lmc"]

        if sampler.lower() in RESEARCH_SAMPLERS:
            if (KEY not in self.options) or (not self.options[KEY]):
                msg = "{} is included for unofficial research purposes only"
                raise ValueError(msg.format(sampler))
            else:
                # With the current implementation, expertMode=True could be
                # added to the calibration arguments with the intent of using a
                # research-grade calibrator but unintentionally allowing the use
                # of a research-grade sampler (or vice versa).  The refactoring
                # will hopefully avoid this ambiguity.
                #
                # Emit warning to extend a helping hand to the experts.
                msg = f"Using unofficial research {sampler} sampler"
                warnings.warn(msg)

        self.method = importlib.import_module('surmise.utilitiesmethods.'
                                              + sampler)

        # update sampler_info with the output of the sampler
        self.sampler_info = self.method.sampler(self.logpost_func,
                                                self.draw_func,
                                                **self.options)

        if 'theta' not in self.sampler_info.keys():
            raise ValueError('A sample from a posterior distribution is required.')

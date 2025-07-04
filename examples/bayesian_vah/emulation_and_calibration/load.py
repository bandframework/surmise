import pandas as pd
class simulation:
    """
    A class to represent a batch of simulation.

    ...

    Attributes
    ----------
    obs : Pandas DataFrame
        final simulation output. [n_designs x n_observables]
    obs_sd: Pandas DataFrame
        final simulation output standard deviation. [n_designs x n_observables]
    design : Pandas DataFrame
        design matrix. [n_designs x n_model_parameters]
    events : Pandas DataFrame
        number of successful events per each  design. [n_designs,2]

    Methods
    -------
    combine(allowed_failure_percentage):
        combine events, design, observable mean and error into a single DataFrame
        taking into account the allowed failure percentage of events per design.

    """
    def __init__(self,file):
        """
        Construct simulation data object.

        Parametrs
        ---------
        file : str
        name of the main simulation output observables file

        """
        # gather all design file info to maps_sim_design
        maps_sim_design = {}
        with open('../simulation_data/map_sim_to_design') as f:
            for l in f:
                l = l.split('\n')[0]
                maps_sim_design[l.split(' ')[1]]=l.split(' ')[2]
        #path to the simulation error output
        sim_path = f'../simulation_data/{file}'
        #path to the simulation error output
        sd_path = '../simulation_data/'+'sd'+file.split('mean')[-1]
        #path to the corresponding design file
        des_path = f'../design_data/{maps_sim_design[file]}'
        #path to the corresponding number of events file
        neve_path = f'../simulation_data/nevents_design/{file}_neve.txt'

        self.obs = pd.read_csv(sim_path, index_col=0)
        self.obs_sd = pd.read_csv(sd_path, index_col=0)
        self.design = pd.read_csv(des_path, delimiter = ' ')
        if 'tau_initial' in self.design:
            self.design = self.design.drop(['tau_initial'], axis=1)
        self.design = self.design.iloc[0:self.obs.shape[0]]
        event_dic = {'design':[],'nevents':[]}
        with open(neve_path) as f:
            for l in f:
                des_number = l.split('/')[-2]
                num_events = l.split(' ')[0]
                event_dic['design'].append(des_number)
                event_dic['nevents'].append( num_events)
        self.events = pd.DataFrame.from_dict(event_dic, dtype=float)
        # Drop last row because we did not consider the last design point when
        # gathering our results. This was by mistake.
        self.events = self.events[:-1]

    def combine(self, allowed_failure_percentage=100):
        """
        Concatanate events, design, observable, observable standard deviation into a single
        Dataframe along rows and only keep the simulations which has events above the specified
        threshold.

        Parametrs:
        ---------
        allowed_failure_percentage : float, optional
            threshold for maximum allowed number of event failures to keep
            in combined simulation data. (Default is 100.0, combine all designs)

        Returns:
        -------
        com_df: Pandas Dataframe
            combined dataframe that has events, design, observable
            mean and standard deviation
        """
        if allowed_failure_percentage >= 0 and allowed_failure_percentage <= 100:
            perc = allowed_failure_percentage
        else:
            raise Exception("Error, Percentage has to be between 0 and 100")
        obs_er = self.obs_sd.add_prefix('sd_')
        com_df = pd.concat([self.events, self.design, self.obs, obs_er], axis=1)
        throw_designs = com_df[com_df['nevents']<(max(com_df.nevents)*(100-perc)/100)].design.values
        print(f'Designs that have more that {perc} failure event rate \n {throw_designs}')
        com_df = com_df[com_df['nevents']>=(max(com_df.nevents)*(100-perc)/100)]
        #throw_designs = com_df[com_df['nevents']<(max(com_df.nevents)*(100-perc)/100)]
        return com_df

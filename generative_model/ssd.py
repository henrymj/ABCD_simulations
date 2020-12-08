# SSD generators

import numpy as np


class fixedSSD:
    def __init__(self, SSDs, **kwargs):
        """[summary]

        Args:
            SSDs (list): a list of SSDs (in which case they are presented with equal frequency)
                TBD(?): or a dictionary containing the SSDs and their relative frequency
        """
        self.SSDs = np.array(SSDs)

    def generate(self, n_stop_trials):
        """generator for fixed SSD design
        - if number of SSDs is not a multiple of n_stop_trials, then extra SSDs are randomly selected

        Args:
            n_stop_trials (int): number of stop trials

        Yields:
            [type]: [description]
        """
        num_repeats = np.ceil(n_stop_trials / len(self.SSDs))
        SSDlist = np.repeat(self.SSDs, num_repeats)
        np.random.shuffle(SSDlist)
        SSDlist = SSDlist[:n_stop_trials]
        for i in range(n_stop_trials):
            yield SSDlist[i]


# based on the ABCD tracking algorithm
class trackingSSD:
    def __init__(self, starting_ssd=50, step_size=50, min_ssd=0, max_ssd=550):
        self.starting_ssd = starting_ssd
        self.step_size = step_size
        self.min_ssd = min_ssd
        self.max_ssd = max_ssd

        self.SSD = starting_ssd

    def update(self, SSD, success):
        """[summary]

        Args:
            SSD (int): current sssd
            success (boolean): success or failure

        Returns:
            SSD (int): updated SSD
        """
        if SSD is None:
            SSD = self.starting_ssd
        elif success:
            SSD = SSD + self.step_size
        else:
            SSD = SSD - self.step_size

        if SSD < self.min_ssd:
            self.SSD = self.min_ssd
        elif SSD > self.max_ssd:
            self.SSD = self.max_ssd
        else:
            self.SSD = SSD

        return(self.SSD)

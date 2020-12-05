# accumulator class

import numpy as np


# class to implement accumulator
class Accumulator:
    def __init__(self, mu, noise_sd, starting_point, max_time,
                 threshold=None, min_accumulator_value=0):
        self.mu = mu
        self.noise_sd = noise_sd
        self.starting_point = starting_point
        self.max_time = max_time
        self.accum_ = None
        self.threshold = threshold
        self.min_accumulator_value = min_accumulator_value

    def run(self):
        """simulate an accumulator process

        Args:
            mu (float): drift rate
            noise_sd (float): noise standard deviation
            starting_point (int): starting point for accumulation (nondecision time for go, SSD for stop), in millseconds
            max_time (int): maximum time point (in milliseconds)
            min_accumulator_value(float): minimum value that accumulator can take.  set to None for unbounded
        """

        accum = np.zeros(self.max_time)
        # for period after start, add noise to mu_go
        accumulation_period = self.max_time - self.starting_point
        noise = np.random.randn(accumulation_period) * self.noise_sd
        drift = np.ones(accumulation_period) * self.mu
        accum[self.starting_point:] = np.cumsum(noise + drift)

        # remove values below minimum
        # vectorized version developed by @henrymj
        if self.min_accumulator_value is not None:
            negative_spots = np.where(accum < self.min_accumulator_value)[0]
            for neg_idx in negative_spots:
                # this will be true for the first index, but not necessarily the remaining indices
                if accum[neg_idx] < 0:
                    accum[neg_idx:] += -(accum[neg_idx])

        self.accum_ = accum

    def threshold_accumulator(self, threshold=None):
        """Compute RT and accuracy from an accumuator trace.

        Args:
            threshold (int, optional): threshold for accumulator. Defaults to None, in which case self.threshold is used

        Returns:
            rt (int): response time in milliseconds
        """
        if threshold is None:
            threshold = self.threshold
        assert threshold is not None

        if self.accum_ is None:
            self.run()

        exceed_threshold = np.where(self.accum_ > threshold)
        if len(exceed_threshold[0]) > 0:
            self.rt_ = np.min(exceed_threshold[0])
        else:
            self.rt_ = None

        return(self.rt_)


if __name__ == '__main__':
    accumulator = Accumulator(mu=0.5, noise_sd=0.3, starting_point=50, max_time=1000)
    accumulator.run()
    rt = accumulator.threshold_accumulator(100)
    print(accumulator.rt_)

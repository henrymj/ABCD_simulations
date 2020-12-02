# accumulator class

import numpy as np


# class to implement accumulator
# this is a 
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
            mu ([type]): drift rate
            noise_sd ([type]): noise standard deviation
            starting_point ([type]): starting point for accumulation (nondecision time for go, SSD for stop), in millseconds
            max_time ([type]): maximum time point (in milliseconds)
        """

        accum = np.zeros(self.max_time)
        # for period after start, add noise to mu_go
        accumulation_period = self.max_time - self.starting_point
        noise = np.random.randn(accumulation_period) * self.noise_sd
        drift = np.ones(accumulation_period) * self.mu
        accum[self.starting_point:] = np.cumsum(noise + drift)
        if self.min_accumulator_value:
            # clip minimum value of accumulator
            accum = np.clip(accum, self.min_accumulator_value, np.inf)
        self.accum_ = accum

    def threshold_accumulator(self, threshold=None):
        """Compute RT and accuracy from an accumuator trace.

        Args:
            threshold ([type], optional): threshold for accumulator. Defaults to None, in which case self.threshold is used
        
        Returns:
            rt (int): response time in milliseconds
            correct (bool): accuracy
        """
        if threshold is None:
            threshold = self.threshold
        assert threshold is not None

        if self.accum_ is None:
            self.run()

        exceed_threshold_abs = np.where(np.abs(self.accum_) > threshold)
        if len(exceed_threshold_abs[0]) > 0:
            self.rt_ = np.min(exceed_threshold_abs[0])
            self.correct_ = self.accum_[self.rt_] >= threshold
        else:
            self.rt_, self.correct_ = None, None

        return(self.rt_, self.correct_)


if __name__ == '__main__':
    accumulator = Accumulator(mu=0.5, noise_sd=0.3, starting_point=50, max_time=1000)
    accumulator.run()
    rt, correct = accumulator._threshold_accumulator(100)
    print(accumulator.rt_, accumulator.correct_)
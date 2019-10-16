__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import numpy as np
import pandas as pd
from abc import abstractmethod

from IPython import get_ipython

class PDF(object):
    """
    To display pdf in modern browsers!
    """
    def __init__(self, pdf, size=(200,200)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)


class RewardMechanism:
    @abstractmethod
    def sample_rew(self):
        """
        A function which when implemented samples or calculates an appropriate reward value.
        :return:
        """
        pass


class UniformReward(RewardMechanism):

    def __init__(self, low, high):
        """
        A reward distribution that returns a uniform reward based on a sample in a a range between
        low nad high.
        :param low: the lower range of the  distribution
        :param high:
        """
        self.low = low
        self.high = high
        self.value = [low, high]

    def sample_rew(self):
        """
        A function to sample a reward based on the definition of the Uniform Reward
        :return: a scalar value
        """
        return np.random.uniform(self.low, self.high, 1)[0]


class DeterministicReward(RewardMechanism):
    def __init__(self, value):
        """
        A reward that return a single scalar value
        :param value:
        """
        self.value = value

    def sample_rew(self):
        """
        Return the value of the deterministic reward
        :return: a scalar float denoting the reward
        """
        return self.value

def random_value_estimates(env):
    """
    Initialized a value estimate dictionary with uniform values.
    :return: the dictionary of states and their values.
    """
    values_estimates = {state : np.random.random(1) for state in env.possible_states}
    return values_estimates

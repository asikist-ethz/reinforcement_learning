__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import numpy as np
import pandas as pd

from utilities import DeterministicReward
from gym.spaces import Discrete
from gym import Env


class FiniteMDPEnv(Env):
    def __init__(self, transitions: list, terminal_state):
        """
        Initialize a FiniteMDP environment based on a transition list and a terminal state
        :param transitions:
        :param terminal_state:
        """

        # generate transitions table
        self.transitions = transitions
        self.terminal_state = terminal_state
        self.transitions_df = pd.DataFrame(transitions)
        self.transitions_df = self.transitions_df[['from_state', 'action', 'to_state', 'prob',
                                                   'reward', 'reward_distro']]
        self.transitions_df['cumul_prob'] = self.transitions_df.groupby(['from_state', 'action'])[
            'prob'].cumsum()

        # determine environment characteristics relevant to gym
        # agent actions, sorted for consistency of action appearance through different transition
        # implementations with same actions
        self.possible_actions = sorted(self.transitions_df['action'].unique().tolist())
        self.possible_states = set(self.transitions_df['from_state'].unique().tolist() +
                                   self.transitions_df['to_state'].unique().tolist())
        self.possible_states = sorted(list(self.possible_states))
        self.action_space = Discrete(len(self.possible_actions))
        self.state_space = Discrete(len(self.possible_states))  # current budget

        # Variables to be initialized in reset
        self.done = None
        self.total_reward = None
        self.time_step = None
        self.current_state = None
        self.reset()

    def step(self, action):
        """
        Applies an action to the current environment, advances the state, calculates reward and
        determines if a terminal state is reached.
        :param action: The action to be taken
        :return: A tuple: The new state representation, the reward, a boolean denoting if
        environment reached terminal state and finally a dictionary with meta information.
        """
        # state transition
        if isinstance(action, int):
            # in case an openai algorithm interacts with this
            action = self.possible_actions[action]

        transition_prob = np.random.random()
        from_state_filt = self.transitions_df['from_state'] == self.current_state
        action_filt = self.transitions_df['action'] == action
        prob_filt = transition_prob <= self.transitions_df['cumul_prob']

        transisition_filt = self.transitions_df[from_state_filt & action_filt & prob_filt]

        current_transition = transisition_filt.loc[transisition_filt['cumul_prob'].idxmin()]

        reward = current_transition['reward_distro'].sample_rew()
        new_state = current_transition['to_state']
        self.current_state = new_state
        self.total_reward += reward
        self.time_step += 1
        self.done = new_state == self.terminal_state
        # the empty dictionary object below is used by gym to store any metadata or extra info
        # you can use it like this as well, and store whatever extra info might be interesting for
        # you! Still, all info relevant to the action choice from your agents needs to be somehow
        # encoded in the state!
        return new_state, reward, self.done, {}

    def reset(self):
        """
        Reset method, useful to reset the environment to an initial state so that a new episode
        can be played.
        :return:
        """
        self.done = False
        self.total_reward = 0
        self.time_step = 0
        self.current_state = np.random.choice(self.transitions_df['from_state'].unique())
        return self.current_state

    def render(self, mode='human'):
        """
        Rendering the environment for an entity.
        :param mode: several modes may be implemented. the env can be rendered e.g. in a way that is
        "human" understandable.
        :return: A visualization or Nothing if the visualization is rendedred internally.
        In the current implementation we return the transition dataframe, the total reward and
        the timesteps.
        """
        return self.transitions_df, self.total_reward, self.time_step

    # some utility functions
    def get_possible_actions(self, from_state):
        """
        Returns a list with possible actions, given a state
        :param from_state:
        :return:
        """
        filt = self.transitions_df['from_state'] == from_state
        return self.transitions_df[filt]['action'].unique().tolist()

    def get_possible_states(self, from_state, action):
        """
        State transitions from a current state and an action.
        :param from_state: the current state
        :param action: the action applies after observing the current state
        :return: a dataframe with a "to_state" column and a probability "column" to transit to that
        state.
        """
        state_filt = self.transitions_df['from_state'] == from_state
        action_filt = self.transitions_df['action'] == action
        return self.transitions_df.loc[state_filt & action_filt, ['to_state', 'prob', 'reward']]


def define_transition(from_state, action, to_state, prob, reward_distro):
    """
    A utility that creates a row of a transition used in the transition table.
    :param from_state: A represenation that the agents can start from
    :param action: A representation of an action
    :param to_state: A representation of a state that an agent can go to
    :param prob: The probability of transition, between
    :param reward_distro: The object that generates rewards
    :return:
    """
    return dict(from_state=from_state,
                to_state=to_state,
                action=action,
                prob=prob,
                reward=reward_distro.value,
                reward_distro=reward_distro
                )


# Now let's make the mdp of the
budget_transitions = [
               define_transition('Low',    'Save',   'Medium',        1, DeterministicReward(1)),
               define_transition('Low',    'Invest', 'Low',         0.9, DeterministicReward(0)),
               define_transition('Low',    'Invest', 'High',        0.1, DeterministicReward(1)),
               define_transition('Medium', 'Save',   'High',        0.2, DeterministicReward(1)),
               define_transition('Medium', 'Save',   'Medium',      0.8, DeterministicReward(0)),
               define_transition('Medium', 'Invest', 'High',        0.4, DeterministicReward(1)),
               define_transition('Medium', 'Invest', 'Low',         0.6, DeterministicReward(-1)),
               define_transition('High',   'Invest', 'Low',         0.3, DeterministicReward(-1)),
               define_transition('High',   'Invest', 'Medium',      0.6, DeterministicReward(-1)),
               define_transition('High',   'Invest', 'Unlimited',   0.1, DeterministicReward(100)),
               define_transition('High',   'Save',   'High',      0.999, DeterministicReward(0)),
               define_transition('High',   'Save',   'Unlimited', 0.001, DeterministicReward(100))]

invest_environment = FiniteMDPEnv(budget_transitions, 'Unlimited')

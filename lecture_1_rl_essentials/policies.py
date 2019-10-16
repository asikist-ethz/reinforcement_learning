from holoviews.ipython import display

__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import numpy as np
import pandas as pd


class ProbabilisticPolicy:
    """
    A probabilistic policy, where each possible action at a state is assigned a non-zero
    probability.
    """
    def __init__(self, env):
        self.policy_table = env.transitions_df[['action', 'from_state']].rename(
            columns={'from_state': 'current_state'})
        self.policy_table = self.policy_table.drop_duplicates()
        self.policy_table['prob'] = 0

    def set_probability(self, current_state, action, prob):
        """
        Set the probability for a selecting an action given a current state
        :param current_state: The representation of the current state
        :param action: The representation of the action
        :param prob: The probability value
        :return: Nothing
        """
        state_filt = self.policy_table['current_state'] == current_state
        action_filt = self.policy_table['action'] == action
        self.policy_table.loc[state_filt & action_filt, 'prob'] = prob

    def validate(self):
        """
        A function to check if the probabilities of selecting an action given a state sum to one!
        :return: True, if the provided probabilities per state sum to one.
        """
        val_test = self.policy_table.groupby('current_state')['prob'].sum()
        return all(val_test > 0.99) and all(val_test < 1.01)

    def act_on_policy(self, state):
        """
        Given a policy table and a current state, take an action
        :param state: the current state
        :return:
        """
        decision_prob = np.random.random()
        policy_table = self.policy_table.copy()
        policy_table['cumul_prob'] = policy_table.groupby(['current_state'])['prob'].cumsum()

        state_filt = policy_table['current_state'] == state
        prob_filt = decision_prob <= policy_table['cumul_prob']
        policy_filt = policy_table[state_filt & prob_filt]
        current_decision = policy_filt.loc[policy_filt['cumul_prob'].idxmin()]['action']
        return current_decision

    def play(self, env, max_steps=300):
        """
        A full playthrough on an environment given a policy table.
        :param env: The environment object
        :param max_steps: The number of steps until the episode is played. May be controlled in
        environment as well.
        :return: The dataframe containing the information of all timesteps played in this environment.
        """
        if not self.validate():
            display(self.policy_table)
            raise ('The provided policy has states that allow no action to be taken! Action '
                   'selection probability per state does not sum to 0.')

        all_experience = []
        current_state = env.reset()
        while current_state is not 'Unlimited':
            old_state = env.current_state
            action = self.act_on_policy(env.current_state)
            current_state, reward, done, _ = env.step(action)
            res = {
                'observed_state': old_state,
                'action': action,
                'new_state': env.current_state,
                'reward': reward,
                'cumul_reward': env.total_reward,
                'done': done
            }
            all_experience.append(res)
            if env.time_step >= max_steps:
                break
        return pd.DataFrame(all_experience)


# some utilities for generating random policies
def any_deterministic_policy(env):
    """
    Generates any deterministic policy, such that the same action is always state given a state.
    :param env: The environment to draw the transition table from.
    :return: The policy table
    """
    random_generated_policy_table = env.transitions_df[['action', 'from_state']].rename(
        columns={'from_state': 'current_state'})
    random_generated_policy_table = random_generated_policy_table.drop_duplicates()
    random_generated_policy_table['prob'] = np.random.random(random_generated_policy_table.shape[0])
    most_probable = random_generated_policy_table.groupby('current_state')['prob'].idxmax()
    random_generated_policy_table.loc[most_probable.values, 'prob'] = 1
    random_generated_policy_table.loc[
        ~random_generated_policy_table.index.isin(most_probable.values), 'prob'] = 0
    random_generated_policy = ProbabilisticPolicy(env)
    random_generated_policy.policy_table = random_generated_policy_table
    return random_generated_policy


def any_stochastic_policy(env):
    """
    Generates a stochastic policy, meaning that each action is assigned a probability to be
    chosen in a uniform manner.
    :param env: The environment to draw the transition table from.
    :return: The policy table
    """
    random_generated_policy_table = env.transitions_df[['action', 'from_state']].rename(
        columns={'from_state': 'current_state'})
    random_generated_policy_table = random_generated_policy_table.drop_duplicates()
    random_generated_policy_table['prob'] = np.random.random(random_generated_policy_table.shape[0])

    probs = random_generated_policy_table.groupby(['current_state', 'action'])['prob'].first()

    denom = random_generated_policy_table.groupby(['current_state'])['prob'].sum()
    probs = probs / denom
    random_generated_policy_table = probs.reset_index()
    random_generated_policy_table['prob'] = probs.reset_index(drop=True)
    random_generated_policy = ProbabilisticPolicy(env)
    random_generated_policy.policy_table = random_generated_policy_table
    return random_generated_policy


def any_esoft_policy(env):
    """
    Generates a stochastic policy, meaning that each action is assigned a probability to be
    chosen in a uniform manner. Only non-zero probabilities are allowed.
    :param env: The environment to draw the transition table from.
    :return: The policy table
    """
    random_generated_policy_table = env.transitions_df[['action', 'from_state']].rename(
        columns={'from_state': 'current_state'})
    random_generated_policy_table = random_generated_policy_table.drop_duplicates()
    random_generated_policy_table['prob'] = np.random.random(random_generated_policy_table.shape[0]) + 0.0001

    probs = random_generated_policy_table.groupby(['current_state', 'action'])['prob'].first()

    denom = random_generated_policy_table.groupby(['current_state'])['prob'].sum()
    probs = probs / denom
    random_generated_policy_table = probs.reset_index()
    random_generated_policy_table['prob'] = probs.reset_index(drop=True)
    random_generated_policy = ProbabilisticPolicy(env)
    random_generated_policy.policy_table = random_generated_policy_table
    return random_generated_policy

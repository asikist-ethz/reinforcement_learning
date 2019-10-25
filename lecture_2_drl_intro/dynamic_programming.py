__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import numpy as np

from policies import any_deterministic_policy
from utilities import random_value_estimates
from plotly import graph_objs as go
from IPython.display import display


class Prediction:
    """
    An abstract class on how value estimation (prediction) is realized
    """

    def estimate(self, policy_table, transition_table):
        """
        A function that estimates the value based on the policy and the transition table.
        :param policy_table: A dataframe that contains the policy and the decision probabilities,
        it should have a "current_state" column containing the representation of state that a
        decision is taken from.
        :param transition_table: a table containing the state transitions
        :return: the dictionary with value estimates.
        """
        pass


class PolicyEvaluation(Prediction):
    """
    A class to determine the value of each state given the dynamic programming update.
    """
    def __init__(self, env, gamma=0.99, theta=0.01, renderer="browser"):
        """
        :param env: The finite MDP environment with known transition probabilities.
        :param gamma: a discounting parameter to determine how much future rewards affect the
        current one.
        :param theta: an upper bound parameter, which is usually a small positive value,
        that determines the accuracy of the approximation
        :param renderer: Where to show plots, e.g. "browser" for opening a browser window or
        "notebook" for showing them inline in notebook.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.convergence_values = dict()
        self.renderer = renderer

    def evaluate(self, policy_table, plot_convergence=False):
        """
        The iterative policy evaluation method
        :param policy_table: A dataframe that contains the policy and the decision probabilities,
        it should have a "current_state" column containing the representation of state that a
        decision is taken from.
        :param plot_convergence: Whether to plot convergence values.
        :return: the dictionary with states and the value estimates
        """
        env = self.env
        gamma = self.gamma
        theta = self.theta
        all_states = set(env.possible_states)
        value_estimates = random_value_estimates(env)
        value_estimates[env.terminal_state] = 0
        all_states.remove(env.terminal_state)  # terminal state is always zero

        self.convergence_values = dict()
        if plot_convergence:
            self.convergence_values['delta'] = []

        while True:  # until convergence
            delta = 0
            # NOTE: All for loops in this method and the rest of Monte Carlo and dynamic
            # programming can be vectorized, i.e. be done by vector/matrix operations in numpy
            # this is expected to speedup runtime of those methods...
            for state in all_states:
                v_old = value_estimates[state]
                possible_actions = policy_table[policy_table['current_state'] == state]

                v_new = 0
                # sum new_value estimate over all actions
                for action in possible_actions['action'].unique():
                    decision_prob = \
                        possible_actions[possible_actions['action'] == action]['prob'].iloc[0]

                    possible_next_states = \
                        env.transitions_df[(env.transitions_df['from_state'] == state) &
                                           (env.transitions_df['action'] == action)][
                            [
                                'to_state',
                                'prob',
                                'reward'
                            ]
                        ]
                    value_estimate_on_action = 0
                    for next_state in possible_next_states['to_state'].unique():
                        next_state_info = possible_next_states[
                            possible_next_states['to_state'] == next_state]
                        expected_reward = next_state_info['reward'].iloc[0]
                        transition_prob = next_state_info['prob'].iloc[0]
                        # get value estimates from other actions.
                        value_estimate_on_action += transition_prob * (
                                expected_reward + gamma * value_estimates[next_state])
                    value_estimate_on_action *= decision_prob

                    v_new += value_estimate_on_action
                # end of value estimate on action

                # use the operation on delta as an upper bound for convergence
                delta = max(delta, abs(v_new - v_old))
                value_estimates[state] = float(v_new)

                # plot stuff
                if plot_convergence:
                    if state not in self.convergence_values:
                        self.convergence_values[state] = []
                    self.convergence_values[state].append(float(abs(v_new - v_old)))

                if delta < theta:
                    # again plot stuff
                    if plot_convergence:
                        traces = []
                        for key, val in self.convergence_values.items():
                            traces.append(go.Scatter(x=np.arange(0, len(val)), y=val, name=key))
                        fig = go.Figure(traces)
                        fig.layout.xaxis.title = '# Iterations'
                        fig.layout.yaxis.title = '|V-v|'
                        fig.show(self.renderer)
                    return value_estimates


class PolicyImprovement:
    """
    A simple methodology to optimize a policy.
    """
    def __init__(self, env):
        """
        :param env: The environment that the policies are based on!
        """

        self.env = env

    def improve(self, value_estimates, policy_table):
        """
        Improve a policy based on current value estimates.
        :param value_estimates: A dictionary with state representations as keys and value
        estimates as values!
        :param policy_table: A dataframe that contains the policy and the decision probabilities,
        it should have a "current_state" column containing the representation of state that a
        decision is taken from.
        :return:
        """
        env = self.env
        policy_stable = True
        all_states = set(env.possible_states)
        all_states.remove(env.terminal_state)  # we don't act after we reach terminal state.

        for state in all_states:
            current_state_info = policy_table[policy_table['current_state'] == state]
            old_action_idx = current_state_info['prob'].idxmax()
            # store old action for later comparison
            old_action = policy_table.loc[old_action_idx, 'action']
            # determine new possible actions
            possible_actions = current_state_info['action'].values
            new_action = old_action
            new_value = -np.infty
            # loop over all possible actions
            for action in possible_actions:
                possible_next_states_info = env.transitions_df[
                    (env.transitions_df['action'] == action) &
                    (env.transitions_df['from_state'] == state)]
                # estimate the action-state value based on future states
                action_value_estimate = 0
                for next_state in possible_next_states_info['to_state'].unique():
                    next_state_info = possible_next_states_info[
                        possible_next_states_info['to_state'] == next_state]
                    reward = next_state_info['reward'].iloc[0]
                    transition_prob = next_state_info['prob'].iloc[0]
                    action_value_estimate += transition_prob * (
                                reward + value_estimates[next_state])
                if new_value <= action_value_estimate:
                    # if an action with higher action-state value occurs, then store it.
                    new_action = action
                    new_value = action_value_estimate

            if old_action != new_action:
                # if the action changes, then the policy is not stable
                # store the change and update the policy
                policy_stable = False
                policy_table.loc[(policy_table['current_state'] == state) &
                                 (policy_table['action'] == old_action), 'prob'] = 0
                policy_table.loc[(policy_table['current_state'] == state) &
                                 (policy_table['action'] == new_action), 'prob'] = 1
        return policy_table, policy_stable


class PolicyIteration:
    """
    A class that combines policy evaluation and policy improvement to find the optimal action.
    """
    def __init__(self, env, policy_evaluation, policy_improvement):
        """
        :param env: The environment to find the optimal value estimates and optimal policy
        :param policy_evaluation: The policy evaluation object to use for value estimation
        :param policy_improvement: hThe policy improvement object to use for policy improvement
        """
        self.env = env
        self.policy_evaluation = policy_evaluation
        self.policy_improvement = policy_improvement

    def iterate(self, initial_policy_table=None, display_policies=False):
        """
        The method that iteratively computes (i) value estimates and (ii) an optimal policy
        until the policy stabilizes.
        :param initial_policy_table: An initial policy table. If None, then a random policy is used.
        :param display_policies: Whether the first and last policy tables are displayed.
        :return:
        """
        env = self.env
        values_estimates = random_value_estimates(env)
        policy_table = [initial_policy_table, any_deterministic_policy(env)][
            initial_policy_table is None]
        policy_table.rename_axis('Initial Policy', axis="columns", inplace=True)
        # for presentation
        if display_policies:
            display(policy_table)

        policy_stable = False
        while not policy_stable:
            # keep iterating between policy evaluation and improvement until you reach a stable
            # policy!
            values_estimates = self.policy_evaluation.evaluate(policy_table)
            policy_table, policy_stable = self.policy_improvement.improve(
                values_estimates,
                policy_table
            )
        policy_table.rename_axis('Final Policy', axis="columns", inplace=True)
        # for presentation
        if display_policies:
            display(policy_table)
        return values_estimates, policy_table


class ValueIteration:
    """
    A faster way to find the optimal policy, without doing a policy improvement step!
    """
    def __init__(self, env, gamma=0.99, theta=0.0001, renderer="browser"):
        """
        :param env: The Finite MDP environment to search the optimal policy for
        :param gamma: a discounting parameter to determine how much future rewards affect the
        current one.
        :param theta: an upper bound parameter, which is usually a small positive value,
        that determines the accuracy of the approximation
        :param renderer: Where to show plots, e.g. "browser" for opening a browser window or
        "notebook" for showing them inline in notebook.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.renderer = renderer
        self.convergence_values = dict()

    def estimate(self, plot_convergence=False):
        """
        Estimate optimal values for each state
        :param plot_convergence: Whether you would like to see convergence plots!
        :return: The optimal value estimates
        """

        gamma = self.gamma
        theta = self.theta
        env = self.env
        all_states = set(env.transitions_df['from_state'].unique()) | set(
            env.transitions_df['to_state'].unique())
        value_estimates = {state: np.random.random() for state in all_states}
        value_estimates[env.terminal_state] = 0
        all_states.remove(env.terminal_state)  # terminal state is always zero

        self.convergence_values = dict()
        if plot_convergence:
            self.convergence_values = {}
        while True:
            delta = 0
            for state in all_states:
                v_old = value_estimates[state]
                # TODO proper action probs
                possible_actions = env.get_possible_actions(state)
                v_new = 0
                for action in possible_actions:
                    possible_next_states = env.get_possible_states(state, action)[['to_state',
                                                                                   'prob',
                                                                                   'reward'
                                                                                   ]]
                    value_estimate_on_action = 0
                    for next_state in possible_next_states['to_state'].unique():
                        next_state_info = possible_next_states[
                            possible_next_states['to_state'] == next_state]
                        expected_reward = next_state_info['reward'].iloc[0]
                        transition_prob = next_state_info['prob'].iloc[0]
                        value_estimate_on_action += transition_prob * (
                                expected_reward + gamma * value_estimates[next_state])

                    # the essential change between value iteration and policy evaluation
                    # using the max assumes the optimal greedy policy
                    # this is why this method doesn't require any input policies
                    v_new = max(v_new, value_estimate_on_action)

                delta = max(delta, abs(v_new - v_old))
                value_estimates[state] = v_new
                if plot_convergence:
                    if state not in self.convergence_values:
                        self.convergence_values[state] = []
                    self.convergence_values[state].append(abs(v_new - v_old))
            # plot stuff

            if delta < theta:
                if plot_convergence:
                    traces = []
                    for k, v_old in self.convergence_values.items():
                        traces.append(go.Scatter(x=np.arange(0, len(v_old)), y=v_old, name=k))
                    fig = go.Figure(traces)
                    fig.layout.xaxis.title = '# Iterations'
                    fig.layout.yaxis.title = '|V-v|'
                    fig.show(self.renderer)
                return value_estimates


def get_greedy_policy_table(env, value_estimates):
    """
    A method to find greedily the optimal policy, given value estimates.
    :param env: The environment to find the best policy in
    :param value_estimates: The dictionary containing the state representations and the values
    estimates for each state!
    :return: the policy table dataframe
    """
    policy_table = any_deterministic_policy(env).policy_table
    policy_table['prob'] = 0
    for state in policy_table['current_state'].unique():
        possible_actions = policy_table[policy_table['current_state'] == state]['action']
        best_action = None
        best_value = -np.infty
        for action in possible_actions:
            current_state_filter = env.transitions_df['from_state'] == state
            action_filter = env.transitions_df['action'] == action
            action_value = 0
            possible_next_states = env.transitions_df[current_state_filter & action_filter]
            for index, next_state in possible_next_states.iterrows():
                action_value += next_state['prob'] * (
                            next_state['reward'] + value_estimates[next_state['to_state']])
            if action_value >= best_value:
                best_value = action_value
                best_action = action
        policy_table.loc[(policy_table['current_state'] == state) &
                         (policy_table['action'] == best_action), 'prob'] = 1

    return policy_table

__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

from policies import any_esoft_policy, ProbabilisticPolicy
from IPython.display import display


class OnPolicyFirstVisitMC:
    """
    A monte carlo method for optimal policy control without knowing state transition
    probabilities (or dynamics).
    """

    def __init__(self, env, epsilon=0.001, gamma=0.99):
        """
        :param env: The environment to optimize on.
        :param epsilon: The fraction of probability that is assigned to choosing an action
        randomly and not greedily. Greedy choice can be choosing the action with highest
        action-state value.
        :param gamma: a discounting parameter to determine how much future rewards affect the
        current one.
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

    def estimate(self, max_episodes=300, display_policies=False):
        env = self.env
        env.reset()
        policy = any_esoft_policy(env)
        policy_table = policy.policy_table

        if display_policies:
            policy_table = policy_table.rename_axis('Initial MC Policy Table', axis='columns')
            display(policy_table)

        action_value_table = env.transitions_df[['from_state', 'action']].copy().drop_duplicates()
        action_value_table['exp_return'] = 0
        action_value_table['total_evals'] = 0
        n_actions = env.transitions_df['action'].nunique()

        # gor a good high number of episodes start searching.
        for episode in range(max_episodes):
            policy = ProbabilisticPolicy(env)
            policy.policy_table = policy_table
            experience = policy.play(env)
            g = 0  # return, or G
            n_exp = experience.shape[0] - 1

            # for all experiences
            for i in range(n_exp):
                # we start from the last experince and go backwards.
                # very usefull to calculate Return for each step with linear complexity!
                curr_index = n_exp - i
                current_exp = experience.iloc[curr_index, :]
                curr_action = current_exp['action']
                curr_state = current_exp['observed_state']

                g = self.gamma * g + current_exp['reward']
                previous_exp = experience.iloc[0:curr_index - 1, :]  # for lookup
                action_filt = (previous_exp['action'] == curr_action)
                state_filt = (previous_exp['observed_state'] == curr_state)
                # look in all earlier timesteps (t_earlier < t_now) and see if the action-state pair
                # appears. If not, time for updating the action-state value estimate.
                if previous_exp[action_filt & state_filt].shape[0] == 0:
                    action_filt_2 = (action_value_table['action'] == curr_action)
                    state_filt_2 = (action_value_table['from_state'] == curr_state)
                    prev_count = action_value_table.loc[action_filt_2 & state_filt_2, 'total_evals']
                    action_value_table.loc[
                        action_filt_2 & state_filt_2, 'total_evals'] = prev_count + 1
                    # Here we use an incremental mean calculation to avoid storing huge Return
                    # lists in memory...
                    prev_mean_g = action_value_table.loc[
                        action_filt_2 & state_filt_2, 'exp_return']
                    action_value_table.loc[action_filt_2 &
                                           state_filt_2, 'exp_return'] = prev_mean_g + (
                                g - prev_mean_g) / (prev_count + 1)

                    all_poss_actions = action_value_table.loc[state_filt_2, :]
                    best_action_idx = all_poss_actions['exp_return'].idxmax()
                    best_action = action_value_table.loc[state_filt_2, 'action'].loc[
                        best_action_idx]

                    # for all possible actions in after observing current state
                    for row_index, row in all_poss_actions.iterrows():
                        c_act = row['action']
                        action_filt_3 = (policy_table['action'] == c_act)
                        state_filt_3 = (policy_table['current_state'] == curr_state)
                        if c_act == best_action:
                            policy_table.loc[action_filt_3 & state_filt_3, 'prob'] = \
                                1 - self.epsilon + self.epsilon / n_actions
                        else:
                            policy_table.loc[action_filt_3 & state_filt_3, 'prob'] = \
                                self.epsilon / n_actions
        if display_policies:
            policy_table = policy_table.rename_axis('Final MC Policy Table', axis='columns')
            display(policy_table)
            display(action_value_table)

        return policy_table, action_value_table

import numpy as np
import torch

from legacy.algorithm.muzero.c_mcts import batch_mcts
from legacy.algorithm.muzero.utils.utils import pad_sampled_actions


def softmax(policy_logit: np.ndarray, available_action: np.ndarray):
    policy = np.exp(policy_logit)
    if available_action is not None:
        policy[available_action == 0] = 0
    policy = policy / policy.sum(axis=-1).reshape(*policy.shape[:-1], 1)
    return policy


class MCTS:

    def __init__(self, act_dim, action_space, pb_c_base, pb_c_init, discount, value_delta_max,
                 value_prefix_horizon, num_threads, num_simulations, mcts_use_sampled_actions, device,
                 **kwargs):
        """Monte Carlo Tree Search for MuZero
        Parameters
        ----------
        discount: float
            discount factor for MDP horizon
        use_mcts: bool
            whether use monte-carlo tree search for roll out
        value_delta_max: float
            value normalization boundary in UCB score computation
        num_simulations: int
            number of searched nodes in MCTS
        pb_c_base: int
            UCB score parameter
        pb_c_init: float
            UCB score parameter
        num_threads: int
            number of parallel threads for mcts
        value_prefix_horizon: int
            horizon of value prefix reset
        mcts_use_sampled_actions: bool
            use sampled actions to do tree search. Recommended for complex action space.
            Reference: Learning and Planning in Complex Action Spaces. http://arxiv.org/abs/2104.06303.
        """
        self.device = device
        self.act_dim = act_dim
        self.action_space = action_space
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.value_delta_max = value_delta_max
        self.value_prefix_horizon = value_prefix_horizon
        self.num_threads = num_threads
        self.num_simulations = num_simulations
        self.mcts_use_sampled_actions = mcts_use_sampled_actions

    def search(self,
               model,
               hidden_state_roots,
               reward_hidden_roots,
               root_value_prefix,
               root_policy,
               root_actions,
               available_action=None):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        """

        with torch.no_grad():
            model.eval()

            # preparation
            num = hidden_state_roots.shape[0]
            device = self.device
            pb_c_base, pb_c_init, discount = self.pb_c_base, self.pb_c_init, self.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            # the data storage of sampled actions
            sampled_actions_pool = [root_actions]
            # minimax value storage
            value_delta_max = self.value_delta_max
            horizons = self.value_prefix_horizon
            # available actions
            if available_action is None:
                available_action = np.ones((num, self.act_dim), dtype=np.int32)
            available_action = available_action.astype(np.int32)

            trees = batch_mcts.Multithread_Batch_MCTS(self.num_threads)  # batch_mcts.Batch_MCTS(num)
            trees.reset(pb_c_base, pb_c_init, discount, value_delta_max, root_value_prefix, root_policy,
                        available_action)

            for index_simulation in range(self.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # traverse to select actions for each root
                # results consist of (node_id, last_action, search_len)
                results = np.array(trees.batch_traverse(), dtype=np.int32)
                parent_ids, last_actions, search_lens = results[:, 0], results[:, 1], results[:, 2]

                last_sampled_actions = []

                # obtain the states for leaf nodes
                for ix, iy in zip(parent_ids, range(num)):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    last_sampled_actions.append(sampled_actions_pool[ix][iy][last_actions[iy]])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(
                    np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(
                    np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                last_sampled_actions = torch.from_numpy(np.asarray(last_sampled_actions)).to(device)

                # evaluation for leaf nodes
                network_output = model.recurrent_inference(hidden_states,
                                                           (hidden_states_c_reward, hidden_states_h_reward),
                                                           last_sampled_actions)

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1)
                value_pool = network_output.value.reshape(-1)
                policy_logits_pool, actions_pool = network_output.policy_logits, network_output.actions
                if self.mcts_use_sampled_actions:
                    # pad zero actions, policy logits and use available mask
                    available_action, policy_logits_pool, actions_pool = pad_sampled_actions(
                        self.act_dim, actions_pool, policy_logits_pool)
                else:
                    available_action = np.ones((num, self.act_dim), dtype=np.int32)
                policy_pool = softmax(policy_logits_pool, available_action).astype(np.float32)
                reward_hidden_nodes = network_output.reward_hidden

                hidden_state_pool.append(hidden_state_nodes)
                sampled_actions_pool.append(actions_pool)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                reward_hidden_nodes[0][:, reset_idx, :] = 0
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32)

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])

                # backpropagation along the search path to update the attributes
                trees.batch_back_propagate(value_prefix_pool, value_pool, policy_pool, is_reset_lst,
                                           available_action.astype(np.int32))
        # For debug, use following to print out search tree:
        # trees.print_all_trees()
        return np.array(trees.batch_get_distribution()), np.array(trees.batch_get_value())

# Algorithms

## Table of Contents:
- [MuZero](#MuZero)

<hr />

## MuZero
MuZero is a model-based RL method that achieves state-of-the-art in Atari Games, originally proposed by DeepMind. Reference: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4). The implementation is based on open-source implementation of EfficientZero, a sample-efficient variant of MuZero. Reference: [Mastering Atari Games with Limited Data](http://arxiv.org/abs/2111.00210), [Open source implementation](https://github.com/YeWR/EfficientZero).

Example configuration of MuZero is provided in config/atari.py - `AtariMuzeroReanalyzeExperiment` and config/gym_mujoco.py - `GymMujocoSampledMuzeroExperiment`. Note that `GymMujocoSampledMuzeroExperiment` could be configured to use either discrete or continuous action space. For a common environment that uses 1-dimension observations, you can safely adopt config of `GymMujocoSampledMuzeroExperiment` by filling in the action space and observation space.

### Compile MCTS Shared Library

To use MuZero, first compile the shared library for monte-carlo tree search(MCTS).
```
cd algorithm/muzero/c_mcts
bash build.sh
```

### Basic Usage

From system side, MuZero requires:
- actor workers to 1) collect trainnig data, 2) perform evaluation with MCTS and 3) perform evaluation without MCTS.
- policy workers for collecting training data and evaluation with or without MCTS.
- buffer workers that store history transitions, reanalyze sample batches (if reanalyze is enabled) and preprocess sample batches for training. If reanalyze is enabled, increasing buffer workers provides faster generation of training samples, since reanalyze is the main bottleneck, especially when batch size is large. If reanalyze is not enabled, you could try lower `gpu` in buffer worker scheduling to save resource usage. 
- trainer workers that update the model given sample from buffer workers.
- eval managers that log evaluation and training results.

From algorithm side, MuZero `Policy` has parameters:
- `td_steps` is the number of steps to compute temperal difference. `num_unroll_steps` is the number of steps the environment model is unrolled when training.
- MCTS related parameters include: 
    - `action_space`: action space defined by openai gym. When using standard MuZero, the action space should be `Discrete`.
    - `act_dim`: When using standard MuZero, `act_dim` should be equivalent to the size of action space. When using Sampled MuZero, `act_dim` is the number of branches at each node when doing MCTS. 
    - `mcts_use_sampled_actions` and `mcts_num_sampled_actions`: Set `mcts_use_sampled_actions` as True when using Sampled MuZero, which adopts sampling when expanding tree node. `mcts_num_sampled_actions` specifies the number of sampled actions when expanding the node. For more details, see section [Sampled MuZero](###SampledMuZero).
    - `num_simulations`: number of search for each input
    - `root_dirichlet_alpha` and `root_exploration_fraction` are used as exploration mechanism.
    - `num_threads`: policy worker uses a multi-threaded implementation of MCTS, `num_threads` specifies the number of threads used in MCTS. Previous experience shows 16 is enough.
- Neural network architecture. Check algorithm/muzero/policy/mlp_policy.py for more details. Current MLP policy uses a Resnet-v2 style architecture. Reference: Appendix A of [Learning and Planning in Complex Action Spaces](http://arxiv.org/abs/2104.06303).
- Value function and reward function. MuZero uses a discrete vectorized form of value and reward functions. In Hanabi, `DiscreteSupport(-25, 25, delta=1)` is enough since the absolute value for value and reward is at most 25. For games that have larger range of reward or value, try use larger support, for example, `DiscreteSupport(-500, 500, delta=1)`.
- Exploration. `visit_softmax_temperature_fn` is a function that outputs a temperature given policy version. `warm_up_version` specifies the verion threshold before which the actors take uniformly random actions.
- `rollout_update_interval` is the interval to update rollout policy. `reanalyze_update_interval` is the interval to update reanalyze policy.
- `reanalyze_ratio_schedule` outputs the ratio of reanalyzed sample given policy version as input ($0\%$ for no-reanalyze, $100\%$ for pure-reanalyze).

MuZero `Trainer` has parameters:
- `td_steps` and `num_unroll_steps`, same as in `Policy`.
- `optimizer_name` and `optimizer_config` for optimizer. Default is SGD. 
- `lr_schedule` specifies the change of learning rate during training. Current types of schedule include `linear`, `cosine` and `decay`. See function `adjust_lr` in `algorithm/muzero/utils/utils.py` for more details.
- `start_train_value` specifies a version threshold before which the model training doesn't take value loss into account. Experience shows this help stablize training.
- Loss coefficient for reward, value and policy: `reward_loss_coeff`, `value_loss_coeff`, `policy_loss_coeff`.
- Loss coefficient for hidden state consistency `consistency_coeff`. This is only enabled when `do_consistency` is `True`. Check algorithm/muzero/atar_policy for more details. 

### Sampled MuZero

Sampled MuZero is proposed to apply MuZero in arbitrary complex action spaces. Refernece: [Learning and Planning in Complex Action Spaces](http://arxiv.org/abs/2104.06303).

We now support some basic action space: `Discrete`, `Box` and `MultiDiscrete`. See `GymMujocoSampledMuzeroExperiment` for an example on using `MultiDiscrete` and `Box` as the action space. 

To add user-defined action space, you should complete following parts:
- Define your action space. `gym.spaces` wrappers are recommended.
- Set up the neural network to parameterize the policy in `algorithm/muzero/utils/act.py`-`ACTLayer`. An input to `ACTLayer` would be a batch of hidden vectors, i.e. a tensor with shape `(batch_size, hidden_dim)`. `ACTLayer` outputs an instance of the class `ActionSampler`.
- Complete the part to sample actions from the policy in function `sample_actions` of `algorithm/muzero/utils/act.py`-`ActionSampler`, and . Output of the function `sample_actions` is two lists of length `batch_size`, `actions` and `policy`. `actions` denotes the sampled actions and its shape is typically `(batch_size, num, *action_shape)`. `policy` denotes the empirical distribution of the sampled actions. For action distributions that would sample duplicated actions, e.g. `Discrete`, `policy` is the count of different sampled actions divided by `num`. For action distributions that do not sample duplicated actions, e.g. `Box`, `policy` is an array of shape `(batch_size, num)` with every item being $\frac{1}{num}$.
- Complete the part to evaluate the log probability of some actions in function `evaluate_actions` of `algorithm/muzero/utils/act.py`-`ActionSampler`. The input `actions` denotes actions to be evaluated and its shape is typically `(batch_size, num, *action_shape)`. The output contains the log probability of these actions and entropy of the policy.

**Notice:** When using Sampled MuZero for tasks that require continuous action space, such as mujoco robotics tasks, it's recommended to use discretized action space rather than the continuous space. If you strongly want to use continuous action space, e.g. `Box`, you should try regularization such as entropy penalty.
import logging

from api.config import *
from api.config import ActorWorker as ActorWorkerConfig
from api.environment import make as make_env
from distributed.system.actor_worker import ActorWorker
import base.namedarray
import distributed.system.inference_stream as inf_

inf_._INLINE_PULL_PARAMETER_ON_START = False

CHECKMARK = "\u2713"
"""Developer's checklist:
This test file is for developers to validate their environment, policy and/or, trainer before
deploying their code to the cluster.

This code is a minimum version of system logic. Advanced usage such as population manager and buffer 
worker are not included.
"""


def test_environment(env_name, env_args, test_stepping):
    action_mask_key = "available_action"
    env_config = Environment(type_=env_name, args=env_args)
    env = make_env(env_config)
    print(f"ENV: {env_name} {env_config}")
    print(f"Environment initialization: {CHECKMARK}")
    step_results = env.reset()
    print(f"Environment reset: {CHECKMARK}")
    assert len(step_results) == env.agent_count
    print(f"Number of agent is consistent: {CHECKMARK}")
    obs_shapes = [
        base.namedarray.from_dict(f"Agent{i}_Obs", sr.obs).shape for i, sr in enumerate(step_results)
        if sr is not None
    ]
    [print(f"Agent {i} observation shape: {s}") for i, s in enumerate(obs_shapes)]

    if test_stepping:
        for _ in range(10):
            actions = []
            for a in range(env.agent_count):
                if step_results[a] is not None:
                    args = (step_results[a].obs[action_mask_key],
                            ) if action_mask_key in step_results[a].obs.keys() else ()
                    actions.append(env.action_spaces[a].sample(*args))
                else:
                    actions.append(None)
            step_results = env.step(actions)
        print(f"Environment steps ok: {CHECKMARK}")
    return obs_shapes


def setup_testing_actor_worker(env_config, policy_config, trainer_config):
    testing_inline_inference = InferenceStream(type_=InferenceStream.Type.INLINE,
                                               stream_name="test",
                                               policy_name="default",
                                               pull_interval_seconds=1,
                                               accept_update_call=True,
                                               param_db=ParameterDB(type_=ParameterDB.Type.LOCAL_TESTING),
                                               policy=policy_config)

    testing_inline_training = SampleStream(type_=SampleStream.Type.INLINE_TESTING,
                                           trainer=trainer_config,
                                           policy=policy_config)

    aw = ActorWorker()
    cfg = ActorWorkerConfig(
        agent_specs=[AgentSpec(
            index_regex=".*",
            inference_stream_idx=0,
            sample_stream_idx=0,
        )],
        env=env_config,
        sample_streams=[testing_inline_training],
        inference_streams=[testing_inline_inference],
        ring_size=1,
        worker_info=WorkerInformation(experiment_name="dev",
                                      trial_name="dev",
                                      worker_type="aw",
                                      worker_index=0))

    aw._configure(cfg)
    return aw


def test_pipeline(env_name, env_args, policy_name, policy_args, trainer_name, trainer_args):

    env_config = Environment(type_=env_name, args=env_args)

    policy_config = Policy(type_=policy_name, args=policy_args)

    trainer_config = Trainer(type_=trainer_name, args=trainer_args)

    aw = setup_testing_actor_worker(env_config, policy_config, trainer_config)

    episodes = 2
    # while episodes > 0:
    print(f"Testing, {episodes} episodes left.")
    for i in range(10000):
        pr = aw._poll()
        if pr.batch_count > 0 and i > 0:
            episodes -= pr.batch_count
            print(f"Testing, {max(episodes, 0)} episodes left.")
        if i > 5000:
            print(f"5 episodes did not finished in 5000 steps. Episode too long?")
        if episodes <= 0:
            break
    else:
        raise RuntimeError(
            "5 episodes did not finish in 10000 steps. "
            "Is your environment hard to end? Consider adding a limit on number of steps or change"
            "the testing experiments.")

    print(f"\nThe following combination passed test! {CHECKMARK} {CHECKMARK} {CHECKMARK}"
          f"\n\nEnvironment \t{env_config} \nPolicy \t\t\t{policy_config} \nTrainer\t\t\t{trainer_config})"
          f"\nYou can not start configuring your experiment.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 1. Testing environment.
    # NOTE: observation_space and action_space are not require for running your experiment.
    #       If you know the system well and feel comfortable directly matching the experiments of your env and policy,
    #       you can pass test_stepping=False, observation space and action space will not be verified.
    obs_dim = test_environment(env_name="atari", env_args=dict(game_name="ALE/Boxing-v5"), test_stepping=True)

    # 2. Testing whole pipeline.
    test_pipeline(env_name="atari",
                  env_args=dict(game_name="ALE/Boxing-v5"),
                  policy_name="atari_naive_rnn",
                  policy_args=dict(obs_dim=obs_dim[0], action_dim=18),
                  trainer_name="mappo",
                  trainer_args=dict())

    # We also support async environment.
    base.namedarray.GENERATED_NAMEDARRAY_CLASSES = {}
    obs_dim = test_environment(env_name="hanabi",
                               env_args=dict(hanabi_name="Hanabi-Full",
                                             num_agents=5,
                                             use_obs_instead_of_state=True),
                               test_stepping=True)

    test_pipeline(env_name="hanabi",
                  env_args=dict(hanabi_name="Hanabi-Full", num_agents=5, use_obs_instead_of_state=True),
                  policy_name="actor-critic-separate",
                  policy_args=dict(obs_dim=obs_dim[0]["obs"][0],
                                   state_dim=obs_dim[0]["state"][0],
                                   action_dim=obs_dim[0]["available_action"][0]),
                  trainer_name="mappo",
                  trainer_args=dict())

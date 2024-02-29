# Distributed MARL

## For algorithm developers

- Our support for multi-agent training goes beyond classic MAPPO experiments.
To unleash full control over your agents, checkout our [experiment configuration doc](distributed/docs/user_guide/config_your_experiment.md).

- We provide a [quick start](distributed/docs/user_guide/dev_quickstart.md) for algorithm developers. Now users could 
migrate their environment, write customized policy and trainer without knowing the details of system implementation.

## Terminology

RL system components:

- [*Controller*](distributed/docs/system_components/02_controller.md), a "control panel" connected to all worker;
- [*Actor worker*](distributed/docs/system_components/03_actor_worker.md), who runs several environments. 
- [*Policy Worker*](distributed/docs/system_components/04_policy_worker.md), who generate actions for the agents.
- [*Buffer Worker*](distributed/docs/system_components/07_buffer_worker.md)，who prepares data for trainers, optional in most cases.
- [*Trainer worker*](distributed/docs/system_components/05_trainer_worker.md), who computes gradients and update parameters.
- [*Eval Manager*](distributed/docs/system_components/06_eval_manager.md), who logs evaluation results and update metadata parameter.
- [*Population Manager*](distributed/docs/system_components/08_pbt_manager.md): who controls the progress of population-based experiments.

Scheduler related:

- *experiment_name*(-e), name as registered by a experiment configuration.
- *trial_name*(-f), name given when launching an experiment.

## Code Structure

- `api`: Development api for algorithm and environments.
- `apps`: Main entry.
- `base`: The base library including anything unrelated to the RL logic; e.g. networking utils, general data structures
  & algorithms.
- `codespace`: where developers should place their code
- `distributed`: Directory for distributed system.
- `legacy`: Implementation of classic algorithm / environments.
- `local`: A local version of the distributed system.
- `scripts`: Scripts for developers.

## Getting Start

See [code-style.md](distributed/docs/cluster.md) for guide on development.
See [cluster.md](distributed/docs/cluster.md) for description on our cluster.

Prerequisite
1. Ask the administrators for an account on the cluster.
2. Setup your VPN. Ask the administrators for details.
3. On your PC, add the following lines to your ~/.ssh/config
```
Host prod 
    HostName 10.210.14.4
    User {YOUR_USER_NAME}
```

First, sync the repo to `frlcpu001`:
```
scripts/sync_repo prod
```
Alternatively you can check out the repo on the server. Make sure to sync or checkout the code to `/home` so that it is
visible on all nodes.

To run a mini experiment:
```
python3 -m apps.main start -e my-atari-exp -f $(whoami)-test --mode slurm --wandb_mode offline
```
This runs the experiment `my-atari-exp` with a trial name `username-test`. Mode should be slurm unless you are running 
the code on you PC or with in a container. You can also config your wandb api key on a 
terminal to allow `--wandb_mode online`:

```shell
# Get your WANDB api key from: https://wandb.ai/authorize
echo "export WANDB_API_KEY=< set your WANDB_API_KEY here>" >> ~/.profile
# Set wandb Host to our proxy.
echo 'export WANDB_BASE_URL="http://proxy.newfrl.com:8081"' >> ~/.profile
```

By default, experiments timeout after 3 days. You could change this value in your experiment configurations.

### System
System documentation is moved to [system documentation](distributed/docs/system_components/01_system_overview.md).

### Monitoring

We use both [wandb](https://wandb.ai) and [Prometheus](http://10.210.14.2:3000). Run `wandb init` and use `--wandb_mode online` to use the former.

The login of prometheus is the same as the cluster. 

Checkout [W\&B configuration](docs/user_guide/config_your_experiment.md#wandb-configuration) on how to customize your wandb_run setup.

Checkout [optimize_your_experiment.md](distributed/docs/user_guide/optimize_your_experiment.md) on how to improve the efficiency of your experiment. 

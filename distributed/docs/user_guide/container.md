# Container

This guide shows how to build, push and use customized images.

## Basic images

Two native images are provided: `marl/marl-cpu` and `marl/marl-gpu`. They can be used as 
```shell
srun ... --container-image=marl/marl-gpu ...
```

- `marl/marl-cpu`: Based on _ubuntu:20.04_, with torch and gym[atari] installed.
- `marl/marl-gpu`: Based on _nvcr.io/nvidia/pytorch:22.06-py3_, with cuda and torch. Requires more memory than `marl-cpu`.

## Build your own

The basic images provide environments for our system to function. If you are building a new environment that has its own dependencies, you will have to build a docker image.
FROM 10.210.14.10:5000/marl/marl-cpu(replace with gpu if CUDA is required.). 

Follow the steps:

### Write Dockerfile
```dockerfile
FROM 10.210.14.10:5000/marl/marl-cpu # replace with GPU if require CUDA.
# Install your dependency
RUN apt update && apt install -y ...
RUN pip3 install ...
```
### Build docker
```shell
docker build -t 10.210.14.10:5000/$USER/my_env:latest
```
### Test your environment
```shell
docker run 10.210.14.10:5000/$USER/my_env:latest python ...
```

### Push your image
```shell
docker push 10.210.14.10:5000/$USER/my_env:latest
```

Change experiment scheduling configuration. Use smac as an example:
```python
# legacy/experiments/smac.py

from api.config import *

ACTOR_IMAGE = "marl/marl-cpu-smac"


class SMACMiniExperiment(Experiment):

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=2,
                              scheduling=Scheduling.actor_worker_default(
                                  container_image=ACTOR_IMAGE,
                                  mem=4000
                                  )
                              ),
            policies=TasksGroup(count=1, scheduling=Scheduling.policy_worker_default()),
            trainers=TasksGroup(count=1, scheduling=Scheduling.trainer_worker_default()),
        )
    def initial_setup(self):
        ...
```

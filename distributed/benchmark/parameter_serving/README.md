# Parameter Serving Benchmark

About:

This benchmark is to simulate so called `staleness` in a distributed reinforcement
learning training loop. 

This benchmark can be run locally or on frl cluster. 
When running on frl cluster, it utilizes slurm to start multiple jobs on multiple nodes.

## Run this benchmark 

```shell
python3 -m distributed.benchmark.parameter_serving.benchmark \
    --mode filesystem --push_frequency 0.2 --data_size 10 \
    --pull_frequency 0.2 --post_frequency 2 --step_frequency 0.005 \
    --max_version 500 --max_time 60 --workers 4 --LOGLEVEL INFO \
    --trial_name 0208-3
```
change `INFO` to `DEBUG` to see verbose info.


When running on frl cluster, argument `--mode` can be changed to `multicast`.


## Pseudo Trainer

The pseudo trainer pushes data to parameter_db per `--push_frequency 0.2` seconds. Each
data is of size `--data_size 10` (MegaBytes). (Note that this is not the final size of the file.
The files size is ~1.5x(data_size) after pickling)

## Pseudo Worker

The benchmark launches `--workers 4` pseudo workers. Workers do not share nodes, to 
change this behavior, specify `--nworkers-per-node` in arguments. Each worker will 
pull the newest checkpoint every `--pull_frequence 0.2` seconds. Every `--step_frequency 0.005`
seconds, it records the checkpoint version of the checkpoint. Every `--post_frequency` seconds,
it posts the average checkpoint version since its latest post. 

## Measurement

The metric `staleness` is measured by pseudo trainer as `current_version - received_version`.
Average is taken over all data received.

## Multicasting

When running on cluster, the `--mode filesystem` can be replaced by `--mode multicast`. The trainer
will start a thread that broadcast the newest parameter version to the workers. In multicast mode, 
workers do not pull parameters, they instead listen to a multicast socket and update its version when
new parameter is ready.

## Known issues

1. Multicast rate is currently limited by hardware to ~300MB/s. The multicast mode does not 
scale properly when the number of workers increase.
2. The throughput of NAS Parameter DB on the cluster is limited by its network bandwidth,
Though workers on the same node can share the file cache created by Linux, the network bandwidth
may get exhausted when multiple applications run on multiple nodes. When we run 3 benchmarks with 
`push_frequency=pull_frequency=0.02,data_size=10`, each with 40 workers. Writing a new checkpoint costs double the time
   (0.32) of when we run 1 benchmark(0.17). 


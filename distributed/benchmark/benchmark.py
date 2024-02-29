import getpass
import hashlib
import logging
import numpy as np
import os
import pickle
import threading
import time

import api.policy
import distributed.base.name_resolve
import base.timeutil
import distributed.base.monitoring
from legacy import experiments
import api.environment
import distributed.system.worker_base
import distributed.system.actor_worker
import distributed.system.policy_worker
import distributed.system.inference_stream
import base.names as names

logger = logging.getLogger("benchmark")


class Benchmark:

    def __init__(self, experiment_name, trial_name, experiment: experiments.Experiment):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.experiment = experiment
        os.environ["WANDB_MODE"] = "offline"
        logger.info("Benchmark experiment: %s %s", self.experiment_name, self.trial_name)

    def benchmark_policy_worker(self, index=0, actors=1, agents=1, ring_size=1, timeout=None):
        distributed.base.name_resolve.clear_subtree(names.trial_root(self.experiment_name, self.trial_name))

        setup = self.experiment.initial_setup()
        setup.set_worker_information(experiment_name=self.experiment_name, trial_name=self.trial_name)
        actor_config = setup.actors[0]
        policy_config = setup.policies[index]
        policy_config.pull_frequency_seconds = 1e9
        logger.info("Select %d-th policy experiments", index)

        worker = distributed.system.policy_worker.PolicyWorker()
        _ = worker.configure(policy_config)
        r = worker.start_monitoring()
        worker.start()
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        logger.info("Started worker, monitoring info: %s", r)

        inference_client = distributed.system.inference_stream.make_client(policy_config.inference_stream,
                                                                           actor_config.worker_info)
        path = ("/tmp/marl_benchmark_req"
                f"_{getpass.getuser()}_{hashlib.sha1(str(actor_config.env).encode()).hexdigest()}.pkl")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                request = pickle.load(f)
        else:
            env = api.environment.make(actor_config.env)
            env_results = env.reset()[0]
            request = api.policy.RolloutRequest(
                obs=env_results.obs,
                policy_state=None,
                is_evaluation=np.array([False]),
                on_reset=np.array([True]),
            )
            with open(path, "wb") as f:
                pickle.dump(request, f)
        logger.info("Generated dummy request, pickled size=%d", len(pickle.dumps(request)))

        try:
            deadline = time.time() + (timeout or base.timeutil.INFINITE_DURATION)
            last_time = time.time()
            samples = distributed.base.monitoring.PrometheusMetricViewer(
                distributed.system.worker_base.METRIC_SAMPLE_COUNT)
            latency = distributed.base.monitoring.PrometheusMetricViewer(
                distributed.system.inference_stream.METRIC_INFERENCE_LATENCY_SECONDS)
            while time.time() < deadline:
                rs = []
                for _ in range(actors):
                    for _ in range(ring_size):
                        for _ in range(agents):
                            rs.append(inference_client.post_request(request))
                for r in rs:
                    while not inference_client.is_ready([r]):
                        time.sleep(0.001)
                    _ = inference_client.consume_result([r])
                if time.time() > last_time + 2:
                    last_time = time.time()
                    qps = samples.rate()
                    logger.info("SamplesPS: %.1f (%.1f rings), InfLatency: %.1fms", qps,
                                qps / actors / ring_size,
                                latency.rate() * 1000)
            logger.info("Time is up, stopping")
            worker.exit()
            thread.join()
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping")
        except TimeoutError as e:
            logger.info("Timeout: %s", e)
            import ipdb
            ipdb.set_trace()

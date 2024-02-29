import json

EXPERIMENT_NAME = "benchmark-param-db-experiment"
POLICY_NAME = "default"
IDENTIFIER = "latest"


def get_sub_request(trial_name):
    return {
        "experiment_name": EXPERIMENT_NAME,
        "trial_name": trial_name,
        "policy_name": POLICY_NAME,
        "tag": IDENTIFIER,
    }


def get_multicast_param_id(trial_name):
    return json.dumps(get_sub_request(trial_name), separators=(',', ':'), sort_keys=True)

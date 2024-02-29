# Evaluation Manager

_"Without proper self-evaluation, failure is inevitable."_

### Initialization(Configuration)
User has to specify the following fields to configure an eval_manger.

1. policy_name
2. eval_sample_stream
3. eval_tag
4. eval_target_tag
5. eval_games_per_version and eval_time_per_version_seconds

### Worker Logic

Evaluation manager accepts samples of two different kinds:

1. Samples of the same version to the current `eval_tag`.
2. Samples of tagged policy versions.

NOTE: eval_manager will discard samples where the policy version is not unique, or it does not match the above.

On receiving 1, the sample is considered as an evaluation result. 
With the specified frequency, data will be logged to W&B.

On receiving 2, eval_manager will extract the `episode_info` from the last step and update the metadata on that version 
accordingly.


# Controller
_"The more you try to control something, the more it controls you."_

### Initialization
- Controller is initialized by a pair of names (-e experiment_name, -f trial_name).
- An empty worker_control_panel and an empty scheduler are initialized.

### Start
- Start method Takes an instance of [ExperimentSetup](../../legacy/experiments/config_.py#L303-L335) and a specified partition. 
This is done automatically by [main_start method](../../../apps/main.py#L93-L106).
- The workers returned by ExperimentSetup is then scheduled, connected to, configured, and 
then instructed to start running.
- The controller will stop the trial if one of the following happens:
    1. The trial timeouts.
    2. Any worker completed/exits with error.
    3. The controller process receives signal SIGINT(KeyboardInterrupt).

### (Reconnect)
- Instead of start a new experiment, a controller can connect to a running experiment through
reconnect.
- Note that multiple controllers can be connected to the same trial. If one controller stop
the trial, others will also quit.

### Stop
To stop a trial, the controller will try the following two methods, until the trial is stopped successfully.
1. The controller will instruct all workers to stop through worker control.
2. If 1.) fails, the controller will ask the scheduler to kill all the workers. This should guarantee that all
workers are stopped.

Finally, the controller exits.


# What's Next
- [Actor Worker](03_actor_worker.md)
- [Policy Worker](04_policy_worker.md)

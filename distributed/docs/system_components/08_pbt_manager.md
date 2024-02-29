# Population Manager

_"Some cool sentences related to population manager."_

### Use case

Run Population-Based Training (PBT) experiments. Supported algorithms are:

* [Vanilla PBT](https://arxiv.org/abs/1711.09846)
* Policy Space Response Oracle ([PSRO](https://arxiv.org/abs/1711.00832))

### Initialization(Configuration)

To configure a population manager, one has to specify

1. population
2. population algorihtm
3. a population sample stream
4. list of actor configurations
5. list of policy configurations
6. list of trainer configurations
7. (optional) list of eval_manager configurations

### Worker Logic

Population manager consumes sample from sample stream, passes it to population algorithm, and get 
`requests`. When `requests` is not None, population manager sends requests to corresponding workers.

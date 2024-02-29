# Buffer Worker

_"Change begins at the end of your comfort zone."_

### Use case

Use buffer worker if you need

1. to reanalyze the sample (e.g. MuZero), or
2. to make changes to your sample (e.g. hindsight relabeling)

before the samples are sent to the trainers.

### Initialization(Configuration)

- To reanalyze
  - specify policy, policy_name, policy_identifier.
- To relabel
  - specify data augmenter.

### Worker Logic
- If configured to run both, relabel is done before reanalyze.
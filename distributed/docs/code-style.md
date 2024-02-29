# Code Style

- Changes are made in separate branches and merge requests are submitted for master.
- Please run `scripts/format` before committing any changes.
- Let's adopt https://google.github.io/styleguide/pyguide.html ?

## Imports

- Let's have two import blocks: import system modules first, and then our modules.
- In each import block, let's put `from ... import ...` all first, then normal imports.
- Otherwise, let's sort imports alphabetically.

# Unit testing

We currently use [Bazel](https://bazel.build/) for unit testing. To set it up, read
[here](https://docs.bazel.build/versions/4.2.1/install.html). Make sure to install version 4.2.1. Then run

```bash
sudo apt install python-is-python3
sudo ln -s /usr/bin/bazel-4.2.1 /usr/bin/bazel
```

Now `cd` into this repo. To run tests:

```bash
bazel test base:testing
bazel test --test_output=errors ... # Test all targets and also write errors to screen.
```

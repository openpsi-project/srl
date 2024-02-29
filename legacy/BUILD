package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

py_library(
    name = "legacy",
    srcs = glob(["**/*.py"]),
    deps = [
        "//api",
        "//base",
        "algorithm/muzero/c_mcts/batch_mcts",
    ],
)

pybind_extension(
    name="algorithm/muzero/c_mcts/batch_mcts",
    srcs=["algorithm/muzero/c_mcts/ctree.cc", "algorithm/muzero/c_mcts/ctree.h"],
)

py_library(
    name="algorithm/muzero/c_mcts/batch_mcts",
    data=[":algorithm/muzero/c_mcts/batch_mcts.so"],
)


py_test(
    name = "actor_critic_policies_test",
    srcs = [
        "tests/actor_critic_policies_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "atari_dqn_policy_test",
    srcs = [
        "tests/atari_dqn_policy_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "curriculum_test",
    srcs = [
        "tests/curriculum_test.py"
    ],
    deps = [
        ":legacy",
        "//api",
        "//base:testing",
    ],
)

py_test(
    name = "conditions_test",
    srcs = [
        "tests/conditions_test.py"
    ],
    deps = [
        ":legacy",
        "//api",
        "//base:testing",
    ],
)


py_test(
    name = "hns_policy_test",
    srcs = [
        "tests/hns_policy_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "meta_solver_test",
    srcs = [
        "tests/meta_solver_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "modules_test",
    srcs = [
        "tests/modules_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "psro_test",
    srcs = [
        "tests/psro_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "qmix_policy_test",
    srcs = [
        "tests/qmix_policy_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "smac_rnn_test",
    srcs = [
        "tests/smac_rnn_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)

py_test(
    name = "vanilla_pbt_test",
    srcs = [
        "tests/vanilla_pbt_test.py"
    ],
    deps = [
        ":legacy",
        "//base:testing",
    ],
)


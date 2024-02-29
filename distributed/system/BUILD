package(default_visibility = ["//apps:__subpackages__"])

py_library(
    name = "system",
    srcs = [
        "actor_worker.py",
        "buffer_worker.py",
        "controller.py",
        "eval_manager.py",
        "inference_stream.py",
        "parameter_db.py",
        "policy_worker.py",
        "sample_stream.py",
        "trainer_worker.py",
        "worker_base.py",
        "worker_control.py",
        "population_manager.py",
    ],
    deps = [
        "//api",
        "//base",
        "//distributed/base",
    ],
)

py_test(
    name = "buffer_worker_test",
    srcs = ["tests/buffer_worker_test.py"],
    deps = [
        ":system",
        "//api:testing",
        "//base:testing",
    ],
)

py_test(
    name = "worker_control_test",
    srcs = ["tests/worker_control_test.py"],
    deps = [
        ":system",
        "//base:testing",
    ],
)

py_test(
    name = "actor_worker_test",
    srcs = ["tests/actor_worker_test.py"],
    deps = [
        ":system",
        "//api:testing",
        "//base",
        "//base:namedarray",
        "//base:testing",
    ],
)

py_test(
    name = "eval_manager_test",
    srcs = ["tests/eval_manager_test.py"],
    deps = [
        ":system",
        "//base:testing",
    ],
)

py_test(
    name = "inference_stream_test",
    srcs = ["tests/inference_stream_test.py"],
    deps = [
        "//api",
        "//api:testing",
        ":system",
        "//base:testing",
    ],
)

py_test(
    name = "sample_stream_test",
    srcs = ["tests/sample_stream_test.py"],
    deps = [
        "//api",
        "//api:testing",
        ":system",
        "//base:namedarray",
        "//base:testing",
    ],
)

py_test(
    name = "parameter_db_test",
    srcs = ["tests/parameter_db_test.py"],
    deps = [
        ":system",
        "//base:testing",
    ],
)

py_test(
    name = "policy_worker_test",
    srcs = ["tests/policy_worker_test.py"],
    deps = [
        ":system",
        "//api",
        "//api:testing",
        "//base:namedarray",
        "//base:testing",
    ],
)

py_test(
    name = "trainer_worker_test",
    srcs = ["tests/trainer_worker_test.py"],
    deps = [
        ":system",
        "//api:testing",
        "//api",
        "//base:namedarray",
        "//base:testing",
    ],
)

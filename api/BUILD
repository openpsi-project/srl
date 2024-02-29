package(default_visibility = ["//visibility:public"])

py_library(
    name = "api",
    srcs = [
        "config.py",
        "curriculum.py",
        "env_utils.py",
        "environment.py",
        "pbt.py",
        "policy.py",
        "trainer.py",
    ],
    deps = [
        "//base",
    ],
)

py_library(
    name = "testing",
    testonly = 1,
    srcs = ["testing/aerochess_env.py",
            "testing/null_trainer.py",
            "testing/random_policy.py",
    ],
)

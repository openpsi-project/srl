package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = [
        "monitoring.py",
        "security.py",
        "name_resolve.py",
    ],
)

py_test(
    name = "name_resolve_test",
    srcs = ["tests/name_resolve_test.py"],
    deps = ["//base",
            ":base"],
)

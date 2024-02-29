package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = [
        "buffer.py",
        "conditions.py",
        "gpu_utils.py",
        "name_resolve.py",
        "namedarray.py",
        "names.py",
        "network.py",
        "numpy_utils.py",
        "segment_tree.py",
        "timeutil.py",
        "cluster.py",
    ],
)

py_library(
    name = "testing",
    testonly = 1,
    srcs = ["testing.py"],
)

py_library(
    name = "namedarray",
    testonly = 1,
    srcs = ["namedarray.py"],
)

py_test(
    name = "unittest_test",
    srcs = ["tests/unittest_test.py"],
)

py_test(
    name = "gpu_utils_test",
    srcs = ["tests/gpu_utils_test.py"],
    deps = [":base"],
)

py_test(
    name = "name_resolve_test",
    srcs = ["tests/name_resolve_test.py"],
    deps = [":base",
            "//base:testing",
            ],
)

py_test(
    name = "namedarray_test",
    srcs = ["tests/namedarray_test.py"],
    deps = [":base"],
)

py_test(
    name = "numpy_utils_test",
    srcs = ["tests/numpy_utils_test.py"],
    deps = [":base"],
)

py_test(
    name = "timeutil_test",
    srcs = ["tests/timeutil_test.py"],
    deps = [":base"],
)

py_test(
    name = "buffer_test",
    srcs = ["tests/buffer_test.py"],
    deps = [":base"],
)

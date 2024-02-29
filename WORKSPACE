load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name="pybind11_bazel",
    strip_prefix="pybind11_bazel-34206c29f891dbd5f6f5face7b91664c2ff7185c",
    urls=["https://github.com/pybind/pybind11_bazel/archive/34206c29f891dbd5f6f5face7b91664c2ff7185c.zip"],
    sha256="8d0b776ea5b67891f8585989d54aa34869fc12f14bf33f1dc7459458dd222e95",
)

http_archive(
    name="pybind11",
    build_file="@pybind11_bazel//:pybind11.BUILD",
    strip_prefix="pybind11-a54eab92d265337996b8e4b4149d9176c2d428a6",
    urls=["https://github.com/pybind/pybind11/archive/a54eab92d265337996b8e4b4149d9176c2d428a6.tar.gz"],
    sha256="c9375b7453bef1ba0106849c83881e6b6882d892c9fae5b2572a2192100ffb8a",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name="local_config_python")

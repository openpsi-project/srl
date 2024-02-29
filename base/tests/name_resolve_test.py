import tempfile
import unittest
import os
import shutil
import mock
import time

from base.network import gethostname
from base.testing import wait_network
import base.name_resolve as name_resolve


class MemoryNameResolveTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory", log_events=True)

    def test_write(self):
        name_resolve.clear_subtree("foo")
        name_resolve.add("foo/0", "bar0")
        self.assertEqual(name_resolve.get("foo/0"), "bar0")

        name_resolve.add("foo/1", "bar1")
        name_resolve.add("foo/2", "bar2")
        name_resolve.add("foo/3", "bar3")
        self.assertSetEqual(set(name_resolve.get_subtree("foo")), {"bar0", "bar1", "bar2", "bar3"})
        self.assertSetEqual(set(name_resolve.find_subtree("foo")), {"foo/0", "foo/1", "foo/2", "foo/3"})

        name_resolve.delete("foo/1")
        name_resolve.delete("foo/2")
        self.assertSetEqual(set(name_resolve.get_subtree("foo")), {"bar0", "bar3"})

        name_resolve.delete("foo/0")
        name_resolve.delete("foo/3")
        self.assertEqual(len(name_resolve.get_subtree("foo")), 0)

    def test_exists(self):
        name_resolve.add("foo", "bar1")

        with self.assertRaises(name_resolve.NameEntryExistsError):
            name_resolve.add("foo", "bar2")

        name_resolve.add("foo", "bar3", replace=True)
        self.assertEqual(name_resolve.get("foo"), "bar3")

    def test_not_found(self):
        name_resolve.add("foo", "bar1")

        with self.assertRaises(name_resolve.NameEntryNotFoundError):
            name_resolve.get("foo2")
        with self.assertRaises(name_resolve.NameEntryNotFoundError):
            name_resolve.delete("foo2")

    def test_reset(self):
        name_resolve.add("foo", "bar")
        self.assertEqual(name_resolve.get("foo"), "bar")

        name_resolve.DEFAULT_REPOSITORY.reset()
        name_resolve.reconfigure(name_resolve.DEFAULT_REPOSITORY_TYPE)
        with self.assertRaises(name_resolve.NameEntryNotFoundError):
            name_resolve.get("foo")

    def test_watch(self):
        foo_func = mock.MagicMock()
        name_resolve.watch_names(["foo"], foo_func, poll_frequency=0.1)
        foo_func.assert_not_called()
        name_resolve.add("foo", "bar", delete_on_exit=True, replace=True)
        wait_network(1)
        foo_func.assert_not_called()
        name_resolve.delete("foo")
        wait_network(1)
        foo_func.assert_called()

    def test_watch_multiple(self):
        foo_func = mock.MagicMock()

        name_resolve.watch_names(["foo1", "foo2"], foo_func, poll_frequency=0.2)
        foo_func.assert_not_called()
        name_resolve.add("foo1", "bar1", keepalive_ttl=0.3, delete_on_exit=True, replace=True)
        wait_network()
        foo_func.assert_not_called()
        name_resolve.add("foo2", "bar2", keepalive_ttl=0.3, delete_on_exit=True, replace=True)
        wait_network(2)
        foo_func.assert_not_called()
        name_resolve.delete("foo1")
        wait_network()
        foo_func.assert_not_called()
        name_resolve.delete("foo2")
        wait_network(2)
        foo_func.assert_called()


if __name__ == '__main__':
    unittest.main()

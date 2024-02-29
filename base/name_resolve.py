# Implements a simple name resolving service, which can be considered as a distributed key-value dict.
from typing import Optional, Callable, List
import logging
import queue
import time
import threading
import random
import uuid

logger = logging.getLogger("name-resolve")


class ArgumentError(Exception):
    pass


class NameEntryExistsError(Exception):
    pass


class NameEntryNotFoundError(Exception):
    pass


class NameRecordRepository:

    def __del__(self):
        try:
            self.reset()
        except Exception as e:
            logger.info(f"Exception ignore when deleting NameResolveRepo {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=None, replace=False):
        """Creates a name record in the central repository.

        In our semantics, the name record repository is essentially a multimap (i.e. Dict[str, Set[str]]).
        This class keeps a single name->value map, where the name can be non-unique while the value has to be.
        The class also deletes the (name, value) pair on exits (__exit__/__del__) if opted-in. In case of
        preventing unexpected exits (i.e. process crashes without calling graceful exits), an user may also
        want to specify time_to_live and call touch() regularly to allow a more consistent

        Args:
            name: The key of the record. It has to be a valid path-like string; e.g. "a/b/c". If the name
                already exists, the behaviour is defined by the `replace` argument.
            value: The value of the record. This can be any valid string.
            delete_on_exit: If the record shall be deleted when the repository closes.
            keepalive_ttl: If not None, adds a time-to-live in seconds for the record. The repository
                shall keep pinging the backend service with at least this frequency to make sure the name
                entry is alive during the lifetime of the repository. On the other hand, specifying this
                prevents stale keys caused by the scenario that a Python process accidentally crashes before
                calling delete().
            replace: If the name already exists, then replaces the current value with the supplied value if
                `replace` is True, or raises exception if `replace` is False.
        """
        raise NotImplementedError()

    def add_subentry(self, name, value, **kwargs):
        """Adds a sub-entry to the key-root `name`. The values is retrievable by get_subtree() given that no
        other entries use the name prefix.
        """
        sub_name = name.rstrip("/") + "/" + str(uuid.uuid4())[:8]
        self.add(sub_name, value, **kwargs)
        return sub_name

    def delete(self, name):
        """Deletes an existing record.
        """
        raise NotImplementedError()

    def clear_subtree(self, name_root):
        """Deletes all records whose names start with the path root name_root; specifically, whose name either
        is `name_root`, or starts with `name_root.rstrip("/") + "/"`.
        """
        raise NotImplementedError()

    def get(self, name):
        """Returns the value of the key. Raises NameEntryNotFoundError if not found.
        """
        raise NotImplementedError()

    def get_subtree(self, name_root):
        """Returns all values whose names start with the path root name_root; specifically, whose name either
        is `name_root`, or starts with `name_root.rstrip("/") + "/"`.
        """
        raise NotImplementedError()

    def find_subtree(self, name_root):
        """Returns all KEYS whose names start with the path root name_root.
        """
        raise NotImplementedError()

    def wait(self, name, timeout=None, poll_frequency=1):
        """Waits until a name appears.

        Raises:
             TimeoutError: if timeout exceeds.
        """
        start = time.monotonic()
        while True:
            try:
                return self.get(name)
            except NameEntryNotFoundError:
                pass
            if timeout is None or timeout > 0:
                time.sleep(poll_frequency + random.random() * .1)  # To reduce concurrency.
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"Timeout waiting for key '{name}' ({self.__class__.__name__})")

    def reset(self):
        """Deletes all entries added via this repository instance's add(delete_on_exit=True).
        """
        raise NotImplementedError()

    def watch_names(self, names: List, call_back: Callable, poll_frequency=15, wait_timeout=300):
        """Watch a name, execute call_back when key is deleted.
        """
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        for name in names:
            t = threading.Thread(target=self._watch_thread_run,
                                 args=(name, wrap_call_back, poll_frequency, wait_timeout),
                                 daemon=True)
            t.start()

    def _watch_thread_run(self, name, call_back, poll_frequency, wait_timeout):
        self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)
        while True:
            try:
                self.get(name)
                time.sleep(poll_frequency + random.random())
            except NameEntryNotFoundError:
                call_back()
                break


class MemoryNameRecordRepository(NameRecordRepository):
    """Stores all the records in a thread-local memory. Note that this is most likely for testing purposes:
    any distributed application is impossible to use this.
    """

    def __init__(self, log_events=False):
        self.__store = {}
        self.__log_events = log_events

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=None, replace=False):
        if self.__log_events:
            print(f"NameResolve: add {name} {value}")
        if name in self.__store and not replace:
            raise NameEntryExistsError(f"K={name} V={self.__store[name]} V2={value}")
        assert isinstance(value, str)
        self.__store[name] = value

    def touch(self, name, value, new_time_to_live):
        raise NotImplementedError()

    def delete(self, name):
        if self.__log_events:
            print(f"NameResolve: delete {name}")
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        del self.__store[name]

    def clear_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: clear_subtree {name_root}")
        name_root = name_root.rstrip("/")
        for name in list(self.__store):
            if name_root == "/" or name == name_root or name.startswith(name_root + "/"):
                del self.__store[name]

    def get(self, name):
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        r = self.__store[name]
        if self.__log_events:
            print(f"NameResolve: get {name} -> {r}")
        return r

    def get_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: get_subtree {name_root}")
        name_root = name_root.rstrip("/")
        rs = []
        for name, value in self.__store.items():
            if name_root == "/" or name == name_root or name.startswith(name_root + "/"):
                rs.append(value)
        return rs

    def find_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: find_subtree {name_root}")
        rs = []
        for name, value in self.__store.items():
            if name.startswith(name_root):
                rs.append(name)
        rs.sort()
        return rs

    def reset(self):
        self.__store = {}


def make_local_repository(**kwargs):
    return MemoryNameRecordRepository(**kwargs)


DEFAULT_REPOSITORY_TYPE = "memory"
DEFAULT_REPOSITORY = make_local_repository()
add = DEFAULT_REPOSITORY.add
add_subentry = DEFAULT_REPOSITORY.add_subentry
delete = DEFAULT_REPOSITORY.delete
clear_subtree = DEFAULT_REPOSITORY.clear_subtree
get = DEFAULT_REPOSITORY.get
get_subtree = DEFAULT_REPOSITORY.get_subtree
find_subtree = DEFAULT_REPOSITORY.find_subtree
wait = DEFAULT_REPOSITORY.wait
reset = DEFAULT_REPOSITORY.reset
watch_names = DEFAULT_REPOSITORY.watch_names


def reconfigure(*args, **kwargs):
    global DEFAULT_REPOSITORY, DEFAULT_REPOSITORY_TYPE
    global add, add_subentry, delete, clear_subtree, get, get_subtree, find_subtree, wait, reset, watch_names
    DEFAULT_REPOSITORY = make_local_repository(**kwargs)
    DEFAULT_REPOSITORY_TYPE = args[0]
    add = DEFAULT_REPOSITORY.add
    add_subentry = DEFAULT_REPOSITORY.add_subentry
    delete = DEFAULT_REPOSITORY.delete
    clear_subtree = DEFAULT_REPOSITORY.clear_subtree
    get = DEFAULT_REPOSITORY.get
    get_subtree = DEFAULT_REPOSITORY.get_subtree
    find_subtree = DEFAULT_REPOSITORY.find_subtree
    wait = DEFAULT_REPOSITORY.wait
    reset = DEFAULT_REPOSITORY.reset
    watch_names = DEFAULT_REPOSITORY.watch_names

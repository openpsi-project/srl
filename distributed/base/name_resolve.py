# Implements a simple name resolving service, which can be considered as a distributed key-value dict.
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
import dataclasses
import os
import socket
import shutil
import redis

import distributed.base.security
import base.timeutil
from base.name_resolve import *

logger = logging.getLogger("name-resolve")


class NfsNameRecordRepository(NameRecordRepository):
    if socket.gethostname().startswith("frl") or socket.gethostname().startswith("dev"):
        RECORD_ROOT = "/data/marl/name_resolve"
    else:
        RECORD_ROOT = "/tmp/marl_name_resolve"

    def __init__(self, **kwargs):
        self.__to_delete = set()

    @staticmethod
    def __dir_path(name):
        return os.path.join(NfsNameRecordRepository.RECORD_ROOT, name)

    @staticmethod
    def __file_path(name):
        return os.path.join(NfsNameRecordRepository.__dir_path(name), "ENTRY")

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=None, replace=False):
        value = str(value)
        path = self.__file_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path) and not replace:
            raise NameEntryExistsError(path)
        local_id = str(uuid.uuid4())[:8]
        with open(path + f".tmp.{local_id}", "w") as f:
            f.write(value)
        os.rename(path + f".tmp.{local_id}", path)
        if delete_on_exit:
            self.__to_delete.add(name)

    def delete(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        os.remove(path)
        while True:
            path = os.path.dirname(path)
            if path == NfsNameRecordRepository.RECORD_ROOT:
                break
            if len(os.listdir(path)) > 0:
                break
            shutil.rmtree(path, ignore_errors=True)
        if name in self.__to_delete:
            self.__to_delete.remove(name)

    def clear_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        if os.path.isdir(dir_path):
            logger.info("Removing name resolve path: %s", dir_path)
            shutil.rmtree(dir_path)

    def get(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        with open(path, "r") as f:
            return f.read().strip()

    def get_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for item in os.listdir(dir_path):
                rs.append(self.get(os.path.join(name_root, item)))
        return rs

    def find_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for item in os.listdir(dir_path):
                rs.append(os.path.join(name_root, item))
        rs.sort()
        return rs

    def reset(self):
        for name in list(self.__to_delete):
            try:
                self.delete(name)
            except NameEntryNotFoundError:
                pass
        self.__to_delete = {}


class RedisNameRecordRepository(NameRecordRepository):
    _IS_FRL = socket.gethostname().startswith("frl")
    REDIS_HOST = "redis" if _IS_FRL else "localhost"
    REDIS_PASSWORD = distributed.base.security.read_key("redis") if _IS_FRL else None
    REDIS_DB = 0
    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        keepalive_ttl: Optional[int] = None
        keeper: Optional[base.timeutil.FrequencyControl] = None

    def __init__(self, **kwargs):
        super().__init__()
        self.__lock = threading.Lock()
        self.__redis = redis.Redis(host=RedisNameRecordRepository.REDIS_HOST,
                                   password=RedisNameRecordRepository.REDIS_PASSWORD,
                                   db=RedisNameRecordRepository.REDIS_DB,
                                   socket_timeout=60,
                                   retry_on_timeout=True,
                                   retry=Retry(ExponentialBackoff(180, 60), 3))
        self.__entries = {}
        self.__keepalive_running = True
        self.__keepalive_thread = threading.Thread(target=self.__keepalive_thread_run, daemon=True)
        self.__keepalive_thread.start()

    def __del__(self):
        self.__keepalive_running = False
        self.__keepalive_thread.join(timeout=5)
        self.reset()
        self.__redis.close()

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=30, replace=False):
        if name.endswith("/"):
            raise ValueError(f"Entry name cannot end with '/': {name}")
        with self.__lock:
            if keepalive_ttl is None:
                if self.__redis.set(name, value, nx=not replace) is None:
                    raise NameEntryExistsError(f"Cannot set Redis key: K={name} V={value}")
                if delete_on_exit:
                    self.__entries[name] = self._Entry(value=value)
            else:
                keepalive_ttl = int(keepalive_ttl * 1000)
                assert keepalive_ttl > 0, f"keepalive_ttl in milliseconds must >0: {keepalive_ttl}"
                # Set the expiration of key to 3*frequency, and we shall touch the key with `frequency`.
                if self.__redis.set(name, value, px=keepalive_ttl, nx=not replace) is None:
                    raise NameEntryExistsError(f"Cannot set Redis key: K={name} V={value}")
                self.__entries[name] = self._Entry(
                    value=value,
                    keepalive_ttl=keepalive_ttl,
                    keeper=base.timeutil.FrequencyControl(frequency_seconds=keepalive_ttl / 1000 / 3))

    def delete(self, name):
        with self.__lock:
            self.__delete_locked(name)

    def __delete_locked(self, name):
        if name in self.__entries:
            del self.__entries[name]
        if self.__redis.delete(name) == 0:
            raise NameEntryNotFoundError(f"No such Redis entry to delete: {name}")

    def clear_subtree(self, name_root):
        with self.__lock:
            count = 0
            for name in list(self.__find_subtree_locked(name_root)):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            logger.info("Deleted %d Redis entries under %s", count, name_root)

    def get(self, name):
        with self.__lock:
            return self.__get_locked(name)

    def __get_locked(self, name):
        r = self.__redis.get(name)
        if r is None:
            raise NameEntryNotFoundError(f"No such Redis entry: {name}")
        return r.decode()

    def get_subtree(self, name_root):
        with self.__lock:
            rs = []
            for name in self.__find_subtree_locked(name_root):
                rs.append(self.__get_locked(name))
            rs.sort()
            return rs

    def find_subtree(self, name_root):
        with self.__lock:
            return list(sorted(self.__find_subtree_locked(name_root)))

    def reset(self):
        with self.__lock:
            count = 0
            for name in list(self.__entries):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            self.__entries = {}
            logger.info("Reset %d saved Redis entries", count)

    def __keepalive_thread_run(self):
        while self.__keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self.__lock:
                for name, entry in self.__entries.items():
                    if entry.keeper is not None and entry.keeper.check():
                        r = self.__redis.set(name, entry.value, px=entry.keepalive_ttl)
                        if r is None:
                            logger.error("Failed touching Redis key: K=%s V=%s", name, entry.value)

    def __find_subtree_locked(self, name_root):
        pattern = name_root.rstrip('/') + "/*"
        return [k.decode() for k in self.__redis.keys(pattern=pattern)]

    def _testonly_drop_cached_entry(self, name):
        """Used by unittest only to simulate the case that the Python process crashes and the key is
        automatically removed after TTL."""
        with self.__lock:
            del self.__entries[name]
            print("Testonly: dropped key:", name)


def make_repository(type_="nfs", **kwargs):
    if type_ == "memory":
        return MemoryNameRecordRepository(**kwargs)
    elif type_ == "nfs":
        return NfsNameRecordRepository(**kwargs)
    elif type_ == "redis":
        return RedisNameRecordRepository(**kwargs)
    else:
        raise NotImplementedError(f"No such name resolver: {type_}")


DEFAULT_REPOSITORY_TYPE = "nfs" if socket.gethostname().startswith("frl") else "nfs"
DEFAULT_REPOSITORY = make_repository(DEFAULT_REPOSITORY_TYPE)
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
    DEFAULT_REPOSITORY = make_repository(*args, **kwargs)
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

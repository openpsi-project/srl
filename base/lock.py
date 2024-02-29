import multiprocessing as mp
from contextlib import contextmanager
import logging

logger = logging.getLogger("Lock")


class ClientServerLock:
    """ A lock object that allows many simultaneous clients, but
    only one server. """

    def __init__(self):
        self._client_ready = mp.Condition(mp.Lock())
        self._clients = mp.Value("i", 0)
        self._waiting_server = mp.Value("i", 0)
        self._waiting_client = mp.Value("i", 0)

    def acquire_client(self):
        """ Acquire a read lock. Blocks only if a thread has
        acquired the write lock. """
        self._client_ready.acquire()
        while self._waiting_server.value > 0:
            self._waiting_client.value += 1
            self._client_ready.wait()
            self._waiting_client.value -= 1
        try:
            self._clients.value += 1
            # print("acquire client", self._readers.value)
        finally:
            self._client_ready.release()

    def release_client(self):
        """ Release a read lock. """
        self._client_ready.acquire()
        try:
            self._clients.value -= 1
            if self._clients.value == 0:
                self._client_ready.notify(self._waiting_client.value + self._waiting_server.value)
            # logger.info(f"Notify, {self._clients.value}")
            # print("release client", self._readers.value)
        finally:
            self._client_ready.release()

    def acquire_server(self):
        """ Acquire a write lock. Blocks until there are no
        acquired read or write locks. """
        self._client_ready.acquire()
        while self._clients.value > 0:
            self._waiting_server.value += 1
            # logger.info(f"acquire server, waiting for {self._clients.value}")
            self._client_ready.wait()
            # logger.info(f"acquired server, wait finished.")
            self._waiting_server.value -= 1
            self._client_ready.notify(self._waiting_client.value + self._waiting_server.value)

    def release_server(self):
        """ Release a write lock. """
        self._client_ready.release()

    @contextmanager
    def client_locked(self):
        try:
            self.acquire_client()
            yield
        finally:
            self.release_client()

    @contextmanager
    def server_locked(self):
        try:
            self.acquire_server()
            yield
        finally:
            self.release_server()


class MultiClientSingleServerLock:

    def __init__(self):
        self.condition = mp.Condition()
        self.num_clients = mp.Value('i', 0)
        self.num_servers = mp.Value('i', 0)

    def acquire_client(self):
        with self.condition:
            print(f"acquire client {self.num_clients.value}")
            while self.num_servers.value > 0:
                self.condition.wait()
            self.num_clients.value += 1
            print(f"acquire client done {self.num_clients.value}")

    def release_client(self):
        with self.condition:
            print(f"release client {self.num_clients.value}")
            self.num_clients.value -= 1
            self.condition.notify_all()
            print(f"release client done {self.num_clients.value}")

    def acquire_server(self):
        with self.condition:
            print(f"acquire server {self.num_servers.value}")
            while self.num_clients.value > 0 or self.num_servers.value > 0:
                self.condition.wait()
            self.num_servers.value += 1
            print(f"acquire server done {self.num_servers.value}")

    def release_server(self):
        with self.condition:
            print(f"release server {self.num_servers.value}")
            self.num_servers.value -= 1
            self.condition.notify_all()
            print(f"release server done {self.num_servers.value}")

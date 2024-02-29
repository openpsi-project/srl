from multiprocessing.shared_memory import SharedMemory
from typing import Optional, List, Tuple, Literal
import json
import logging
import numpy as np
import pickle
import threading
import time
import zmq
import socket

import base.namedarray
import base.names
import base.numpy_utils
import distributed.base.name_resolve  # TODO: resolve import distributed

logger = logging.getLogger("SharedMemory")


class SharedMemoryDockException(Exception):
    pass


class SharedMemoryDock:

    def __init__(
        self,
        qsize: int,
        shm_names: List[str],
        keys: List[str],
        dtypes: List[np.dtype],
        shapes: List[Tuple[int]],
        second_dim_index: bool = True  # whether to use second dimension for batch indices
        # should be true for sample stream, false for inference stream
    ):
        # Initialize shared memory. Create if not exist, else attach to existing shared memory block.
        self._shms = []
        for shm_name, dtype, shape in zip(shm_names, dtypes, shapes):
            if dtype is None:
                self._shms.append(None)
                continue
            try:
                # logger.info(F"Creating shared memory block {qsize}, {dtype}, {shape}")
                shm = SharedMemory(name=shm_name,
                                   create=True,
                                   size=qsize * base.numpy_utils.dtype_to_num_bytes(dtype) * np.prod(shape))
            except FileExistsError:
                shm = SharedMemory(name=shm_name, create=False)
            self._shms.append(shm)

        self._keys = keys
        self._dtypes = dtypes
        self._shapes = shapes
        self._second_dim_index = second_dim_index

        if self._second_dim_index:
            self._buffer = [
                np.frombuffer(shm.buf, dtype=dtype).reshape(shape[0], qsize, *shape[1:])
                if dtype is not None else None for shm, dtype, shape in zip(self._shms, dtypes, shapes)
            ]
        else:
            self._buffer = [
                np.frombuffer(shm.buf, dtype=dtype).reshape(qsize, *shape) if dtype is not None else None
                for shm, dtype, shape in zip(self._shms, dtypes, shapes)
            ]

        logger.debug("SharedMemoryDock buffer initialized.")

    def put(self, idx, x: base.namedarray.NamedArray, sort=True):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            if sort:
                idx = np.sort(idx)
        flattened_x = base.namedarray.flatten(x)
        assert len(flattened_x) == len(self._buffer)
        for (k, v), buf in zip(flattened_x, self._buffer):
            if buf is None:
                assert v is None, f"Buffer is none but value is not none, {k}"
            else:
                if self._second_dim_index:
                    # Following the convention of base.buffer, we regard the second dim as batch dimension.
                    buf[:, idx] = v
                else:
                    buf[idx] = v

    def get(self, idx):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            if len(idx) == 0:
                raise Exception("Cannot get with empty list idx!")
            idx = np.sort(idx)
        # idx here could be list or int
        if self._second_dim_index:
            xs = [(key, buf[:, idx]) if buf is not None else (key, None)
                  for buf, key in zip(self._buffer, self._keys)]
        else:
            xs = [(key, buf[idx]) if buf is not None else (key, None)
                  for buf, key in zip(self._buffer, self._keys)]
        res = base.namedarray.from_flattened(xs)
        # logger.info("shared memory get time %f", time.monotonic() - st)
        return res

    def get_key(self, key, idx=None):
        """Return numpy array with key. """
        buffer_idx = self._keys.index(key)
        if idx is None:
            return self._buffer[buffer_idx]
        else:
            return self._buffer[buffer_idx][idx]

    def put_key(self, key, idx, value):
        """Put numpy array with key. """
        buffer_idx = self._keys.index(key)
        self._buffer[buffer_idx][idx] = value

    def close(self):
        # Try release all shared memory blocks.
        self._buffer = []
        for shm in self._shms:
            if not shm:
                continue
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = []


class SharedMemoryRpcServer:

    def __init__(self, experiment_name, trial_name, server_type, stream_name):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._port = self._socket.bind_to_random_port(f"tcp://*")

        _name = base.names.shared_memory_dock_server(experiment_name, trial_name, server_type, stream_name)
        distributed.base.name_resolve.clear_subtree(name_root=_name)
        distributed.base.name_resolve.add(name=_name, value=str(self._port), keepalive_ttl=15)
        logger.info(f"name resolve put {_name}, {self._port}")

        self._handlers = {}

        self._server_thread = threading.Thread(target=self.handle, daemon=True)

        # Profiling
        self._server_counter = 0
        self._avg_recv_time = 0
        self._avg_send_time = 0
        self._avg_pickle_time = 0
        self._avg_handle_time = 0
        self._avg_total_time = 0

    def start(self):
        self._server_thread.start()

    def register_handler(self, name, handler):
        self._handlers[name] = handler

    @property
    def port(self):
        return self._port

    @property
    def messages_handled(self):
        return self._server_counter

    @property
    def timing(self):
        return self._avg_recv_time, self._avg_send_time, self._avg_pickle_time, self._avg_handle_time, self._avg_total_time

    def handle(self):
        while True:
            # Persistent server, blocking.
            t0 = time.monotonic()
            data = self._socket.recv()
            t1 = time.monotonic()
            # Handle requests sequentially.
            command, kwargs = pickle.loads(data)
            t2 = time.monotonic()
            logger.debug("Handle request: %s, len(data)=%d", command, len(data))
            if command in self._handlers:
                try:
                    response = self._handlers[command](**kwargs)
                    logger.debug("Handle request: %s, ok", command)
                except Exception as e:
                    logger.error("Handle request: %s, error", command)
                    logger.error(e, exc_info=True)
                    response = e
            else:
                logger.error("Handle request: %s, no such command", command)
                response = KeyError(f'No such command: {command}')
            t3 = time.monotonic()
            response = pickle.dumps(response)
            t4 = time.monotonic()
            self._socket.send(response)
            t5 = time.monotonic()

            logger.debug("Handle request: %s, sent reply %s", command, response)
            self._server_counter += 1
            self._avg_recv_time = (self._avg_recv_time * (self._server_counter - 1) +
                                   (t1 - t0)) / self._server_counter
            self._avg_pickle_time = (self._avg_pickle_time * (self._server_counter - 1) +
                                     (t2 - t1 + t4 - t3)) / self._server_counter
            self._avg_send_time = (self._avg_send_time * (self._server_counter - 1) +
                                   (t5 - t4)) / self._server_counter
            self._avg_handle_time = (self._avg_handle_time * (self._server_counter - 1) +
                                     (t3 - t2)) / self._server_counter
            self._avg_total_time = (self._avg_total_time * (self._server_counter - 1) +
                                    (t5 - t0)) / self._server_counter


class SharedMemoryRpcClient:

    def __init__(self, experiment_name, trial_name, server_type, stream_name):
        _name = base.names.shared_memory_dock_server(experiment_name, trial_name, server_type, stream_name)
        logger.info(f"name resolve wait for {_name}")
        server_port = distributed.base.name_resolve.wait(_name, timeout=5, poll_frequency=0.1)
        logger.info(f"name resolve wait for {_name}, get {server_port}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://localhost:{server_port}")

    def call(self, command, kwargs):
        self._socket.send(pickle.dumps((command, kwargs)))
        response = pickle.loads(self._socket.recv())
        return response


class SharedMemoryDockServer(SharedMemoryRpcServer):

    def __init__(self, experiment_name, trial_name, stream_name, qsize, reuses=1):
        super().__init__(experiment_name, trial_name, "sample", stream_name)
        self._qsize = qsize

        self._is_writable = np.ones(qsize, dtype=np.uint8)
        self._is_readable = np.zeros(qsize, dtype=np.uint8)
        self._is_being_read = np.zeros(qsize, dtype=np.uint8)
        self._is_being_written = np.zeros(qsize, dtype=np.uint8)

        self._time_stamp = np.zeros(qsize, dtype=np.int64)
        self._max_reuse = reuses
        self._reuses = np.zeros(qsize, dtype=np.int64)

        self.register_handler("acquire_read", self.acquire_read)
        self.register_handler("release_read", self.release_read)
        self.register_handler("acquire_write", self.acquire_write)
        self.register_handler("release_write", self.release_write)

    def acquire_write(
        self,
        batch_size: int = 1,
        allow_overwrite: bool = True,
        preference: Optional[Literal["uniform", "fresh", "old", "less_reuses_left",
                                     "more_reuses_left"]] = "old",
    ) -> Optional[List[int]]:
        """Acquire writable slots in shared memory.

        Args:
            batch_size (int): The number of slots to be allocated.
            allow_overwrite (bool): Whether to overwrite slots that have been written but
                not read by consumers. Defaults to True.
            preference (Literal): Determine to overwrite which kind of samples.

        Returns:
            Optional[List[int]]: Slot indices or None if not enough available slots.
        """
        writable_slots = np.nonzero(self._is_writable)[0]
        readable_slots = np.nonzero(self._is_readable)[0]

        if not allow_overwrite and len(writable_slots) < batch_size:
            return None

        if allow_overwrite and len(writable_slots) + len(readable_slots) < batch_size:
            return None

        slot_ids = list(writable_slots[:batch_size])
        # writable -> busy
        self._is_writable[slot_ids] = 0
        if len(slot_ids) < batch_size:

            if preference == "old":
                # replace the oldest readable slot, in a FIFO pattern
                slot_ids_ = readable_slots[np.argsort(self._time_stamp[readable_slots])[:batch_size -
                                                                                        len(slot_ids)]]
            elif preference == "fresh":
                slot_ids_ = readable_slots[np.argsort(-self._time_stamp[readable_slots])[:batch_size -
                                                                                         len(slot_ids)]]
            elif preference == "less_reuses_left":
                slot_ids_ = readable_slots[np.argsort(self._reuses[readable_slots])[:batch_size -
                                                                                    len(slot_ids)]]
            elif preference == "more_reuses_left":
                slot_ids_ = readable_slots[np.argsort(-self._reuses[readable_slots])[:batch_size -
                                                                                     len(slot_ids)]]
            elif preference == "uniform":
                slot_ids_ = np.random.choice(readable_slots, batch_size - len(slot_ids), replace=False)
            else:
                raise NotImplementedError(f"Unknown write preference {preference}.")

            # readable -> busy
            self._is_readable[slot_ids_] = 0
            self._time_stamp[slot_ids_] = 0
            self._reuses[slot_ids_] = 0
            slot_ids += list(slot_ids_)

        self._is_being_written[slot_ids] = 1
        assert (self._is_writable + self._is_readable + self._is_being_read +
                self._is_being_written == 1).all()
        return slot_ids

    def release_write(self, slot_ids: List[int]):
        """Release slots that is being written.

        Args:
            slot_ids (List[int]): Slot IDs to be released.
        """
        if np.any(np.array(slot_ids) >= self._qsize) or np.any(np.array(slot_ids) < 0):
            raise ValueError(f"Slot ID can only in the interval [0, qsize). "
                             f"Input {slot_ids}, qsize {self._qsize}.")
        if not np.all(self._is_being_written[slot_ids]):
            raise RuntimeError("Can't release slots that are not being written!")
        self._is_readable[slot_ids] = 1
        self._is_being_written[slot_ids] = 0
        self._time_stamp[slot_ids] = time.monotonic_ns()
        self._reuses[slot_ids] = self._max_reuse

    def acquire_read(
        self,
        batch_size: int = 1,
        preference: Literal["uniform", "fresh", "old", "less_reuses_left", "more_reuses_left"] = "fresh",
    ) -> Optional[List[int]]:
        """Acquire readable slot IDs in shared memory.

        Args:
            batch_size (int): The number of slots to be allocated.
            preference (Literal): Determine to read which kind of samples.

        Returns:
            Optional[List[int]]: Slot indices or None if not enough available slots.
        """
        # st = time.monotonic()
        readable_slots = np.nonzero(self._is_readable)[0]
        if len(readable_slots) < batch_size:
            return None

        # t1 = time.monotonic()

        if preference == "old":
            slot_ids = readable_slots[np.argsort(self._time_stamp[readable_slots])[:batch_size]]
        elif preference == "fresh":
            slot_ids = readable_slots[np.argsort(-self._time_stamp[readable_slots])[:batch_size]]
        elif preference == 'less_reuses_left':
            slot_ids = readable_slots[np.argsort(self._reuses[readable_slots])[:batch_size]]
        elif preference == "more_reuses_left":
            slot_ids = readable_slots[np.argsort(-self._reuses[readable_slots])[:batch_size]]
        elif preference == "uniform":
            slot_ids = np.random.choice(readable_slots, replace=False)
        else:
            raise NotImplementedError(f"Unknown read preference {preference}.")

        # t2 = time.monotonic()

        self._is_readable[slot_ids] = 0
        self._is_being_read[slot_ids] = 1
        assert (self._is_writable + self._is_readable + self._is_being_read +
                self._is_being_written == 1).all()

        # t3 = time.monotonic()
        # print(f"acquire_read: {t1-st}, {t2-t1}, {t3-t2}, total {t3-st}")
        return slot_ids

    def release_read(self, slot_ids: List[int]):
        """Release slots that is being read.

        Args:
            slot_ids (List[int]): Slot IDs to be released.
        """
        # t0 = time.monotonic()
        if np.any(np.array(slot_ids) >= self._qsize) or np.any(np.array(slot_ids) < 0):
            raise ValueError(f"Slot ID can only in the interval [0, qsize). "
                             f"Input {slot_ids}, qsize {self._qsize}.")
        if not np.all(self._is_being_read[slot_ids]):
            raise RuntimeError("Can't release slots that are not being read!")
        # t1 = time.monotonic()
        self._reuses[slot_ids] -= 1
        used_up_slots = np.array(slot_ids)[self._reuses[slot_ids] == 0]
        self._is_writable[used_up_slots] = 1
        self._is_being_read[used_up_slots] = 0
        self._time_stamp[used_up_slots] = 0
        # t2 = time.monotonic()

        remaining_slots = np.array(slot_ids)[self._reuses[slot_ids] > 0]
        self._is_being_read[remaining_slots] = 0
        self._is_readable[remaining_slots] = 1
        # t3 = time.monotonic()
        assert (self._is_writable + self._is_readable + self._is_being_read +
                self._is_being_written == 1).all()
        # t4 = time.monotonic()


class RequestSharedMemoryFull(Exception):
    pass


class RequestSharedMemoryEmpty(Exception):
    pass


class ResponseSharedMemoryFull(Exception):
    pass


class ResponseSharedMemoryEmpty(Exception):
    pass


class InferencePinnedSharedMemoryServer:

    def __init__(self):
        pass


class ReqRespSharedMemoryDockServer(SharedMemoryRpcServer):
    """ Shared memory dock server with bidirectional req/rep pattern for data. 
    Used for inference stream.
    """

    # client side write, one dock:
    # 1. client flush, acquire_write
    # 2. server find ready_to_write index, return to clinet
    # 3. client write to index
    # 4. client release_write
    # 5. server mark request as ready to be read by inference server

    # client side read, one dock:
    # 1. client poll_responses
    # 2. server find all ready_to_read index to the specific client, return to client
    # 3. client read from index, cache reference
    # 4. agent read from client, consume result, release_read

    def __init__(self, experiment_name, trial_name, stream_name, qsize, reuses=1):
        super().__init__(experiment_name, trial_name, "inference", stream_name)

        self.register_handler("request_acquire_write", self._request_acquire_write)
        self.register_handler("request_release_write", self._request_release_write)
        self.register_handler("request_acquire_read", self._request_acquire_read)
        self.register_handler("request_release_read", self._request_release_read)

        self.register_handler("response_acquire_write", self._response_acquire_write)
        self.register_handler("response_release_write", self._response_release_write)
        self.register_handler("response_acquire_read", self._response_acquire_read)
        self.register_handler("response_release_read", self._response_release_read)

        self.__qsize = qsize

        # Indices for shared memory.
        # Since multi-thread, add lock
        self._request_index_lock = threading.Lock()
        self._request_is_writable = np.ones(qsize, dtype=np.uint8)
        self._request_is_readable = np.zeros(qsize, dtype=np.uint32)
        self._request_is_being_written = np.zeros(qsize, dtype=np.uint8)
        self._request_is_being_read = np.zeros(qsize, dtype=np.uint8)

        self._response_index_lock = threading.Lock()
        self._response_is_writable = np.ones(qsize, dtype=np.uint8)
        self._response_is_readable = np.zeros(qsize, dtype=np.uint32)
        self._response_is_being_read = np.zeros(qsize, dtype=np.uint8)
        self._response_is_being_written = np.zeros(qsize, dtype=np.uint8)

    @property
    def qsize(self):
        return self.__qsize

    def _check_requestuest_index_legit(self):
        indices = [
            self._request_is_writable, self._request_is_readable, self._request_is_being_read,
            self._request_is_being_written
        ]
        masked = [np.where(x != 0, 1, 0) for x in indices]
        debug = np.where(sum(masked) != 1, 1, 0)
        debug_indices = np.nonzero(debug)[0]
        if not (sum(masked) == 1).all():
            logger.error(f"REQ wrong indices: {debug_indices}")
            logger.error(f"writable: {self._request_is_writable[debug_indices]}")
            logger.error(f"masked: {masked[0][debug_indices]}")
            logger.error(f"readable: {self._request_is_readable[debug_indices]}")
            logger.error(f"masked: {masked[1][debug_indices]}")
            logger.error(f"being_read: {self._request_is_being_read[debug_indices]}")
            logger.error(f"masked: {masked[2][debug_indices]}")
            logger.error(f"being_written: {self._request_is_being_written[debug_indices]}")
            logger.error(f"masked: {masked[3][debug_indices]}")
        assert (sum(masked) == 1).all(), "Request index is not legit, check implementation."

    def _check_response_index_legit(self):
        indices = [
            self._response_is_writable, self._response_is_readable, self._response_is_being_read,
            self._response_is_being_written
        ]
        masked = [np.where(x != 0, 1, 0) for x in indices]
        debug = np.where(sum(masked) != 1, 1, 0)
        debug_indices = np.nonzero(debug)[0]
        if not (sum(masked) == 1).all():
            logger.error(f"REP wrong indices: {debug_indices}")
            logger.error(f"writable: {self._response_is_writable[debug_indices]}")
            logger.error(f"masked: {masked[0][debug_indices]}")
            logger.error(f"readable: {self._response_is_readable[debug_indices]}")
            logger.error(f"masked: {masked[1][debug_indices]}")
            logger.error(f"being_read: {self._response_is_being_read[debug_indices]}")
            logger.error(f"masked: {masked[2][debug_indices]}")
            logger.error(f"being_written: {self._response_is_being_written[debug_indices]}")
            logger.error(f"masked: {masked[3][debug_indices]}")
        assert (sum(masked) == 1).all(), "Response index is not legit, check implementation."

    def _request_acquire_write(self, num_requests):
        """ Called when inference client starts calling flush(). 
        Return a list of indices that tells client where to write the requests. 
        """
        with self._request_index_lock:
            writable_indices = np.nonzero(self._request_is_writable)[0]
            if len(writable_indices) < num_requests:
                return []
            else:
                write_indices = writable_indices[:num_requests]
                self._request_is_writable[write_indices] = 0
                self._request_is_being_written[write_indices] = 1

                # print(f"acquire write for {write_indices}")
                # print(f"not writable indices {np.where(self._request_is_writable==0)}")

                self._check_requestuest_index_legit()
                return write_indices

    def _request_release_write(self, indices, client_id):
        """ Called when inference client finishes calling flush().
        """
        with self._request_index_lock:
            assert self._request_is_being_written[indices].all() == 1, \
                "Can not release write on indices that are not being written."
            self._request_is_being_written[indices] = 0
            self._request_is_readable[indices] = client_id

            # print(f"release write for {indices}")
            # print(f"not writable indices {np.where(self._request_is_writable==0)}")
            self._check_requestuest_index_legit()

    def _request_acquire_read(self, num_requests, read_all=False):
        """ Called when inference server starts calling poll_request().
        Return num_request readable indices if read_all = False.
        Return all readable indices if read_all = True.
        """
        with self._request_index_lock:
            readable_indices = np.nonzero(self._request_is_readable)[0]
            if (not read_all and len(readable_indices) < num_requests) \
               or (read_all and len(readable_indices) == 0):
                return []

            read_indices = readable_indices if read_all else readable_indices[:num_requests]
            self._request_is_readable[read_indices] = 0
            self._request_is_being_read[read_indices] = 1

            # print(f"acquire read for {read_indices}")
            # print(f"not writable indices {np.where(self._request_is_writable==0)}")
            self._check_requestuest_index_legit()
            return read_indices

    def _request_release_read(self, indices):
        """ Called when inference server finishes calling poll_request()
        """
        with self._request_index_lock:
            self._request_is_writable[indices] = 1
            self._request_is_being_read[indices] = 0
            # print(f"release read for {indices}")
            # print(f"not writable indices {np.where(self._request_is_writable==0)}")
            self._check_requestuest_index_legit()

    def _response_acquire_write(self, num_responses):
        """ Called when inference server starts calling respond()
        """
        with self._response_index_lock:
            writable_indices = np.nonzero(self._response_is_writable)[0]
            if len(writable_indices) < num_responses:
                raise ResponseSharedMemoryEmpty()
            write_indices = writable_indices[:num_responses]
            self._response_is_being_written[write_indices] = 1
            self._response_is_writable[write_indices] = 0
            self._check_response_index_legit()
            return write_indices

    def _response_release_write(self, indices, client_ids):
        """ Called when inference srever finishes calling respond()
        """
        assert len(indices) == len(
            client_ids), "Number of responses client ids does not match length of response indices."
        with self._response_index_lock:
            self._response_is_being_written[indices] = 0
            self._response_is_readable[indices] = client_ids
            self._check_response_index_legit()

    def _response_acquire_read(self, client_id):
        """ Called when client calls poll_responses(). 
        Return a list of indices that tells client where the responses (with input client id) are stored.
        """
        with self._response_index_lock:
            # print(np.nonzero(self._response_is_readable)[0], client_id)
            # print(self._response_is_readable)
            client_id_mask = np.where(self._response_is_readable == client_id, self._response_is_readable,
                                      np.zeros(self._response_is_readable.shape))
            # print(f"client id mask {client_id_mask}")
            readable_indices = np.nonzero(client_id_mask)[0]
            self._response_is_being_read[readable_indices] = 1
            self._response_is_readable[readable_indices] = 0
            # print(f"acquire read {readable_indices}")

            self._check_response_index_legit()
            return readable_indices

    def _response_release_read(self, indices):
        """ Called when client calls consume result.
        Acknowledge server that responses on position `indices` are already read.
        Server set these indices writable again.
        """
        with self._response_index_lock:
            assert self._response_is_being_read[indices].all() == 1, \
                "Can not release write on indices that are not being written."
            self._response_is_writable[indices] = 1
            self._response_is_being_read[indices] = 0

            self._check_response_index_legit()


class PinnedSharedMemoryServer(SharedMemoryRpcServer):
    """ Handle pinned shared memory requests and responses.
    """

    def __init__(self, experiment_name, trial_name, stream_name):
        super().__init__(experiment_name, trial_name, "inference", stream_name)

        self._total_num_agents = 0
        self._response_dock_acquired = False
        self.register_handler("register_agent", self._register_agent)
        self.register_handler("get_qsize", self._get_qsize)

    def _get_qsize(self):
        # logger.info(f"Getting qsize {self._total_num_agents}")
        return self._total_num_agents

    def _register_agent(self):
        """ Called when configuring actors.
        Register agents for one actor worker using this inference client/server.
        """
        logger.info(f"Registering agent {self._total_num_agents}.")
        index = self._total_num_agents
        self._total_num_agents += 1
        return index


def writer_make_shared_memory_dock(x, qsize, experiment_name, trial_name, stream_name, second_dim_index=True):
    name = base.names.shared_memory(experiment_name, trial_name, stream_name)
    shm_name = name.strip("/").replace("/", "_")
    # logger.info(f"DEBUG, writer make QSIZE = {qsize}")
    try:
        # If exists, get shared memory info from name resolve
        json_str = distributed.base.name_resolve.get(name)
        shm_names, keys, dtypes, shapes = json.loads(json_str)
        dtypes = [base.numpy_utils.decode_dtype(s) if s is not None else None for s in dtypes]
    except distributed.base.name_resolve.NameEntryNotFoundError:
        # If not, parse shared memory info from input namedarray
        keys, values = zip(*base.namedarray.flatten(x))
        # for key, value in zip(keys, values):
        #     if isinstance(value, np.ndarray):
        #         print(key, value.shape, value.dtype)
        #     else:
        #         print(key, value)
        shm_names = [f"{shm_name}_{key}" for key in keys]
        dtypes = [value.dtype if value is not None else None for value in values]
        shapes = [value.shape if value is not None else None for value in values]
        dtype_strs = [base.numpy_utils.encode_dtype(dt) if dt is not None else None for dt in dtypes]
        try:
            distributed.base.name_resolve.add(name=name,
                                              value=json.dumps([shm_names, keys, dtype_strs, shapes]),
                                              keepalive_ttl=15)
        except distributed.base.name_resolve.NameEntryExistsError:
            pass

    # Initialize shared memory dock
    return SharedMemoryDock(qsize, shm_names, keys, dtypes, shapes, second_dim_index)


class SharedMemoryWriter:

    def __init__(self, qsize, experiment_name, trial_name, stream_name):
        self.__rpc_client = SharedMemoryRpcClient(experiment_name, trial_name, "sample", stream_name)

        self._qsize = qsize
        self.__experiemnt_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self._shm_dock = None

    @property
    def qsize(self):
        return self._qsize

    def write(self, x: base.namedarray.NamedArray):
        if self._shm_dock is None:  # the first call, initialize shared memory
            self._shm_dock = writer_make_shared_memory_dock(x, self._qsize, self.__experiemnt_name,
                                                            self.__trial_name, self.__stream_name)

        while True:
            slot_ids = self.__rpc_client.call("acquire_write", dict(batch_size=1, allow_overwrite=False))
            if slot_ids is not None:
                break
            else:
                time.sleep(0.005)

        # TODO: batch write with flattened slicing
        slot_id = slot_ids[0]
        self._shm_dock.put(slot_id, x)

        self.__rpc_client.call("release_write", dict(slot_ids=slot_ids))

    def close(self):
        self._shm_dock.close()


class NothingToRead(Exception):
    pass


def reader_make_shared_memory_dock(qsize,
                                   experiment_name,
                                   trial_name,
                                   stream_name,
                                   second_dim_index=True,
                                   timeout=None):
    # logging.info(f"DEBUG, reader make QSIZE = {qsize}")
    name = base.names.shared_memory(experiment_name, trial_name, stream_name)
    try:
        dock_info = distributed.base.name_resolve.wait(name, timeout=timeout)
    except TimeoutError:
        return None
    logger.debug("SharedMemoryReader name resolve done.")
    try:
        shm_names, keys, dtypes, shapes = json.loads(dock_info)
        dtypes = [base.numpy_utils.decode_dtype(s) if s is not None else None for s in dtypes]
        # Initialize shared memory dock.
        return SharedMemoryDock(qsize, shm_names, keys, dtypes, shapes, second_dim_index)
    except Exception as e:
        logger.error("SharedMemoryReader failed to initialize shared memory dock.")
        raise e


class SharedMemoryReader:

    def __init__(self, qsize, experiment_name, trial_name, stream_name, batch_size):
        self.__rpc_client = SharedMemoryRpcClient(experiment_name, trial_name, "sample", stream_name)

        self._qsize = qsize
        self._batch_size = batch_size
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self._shm_dock = None

    @property
    def qsize(self):
        return self._qsize

    def read(self):
        if self._shm_dock is None:  # Lazy initialize shared memory dock
            self._shm_dock = reader_make_shared_memory_dock(self._qsize, self.__experiment_name,
                                                            self.__trial_name, self.__stream_name)

        cmd, kwargs = "acquire_read", dict(batch_size=self._batch_size if self._batch_size is not None else 1)
        slot_ids = self.__rpc_client.call(cmd, kwargs)
        if slot_ids is None:
            raise NothingToRead()

        if self._batch_size is None:
            x = self._shm_dock.get(slot_ids[0])
        else:
            x = self._shm_dock.get(slot_ids)

        cmd, kwargs = "release_read", dict(slot_ids=slot_ids)
        self.__rpc_client.call(cmd, kwargs)

        return x

    def close(self):
        self._shm_dock.close()

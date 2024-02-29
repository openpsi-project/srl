import concurrent.futures
import unittest
import unittest.mock
import threading

from distributed.system.worker_control import *
import distributed.base.name_resolve


class SocketServerTest(unittest.TestCase):

    def setUp(self):
        self.exp_name = "test_exp"
        self.trial_name = "test"
        distributed.base.name_resolve.reconfigure("memory", log_events=True)
        distributed.base.name_resolve.add(names.worker_status(self.exp_name, self.trial_name, "ctl"),
                                          value="READY",
                                          delete_on_exit=True)

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _run_server(server, max_requests=None):
        count = 0
        while max_requests is None or count < max_requests:
            count += server.handle_requests()
            time.sleep(0.01)

    def make_server(self, name):
        return ZmqServer(experiment_name=self.exp_name, trial_name=self.trial_name, worker_name=name)

    def make_control(self):
        return ZmqWorkerControl(experiment_name=self.exp_name, trial_name=self.trial_name)

    def test_simple(self):
        mock = unittest.mock.MagicMock(return_value={"x": 42, "y": "bar"})
        server = self.make_server("w0")
        server.register_handler("foo", mock)

        def run_client():
            control = self.make_control()
            assert len(control.connect(["w0"], timeout=1)) == 1
            return control.request("w0", "foo", a=50, b=40)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future = executor.submit(run_client)
            self._run_server(server, max_requests=1)
            r = future.result()
        self.assertEqual(r, {"x": 42, "y": "bar"})
        mock.assert_called_with(a=50, b=40)

    def test_multiple_requests(self):
        mock = unittest.mock.Mock(side_effect=lambda x, y: x * y)
        server = self.make_server("w0")
        server.register_handler("times", mock)

        def run_client():
            control = self.make_control()
            assert len(control.connect(["w0"], timeout=1)) == 1
            self.assertEqual(control.request("w0", "times", x=5, y=6), 30)
            self.assertEqual(control.request("w0", "times", x=-1, y=-1), 1)
            self.assertEqual(control.request("w0", "times", x=2, y=4), 8)
            self.assertEqual(control.request("w0", "times", x=7, y=4), 28)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future = executor.submit(run_client)
            self._run_server(server, max_requests=4)
            _ = future.result()

    def test_multiple_servers(self):
        n = 8
        done = False
        sems = [threading.Semaphore(value=0) for _ in range(n)]

        def run_server(index):
            mock = unittest.mock.MagicMock(return_value=index)
            server = self.make_server(f"w{index}")
            server.register_handler("index", mock)
            sems[index].release()
            while not done:
                server.handle_requests(1)
                time.sleep(0.01)

        threads = []
        for i in range(n):
            threads.append(threading.Thread(target=run_server, args=(i,), daemon=True))
            threads[-1].start()
        for sem in sems:  # Wait until all servers start.
            sem.acquire()

        control = self.make_control()
        rs = control.auto_connect()
        print(rs)
        self.assertSetEqual(set(rs), {f"w{i}" for i in range(n)})

        rs = [r.result for r in control.group_request("index", timeout=1)]
        self.assertSetEqual(set(rs), set(range(n)))

        rs = control.group_request("index", worker_names=["w1", "w4", "w5"], timeout=1)
        rs = [r.result for r in rs]
        self.assertSetEqual(set(rs), {1, 4, 5})

        rs = control.group_request("index", worker_regex="w[4-6]", timeout=1)
        rs = [r.result for r in rs]
        self.assertSetEqual(set(rs), {4, 5, 6})

        done = True

    def test_server_exception(self):
        mock = unittest.mock.Mock(side_effect=KeyError("foobar"))
        server = self.make_server("w0")
        server.register_handler("error", mock)

        def run_client():
            control = self.make_control()
            assert len(control.connect(["w0"], timeout=1)) == 1
            return control.request("w0", "error")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future = executor.submit(run_client)
            self._run_server(server, max_requests=1)
            with self.assertRaisesRegex(KeyError, "foobar"):
                _ = future.result()


if __name__ == "__main__":
    unittest.main()

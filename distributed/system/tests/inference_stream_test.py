import mock
import sys
import unittest

from distributed.system.inference_stream import *
from base.testing import get_testing_port, wait_network
from unittest import mock
import api.testing.random_policy
import api.config
import api.policy
import distributed.base.name_resolve as name_resolve
import distributed.system.parameter_db


def make_test_server(type_="ip", port=32342, policy_name="test_policy", batch_size=10, rank=0):
    if type_ == 'ip':
        return make_server(spec=api.config.InferenceStream(
            type_=api.config.InferenceStream.Type.IP, stream_name=policy_name, address=f"*:{port}"))
    elif type_ == "name":
        return make_server(spec=api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME,
                                                           stream_name=policy_name),
                           worker_info=api.config.WorkerInformation(experiment_name="test_exp",
                                                                    trial_name="test_run",
                                                                    worker_index=rank))


def make_test_client(type_="ip",
                     port=None,
                     policy_name="test_policy",
                     batch_size=10,
                     rank=0,
                     foreign_policy=None,
                     accept_update_call=True):
    if type_ == 'ip':
        return make_client(spec=api.config.InferenceStream(
            type_=api.config.InferenceStream.Type.IP,
            stream_name=policy_name,
            address=f"localhost:{port}",
        ))
    elif type_ == "name":
        return make_client(spec=api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME,
                                                           stream_name=policy_name),
                           worker_info=api.config.WorkerInformation(experiment_name="test_exp",
                                                                    trial_name="test_run",
                                                                    worker_index=rank))
    elif type_ == "inline":
        return make_client(spec=api.config.InferenceStream(
            type_=api.config.InferenceStream.Type.INLINE,
            stream_name=policy_name + "inline",
            policy=api.config.Policy(type_="random_policy", args=dict(action_space=10)),
            param_db=api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM,
                                            policy_name=policy_name),
            pull_interval_seconds=100,
            foreign_policy=foreign_policy,
            accept_update_call=accept_update_call,
            policy_name=policy_name),
                           worker_info=api.config.WorkerInformation(experiment_name="test_exp",
                                                                    trial_name="test_run",
                                                                    worker_index=rank))


class IpInferenceStreamTest(unittest.TestCase):

    def setUp(self) -> None:
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        IpInferenceClient._shake_hand = mock.Mock()
        sys.modules["gfootball"] = mock.Mock()

    def test_simple_pair(self):
        port = get_testing_port()

        server = make_test_server(port=port)
        client = make_test_client(port=port)

        # No request in the queue now.
        self.assertEqual(len(server.poll_requests()), 0)

        # Post two requests from the client. The server should be able to see.
        id1 = client.post_request(api.policy.RolloutRequest(obs=np.array(["foo"]),
                                                            policy_state=np.array(["foo"])),
                                  flush=False)
        id2 = client.post_request(api.policy.RolloutRequest(obs=np.array(["bar"]),
                                                            policy_state=np.array(["bar"])),
                                  flush=False)
        client.flush()
        wait_network()
        request_batch = server.poll_requests()

        self.assertEqual(len(request_batch), 1)  # One Bundle with two requests
        self.assertEqual(request_batch[0].length(), 2)  # One Bundle with two requests
        self.assertEqual((request_batch[0].request_id[0, 0], request_batch[0].obs[0, 0]), (id1, 'foo'))
        self.assertEqual((request_batch[0].request_id[1, 0], request_batch[0].obs[1, 0]), (id2, 'bar'))

        # No reply from the server yet.
        self.assertFalse(client.is_ready([id1]))
        self.assertFalse(client.is_ready([id2]))

        # Reply to one request.
        server.respond(
            api.policy.RolloutResult(action=np.array([[24]]),
                                     client_id=np.array([[client.client_id]], dtype=np.int32),
                                     request_id=np.array([[id1]], dtype=np.int64)))
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1]))
        self.assertFalse(client.is_ready([id2]))
        self.assertFalse(client.is_ready([id1, id2]))

        # Reply to the other.
        server.respond(
            api.policy.RolloutResult(action=np.array([[42]]),
                                     client_id=np.array([[client.client_id]], dtype=np.int32),
                                     request_id=np.array([[id2]], dtype=np.int64)))
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1]))
        self.assertTrue(client.is_ready([id2]))
        self.assertTrue(client.is_ready([id1, id2]))

        results = client.consume_result([id1, id2])
        self.assertEqual(results[0].action[0], 24)
        self.assertEqual(results[1].action[0], 42)

    def test_multiple_client_single_server(self):
        port = get_testing_port()

        server = make_test_server(port=port)
        client_list = [
            make_test_client(port=port),
            make_test_client(port=port),
        ]

        # send requests from two clients
        id11 = client_list[0].post_request(api.policy.RolloutRequest(obs=np.array(["foo1"]),
                                                                     policy_state=np.array(["foo1"])),
                                           flush=False)
        id12 = client_list[0].post_request(api.policy.RolloutRequest(obs=np.array(["foo2"]),
                                                                     policy_state=np.array(["foo2"])),
                                           flush=False)
        id21 = client_list[1].post_request(api.policy.RolloutRequest(obs=np.array(["bar1"]),
                                                                     policy_state=np.array(["bar1"])),
                                           flush=False)
        id22 = client_list[1].post_request(api.policy.RolloutRequest(obs=np.array(["bar2"]),
                                                                     policy_state=np.array(["bar2"])),
                                           flush=False)
        [c.flush() for c in client_list]

        wait_network()
        request_bundle = server.poll_requests()

        # should have 4 requests
        self.assertEqual(len(request_bundle), 2)
        self.assertEqual(request_bundle[0].length(0), 2)
        self.assertEqual(request_bundle[1].length(0), 2)
        server.respond(
            api.policy.RolloutResult(action=np.array([[24]]),
                                     client_id=np.array([[client_list[0].client_id]], dtype=np.int32),
                                     request_id=np.array([[id11]], dtype=np.int64)))
        wait_network()
        # client1: the first request is ready but not the second
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[0].is_ready([id11]))
        self.assertFalse(client_list[0].is_ready([id11, id12]))
        server.respond(
            api.policy.RolloutResult(action=np.array([[224]]),
                                     client_id=np.array([[client_list[0].client_id]], dtype=np.int32),
                                     request_id=np.array([[id12]], dtype=np.int64)))
        wait_network()
        # client1: both requests are ready
        # but nothing on client2
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[0].is_ready([id11, id12]))
        self.assertFalse(client_list[1].is_ready([id21, id22]))

        server.respond(
            api.policy.RolloutResult(action=np.array([[224]]),
                                     client_id=np.array([[client_list[1].client_id]], dtype=np.int32),
                                     request_id=np.array([[id21]], dtype=np.int64)))
        wait_network()
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[1].is_ready([id21]))
        self.assertFalse(client_list[1].is_ready([id21, id22]))
        server.respond(
            api.policy.RolloutResult(action=np.array([[224]]),
                                     client_id=np.array([[client_list[1].client_id]], dtype=np.int32),
                                     request_id=np.array([[id22]], dtype=np.int64)))
        wait_network()
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[1].is_ready([id21, id22]))


class InlineInferenceServerTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory")
        name_resolve.clear_subtree("inference_stream")
        self.param_db = mock.MagicMock()
        distributed.system.parameter_db.make_db = mock.MagicMock(return_value=self.param_db)

    def tearDown(self):
        name_resolve.clear_subtree("inference_stream")

    def test_simple_pair(self):
        server = make_test_server(type_="name")
        client = make_test_client(type_="name")

        # No request in the queue now.
        self.assertEqual(len(server.poll_requests()), 0)

        # Post two requests from the client. The server should be able to see.
        id1 = client.post_request(api.policy.RolloutRequest(obs=np.array(["foo"]),
                                                            policy_state=np.array(["foo"])),
                                  flush=False)
        client.flush()
        wait_network()
        request_batch = server.poll_requests()

        self.assertEqual(len(request_batch), 1)  # One Bundle with two requests
        self.assertEqual(request_batch[0].length(), 1)  # One Bundle with two requests
        self.assertEqual((request_batch[0].request_id[0, 0], request_batch[0].obs[0, 0]), (id1, 'foo'))

        # No reply from the server yet.
        self.assertFalse(client.is_ready([id1]))

        # Reply to one request.
        server.respond(
            api.policy.RolloutResult(action=np.array([[24]]),
                                     client_id=np.array([[client.client_id]], dtype=np.int32),
                                     request_id=np.array([[id1]], dtype=np.int64)))
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1]))

    def test_name_resolving_multiple(self):
        server0 = make_test_server(type_="name")
        server1 = make_test_server(type_="name")
        client0 = make_test_client(type_="name", rank=0)
        client1 = make_test_client(type_="name", rank=1)

        client0.post_request(api.policy.RolloutRequest(obs=np.array(["foo"]), policy_state=np.array(["foo"])),
                             flush=False)
        client0.flush()
        wait_network()
        n0 = len(server0.poll_requests())
        n1 = len(server1.poll_requests())
        self.assertEqual(n0 + n1, 1)

        client1.post_request(api.policy.RolloutRequest(obs=np.array(["bar"]), policy_state=np.array(["foo"])),
                             flush=False)
        client1.flush()
        wait_network()
        m0 = len(server0.poll_requests())
        m1 = len(server1.poll_requests())
        self.assertEqual(m0 + m1, 1)
        self.assertEqual(m0 + n0, 1)
        self.assertEqual(m1 + n1, 1)

    def test_set_constant(self):
        server = make_test_server(type_="name")
        client = make_test_client(type_="name")

        with self.assertRaises(name_resolve.NameEntryNotFoundError):
            client.get_constant("default_state")

        x = np.random.randn(10, 4)
        server.set_constant("default_state", x)
        y = client.get_constant("default_state")
        np.testing.assert_array_equal(x, y)

    def test_inline_client(self):
        ckpt = {"steps": 1, "state_dict": "null"}
        self.param_db.get = mock.MagicMock(return_value=ckpt)
        client = make_test_client("inline")
        id1 = client.post_request(api.policy.RolloutRequest(obs=np.array(["foo"]),
                                                            policy_state=np.array(["foo"])),
                                  flush=False)
        id2 = client.post_request(api.policy.RolloutRequest(obs=np.array(["bar"]),
                                                            policy_state=np.array(["bar"])),
                                  flush=False)
        self.assertFalse(client.is_ready([id1]))
        self.assertFalse(client.is_ready([id2]))
        client.flush()
        self.param_db.get.assert_called_once()
        self.assertTrue(client.is_ready([id1]))
        self.assertTrue(client.is_ready([id2]))
        rollout_results = client.consume_result([id1, id2])
        self.assertEqual(len(rollout_results), 2)
        self.assertFalse(client.is_ready([id1]))
        self.assertFalse(client.is_ready([id2]))

    def test_load_param(self):
        client = make_test_client(
            "inline",
            foreign_policy=api.config.ForeignPolicy(
                foreign_experiment_name="foo",
                foreign_trial_name="bar",
                foreign_policy_name="p",
                param_db=api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM),
                foreign_policy_identifier="i"))
        self.param_db.get.assert_not_called()
        client.load_parameter()
        self.param_db.get.assert_called_once()

        pi = self.param_db.get.call_args_list[0][1]
        self.assertEqual(pi["name"], "p")
        self.assertEqual(pi["identifier"], "i")

    def test_get_file(self):
        client = make_test_client(
            "inline",
            foreign_policy=api.config.ForeignPolicy(
                foreign_experiment_name="foo",
                foreign_trial_name="bar",
                foreign_policy_name="p",
                param_db=api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM),
                absolute_path="pax",
                foreign_policy_identifier="i"))
        self.param_db.get.assert_not_called()
        client.load_parameter()
        self.param_db.get.assert_not_called()
        self.param_db.get_file.assert_called_once_with("pax")

    def test_foreign_db(self):
        distributed.system.parameter_db.make_db = mock.MagicMock()
        client = make_test_client("inline",
                                  foreign_policy=api.config.ForeignPolicy(foreign_experiment_name="foo",
                                                                          foreign_trial_name="bar",
                                                                          foreign_policy_name="p",
                                                                          foreign_policy_identifier="i"))
        wi = distributed.system.parameter_db.make_db.call_args_list[0][1]["worker_info"]
        self.assertEqual(wi.experiment_name, "foo")
        self.assertEqual(wi.trial_name, "bar")


if __name__ == '__main__':
    unittest.main()

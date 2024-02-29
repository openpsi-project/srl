import io
import json
import threading

import mock
import time
import torch
import unittest
import zmq

import api.config
import distributed.service.parameter_service as parameter_service
import distributed.system.parameter_db as db


class TestPubInstance(unittest.TestCase):

    def setUp(self) -> None:
        self.test_policy_name = 'test_policy'
        self.test_sub_req = {
            "user_name": "test_user",
            "experiment_name": "test_exp",
            "trial_name": "test_trial",
            "policy_name": self.test_policy_name,
            "tag_name": "latest",
        }
        self.test_db = db.make_db(spec=api.config.ParameterDB(
            type_=api.config.ParameterDB.Type.LOCAL_TESTING))
        parameter_service.make_db = mock.MagicMock(return_value=self.test_db)

    def push_parameter(self, version):
        test_policy_version = version
        test_model_param = torch.randn(10)
        self.test_db.push(name=self.test_policy_name,
                          checkpoint=test_model_param,
                          version=str(test_policy_version))
        return test_model_param

    def make_subscriber(self, pub_address, topic):
        c = zmq.Context()
        s = c.socket(zmq.SUB)
        s.connect(pub_address)
        s.setsockopt(zmq.SUBSCRIBE, topic)
        s.setsockopt(zmq.RCVTIMEO, 2000)
        return c, s

    def make_pub_instance(self, sub_req):
        p = parameter_service.ParameterService.PubInstance(sub_req)
        return p

    def test_tcp_pub_run(self):
        sub_req = {
            "client_type": "local",
            **self.test_sub_req,
        }
        p = self.make_pub_instance(sub_req)
        pub_address = p.get_pub_address()
        topic = p.get_topic()
        self.assertTrue(pub_address.startswith('tcp'))

        context1, sub1 = self.make_subscriber(pub_address, topic)
        parameter0 = self.push_parameter(0)
        p.start()
        data0 = sub1.recv()[len(topic):]
        torch.testing.assert_close(parameter0, torch.load(io.BytesIO(data0), map_location="cpu"))

        context2, sub2 = self.make_subscriber(pub_address, topic)
        parameter1 = self.push_parameter(1)
        data1 = sub1.recv()[len(topic):]
        torch.testing.assert_close(parameter1, torch.load(io.BytesIO(data1), map_location="cpu"))
        data1 = sub2.recv()[len(topic):]
        torch.testing.assert_close(parameter1, torch.load(io.BytesIO(data1), map_location="cpu"))

        sub1.close(linger=False)
        sub2.close(linger=False)
        context1.destroy(linger=False)
        context2.destroy(linger=False)
        p.stop()


class TestParameterService(TestPubInstance):

    def setUp(self) -> None:
        super(TestParameterService, self).setUp()

    def make_connect_to_server(self, s):
        connect_request = {
            "type": "connect",
        }
        rep = s._get_response(connect_request)
        return rep

    def make_valid_subscribe_to_server(self, s, client):
        valid_subscribe_request = {
            "type": "subscribe",
            "client_id": client,
            "client_type": 'local',
            **self.test_sub_req,
        }
        rep = s._get_response(valid_subscribe_request)
        return rep

    def make_valid_unsubscribe_to_server(self, s, client, sub_key):
        valid_unsub_request = {
            "type": "unsubscribe",
            "client_id": client,
            "sub_key": sub_key,
        }
        rep = s._get_response(valid_unsub_request)
        return rep

    def make_valid_touch_to_server(self, s, client):
        valid_touch_request = {
            'type': 'touch',
            'client_id': client,
        }
        rep = s._get_response(valid_touch_request)
        return rep

    def set_time(self, now_time: float):
        time.time = lambda: now_time

    def advance_time(self, seconds):
        cur_time = time.time()
        time.time = lambda: cur_time + seconds

    def test_server_connect(self):
        s = parameter_service.make_service()
        rep = self.make_connect_to_server(s)
        self.assertTrue('status' in rep.keys())
        self.assertTrue('client_id' in rep.keys())
        self.assertEqual(rep['status'], 'ok')
        self.test_client = rep['client_id']

    def test_server_subscribe(self):
        s = parameter_service.make_service()
        rep = self.make_connect_to_server(s)
        client = rep['client_id']

        rep = self.make_valid_subscribe_to_server(s, client)
        self.assertTrue('status' in rep.keys())
        self.assertTrue('pub_address' in rep.keys())
        self.assertTrue('sub_key' in rep.keys())
        self.assertEqual(rep['status'], 'ok')

        invalid_subscribe_request = {
            "type": "subscribe",
            "client_id": 'test',
            "client_type": 'local',
            **self.test_sub_req,
        }
        rep = s._get_response(invalid_subscribe_request)
        self.assertEqual(rep['status'], 'error')
        s.interrupt()

    def test_server_unsubscribe(self):
        s = parameter_service.make_service()
        rep = self.make_connect_to_server(s)
        client = rep['client_id']
        rep = self.make_valid_subscribe_to_server(s, client)
        sub_key = rep['sub_key']

        rep = self.make_valid_unsubscribe_to_server(s, client, sub_key)
        self.assertTrue('status' in rep.keys())
        self.assertEqual(rep['status'], 'ok')

        invalid_unsub_request = {
            "type": "unsubscribe",
            "client_id": client,
            "sub_key": "test",
        }
        rep = s._get_response(invalid_unsub_request)
        self.assertEqual(rep['status'], 'error')
        s.interrupt()

    def test_server_touch(self):
        s = parameter_service.make_service()

        invalid_touch_request = {
            'type': 'touch',
            'client_id': 'test',
        }
        rep = s._get_response(invalid_touch_request)
        self.assertEqual(rep['status'], 'error')

        rep = self.make_connect_to_server(s)
        client = rep['client_id']

        rep = self.make_valid_touch_to_server(s, client)
        self.assertTrue(rep['status'], 'ok')

    def test_server_manage_publisher(self):
        s = parameter_service.make_service()
        self.assertEqual(s.active_clients_num(), 0)
        self.assertEqual(s.sleeping_clients_num(), 0)
        self.assertEqual(s.pub_instance_num(), 0)

        rep = self.make_connect_to_server(s)
        client1 = rep['client_id']
        self.assertEqual(s.active_clients_num(), 0)
        self.assertEqual(s.sleeping_clients_num(), 1)
        self.assertEqual(s.pub_instance_num(), 0)

        rep = self.make_valid_subscribe_to_server(s, client1)
        sub_key1 = rep['sub_key']
        self.assertEqual(s.active_clients_num(), 1)
        self.assertEqual(s.sleeping_clients_num(), 0)
        self.assertEqual(s.pub_instance_num(), 1)

        rep = self.make_connect_to_server(s)
        client2 = rep['client_id']
        self.assertNotEqual(client1, client2)
        rep = self.make_valid_subscribe_to_server(s, client2)
        sub_key2 = rep['sub_key']
        self.assertEqual(sub_key1, sub_key2)
        self.assertEqual(s.active_clients_num(), 2)
        self.assertEqual(s.sleeping_clients_num(), 0)
        self.assertEqual(s.pub_instance_num(), 1)

        self.advance_time(parameter_service.SUBSCRIPTION_MAX_TTL_SECONDS)
        self.make_valid_touch_to_server(s, client1)
        s._manage_publishers()
        self.assertEqual(s.active_clients_num(), 1)
        self.assertEqual(s.sleeping_clients_num(), 1)
        self.assertEqual(s.pub_instance_num(), 1)

        self.advance_time(parameter_service.SUBSCRIPTION_MAX_TTL_SECONDS)
        s._manage_publishers()
        self.assertEqual(s.active_clients_num(), 0)
        self.assertEqual(s.sleeping_clients_num(), 2)
        self.assertEqual(s.pub_instance_num(), 0)

    def test_server_run(self):
        s = parameter_service.make_service(address='127.0.0.1', port=3000)
        t = threading.Thread(target=s.run)
        t.start()
        ctx = zmq.Context()
        client_socket = ctx.socket(zmq.REQ)
        client_socket.setsockopt(zmq.RCVTIMEO, 2000)
        client_socket.connect(f"tcp://127.0.0.1:3000")

        invalid_request1 = {
            'typ': 'connect',
        }
        client_socket.send(json.dumps(invalid_request1).encode("ascii"))
        rep = json.loads(client_socket.recv())
        self.assertEqual(rep['status'], 'error')

        invalid_request2 = {
            'type': 'test_type',
        }
        client_socket.send(json.dumps(invalid_request2).encode("ascii"))
        rep = json.loads(client_socket.recv())
        self.assertEqual(rep['status'], 'error')

        connect_request = {'type': 'connect'}
        client_socket.send(json.dumps(connect_request).encode("ascii"))
        rep = json.loads(client_socket.recv())
        self.assertEqual(rep['status'], 'ok')
        client = rep['client_id']

        subscription_request = {
            'type': 'subscribe',
            "client_id": client,
            "client_type": 'local',
            **self.test_sub_req,
        }
        client_socket.send(json.dumps(subscription_request).encode("ascii"))
        rep = json.loads(client_socket.recv())
        self.assertEqual(rep['status'], 'ok')
        sub_key = rep['sub_key']

        client_sub_socket = ctx.socket(zmq.SUB)
        client_sub_socket.setsockopt(zmq.SUBSCRIBE, sub_key.encode("ascii"))
        client_sub_socket.connect(rep['pub_address'])
        parameter = self.push_parameter(0)
        data = client_sub_socket.recv()[len(sub_key.encode("ascii")):]
        torch.testing.assert_close(parameter, torch.load(io.BytesIO(data), map_location="cpu"))

        unsub_request = {
            'type': 'unsubscribe',
            "client_id": client,
            "sub_key": sub_key,
        }
        client_socket.send(json.dumps(unsub_request).encode("ascii"))
        rep = json.loads(client_socket.recv())
        self.assertEqual(rep['status'], 'ok')

        s.interrupt()
        client_socket.close()
        ctx.destroy()


if __name__ == '__main__':
    unittest.main()

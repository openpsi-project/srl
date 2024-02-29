import threading

import mock
import os
import shutil
import torch
import torch.nn as nn
import tempfile
import unittest.mock

from distributed.system.parameter_db import PytorchFilesystemParameterDB as FilesystemParameterDB
from base.testing import wait_network
import api.config
import distributed.base.name_resolve
import base.names
import base.network
import distributed.system.parameter_db


def get_test_param():
    return {
        "steps": 0,
        "state_dict": {
            "linear_weights": torch.randn(10, 10)
        },
    }


def weights_eq(x1, x2):
    return torch.all(x1["linear_weights"] == x2["linear_weights"])


class PytorchTestModule(nn.Module):
    """Two layer test network.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, data):
        return self.linear2(self.linear1(data))


def make_test_db(type_="filesystem", policy_name="test_policy", experiment_name="test_exp"):
    if type_ == "filesystem":
        return distributed.system.parameter_db.make_db(
            api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM, policy_name=policy_name),
            api.config.WorkerInformation(experiment_name=experiment_name,
                                         trial_name="test_run",
                                         worker_index=0))
    if type_ == "metadata":
        return distributed.system.parameter_db.make_db(
            api.config.ParameterDB(type_=api.config.ParameterDB.Type.METADATA, policy_name=policy_name),
            api.config.WorkerInformation(experiment_name=experiment_name,
                                         trial_name="test_run",
                                         worker_index=0))


class FilesystemParameterDBTest(unittest.TestCase):

    def setUp(self):
        self.__tmp = tempfile.TemporaryDirectory()
        FilesystemParameterDB.ROOT = os.path.join(self.__tmp.name, "checkpoints")
        self.__version = 0

    def get_version(self):
        self.__version += 1
        return str(self.__version)

    def test_trivial(self):
        db = make_test_db()

        ckpt = get_test_param()
        v1 = db.push("test", ckpt, self.get_version())
        self.assertTrue(weights_eq(db.get(name="test")["state_dict"], ckpt["state_dict"]))
        self.assertListEqual(db.list_versions("test"), [v1])
        self.assertListEqual(db.list_tags("test"), [("latest", v1)])

        db.clear("test")
        self.assertListEqual(db.list_versions("test"), [])
        self.assertListEqual(db.list_tags("test"), [])

    def test_latest(self):
        db = make_test_db()

        latest_param = None
        latest_version = None
        for i in range(10):
            latest_param = get_test_param()
            latest_version = db.push("test", latest_param, self.get_version())
        self.assertTrue(weights_eq(db.get(name="test")["state_dict"], latest_param["state_dict"]))
        self.assertListEqual(db.list_tags("test"), [("latest", latest_version)])

    def test_gc(self):
        db = make_test_db()

        for i in range(10):
            _ = db.push("test", get_test_param(), version=self.get_version())
        self.assertEqual(len(db.list_versions("test")), 10)

        db.gc("test", max_untagged_version_count=5)
        self.assertEqual(len(db.list_versions("test")), 6)  # Excluding "latest".

        for i in range(10):
            _ = db.push("test", get_test_param(), version=self.get_version(), tags=f"tag{i}")
        self.assertEqual(len(db.list_versions("test")), 16)

        db.gc("test", max_untagged_version_count=5)
        self.assertEqual(len(db.list_versions("test")), 15)  # 10 with tags, and 5 untagged.

    def test_has_tag(self):
        db = make_test_db()
        self.assertFalse(db.has_tag("test", "latest"))

        _ = db.push("test", get_test_param(), version=self.get_version())
        self.assertTrue(db.has_tag("test", "latest"))

    def test_push_version(self):
        db = make_test_db()
        ckpt = get_test_param()
        v = self.get_version()
        v1 = db.push("test", ckpt, version=v)
        self.assertTrue(weights_eq(db.get(name="test")["state_dict"], ckpt["state_dict"]))
        self.assertTrue(weights_eq(db.get(name="test", identifier=v)["state_dict"], ckpt["state_dict"]))

    def test_purge(self):
        db = make_test_db()
        _ = db.push("test", get_test_param(), version=self.get_version())
        db.get("test")
        db.purge(experiment_name="test_exp", trial_name="test_run")
        self.assertRaises(FileNotFoundError, db.get, "test")

    def test_list_names(self):
        db = make_test_db()
        db.push("test1", get_test_param(), version=str(0))
        db.push("test2", get_test_param(), version=str(0))
        self.assertSetEqual({"test1", "test2"}, set(db.list_names()))


class OSWorkaroundTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.load = mock.MagicMock(side_effect=[OSError, ""])
        os.readlink = mock.MagicMock(side_effect=[OSError, ""])
        os.symlink = mock.MagicMock()
        shutil.move = mock.MagicMock()

    def test_workaround_os_error(self):
        db = make_test_db()
        db.push("test", get_test_param(), "123")
        db.get("test")
        db.tag("test", "latest", "new_tag")


class MetadataParameterDBTest(unittest.TestCase):

    def setUp(self):
        self.__tmp = tempfile.TemporaryDirectory()
        FilesystemParameterDB.ROOT = os.path.join(self.__tmp.name, "checkpoints")
        db = make_test_db(type_="metadata")
        try:
            db.clear("test")
        except FileNotFoundError:
            pass

    def test_trivial(self):
        db = make_test_db(type_="metadata")
        ckpt = get_test_param()
        v1 = db.push("test", ckpt, version="12023", metadata={"win": 0})
        self.assertTrue(
            weights_eq(
                db.get(name="test", identifier={"$match": {
                    "version": "test_version"
                }})["state_dict"], ckpt["state_dict"]))
        self.assertListEqual(db.list_versions("test"), [v1])
        self.assertListEqual(db.list_tags("test"), [("latest", v1)])

        db.get(name="test", identifier={"$match": {"version": "test_version"}})

        db.clear("test")

    def test_get_fail(self):
        db = make_test_db(type_="metadata")
        self.assertRaises(ValueError,
                          lambda: db.get(name="test", identifier={"$match": {
                              "version": "not_version"
                          }}))
        ckpt = get_test_param()
        v1 = db.push("test", ckpt, version="12023", metadata={"win": 0})
        self.assertTrue(
            weights_eq(
                db.get(name="test", identifier={"$match": {
                    "loss": 0
                }})["state_dict"], ckpt["state_dict"]))

    def test_update(self):
        db = make_test_db(type_="metadata")
        ckpt = get_test_param()
        v1 = db.push("test", ckpt, version="12023", metadata={"win": 0})
        self.assertRaises(FileNotFoundError, lambda: db.update_metadata("test", "32021", {"win": 0}))
        db.update_metadata("test", version="12023", metadata={"win": 1})
        wait_network()
        v1_metadata = db._get_metadata("test", "12023")
        self.assertEqual(v1_metadata["win"], 0.5)
        db.clear("test")
        self.assertRaises(FileNotFoundError, lambda: db._get_metadata("test", "12023"))

    def test_first_update(self):
        db = make_test_db(type_="metadata")
        ckpt = get_test_param()
        v1 = db.push("test", ckpt, version="12023", tags="random_test")
        db.update_metadata("test", version="12023", metadata={"win": 0})
        v1_metadata = db._get_metadata("test", "12023")
        self.assertEqual(v1_metadata["win"], 0)
        db.update_metadata("test", version="12023", metadata={"win": 1})
        v1_metadata = db._get_metadata("test", "12023")
        self.assertEqual(v1_metadata["win"], 0.5)
        db.clear("test")

    def test_get_param(self):
        db = make_test_db(type_="metadata")
        ckpt0, ckpt1, ckpt2 = get_test_param(), get_test_param(), get_test_param()
        v0 = db.push("test", ckpt0, version="0", metadata={"win": 0, "return": 50})
        v1 = db.push("test", ckpt1, version="1", metadata={"win": 0.5, "return": 200})
        v2 = db.push("test", ckpt2, version="2", metadata={"win": 1, "return": 10})
        should_be_v0 = db.get(name="test", identifier={"$match": {"md.win": {"$lt": 0.1}}})
        self.assertTrue(weights_eq(should_be_v0["state_dict"], ckpt0["state_dict"]))
        should_be_v1 = db.get("test", {"$match": {"md.win": {"$gt": 0.4, "$lt": 0.6}}})
        self.assertTrue(weights_eq(should_be_v1["state_dict"], ckpt1["state_dict"]))
        should_be_v1 = db.get("test", {"$match": {"md.win": {"$gte": 0.5}, "md.return": {"$gt": 150}}})
        self.assertTrue(weights_eq(should_be_v1["state_dict"], ckpt1["state_dict"]))
        can_be_any = db.get("test", {"$match": {}})
        self.assertTrue(
            any([weights_eq(can_be_any["state_dict"], c["state_dict"]) for c in [ckpt0, ckpt1, ckpt2]]))
        self.assertTrue(
            any([weights_eq(can_be_any["state_dict"], c["state_dict"]) for c in [ckpt0, ckpt1, ckpt2]]))

        should_be_v1 = db.get("test", [{"$sort": {"md.return": -1}}, {"$limit": 1}])
        self.assertTrue(weights_eq(should_be_v1["state_dict"], ckpt1["state_dict"]))
        should_be_v0 = db.get("test", [{
            "$match": {
                "md.return": {
                    "$gt": 20
                }
            }
        }, {
            "$sort": {
                "md.win": 1
            }
        }, {
            "$limit": 1
        }])
        self.assertTrue(weights_eq(should_be_v0["state_dict"], ckpt0["state_dict"]))
        db.clear("test")

    def test_gc(self):
        db = make_test_db(type_="metadata")

        ckpt0, ckpt1, ckpt2 = get_test_param(), get_test_param(), get_test_param()
        v0 = db.push("test", ckpt0, version="0", metadata={"win": 0, "return": 50})
        v1 = db.push("test", ckpt1, version="1", metadata={"win": 0.5, "return": 200})
        v2 = db.push("test", ckpt2, version="2", metadata={"win": 1, "return": 10})
        query = [{"$match": {"md.return": {"$gt": 5}}}, {"$sort": {"md.win": 1}}, {"$limit": 1}]
        should_be_v0 = db.get("test", query)
        self.assertTrue(weights_eq(should_be_v0["state_dict"], ckpt0["state_dict"]))
        db.gc(name="test", max_untagged_version_count=1)
        # GC keeps a latest version(v2) and one recent version(v1)
        should_be_v1 = db.get("test", query)
        self.assertTrue(weights_eq(should_be_v1["state_dict"], ckpt1["state_dict"]))
        self.assertRaises(FileNotFoundError, db.get, "test", "0")

    def test_fcp_mongodb_query(self):
        db = make_test_db(type_="metadata")
        ckpt0, ckpt1, ckpt2, ckpt3, ckpt4 = get_test_param(), get_test_param(), get_test_param(
        ), get_test_param(), get_test_param()
        v0 = db.push("test", ckpt0, version="0", metadata={"episode_return": 0.2, "return": 50})
        v1 = db.push("test", ckpt1, version="1", metadata={"episode_return": 0.5, "return": 200})
        v2 = db.push("test", ckpt2, version="2", metadata={"episode_return": 1, "return": 150})
        v3 = db.push("test", ckpt3, version="3", metadata={"episode_return": 10, "return": 0})
        v4 = db.push("test", ckpt4, version="4", metadata={"episode_return": 20, "return": 300})
        q1 = [
            {
                "$group": {
                    "_id": "null",
                    "max_score": {
                        "$max": "$md.episode_return"
                    },
                    "data": {
                        "$push": "$$ROOT"
                    }
                }
            },
            {
                "$unwind": "$data"
            },
            {
                "$addFields": {
                    "diff": {
                        "$abs": {
                            "$subtract": ["$data.md.episode_return", {
                                "$divide": ["$max_score", 2]
                            }]
                        }
                    }
                }
            },
            {
                "$sort": {
                    "diff": 1
                }
            },
            {
                "$limit": 1
            },
            {
                "$addFields": {
                    "version": "$data.version"
                }
            },
        ]
        should_be_v3 = db.get("test", q1)
        self.assertTrue(weights_eq(should_be_v3["state_dict"], ckpt3["state_dict"]))

        q2 = [{"$sort": {"md.episode_return": -1}}, {"$limit": 1}]
        q3 = [{"$sort": {"md.episode_return": 1}}, {"$limit": 1}]
        q_total = [{
            "$facet": {
                "min": q1,
                "max": q2,
                "avg": q3,
            }
        }, {
            "$unwind": "$min"
        }, {
            "$unwind": "$max"
        }, {
            "$unwind": "$avg"
        }, {
            "$addFields": {
                "version": ["$min.version", "$max.version", "$avg.version"]
            }
        }, {
            "$unwind": "$version"
        }]
        results = [db.get("test", q_total) for _ in range(32)]
        self.assertTrue(any([weights_eq(r["state_dict"], ckpt0["state_dict"]) for r in results]))
        self.assertTrue(any([weights_eq(r["state_dict"], ckpt3["state_dict"]) for r in results]))
        self.assertTrue(any([weights_eq(r["state_dict"], ckpt4["state_dict"]) for r in results]))
        for r in results:
            self.assertTrue(
                any([
                    weights_eq(r["state_dict"], ckpt0["state_dict"]),
                    weights_eq(r["state_dict"], ckpt3["state_dict"]),
                    weights_eq(r["state_dict"], ckpt4["state_dict"])
                ]))


class MultiCastParamServerTest(unittest.TestCase):

    def setUp(self) -> None:
        distributed.base.name_resolve.reconfigure("memory", log_events=True)
        self.experiment_name = "test_exp"
        self.trial_name = "test_trial"
        distributed.base.name_resolve.clear_subtree(
            base.names.trial_root(experiment_name=self.experiment_name, trial_name=self.trial_name))
        self.worker_info = api.config.WorkerInformation(experiment_name=self.experiment_name,
                                                        trial_name=self.trial_name)
        distributed.system.parameter_db.PARAMETER_CLIENT_RECV_TIMEO = 100

    def test_server_tcp(self):
        base.network.gethostname = mock.MagicMock(return_value="localhost")
        server = distributed.system.parameter_db.make_server(api.config.ParameterServer(
            type_=api.config.ParameterServer.Type.MultiCast,
            backend_db=api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM)),
                                                             worker_info=self.worker_info)

        threading.Thread = mock.MagicMock()
        call_back_mock1 = mock.MagicMock()
        call_back_mock2 = mock.MagicMock()
        client1 = distributed.system.parameter_db.make_client(
            spec=api.config.ParameterServiceClient(type_=api.config.ParameterServiceClient.Type.MultiCast),
            worker_info=self.worker_info)
        client1.subscribe(self.experiment_name,
                          self.trial_name,
                          "test_policy",
                          "latest",
                          callback_fn=call_back_mock1)
        server.update_subscription()
        self.assertEqual(len(server.serving_instances), 1)
        client2 = distributed.system.parameter_db.make_client(
            spec=api.config.ParameterServiceClient(type_=api.config.ParameterServiceClient.Type.MultiCast),
            worker_info=self.worker_info)
        client2.subscribe(self.experiment_name,
                          self.trial_name,
                          "test_policy",
                          "latest",
                          callback_fn=call_back_mock2)
        other_db = distributed.system.parameter_db.make_db(
            api.config.ParameterDB(type_=api.config.ParameterDB.Type.FILESYSTEM),
            worker_info=api.config.WorkerInformation(experiment_name="test_exp", trial_name="test_trial"))
        server.update_subscription()
        self.assertEqual(len(server.serving_instances), 1)
        params = torch.randn(10)
        other_db.push("test_policy", params, version=str(0))
        for k, v in client1.subscriptions.items():
            v.run()
        for k, v in client2.subscriptions.items():
            v.run()
        wait_network()
        server.serve_all()
        wait_network()

        client1.poll()
        client2.poll()
        torch.testing.assert_close(params, call_back_mock1.call_args_list[0][0][0])
        torch.testing.assert_close(params, call_back_mock2.call_args_list[0][0][0])


if __name__ == "__main__":
    unittest.main()

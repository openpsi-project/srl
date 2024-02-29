import tempfile
import unittest
import os
import shutil
import redis
import time

import distributed.base.name_resolve as name_resolve


class NfsNameResolveTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("nfs")
        self.test_dir = tempfile.TemporaryDirectory()
        name_resolve.NfsNameRecordRepository.RECORD_ROOT = os.path.join(self.test_dir.name, "name_resolve")

    def tearDown(self):
        shutil.rmtree(self.test_dir.name)
        self.test_dir.cleanup()


class RedisNameResolveTest(unittest.TestCase):

    def setUp(self):
        name_resolve.RedisNameRecordRepository.REDIS_HOST = "redis"
        name_resolve.RedisNameRecordRepository.REDIS_DB = 10
        name_resolve.RedisNameRecordRepository.KEEPALIVE_POLL_FREQUENCY = 0.01

        with redis.Redis(host=name_resolve.RedisNameRecordRepository.REDIS_HOST,
                         password=name_resolve.RedisNameRecordRepository.REDIS_PASSWORD,
                         db=name_resolve.RedisNameRecordRepository.REDIS_DB) as r:
            for key in list(r.keys()):
                r.delete(key)
        name_resolve.reconfigure("redis")

    def tearDown(self):
        name_resolve.reset()
        del name_resolve.DEFAULT_REPOSITORY

    def test_keepalive(self):
        name_resolve.add("foo1", "bar1", keepalive_ttl=0.1)
        name_resolve.add("foo2", "bar2", keepalive_ttl=0.1)
        name_resolve.DEFAULT_REPOSITORY._testonly_drop_cached_entry("foo1")

        time.sleep(0.2)
        with self.assertRaises(name_resolve.NameEntryNotFoundError):
            name_resolve.get("foo1")
        self.assertEqual(name_resolve.get("foo2"), "bar2")


if __name__ == '__main__':
    unittest.main()

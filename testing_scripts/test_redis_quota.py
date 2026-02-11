import unittest
import os
import sys
import time
import shutil

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager

class TestRedisQuota(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.qm = RedisQuotaManager(self.test_dir)
        
        # Manually inject a test model
        self.qm.redis.set("model:test-redis:rpm:limit", 10)
        self.qm.redis.set("model:test-redis:rpd:limit", 100)
        self.qm.redis.set("model:test-redis:config_file", "dummy.json")
        self.qm.redis.set("config:dummy.json:endpoint", "http://test")
        self.qm.redis.set("config:dummy.json:key", "testkey")
        
    def tearDown(self):
        self.qm.close()
        # Clean up dump file
        dump_file = os.path.join(self.test_dir, "redis_dump.json")
        if os.path.exists(dump_file):
            os.remove(dump_file)

    def test_atomic_update(self):
        # 1. Update quota
        self.qm.update_quota("test-redis", request_count=5)
        
        # 2. Verify value
        val = int(self.qm.redis.get("quota:test-redis:rpm:used"))
        self.assertEqual(val, 5)
        
        # 3. Verify TTL
        ttl = self.qm.redis.ttl("quota:test-redis:rpm:used")
        self.assertTrue(0 < ttl <= 60)

    def test_availability_check(self):
        # 1. Initially available (0/10)
        self.assertTrue(self.qm.is_model_available("test-redis"))
        
        # 2. Push to limit (9/10 = 90%)
        self.qm.update_quota("test-redis", request_count=9)
        
        # 3. Should be unavailable now (>= 90%)
        self.assertFalse(self.qm.is_model_available("test-redis"))

    def test_persistence(self):
        # 1. Set a value
        self.qm.redis.set("test_key", "test_val")
        self.qm._save_state()
        
        # 2. Create new instance
        qm2 = RedisQuotaManager(self.test_dir)
        val = qm2.redis.get("test_key").decode('utf-8')
        
        self.assertEqual(val, "test_val")
        qm2.close()

if __name__ == '__main__':
    unittest.main()

import time
import unittest


class TimeTrackingTestCase(unittest.TestCase):
    def setUp(self):
        self.test_start_time = time.time()
        print(f"\nTest start: {self.id()}")

    def tearDown(self):
        test_end_time = time.time()
        test_runtime = test_end_time - self.test_start_time
        minutes, seconds = divmod(test_runtime, 60)
        print(f"Test {self.id()}: {int(minutes)} min {int(seconds)} sec")


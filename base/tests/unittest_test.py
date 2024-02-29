import unittest


class UnittestDemo(unittest.TestCase):

    def setUp(self):
        # Run setup code here for each test case.
        pass

    def tearDown(self):
        # Clean up environment here.
        pass

    def test_foo(self):
        self.assertTrue(2 > 1)
        self.assertEqual(42, 42)
        self.assertSetEqual({1, 3, 2}, {1, 2, 3})


if __name__ == '__main__':
    unittest.main()

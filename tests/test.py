import unittest
unittest.TestLoader.sortTestMethodsUsing = None

class TestTest(unittest.TestCase):

    def test_test(self):
        print("test")

if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestTest('test_test'))
    runner = unittest.TextTestRunner()
    runner.run(suite)

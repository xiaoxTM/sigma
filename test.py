import unittest
import sys
import os.path
base = os.path.dirname(os.path.realpath(__file__))
base_path = base.rsplit('/',1)[0]
print('base path:',base_path)
sys.path.append(base_path)

if __name__ == '__main__':
    tests = unittest.TestLoader().discover('tests')
    testrunner = unittest.TextTestRunner(verbosity=2)

    for test in tests:
        testcases = list(test.__iter__())
        if len(testcases) > 0:
            print('='*20)
            testrunner.run(test)

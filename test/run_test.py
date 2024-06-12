import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import colors
import unittest


if __name__ == '__main__':
    tests = unittest.TestLoader().discover('sigma')
    testrunner = unittest.TextTestRunner(verbosity=2)

    for test in tests:
        testcases = list(test.__iter__())
        if len(testcases) > 0:
            print('='*20)
            testrunner.run(test)

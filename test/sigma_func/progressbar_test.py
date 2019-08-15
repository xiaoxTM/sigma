import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import helpers
import time

progressor, total_iterations = helpers.line(None,
                                            feedbacks=True,
                                            brief=True,
                                            epochs=None,
                                            iterations=10,
                                            timeit=True)
progressor = progressor()
global_idx, element, epoch, iteration = next(progressor)
while epoch < 1:
    time.sleep(0.5)
    global_idx, element, epoch, iteration = progressor.send('test')

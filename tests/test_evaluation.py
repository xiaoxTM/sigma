import unittest

from sigma.metrics import evaluation as evals
import numpy as np


class EvaluationTest(unittest.TestCase):
    def test_hitmap(self):
        #predict=np.random.randint(0, 40, size=50)
        #label = np.random.randint(0, 40, size=50)
        result = [1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1]
        with self.subTest():
            predict = np.asarray([0,3,2,3,1], dtype=np.int32)
            label   = np.asarray([0,2,3,3,2], dtype=np.int32)
            matrix = evals.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,0,1],[0,1,0,0]], dtype=np.int32)
            label   = np.asarray([0,2,3,3,2], dtype=np.int32)
            matrix = evals.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([0,3,2,3,1], dtype=np.int32)
            label   = np.asarray([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0]], dtype=np.int32)
            matrix = evals.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,0,1],[0,1,0,0]], dtype=np.int32)
            label   = np.asarray([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0]], dtype=np.int32)
            matrix = evals.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)

    def test_accuracy(self):
        with self.subTest():
            pred = np.array([1,3,2,4,5,4,7], dtype=np.int32)
            labels = np.array([1,4,2,4,6,7,7], dtype=np.int32)
            accuracy = evals.accuracy(pred, labels)
            self.assertEqual(accuracy, 4.0 / 7.0)
        with self.subTest():
            pred = np.array([[0,0,1,0],
                                               [0,1,0,0],
                                               [0,1,0,0],
                                               [0,0,0,1],
                                               [1,0,0,0],
                                               [1,0,0,0]])
            labels = np.array([[0,0,1,0],
                                                  [0,1,0,0],
                                                  [0,0,1,0],
                                                  [0,1,0,0],
                                                  [0,0,0,1],
                                                  [1,0,0,0]])
            accuracy = evals.accuracy(pred, labels)
            self.assertEqual(accuracy, 3.0 / 6.0)

    def test_accuracy_from_hitmap(self):
        hitmap = np.array([[10, 1, 0, 1, 0],
                                                [  2, 6, 0, 2, 0],
                                                [  1, 0, 8, 0, 1],
                                                [  0, 1, 0, 9, 0],
                                                [  2, 3, 0, 1,10]])
        acc, apc = evals.accuracy_from_hitmap(hitmap, label='row')
        _, apc_c = evals.accuracy_from_hitmap(hitmap, label='col')
        with self.subTest():
            self.assertEqual(acc, (10.0+6+8+9+10)  / (12.0+10+10+10+16))
        with self.subTest():
            self.assertListEqual(apc.tolist(), [10.0/15, 6.0/11, 1.0, 9.0/13, 10.0/11])
        with self.subTest():
            self.assertListEqual(apc_c.tolist(), [10.0/12, 6.0/10,8.0/10, 9.0/10,10.0/16])

    def test_mean_perclass_accuracy(self):
        with self.subTest():
            data = [0.2, 0.3, 0.45, 0.34, 0.76, 0.43]
            res = evals.mean_perclass_accuracy(data)
            self.assertEqual(res, (0.2+0.3+0.45+0.34+0.76+0.43) / 6)
        with self.subTest():
            data = [0.2, 0.3, 0.45, np.nan, 0.76, np.nan]
            res = evals.mean_perclass_accuracy(data)
            self.assertEqual(res, (0.2+0.3+0.45+0.76)/4)
from sigma.utils.params import *
import unittest

@defaultable
def test1(a,b=2):
    return a,b

def test2(a,b,c,d=2,e=3,f=4):
    return a,b,c,d,e,f

class ParamsTest(unittest.TestCase):
    def test_defaultable(self):
        with defaults(b=3):
            a,b = test1(a=4)
            self.assertEqual(b,3)

    def test_split_params(self):
        params = split_params('[10;20]')
        self.assertListEqual(params, ['10','20'])

    def test_split_by_comma(self):
        params = split_by_comma('T_max=20,steps=[10,20,30],splits="\[",eta_min=20')
        self.assertEqual(params[0],'T_max=20')
        self.assertEqual(params[1],'steps=[10,20,30]')
        self.assertEqual(params[2],'splits="["')
        self.assertEqual(params[3],'eta_min=20')
        params = split_by_comma('ca(T_max=20,steps=[10,20,30],splits="\[",eta_min=20)')
        self.assertEqual(len(params),4)
        self.assertEqual(params[0],'ca(T_max=20')
        self.assertEqual(params[3],'eta_min=20)')

    def test_parse_params(self):
        params = parse_params('sgd(momentum=0.9,weight_decay=0.001)')
        self.assertEqual(params[0], 'sgd')
        self.assertDictEqual(params[1], {'momentum':0.9,'weight_decay':0.001})

    def test_expand_param(self):
        self.assertListEqual(expand_param(20,3), [20,20,20])

    def test_expand_params(self):
        o1,o2 = expand_params([[20],[30,4]], 3)
        self.assertListEqual(o1, [20,20,20])
        self.assertListEqual(o2, [30,4,4])

    def test_merge_args(self):
        merged = merge_args([20,30,40],dict(e=2,d=4,f=0),test2)
        self.assertDictEqual(merged, {'a':20,'b':30,'c':40,'d':4,'e':2,'f':0})

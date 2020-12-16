import numpy as np
import os
import subprocess
import sys
import copy
import logging
import random_sampler

class GeneticOptimizer():
    def __init__(self, num_pops, num_gene, selection='sus'):
        self._num_pops = num_pops # number of population, i.e., the size/capacity of population
        self._num_gene = num_gene # size of gene of each individal
        self._pops = None
        self._select = random_sampler.sus

    def init(self):
        self._pops = np.random.randint(0,2,size=(self._num_pops,self._num_gene))
        fitness = self.fit()
        return fitness

    def best_gene(self, fitness):
        return fitness.max(),self._pops[fitness.argmax()]

    def select(self, fitness):
        return self._select(fitness)

    def fit(self):
        raise NotImplementedError('`fit` function for GeneticOptimizer is not implemented')

    def cross_over(self, parents, rate=0.25):
        # parents: np.ndarray with shape:
        #        number-of-parents x 2
        #    where number-of-parents generally
        #    is half of the population
        assert isinstance(parents, np.ndarray)
        assert len(parents.shape)==2
        rates = np.random.rand(parents.shape[0])
        idx = np.argwhere(rates < rate)
        pos0 = np.random.randint(1,self._num_gene,parents.shape[0])
        pos1 = np.random.randint(1,self._num_gene,parents.shape[0])
        ends = np.maximum(pos0, pos1)
        begs = np.minimum(pos0, pos1)
        # shape of p1, p2:
        #     number-of-parents//2 x number-of-gene
        p0 = copy.deepcopy(self._pops[parents[:,0]])
        p1 = copy.deepcopy(self._pops[parents[:,1]])
        for beg,end in zip(begs, ends):
            p0[idx,beg:end] = self._pops[parents[:,1]][idx,beg:end]
            p1[idx,beg:end] = self._pops[parents[:,0]][idx,beg:end]
        return np.concatenate((p0, p1),axis=0)

    def mutate(self, offsprings, rate=0.01):
        # offsprings: np.ndarray with shape:
        #      number-of-pops x number-of-gene
        rates = np.random.rand(*offsprings.shape)
        idx = np.where(rates < rate)
        #for i,os in zip(idx,offsprings):
        #    os[i] = (os[i] + 1) % 2
        offsprings[idx] = (offsprings[idx] + 1) % 2
        return offsprings

    def run(self, num_generations, crate, mrate):
        logging.debug('initializing ...')
        fitness = self.init()
        best_fitness, best_gene = self.best_gene(fitness)
        for generation in range(num_generations):
            logging.debug('    {}-th generation'.format(generation))
            logging.debug('    selecting ...')
            parents=self.select(fitness)
            logging.debug('    crossing over ...')
            offsprings=self.cross_over(parents,crate)
            logging.debug('    mutating ...')
            self.mutate(offsprings,mrate)
            logging.debug('    fitting ...')
            fitness = self.fit()
            best_fitness, best_gene = self.best_gene(fitness)
        return best_fitness, best_gene

    def __call__(self, num_generations, crate, mrate):
        return self.run(num_generations, crate, mrate)

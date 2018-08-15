from deap import base, creator
from deap import tools

import random

from kddcup.core.data import KddCupData
from kddcup.core.model import KddCupModel
from kddcup.core.constants import kddcup_properties as properties


creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

class Individual(creator.Individual):
    def __init__(self, inputs, targets=[], brain=[], *args, **kwargs):
        super(Individual, self).__init__(*args, **kwargs)
        self[:] = inputs
        if targets:
            self.targets = targets
        else:
            self.targets = ['normal.', 'other.']
        if not brain:
            self.brain = [{'neurons': 1, 'activation': 'relu'}]
        else:
            self.brain = brain


    def mate(self, other):
        return tools.cxTwoPoint(self, other)

    def mutate(self, mutator, union_difference_prob=.5): # We will use union or difference 50% of the time
        mutated = []
        if random.random() < union_difference_prob:
            mutated = list(set(self).union(set(mutator)))
        else:
            mutated = list(set(self).difference(set(mutator)))
        return mutated

    def evaluate(self, train_env, test_env, surface=None, hops=None, 
                 iterations=1, verbose=0, test_surface=None, test_hops=None):

        self.fitness.values = KddCupModel(inputs=self, targets=self.targets, layers=self.brain)\
                .train(KddCupData(train_env, nrows=surface, batch=hops), epochs=iterations, verbose=verbose)\
                .test(KddCupData(test_env, nrows=test_surface, batch=test_hops))\
                .print()\
                ['loss', 'accuracy']

class Population(list):
    def __init__(self, space=properties[:-1], size=41, targets=[], brain=[]):
        self.space = space
        self[:] = [Individual(self.__random_inputs(), targets=targets, brain=brain) for i in range(size)]

    def __random_inputs(self, size=None):
        if not size:
            size = random.randint(1, len(self.space))
        properties = random.sample(self.space, k=size)
        return properties

    def evaluation(self, train_env, test_env, surface=None, hops=None,
                   iterations=1, verbose=0, test_surface=None, test_hops=None):
        for ind in self:
            ind.evaluate(train_env=train_env, test_env=test_env, 
                        surface=surface, iterations=iterations, 
                        verbose=verbose, test_surface=test_surface)
        self.train_env = train_env
        self.test_env = test_env
        return self

    def __re_evaluation(self):
        invalid_ind = [ind for ind in self if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.evaluate(train_env=self.train_env, test_env=self.test_env)
        return self


    def selection(self):
        offspring = tools.selTournament(self, k=len(self), tournsize=3)
        self[:] = base.Toolbox().clone(offspring)
        return self

    def crossover(self, CXPB=0.5):
        for child1, child2 in zip(self[::2], self[1::2]):
            if random.random() < CXPB:
                child1.mate(child2)
                del child1.fitness.values
                del child2.fitness.values
        return self

    def mutation(self, MUTPB=0.2):
        for mutant in self:
            if random.random() < MUTPB:
                mutant.mutate(self.__random_inputs(size=5))
                del mutant.fitness.values
        return self.__re_evaluation()


    def evolve(self, train_env, test_env, surface=None, hops=None,
                iterations=1, verbose=0, test_surface=None, test_hops=None, 
                MUTPB=0.2, CXPB=0.5, NGEN=2):
        if NGEN == 0:
            return self 
        print("\nGENERATION ", NGEN, '\n')
        print("    Population: ", [len(p) for p in self])
        return self.evaluation(train_env=train_env, test_env=test_env,
                                surface=None, hops=None, iterations=1,
                                verbose=0, test_surface=None, test_hops=None)\
                   .selection()\
                   .crossover(CXPB=CXPB)\
                   .mutation(MUTPB=MUTPB)\
                   .evolve(train_env, test_env, NGEN=NGEN-1)

    
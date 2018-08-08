from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense
from keras import Sequential

import random

from deap import base, creator
from deap import tools

from data import KddCupData
from model import KddCupModel

env = KddCupData(filename='data/kddcup.data_10_percent_corrected', batch_size=10000)
env_test = KddCupData(filename='data/corrected')
properties = env.properties
# targets = env.attack_types

brain = [{'neurons': 8, 'activation': 'relu'},
         {'neurons': 4, 'activation': 'relu'}]
# brain = KddCupModel(inputs=properties, targets=targets, layers=cells)


def random_inputs(the_set=properties, size=None):
    if not size:
        size = random.randint(1, len(the_set))
    properties = random.sample(the_set, k=size)
    return properties

# CREATE TYPES
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# FITNESS OPERATOR

# individual = random_inputs()
# print(individual)

def evaluate(individual):
    model = KddCupModel(inputs=individual, layers=brain)
    model.train(env, batch_size=256, epochs=10, verbose=1)
    _, acc = model.test(env_test)
    return acc,

# MUTATION OPERATOR
def mutate(individual, union_difference_prob=.5): # We will use union or difference 50% of the time
    mutator = random_inputs(individual, size=5)
#     print('mutator: ', mutator, '\n', len(mutator))
    mutated = []
    if random.random() < union_difference_prob:
        mutated = list(set(individual).union(set(mutator)))
    else:
        mutated = list(set(individual).difference(set(mutator)))
    return mutated

# ALGORITHM

CXPB, MUTPB, NGEN = 0.5, 0.2, 50

toolbox = base.Toolbox()

pop = None

for g in range(NGEN):
    # INITIALISATION
    
    print("\nGENERATION ", g, '\n')

#     toolbox.register("attribute", lambda : random.sample(range(1, 100), random.randint(1,MAX_IND_SIZE)))
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda : random_inputs(size=random.randint(1, 41)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
   
    # ALGORITHM
    
    if not pop:
        pop = toolbox.population(n=41)
    print("    Population: ", [len(p) for p in pop])
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    
    
    # Unregistering tools
    toolbox.unregister("individual")
    toolbox.unregister("population")
    toolbox.unregister("mate")
    toolbox.unregister("mutate")
    toolbox.unregister("select")
    toolbox.unregister("evaluate")
    

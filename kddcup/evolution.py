from deap import base, creator
from deap import tools

import random

from core import KddCupData, KddCupModel

properties = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', \
              'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', \
              'num_failed_logins', 'logged_in', 'num_compromised', \
              'root_shell', 'su_attempted', 'num_root', 'num_file_creations', \
              'num_shells', 'num_access_files', 'num_outbound_cmds', \
              'is_host_login', 'is_guest_login', 'count', 'srv_count', \
              'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', \
              'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', \
              'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', \
              'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', \
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', \
              'dst_host_srv_serror_rate', 'dst_host_rerror_rate', \
              'dst_host_srv_rerror_rate', 'attack_type']

brain = [{'neurons': 8, 'activation': 'relu'},
         {'neurons': 4, 'activation': 'relu'}]

targets = ['normal.', 'other.']


# RANDOM INDIVIDUAL
def random_inputs(the_set=properties[:-1], size=None):
    if not size:
        size = random.randint(1, len(the_set))
    properties = random.sample(the_set, k=size)
    return properties


# FITNESS OPERATOR
def evaluate(individual):
    targets = ['normal.', 'other.']
    return KddCupModel(inputs=individual, targets=targets, layers=brain)\
                .train(KddCupData(filename='data/kddcup.data_10_percent_corrected', nrows=10000), epochs=5)\
                .test(KddCupData(filename='data/corrected', nrows=1000))\
                ['accuracy'],

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
def evolution():

    # CREATE TYPES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 2

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
    return pop

if __name__ == '__main__':
    print(evolution())

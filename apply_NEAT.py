################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy
import neat
sys.path.insert(0, 'evoman')
from environment import Environment
from specialist_controller import NEAT_Controls
import csv
import numpy as np



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemy = 2

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=NEAT_Controls(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def eval_genomes(genomes, config):
    fitness_array = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.fitness = simulation(env, genome)
        fitness_array.append(genome.fitness)

    #save the fitness data
    with open('testing_data_NEAT_3.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([np.max(fitness_array),
                         np.mean(fitness_array),
                         np.std(fitness_array)])


generations = 2
def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play mega man.
    It uses the config file named config-feedforward.txt
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    with open('testing_data_NEAT_3.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([enemy, generations])

    # Run for generations.
    winner = p.run(eval_genomes, generations)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))





if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

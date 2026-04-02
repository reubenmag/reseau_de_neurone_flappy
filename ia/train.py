import sys
import os
import neat
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
N_GENERATIONS = 100


def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    state = env.reset()
    done = False

    while not done:
        output = net.activate(state)
        action = 1 if output[0] > 0.5 else 0
        state, reward, done = env.step(action)

    return env.frames + 500 * env.score

generation = [0]

def eval_genomes(genomes, config):
    best_gen_fitness = -1
    best_gen_genome = None
    
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)
        if genome.fitness > best_gen_fitness:
            best_gen_fitness = genome.fitness
            best_gen_genome = genome

    path = os.path.join(os.path.dirname(__file__), 'checkpoints', f'best_gen_{generation}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(best_gen_genome, f)
    
    generation[0] =+1

def plot_stats(stats, output_path):
    generations = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    species_counts = [len(sizes) for sizes in stats.get_species_sizes()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(generations, best_fitness, label='Fitness maximale')
    ax1.plot(generations, avg_fitness, label='Fitness moyenne')
    ax1.set_ylabel('Fitness')
    ax1.legend()

    ax2.plot(generations, species_counts, color='green', label='Nombre d\'especes')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Especes')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Courbe sauvegardee dans {output_path}")


def run():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'checkpoints'), exist_ok=True)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    best = population.run(eval_genomes, N_GENERATIONS)

    genome_path = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
    with open(genome_path, 'wb') as f:
        pickle.dump(best, f)

    plot_stats(stats, os.path.join(os.path.dirname(__file__), 'fitness_courbe_relu1.png'))

    print(f"\nMeilleur genome sauvegarde dans {genome_path}")
    print(f"Fitness du meilleur genome : {best.fitness:.1f}")


if __name__ == '__main__':
    run()
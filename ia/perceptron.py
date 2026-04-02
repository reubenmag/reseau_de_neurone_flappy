import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

class Perceptron:
    def __init__(self, n_inputs=5):
        self.weights = np.random.uniform(-1, 1, n_inputs)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        # Sigmoïde : ramène la sortie entre 0 et 1
        return np.tanh(z)

    def decide(self, x):
        return 1 if self.forward(x) > 0 else 0



def run(n_games=100):
    env = FlappyBirdEnv()
    net = Perceptron()
    scores = []

    best_score = -1
    best_weights = None
    best_bias = None

    for _ in range(100):
        net = Perceptron()
        state = env.reset()
        done = False
        while not done:
            action = net.decide(state)
            state, reward, done = env.step(action)
        if env.score > best_score:
            best_score = env.score
            best_weights = net.weights.copy()
            best_bias = net.bias

    print(f"Meilleur score : {best_score}")
    print(f"Poids : {best_weights}")
    print(f"Biais : {best_bias}")




if __name__ == '__main__':
    run()
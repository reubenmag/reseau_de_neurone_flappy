# Entraîner un réseau de neurones à jouer à Flappy Bird

---

```
A LIRE AVANT DE COMMENCER
INTERDICTION TOTALE DES LLM pour de la production de code


L'usage de tout outil d'IA générative est STRICTEMENT INTERDIT
pendant cette session de 7 heures. Sont interdits sans exception :
ChatGPT, Claude, Copilot, Mistral, Gemini, Perplexity, Codeium,
Tabnine, et tout autre LLM ou assistant IA.

Cela inclut : génération de code, débogage assisté, et toute autre forme
d'assistance par IA, directe ou indirecte dans la production de code.

Ce document a été conçu pour vous guider sans assistance externe.
Toute utilisation détectée sera considérée comme une fraude.
```

---

## Table des matières

1. [Étape 0 — Prérequis et mise en place (30 min)](#étape-0)
2. [Étape 1 — Bot naïf : règles codées en dur (1 h)](#étape-1)
3. [Étape 2 — Premier réseau de neurones : perceptron sans entraînement (1 h 30)](#étape-2)
4. [Étape 3 — NEAT : neuroévolution de base (2 h)](#étape-3)
5. [Étape 4 — Amélioration de la fitness et des hyperparamètres (1 h 30)](#étape-4)
6. [Étape 5 — Visualisation et analyse (30 min)](#étape-5)
7. [Glossaire](#glossaire)

---

## Étape 0 — Prérequis et mise en place }

**Durée estimée : 30 minutes**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 0.1 Récupérer le code source

Le code source du projet est disponible à l'adresse suivante :

```
https://github.com/HorHakim/flappy_bird
```

Clonez le dépôt dans un dossier de votre choix :

```bash
git clone https://github.com/HorHakim/flappy_bird.git
cd flappy_bird
```

La structure obtenue est la suivante :

```text
flappy_bird/
├── game/
│   ├── main.py
│   ├── game_engine.py
│   ├── config.py
│   └── requirements.txt
├── ia/
│   ├── train.py
│   ├── play_ia.py
│   ├── neat_config.txt
│   └── requirements.txt
├── pedagogique.md
└── README.md
```

### 0.2 Vérifier la version de Python

Ouvrez un terminal et exécutez la commande suivante :

```bash
python --version
```

Sur certains systèmes, la commande est `python3` :

```bash
python3 --version
```

La version affichée doit être 3.9 ou supérieure. Si ce n'est pas le cas, téléchargez Python depuis `https://www.python.org/downloads/` et installez-le.

**ATTENTION.** Sur Windows, si la commande `python` ouvre le Microsoft Store au lieu d'afficher une version, Python n'est pas installé correctement. Suivez les instructions du site officiel pour l'installation sous Windows et cochez la case "Add Python to PATH" lors de l'installation.

### 0.3 Créer un environnement virtuel

Un environnement virtuel est un dossier isolé qui contient une installation Python et ses paquets, indépendamment du reste du système. Cela évite les conflits entre versions de bibliothèques.

**Linux / macOS :**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows :**

```bash
python -m venv venv
venv\Scripts\activate
```

Une fois activé, le nom de l'environnement `(venv)` apparaît au début de chaque ligne du terminal.

### 0.4 Installer les dépendances

Depuis la racine du dépôt, avec l'environnement virtuel activé :

```bash
pip install pygame numpy neat-python matplotlib
```

Vérifiez que l'installation s'est déroulée sans erreur. La sortie doit se terminer par une ligne du type :

```text
Successfully installed ...
```

**ATTENTION.** Si vous obtenez une erreur `ModuleNotFoundError` plus tard, c'est que la dépendance n'est pas installée dans votre environnement virtuel actif. Vérifiez que le préfixe `(venv)` est présent dans votre terminal avant de relancer `pip install`.

### 0.5 Comprendre les règles de Flappy Bird

Flappy Bird est un jeu simple dont les règles sont les suivantes :

- Un oiseau tombe vers le bas sous l'effet de la gravité.
- Le joueur appuie sur une touche (ESPACE ou clic gauche) pour faire sauter l'oiseau vers le haut.
- Des paires de tuyaux apparaissent depuis la droite et défilent vers la gauche. Chaque paire comporte un espace vide (le "gap") que l'oiseau doit traverser.
- Le score augmente d'un point à chaque paire de tuyaux franchie.
- La partie se termine si l'oiseau touche un tuyau, le sol ou le plafond.

### 0.6 Lancer le jeu humain

```bash
cd game
python main.py
```

Contrôles : ESPACE ou clic gauche pour sauter. ECHAP pour quitter.

**RESULTAT ATTENDU.** La fenêtre pygame s'ouvre. L'oiseau tombe sous l'effet de la gravité. Les tuyaux défilent de droite à gauche. Le score s'affiche en haut de l'écran.

### 0.7 Exercice pratique

Jouez quelques parties et notez les paramètres qui vous semblent influencer la difficulté : la vitesse des tuyaux, la taille du gap entre les tuyaux, la fréquence à laquelle de nouveaux tuyaux apparaissent. Ces paramètres sont définis dans `game/config.py`. Ouvrez ce fichier et identifiez les constantes correspondantes.

---

## Étape 1 — Bot naïf : règles codées en dur 

**Durée estimée : 1 heure**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 1.1 Qu'est-ce qu'un agent ?

Un agent est un programme qui observe l'état d'un environnement et choisit une action à effectuer. Dans ce projet, l'environnement est le jeu Flappy Bird. L'état est une description numérique de la situation actuelle du jeu. L'action est binaire : soit l'oiseau saute, soit il ne fait rien.

Avant d'entraîner quoi que ce soit, on peut écrire manuellement une stratégie : un ensemble de règles fixes qui décident quand sauter. C'est ce qu'on appelle un bot naïf ou bot à règles codées en dur.

### 1.2 L'interface FlappyBirdEnv

Le fichier `game/game_engine.py` expose une classe `FlappyBirdEnv`. Cette classe permet de faire tourner le jeu sans affichage graphique (mode "headless"), ce qui est indispensable pour entraîner une IA rapidement.

Elle fournit trois méthodes :

- `reset()` : réinitialise le jeu et retourne le vecteur d'état initial.
- `step(action)` : avance d'un frame. Prend une action en entrée (0 = ne rien faire, 1 = sauter) et retourne un triplet `(état, récompense, done)`. `done` vaut `True` si la partie est terminée.
- `get_state()` : retourne le vecteur d'état courant sans avancer le jeu.

### 1.3 Le vecteur d'état

Le vecteur d'état est une liste de 5 valeurs numériques, toutes normalisées entre -1 et 1 (ou 0 et 1 selon la valeur) :

1. Position verticale de l'oiseau divisée par la hauteur de l'écran.
2. Vélocité verticale de l'oiseau divisée par la vélocité maximale.
3. Distance horizontale au prochain tuyau divisée par la largeur de l'écran.
4. Différence entre la position de l'oiseau et le bord haut du gap, divisée par la hauteur de l'écran.
5. Différence entre le bord bas du gap et la position de l'oiseau, divisée par la hauteur de l'écran.

Les valeurs 4 et 5 sont positives si l'oiseau est à l'intérieur du gap, négatives s'il est en dehors.

### 1.4 Code du bot naïf

La règle est simple : si l'oiseau est au-dessus du centre du gap (valeur 4 négative, c'est-à-dire que l'oiseau est plus haut que le bord haut du gap), ne pas sauter. Sinon, sauter.

Créez un fichier `ia/naive_bot.py` avec le contenu suivant :

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

def naive_action(state):
    bird_above_gap_top = state[3]
    if bird_above_gap_top < 0:
        return 0
    return 1

def run(n_games=5):
    env = FlappyBirdEnv()
    scores = []

    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action = naive_action(state)
            state, reward, done = env.step(action)
        scores.append(env.score)
        print(f"Partie {i + 1} : score = {env.score}")

    print(f"\nScore moyen sur {n_games} parties : {sum(scores) / len(scores):.1f}")

if __name__ == '__main__':
    run()
```

Exécutez ce fichier depuis la racine du dépôt :

```bash
python ia/naive_bot.py
```

**RESULTAT ATTENDU.** Le terminal affiche le score de chaque partie. Le bot survive quelques secondes mais finit toujours par mourir. Le score moyen est généralement compris entre 0 et 5.

### 1.5 Ce qui ne va pas

Cette règle présente un défaut fondamental : elle ignore la vélocité de l'oiseau. L'oiseau ne s'arrête pas instantanément quand on cesse de sauter : il continue sur sa lancée. Si l'oiseau monte trop vite, il dépasse le bord haut du gap avant que la règle ne réagisse. Cela provoque des oscillations et des collisions.

La règle ne s'adapte pas non plus à la difficulté progressive : lorsque les tuyaux deviennent plus rapides ou le gap plus étroit, la même règle produit de moins bons résultats.

C'est précisément ce problème que le réseau de neurones devra résoudre en apprenant à prendre en compte plusieurs signaux simultanément.

### 1.6 Ce que vous devez observer

**CE QUE VOUS DEVEZ OBSERVER.** Le bot survit quelques secondes mais finit toujours par mourir, souvent en touchant un tuyau après une oscillation trop prononcée. Le comportement est répétitif et prévisible : l'oiseau monte, dépasse le centre du gap, cesse de sauter, redescend, puis saute de nouveau trop tard.

### 1.7 Pour aller plus loin

**POUR ALLER PLUS LOIN.** Modifiez la fonction `naive_action` pour qu'elle tienne compte de la vélocité (valeur 1 du vecteur d'état). Par exemple : si l'oiseau est légèrement en dessous du centre du gap mais monte déjà vite, ne pas sauter. Observez si le score moyen s'améliore.

### 1.8 Exercices pratiques

**Exercice 1.** La règle actuelle utilise uniquement `state[3]`. Ajoutez une condition sur `state[1]` (la vélocité) : ne sauter que si la vélocité est inférieure à un seuil de votre choix, par exemple 0.3. Testez avec plusieurs valeurs de seuil.

**Correction suggérée.**

```python
def naive_action(state):
    bird_above_gap_top = state[3]
    velocity = state[1]
    if bird_above_gap_top < 0 or velocity < -0.3:
        return 0
    return 1
```

**Exercice 2.** Faites tourner le bot sur 50 parties et calculez le score maximal, le score minimal et l'écart-type. Utilisez le module `statistics` de Python standard.

**Correction suggérée.**

```python
import statistics

scores = []
for i in range(50):
    state = env.reset()
    done = False
    while not done:
        action = naive_action(state)
        state, reward, done = env.step(action)
    scores.append(env.score)

print(f"Max : {max(scores)}, Min : {min(scores)}, Ecart-type : {statistics.stdev(scores):.2f}")
```

---

## Étape 2 — Premier réseau de neurones : perceptron sans entraînement 

**Durée estimée : 1 heure 30 minutes**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 2.1 Qu'est-ce qu'un neurone artificiel ?

Un neurone artificiel est une unité de calcul qui reçoit plusieurs valeurs en entrée, les combine, et produit une seule valeur en sortie.

Le calcul se déroule en deux étapes :

1. Somme pondérée : chaque entrée x_i est multipliée par un poids w_i, et on ajoute un biais b. Cela donne z = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b.
2. Activation : on applique une fonction d'activation f à z. La sortie du neurone est f(z).

Analogie : imaginez une balance à plusieurs plateaux. Chaque plateau reçoit une information (la position de l'oiseau, sa vitesse, etc.). Le poids représente l'importance accordée à cette information. Plus un poids est élevé, plus ce plateau penche la balance. Le biais représente le point d'équilibre de départ de la balance.

### 2.2 Les fonctions d'activation

Une fonction d'activation introduit une non-linéarité dans le calcul. Sans elle, un réseau de plusieurs couches serait équivalent à un seul neurone linéaire. Les trois fonctions les plus courantes sont :

- Sigmoïde : f(z) = 1 / (1 + exp(-z)). Produit une sortie entre 0 et 1. Utilisée pour les sorties binaires.
- Tanh : f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)). Produit une sortie entre -1 et 1.
- ReLU : f(z) = max(0, z). Produit 0 si z est négatif, z sinon. Très utilisée dans les couches cachées.

### 2.3 Le perceptron

Un perceptron est le réseau de neurones le plus simple. Il se compose d'une couche d'entrée et d'une couche de sortie, sans couche intermédiaire (appelée couche cachée). Chaque entrée est directement connectée à chaque neurone de sortie.

Dans notre cas : 5 entrées (le vecteur d'état), 1 sortie (la décision de sauter ou non).

### 2.4 Code du perceptron non entraîné

Créez un fichier `ia/perceptron.py` avec le contenu suivant :

```python
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
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, x):
        return 1 if self.forward(x) > 0.5 else 0


def run(n_games=10):
    env = FlappyBirdEnv()
    net = Perceptron()
    scores = []

    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action = net.decide(state)
            state, reward, done = env.step(action)
        scores.append(env.score)
        print(f"Partie {i + 1} : score = {env.score}")

    print(f"\nScore moyen sur {n_games} parties : {sum(scores) / len(scores):.1f}")

if __name__ == '__main__':
    run()
```

Exécutez depuis la racine du dépôt :

```bash
python ia/perceptron.py
```

**RESULTAT ATTENDU.** Le terminal affiche les scores de 10 parties. Les scores sont proches de 0, souvent tous nuls. Le comportement de l'oiseau est erratique : il saute ou ne saute pas de façon apparemment aléatoire.

### 2.5 Ce qui ne va pas

Les poids sont initialisés de façon aléatoire. Le réseau n'a aucune connaissance du jeu : il prend des décisions arbitraires. Les résultats sont inférieurs au bot naïf, parfois même nuls.

C'est attendu. L'entraînement consiste précisément à trouver des valeurs de poids qui produisent de bonnes décisions, c'est-à-dire à remplacer les valeurs aléatoires par des valeurs utiles. NEAT, que vous allez utiliser à l'étape suivante, est un algorithme qui fait exactement cela.

### 2.6 Ce que vous devez observer

**CE QUE VOUS DEVEZ OBSERVER.** Les scores sont proches de 0 sur les 10 parties. L'oiseau meurt rapidement, souvent avant d'atteindre le premier tuyau. En relançant le script plusieurs fois, les résultats varient à chaque exécution car les poids sont réinitialisés aléatoirement.

### 2.7 Pour aller plus loin

**POUR ALLER PLUS LOIN.** Modifiez manuellement les valeurs de `self.weights` et `self.bias` au lieu de les initialiser aléatoirement. Par exemple, essayez de donner un poids fortement positif à l'entrée 4 (`state[3]`, la distance au bord haut du gap). Observez si le comportement de l'oiseau change de façon prévisible.

### 2.8 Exercices pratiques

**Exercice 1.** Modifiez la classe `Perceptron` pour utiliser `tanh` comme fonction d'activation à la place de la sigmoïde. La formule de tanh est `np.tanh(z)`. La décision devient : sauter si la sortie est supérieure à 0. Comparez les scores avec la sigmoïde.

**Correction suggérée.**

```python
def forward(self, x):
    z = np.dot(self.weights, x) + self.bias
    return np.tanh(z)

def decide(self, x):
    return 1 if self.forward(x) > 0 else 0
```

**Exercice 2.** Faites tourner 100 perceptrons différents (chacun avec des poids aléatoires différents) sur 1 partie chacun. Affichez les poids du perceptron qui a obtenu le meilleur score.

**Correction suggérée.**

```python
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
```

---

## Étape 3 — NEAT : neuroévolution de base 

**Durée estimée : 2 heures**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 3.1 Les algorithmes génétiques

Un algorithme génétique est une méthode d'optimisation inspirée de la sélection naturelle. Au lieu de calculer directement la meilleure solution, il fait évoluer une population de solutions candidates au fil de plusieurs générations.

Le principe repose sur quatre mécanismes :

- **Évaluation** : chaque individu est testé et reçoit un score appelé fitness. Plus la fitness est élevée, meilleures sont les performances de l'individu.
- **Sélection** : les individus avec la fitness la plus élevée ont plus de chances de se reproduire.
- **Croisement (crossover)** : deux individus parents échangent des parties de leurs informations génétiques pour produire un individu enfant.
- **Mutation** : des modifications aléatoires sont appliquées aux individus enfants. La mutation introduit de la nouveauté et évite que la population converge trop vite vers une solution médiocre.

Analogie : imaginez un éleveur de chevaux de course. Il sélectionne les chevaux les plus rapides, les fait reproduire entre eux, et obtient ainsi une descendance tendanciellement plus rapide à chaque génération.

### 3.2 Les termes clés de NEAT

Avant de continuer, voici les définitions des termes utilisés par la bibliothèque NEAT :

- **Population** : l'ensemble des individus d'une génération. Dans notre cas, chaque individu est un réseau de neurones.
- **Génome** : la description complète d'un individu. Il contient la liste des noeuds (neurones) et des connexions (liens entre neurones), ainsi que leurs poids.
- **Fitness** : le score attribué à un individu après évaluation. C'est le signal qui guide l'évolution.
- **Espèce (spéciation)** : NEAT regroupe les individus similaires en espèces. Les individus ne sont en compétition directe qu'au sein de leur espèce. Cela protège les innovations récentes qui n'ont pas encore eu le temps d'être optimisées.
- **Génération** : un cycle complet comprenant l'évaluation, la sélection et la reproduction de toute la population.

### 3.3 Ce que NEAT ajoute aux algorithmes génétiques classiques

NEAT (NeuroEvolution of Augmenting Topologies) fait évoluer simultanément les poids et la structure du réseau de neurones. Un algorithme génétique classique ne modifie que les poids d'une structure fixe. NEAT peut ajouter des noeuds et des connexions au fil des générations, permettant au réseau de devenir plus complexe si nécessaire.

### 3.4 Le fichier de configuration NEAT

Créez le fichier `ia/neat_config.txt` avec le contenu suivant :

```text
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 50
reset_on_extinction   = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.5

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = 5
num_outputs             = 1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
```

Ce fichier définit tous les paramètres de l'algorithme NEAT. Les lignes importantes pour cette étape sont :

- `pop_size = 50` : la population contient 50 individus par génération.
- `num_inputs = 5` : le réseau reçoit 5 valeurs en entrée (le vecteur d'état).
- `num_outputs = 1` : le réseau produit une sortie (la décision de sauter).
- `num_hidden = 0` : aucune couche cachée au départ. NEAT peut en ajouter par mutation.

### 3.5 Code d'entraînement de base

Créez le fichier `ia/train.py` avec le contenu suivant :

```python
import sys
import os
import neat
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
N_GENERATIONS = 20


def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    state = env.reset()
    done = False

    while not done:
        output = net.activate(state)
        action = 1 if output[0] > 0.5 else 0
        state, reward, done = env.step(action)

    # frames survécus + 500 * tuyaux franchis
    return env.frames + 500 * env.score


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)


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
        generation_interval=5,
        filename_prefix=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    best = population.run(eval_genomes, N_GENERATIONS)

    output_path = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(best, f)

    print(f"\nMeilleur genome sauvegarde dans {output_path}")
    print(f"Fitness du meilleur genome : {best.fitness:.1f}")


if __name__ == '__main__':
    run()
```

Exécutez depuis la racine du dépôt :

```bash
python ia/train.py
```

**RESULTAT ATTENDU.** Le terminal affiche la progression de l'entraînement génération par génération. Voici un exemple de sortie typique des premières générations :

```text
****** Running generation 0 ******

Population's average fitness: 45.23 stdev: 12.11
Best fitness: 134.00 - size: (1, 5) - species 1 - id 23
Species 0     2    10: mean fitness  45.23, best fitness  134.00

****** Running generation 1 ******

Population's average fitness: 67.45 stdev: 18.33
Best fitness: 210.00 - size: (1, 5) - species 1 - id 31
```

La fitness maximale doit augmenter au fil des générations, même si la progression n'est pas monotone.

### 3.6 Ce qui ne va pas

20 générations sont souvent insuffisantes pour que l'IA franchisse régulièrement des tuyaux. La fitness peut stagner si plusieurs espèces s'éteignent ou si la population converge vers un minimum local. Certaines exécutions donnent de bons résultats, d'autres non, en raison de l'initialisation aléatoire. C'est normal et attendu. L'étape 4 traitera ces problèmes.

### 3.7 Ce que vous devez observer

**CE QUE VOUS DEVEZ OBSERVER.** La fitness moyenne et la fitness maximale augmentent globalement au fil des générations, même si des régressions sont possibles. Le nombre d'espèces varie. Une stagnation peut survenir si la fitness maximale ne progresse pas pendant plusieurs générations consécutives.

### 3.8 Pour aller plus loin

**POUR ALLER PLUS LOIN.** Modifiez le facteur de récompense des tuyaux franchis. Remplacez `500 * env.score` par `1000 * env.score` et relancez l'entraînement. Observez si la convergence est plus rapide ou plus lente.

### 3.9 Exercices pratiques

**Exercice 1.** Ajoutez un affichage à la fin de chaque génération qui indique le nombre de tuyaux franchis par le meilleur individu. Hint : la fitness est `frames + 500 * score`, donc `score = (fitness - frames) / 500`. Mais il est plus simple de modifier `evaluate_genome` pour qu'elle retourne également le score.

**Correction suggérée.** Modifiez `eval_genomes` pour suivre le meilleur score de la génération :

```python
def eval_genomes(genomes, config):
    best_score_gen = 0
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)
        score = (genome.fitness - 0) // 500
        if score > best_score_gen:
            best_score_gen = score
    print(f"  Meilleur score (tuyaux) cette generation : {int(best_score_gen)}")
```

**Exercice 2.** Modifiez la fonction de fitness pour pénaliser les sauts inutiles. Ajoutez un compteur de sauts dans `evaluate_genome` et soustrayez un petit malus (par exemple 0.1 par saut) à la fitness finale.

**Correction suggérée.**

```python
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    state = env.reset()
    done = False
    jumps = 0

    while not done:
        output = net.activate(state)
        action = 1 if output[0] > 0.5 else 0
        if action == 1:
            jumps += 1
        state, reward, done = env.step(action)

    return env.frames + 500 * env.score - 0.1 * jumps
```

---

## Étape 4 — Amélioration de la fitness et des hyperparamètres 

**Durée estimée : 1 heure 30 minutes**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 4.1 Pourquoi la fitness est-elle aussi importante que l'algorithme ?

La fonction de fitness est le seul signal que l'algorithme NEAT reçoit pour juger si un individu est bon ou mauvais. Si elle mesure la mauvaise chose, l'algorithme optimise la mauvaise chose.

Exemple : une fitness qui récompense uniquement les frames survécus (sans bonus pour les tuyaux franchis) peut produire un oiseau qui survit longtemps en restant immobile dans une zone sûre, sans jamais tenter de franchir un tuyau.

### 4.2 Les hyperparamètres

Les hyperparamètres sont les réglages de l'algorithme lui-même, distincts des poids du réseau que l'algorithme apprend. Dans NEAT, les principaux hyperparamètres sont :

- `pop_size` : la taille de la population. Une population plus grande explore plus d'solutions en parallèle, mais chaque génération prend plus de temps.
- `weight_mutate_rate` : la probabilité que chaque poids soit muté à chaque génération. Une valeur trop élevée rend l'apprentissage chaotique. Une valeur trop faible ralentit la convergence.
- `compatibility_threshold` : le seuil à partir duquel deux individus sont considérés comme appartenant à des espèces différentes. Une valeur trop basse crée beaucoup d'espèces minuscules.

**Méthode d'ajustement.** Modifier un seul paramètre à la fois. Observer l'effet sur 30 générations avant de toucher au suivant. Conclure avant de passer au paramètre suivant.

### 4.3 Modifications à apporter

Trois modifications améliorent significativement les résultats :

1. Porter la population à 100 individus.
2. Renforcer le bonus pour les tuyaux franchis (facteur 500 ou plus).
3. Augmenter le nombre de générations à 100.

Modifiez `ia/neat_config.txt` : changez `pop_size = 50` en `pop_size = 100`.

Créez ou remplacez `ia/train.py` avec le code suivant :

```python
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


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)


def plot_stats(stats, output_path):
    generations = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label='Fitness maximale')
    plt.plot(generations, avg_fitness, label='Fitness moyenne')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution de la fitness par generation')
    plt.legend()
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

    plot_stats(stats, os.path.join(os.path.dirname(__file__), 'fitness_courbe.png'))

    print(f"\nMeilleur genome sauvegarde dans {genome_path}")
    print(f"Fitness du meilleur genome : {best.fitness:.1f}")


if __name__ == '__main__':
    run()
```

Lancez l'entraînement :

```bash
python ia/train.py
```

**ATTENTION.** L'entraînement sur 100 générations avec une population de 100 individus peut prendre entre 5 et 30 minutes selon la puissance de votre machine. Ne fermez pas le terminal pendant l'exécution.

**RESULTAT ATTENDU.** La fitness maximale dépasse régulièrement 1000 vers la génération 30-50. Un fichier `ia/fitness_courbe.png` est créé à la fin de l'entraînement. Il montre la progression de la fitness maximale et moyenne par génération.

### 4.4 Ce qui ne va pas encore

L'entraînement reste non déterministe : deux exécutions avec les mêmes paramètres produisent des résultats différents. NEAT est sensible à l'initialisation aléatoire et aux événements de mutation qui surviennent en début d'entraînement. Pour obtenir des résultats reproductibles, il faudrait fixer la graine aléatoire (`random.seed`, `numpy.random.seed`), mais cela dépasse le cadre de ce TP.

### 4.5 Ce que vous devez observer

**CE QUE VOUS DEVEZ OBSERVER.** La fitness maximale converge en 30 à 50 générations. Le meilleur individu franchit régulièrement entre 10 et 30 tuyaux. La courbe de fitness moyenne suit la courbe maximale avec un décalage, ce qui indique que les bonnes stratégies se propagent dans la population.

### 4.6 Pour aller plus loin

**POUR ALLER PLUS LOIN.** Testez la fonction d'activation `relu` à la place de `tanh` dans `neat_config.txt` (ligne `activation_options = tanh` devient `activation_options = relu` et `activation_default = tanh` devient `activation_default = relu`). Comparez les courbes de convergence sur 50 générations.

### 4.7 Exercices pratiques

**Exercice 1.** Ajoutez une deuxième courbe au graphique montrant le nombre d'espèces par génération. La méthode `stats.get_species_sizes()` retourne une liste de listes (une par génération, contenant la taille de chaque espèce).

**Correction suggérée.**

```python
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
```

**Exercice 2.** Modifiez le script pour sauvegarder le meilleur génome de chaque génération dans un fichier séparé, en plus du fichier `best_genome.pkl` final.

**Correction suggérée.**

```python
def eval_genomes(genomes, config, generation):
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
```

---

## Étape 5 — Visualisation et analyse 

**Durée estimée : 30 minutes**

**Rappel : pas de LLM pour cette étape. Lisez, réfléchissez, essayez.**

### 5.1 Charger le meilleur génome et le faire jouer

Créez le fichier `ia/play_ia.py` avec le contenu suivant :

```python
import sys
import os
import neat
import pickle
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv
from config import WIDTH, HEIGHT, FPS

GENOME_PATH = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')


def load_genome_and_config():
    with open(GENOME_PATH, 'rb') as f:
        genome = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    return genome, config


def play(genome, config):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Flappy Bird - IA')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 20)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv(render=True, screen=screen)

    running = True
    while running:
        state = env.reset()
        done = False
        frames = 0

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            output = net.activate(state)
            action = 1 if output[0] > 0.5 else 0
            state, reward, done = env.step(action)
            frames += 1

            overlay_lines = [
                f'Score : {env.score}',
                f'Frames : {frames}',
                f'Sortie reseau : {output[0]:.3f}',
            ]
            for i, line in enumerate(overlay_lines):
                surf = font.render(line, True, (255, 255, 255))
                screen.blit(surf, (10, 10 + i * 24))

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    genome, config = load_genome_and_config()
    play(genome, config)
```

Exécutez depuis la racine du dépôt :

```bash
python ia/play_ia.py
```

**ATTENTION.** Ce script requiert que `ia/best_genome.pkl` existe. Si ce n'est pas le cas, relancez d'abord `python ia/train.py`.

**RESULTAT ATTENDU.** La fenêtre pygame s'ouvre. L'IA joue automatiquement. Un overlay affiche le score courant, le nombre de frames survécus et la valeur de sortie du réseau à chaque frame. ECHAP pour quitter.

### 5.2 Visualiser le réseau de neurones

Le module `visualize` distribué avec les exemples de `neat-python` permet de dessiner la topologie du réseau. Téléchargez le fichier `visualize.py` depuis :

```
https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py
```

Placez-le dans `ia/`. Puis créez un fichier `ia/visualize_genome.py` :

```python
import sys
import os
import neat
import pickle

sys.path.insert(0, os.path.dirname(__file__))
import visualize

GENOME_PATH = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

with open(GENOME_PATH, 'rb') as f:
    genome = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

visualize.draw_net(config, genome, view=True, filename='ia/reseau')
```

Exécutez :

```bash
python ia/visualize_genome.py
```

**RESULTAT ATTENDU.** Un schéma du réseau de neurones s'affiche ou est sauvegardé dans `ia/reseau.svg`. Les noeuds d'entrée sont à gauche, le noeud de sortie à droite. Les connexions en vert ont un poids positif (elles activent la sortie), celles en rouge ont un poids négatif (elles l'inhibent). L'épaisseur de chaque connexion est proportionnelle à la valeur absolue de son poids.

### 5.3 Interpréter le schéma

Observez quelles entrées sont connectées à la sortie et avec quel poids. Une entrée avec un poids proche de zéro est peu utilisée par le réseau. Une entrée avec un poids fortement positif ou négatif est déterminante pour la décision.

Comparez avec le bot naïf de l'étape 1 : le réseau a-t-il appris à utiliser la vélocité (entrée 2) que le bot naïf ignorait ?

### 5.4 Limites de NEAT

NEAT présente plusieurs limitations importantes à connaître :

**Absence de mémoire temporelle.** NEAT produit des réseaux sans récurrence par défaut. Le réseau ne dispose d'aucune mémoire des états passés. Chaque décision est prise uniquement à partir de l'état courant.

**Sensibilité aux hyperparamètres.** Les résultats varient fortement selon `pop_size`, `weight_mutate_rate` et `compatibility_threshold`. Il n'existe pas de réglage universel optimal.

**Lenteur sur des environnements complexes.** NEAT évalue chaque individu de la population séquentiellement. Dans des environnements nécessitant de longues simulations ou une entrée haute dimension (images, par exemple), le temps de calcul devient prohibitif.

### 5.5 Comparaison avec l'apprentissage par renforcement profond

L'apprentissage par renforcement profond (deep RL) est une famille d'approches différente de NEAT. Les deux principales différences sont :

- **DQN (Deep Q-Network)** : l'agent apprend une fonction qui associe à chaque état-action une valeur (la récompense future attendue). Il utilise cette fonction pour choisir l'action la plus prometteuse. L'entraînement repose sur la descente de gradient et non sur l'évolution génétique.
- **PPO (Proximal Policy Optimization)** : l'agent apprend directement une politique (une fonction qui associe un état à une action), en optimisant une fonction d'objectif qui garantit que les mises à jour ne sont pas trop brutales. PPO est plus stable que DQN sur de nombreux environnements.

NEAT est simple à mettre en oeuvre et ne nécessite pas de différencier la fonction de récompense. DQN et PPO sont plus efficaces sur des environnements complexes mais requièrent une implémentation plus sophistiquée.

### 5.6 Pistes pour aller plus loin

**POUR ALLER PLUS LOIN.**

- Utilisez la bibliothèque `stable-baselines3` pour entraîner un agent DQN ou PPO sur le même environnement `FlappyBirdEnv`. Comparez la courbe de progression avec celle de NEAT.
- Expérimentez la récompense par curiosité : au lieu de récompenser uniquement le franchissement de tuyaux, récompensez l'agent pour avoir visité des états rares. Cela peut améliorer l'exploration en début d'entraînement.
- Entraînez plusieurs populations NEAT en parallèle en utilisant le module `multiprocessing` de Python. Combinez ensuite les meilleurs génomes.

### 5.7 Ce que vous devez observer

**CE QUE VOUS DEVEZ OBSERVER.** L'IA joue indéfiniment ou pendant un temps très long. Le réseau visualisé est simple : peu de connexions, souvent sans couche cachée supplémentaire si l'entraînement a convergé rapidement. Certaines entrées peuvent ne pas être utilisées si elles n'ont pas contribué à la fitness.

### 5.8 Exercices pratiques

**Exercice 1.** Modifiez `play_ia.py` pour afficher, à chaque frame, quelle action le réseau choisit (0 ou 1) et la valeur brute de sortie avant le seuil de 0.5.

**Correction suggérée.** L'overlay inclut déjà `output[0]`. Ajoutez :

```python
action_label = 'SAUT' if action == 1 else 'ATTENTE'
surf = font.render(f'Action : {action_label}', True, (255, 255, 0))
screen.blit(surf, (10, 10 + 3 * 24))
```

**Exercice 2.** Chargez deux génomes sauvegardés à des générations différentes (par exemple la génération 10 et la génération 90) et faites-les jouer en alternance, une partie chacun. Comparez visuellement leur comportement.

**Correction suggérée.**

```python
import glob

checkpoint_files = sorted(glob.glob('ia/checkpoints/best_gen_*.pkl'))
genome_gen10_path = checkpoint_files[10] if len(checkpoint_files) > 10 else checkpoint_files[-1]
genome_gen90_path = checkpoint_files[90] if len(checkpoint_files) > 90 else checkpoint_files[-1]

with open(genome_gen10_path, 'rb') as f:
    genome_early = pickle.load(f)
with open(genome_gen90_path, 'rb') as f:
    genome_late = pickle.load(f)

for genome in [genome_early, genome_late]:
    play(genome, config)
```

---

## Glossaire {#glossaire}

**Activation.** Fonction mathématique appliquée à la sortie d'un neurone avant de la transmettre à la couche suivante. Elle introduit une non-linéarité dans le réseau. Exemples : sigmoïde, tanh, ReLU.

**Agent.** Programme qui observe l'état d'un environnement et choisit une action à effectuer. Dans ce projet, l'agent est le réseau de neurones qui décide si l'oiseau doit sauter.

**Algorithme génétique.** Méthode d'optimisation inspirée de la sélection naturelle. Elle fait évoluer une population de solutions candidates au fil de plusieurs générations en appliquant sélection, croisement et mutation.

**Biais.** Valeur ajoutée à la somme pondérée des entrées d'un neurone, indépendamment des entrées. Il permet au neurone de produire une sortie non nulle même si toutes les entrées sont nulles.

**Connexion.** Lien entre deux noeuds d'un réseau de neurones. Chaque connexion est associée à un poids qui module l'influence du noeud source sur le noeud cible.

**Croisement (crossover).** Opération génétique qui combine les informations de deux individus parents pour produire un individu enfant. Dans NEAT, cela signifie combiner les gènes (noeuds et connexions) de deux génomes.

**Environnement.** Dans le cadre de l'apprentissage par renforcement, l'environnement est le système avec lequel l'agent interagit. Dans ce projet, l'environnement est le jeu Flappy Bird, représenté par la classe `FlappyBirdEnv`.

**Espèce.** Sous-groupe d'individus génétiquement similaires au sein d'une population NEAT. La spéciation protège les innovations récentes en limitant la compétition directe aux individus d'une même espèce.

**Fitness.** Score numérique attribué à un individu après évaluation. Il mesure la qualité de ses performances. Dans ce projet, la fitness est une combinaison du nombre de frames survécus et du nombre de tuyaux franchis.

**Génome.** Description complète d'un individu dans NEAT. Il contient la liste des noeuds et des connexions, avec leurs paramètres (poids, biais, type d'activation).

**NEAT (NeuroEvolution of Augmenting Topologies).** Algorithme génétique qui fait évoluer simultanément les poids et la structure d'un réseau de neurones. Il peut ajouter des noeuds et des connexions au fil des générations.

**Neurone artificiel.** Unité de calcul qui reçoit plusieurs entrées, calcule une somme pondérée à laquelle s'ajoute un biais, et applique une fonction d'activation au résultat.

**Overfitting.** Phénomène par lequel un modèle apprend trop précisément les données d'entraînement et perd en capacité à généraliser à de nouvelles situations. Dans NEAT, il peut se manifester si le réseau se spécialise trop sur les conditions exactes de l'entraînement.

**Perceptron.** Réseau de neurones constitué d'une couche d'entrée directement reliée à une couche de sortie, sans couche cachée intermédiaire.

**Poids.** Valeur numérique associée à une connexion entre deux neurones. Elle détermine l'influence de la valeur du neurone source sur le neurone cible. Les poids sont les paramètres que l'algorithme d'entraînement cherche à optimiser.

**Population.** Ensemble des individus (génomes) présents à une génération donnée. NEAT évalue tous les individus d'une population avant de passer à la génération suivante.

**Récompense.** Signal numérique renvoyé par l'environnement à chaque étape, indiquant à l'agent la qualité de l'action qu'il vient d'effectuer. Dans ce projet, la récompense est +1 à chaque tuyau franchi et 0 sinon.

**Réseau de neurones.** Ensemble de neurones artificiels organisés en couches et reliés par des connexions pondérées. Un signal se propage de la couche d'entrée vers la couche de sortie en passant par les couches intermédiaires.

**Spéciation.** Mécanisme de NEAT qui regroupe les individus en espèces selon leur similarité structurelle. Les individus ne sont en compétition directe qu'au sein de leur espèce, ce qui protège les innovations récentes.

**Taux de mutation.** Hyperparamètre qui contrôle la probabilité qu'un gène soit modifié aléatoirement lors de la reproduction. Un taux élevé favorise l'exploration mais réduit la stabilité de l'apprentissage.

**Vecteur d'état.** Liste de valeurs numériques qui décrit la situation actuelle du jeu. Dans ce projet, il contient 5 valeurs : la position et la vélocité de l'oiseau, sa distance au prochain tuyau et sa position relative par rapport au gap.
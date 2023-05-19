import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

from Wrappers import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform


def create_cnn():
    """Creates and initialises CNNs"""
    initializer = GlorotUniform()
    model = Sequential([
        Conv2D(32, 8, 4, activation='relu', kernel_initializer=initializer),
        Conv2D(64, 4, 2, activation='relu', kernel_initializer=initializer),
        Conv2D(64, 3, 1, activation='relu', kernel_initializer=initializer),
        Dense(512, activation='relu'),
        Flatten(),
        Dense(14)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model


def populate(population_size):
    """Creates a glorot-initialised population of CNNs"""
    return [create_cnn() for _ in range(population_size)]


def evaluate(num_eps, models):
    """Evaluates the fitness of the models"""
    scores = []
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)

    for model in models:
        ep_score = []
        for _ in range(num_eps):
            state = env.reset()
            eps = 0
            while True:
                state_tensor = np.reshape(state, (-1, 84, 84, 4))
                action = np.argmax(model(state_tensor))
                n_state, reward, done, _ = env.step(action)
                eps += reward
                state = n_state
                if done:
                    ep_score.append(eps)
                    break

        scores.append(np.mean(ep_score))

    return scores


def add_noise(model, sigma):
    """Adds sigma-scaled noise to the model's parameters"""
    for layer in model.layers:
        for weight in layer.trainable_weights:
            weight.assign_add(sigma * tf.random.normal(weight.shape))
    return model


def evolution_selection(models, scores):
    """Ranks the models and their scores in descending order"""
    res = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    return list(zip(*res))


def random_agent(episodes):
    """Agent to choose actions at random for the specified number of episodes, returns a
    baseline"""
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)

    r_rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            _, reward, done, _ = env.step(env.action_space.sample())
            ep_reward += reward
            if done:
                r_rewards.append(ep_reward)
                break
    return r_rewards


def crossover(parent1, parent2):
    """Performs uniform crossover between two parent models"""
    child_model = create_cnn()

    for i, layer in enumerate(child_model.layers):
        weights1 = parent1.layers[i].get_weights()
        weights2 = parent2.layers[i].get_weights()
        new_weights = []

        for w1, w2 in zip(weights1, weights2):
            if len(w1.shape) > 0:
                mask = np.random.randint(2, size=w1.shape)
                w_new = w1 * mask + w2 * (1 - mask)
            else:
                w_new = w1 if random.random() < 0.5 else w2
            new_weights.append(w_new)

        layer.set_weights(new_weights)
    return child_model


def mutate(model, sigma, method='noise'):
    if method == 'noise':
        return add_noise(model, sigma)
    elif method == 'reset':
        return reset_weights(model, sigma)
    elif method == 'creep':
        return creep_mutation(model, sigma)
    else:
        raise ValueError(f'Unknown mutation method: {method}')

def reset_weights(model, sigma):
    for layer in model.layers:
        for weight in layer.trainable_weights:
            if random.random() < sigma:
                weight.assign(tf.random.normal(weight.shape))
    return model

def creep_mutation(model, sigma):
    for layer in model.layers:
        for weight in layer.trainable_weights:
            if random.random() < sigma:
                weight.assign_add(np.random.normal(loc=0.0, scale=sigma))
    return model

def select_parents(models, scores, num_parents, method='rank'):
    models, scores = evolution_selection(models, scores)

    if method == 'rank':
        return models[:num_parents]
    elif method == 'tournament':
        return tournament_selection(models, scores, num_parents)
    elif method == 'roulette':
        return roulette_wheel_selection(models, scores, num_parents)
    else:
        raise ValueError(f'Unknown selection method: {method}')

def tournament_selection(models, scores, num_parents):
    parents = []
    for _ in range(num_parents):
        competitors = random.sample(list(zip(models, scores)), 3)
        competitors.sort(key=lambda x: x[1], reverse=True)
        parents.append(competitors[0][0])
    return parents

def roulette_wheel_selection(models, scores, num_parents):
    total_fitness = sum(scores)
    rel_fitness = [f/total_fitness for f in scores]
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    parents = []
    for _ in range(num_parents):
        r = random.random()
        for (i, model) in enumerate(models):
            if r <= probs[i]:
                parents.append(model)
                break
    return parents

def early_stopping_evaluate(models, num_eps, num_episodes_to_check=2, threshold=0.2):
    scores = []
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)

    for model in models:
        ep_score = []
        for ep in range(num_eps):
            state = env.reset()
            eps = 0
            while True:
                state_tensor = np.reshape(state, (-1, 84, 84, 4))
                action = np.argmax(model(state_tensor))
                n_state, reward, done, _ = env.step(action)
                eps += reward
                state = n_state
                if done:
                    ep_score.append(eps)
                    break
            if ep == num_episodes_to_check and np.mean(ep_score) < threshold:
                break
        scores.append(np.mean(ep_score))
    return scores

def algorithm(sigma, population_size, episodes, num_elites, runs, num_parents):
    if num_elites >= population_size:
        raise ValueError("Number of elites must be less than population size.")

    models = populate(population_size)
    best_scores = []
    for _ in range(runs):
        tf.keras.backend.clear_session()
        scores = early_stopping_evaluate(models, episodes)
        parents = select_parents(models, scores, num_parents)
        for x in range(num_elites, population_size):
            a = random.choice(parents)
            models[x] = mutate(a, sigma)
        models, scores = evolution_selection(models, scores)
        best_scores.append(scores[0])
    return best_scores

results = algorithm(sigma=0.002, population_size=100, episodes=2, num_elites=1, runs=10, num_parents=4)

plt.plot(results)
plt.axhline(y=np.mean(random_agent(10)), color='r', linestyle='-')
plt.legend(['Genetic Algo', 'Random agent mean score'])
plt.ylabel('Highest mean score')
plt.xlabel('Generation Number')
plt.title('Highest mean score per generation')
plt.show()



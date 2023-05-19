# AtariGeneticAlgorithm

This project demonstrates the use of a genetic algorithm to evolve a population of Convolutional Neural Networks (CNNs) playing the Breakout Atari game, using the gym learning environment. The program uses TensorFlow and Keras to build and manage the models, and the gym environment to provide the game interface.

The script operates in several stages:

    Initialization: A population of CNN models is created, each with the same architecture but different initialized weights.

    Evaluation: Each model plays a specified number of games of Breakout. The fitness of a model is defined as the mean score across these games.

    Selection: The models are ranked by their fitness scores. The best-performing models are selected to be parents for the next generation.

    Mutation: New models are created for the next generation by mutating the selected parents. This involves adding noise to the weights of the parent models. Some preliminary code included on different mutation methods, including crossover and creep_mutation.

    Replacement: The least fit individuals in the population are replaced by the newly created individuals.

    Repeat: The process of evaluation, selection, mutation, crossover, and replacement is repeated for a number of generations.

At the end of the run, the script plots the highest fitness score in each generation, providing a visualization of the evolution of the models over time.

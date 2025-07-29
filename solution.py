'''
                            INTRODUCTION TO ARTIFICIAL INTELLIGENCE - LAB ASSIGNMENT 4
                                            ZIBOROV NIKITA, FER UNIZG

used literature sources:
1. Lectures 11 and 12 -IAI FER UNIZG
2. https://realpython.com/python-ai-neural-network/
3. https://www.quora.com/Can-you-implement-any-deep-learning-model-using-only-numpy-as-your-tool
4. https://www.w3schools.com/python/numpy/default.asp
5. https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
6. https://numpy.org/devdocs//user/basics.types.html
7. https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
8. https://habr.com/ru/articles/887400/
9. https://neurohive.io/ru/tutorial/prostaja-nejronnaja-set-python/

                                                                                                                    '''
import argparse
import csv
import numpy as np

def readfile(filename): #firstly, basics - function to read data from csv file
    f = open(filename, 'r')
    data=[]
    reader = csv.reader(f) #method reader for csv files
    next(reader) #skipping first line - nothing interesting here... just 'x' and 'y'...
    for row in reader: data.append(row)
    '''NOTE: since 3rd lab, discovering different sources, almost in every of them pandas library is used. however, due to
    task impossibility to use another external libraries (which makes this task harder!), i should do something else.
    at this point, i tried to implement it through numpy array (source 4):                                          '''

    data = np.array(data)
    data = data.astype(np.double) #source 4 - because of numbers, we should rearrange type to double (or float?)
    #basically, just returning x and y values (x is vector, y is always last, just assuming that fact):
    return data[:-1], data[-1]

#at this point, i am at first point of task - making neural network (NN) for generic algorithm (GA)
#for usability (and by comfort, im lazy to make it other way) creating separate class:
class NN:
    #like from previous lab, every class begins with init function and so-called hyperparameters:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights = [] #initializing list of weight matrices for each layer
        self.bias = [] #initializing list of biases (bias vectors its called i hope) for each layer
        self.input_size = input_size #initializing rest "hyperparameters" for NN class
        self.output_size = output_size
        self.fill(input_size, hidden_size, output_size)
        #actually, what am i doing here - trying to implement algorithm from lecture: imagine having matrix of
        #necessary shape, then filling it with values of weights and biases, implementation - below:
    def fill(self, input_size, hidden_size, output_size):
        out = input_size
        for hidden in hidden_size:
            '''at this point, iterating every "layer", i wanted to implement, citation from lab task:
            The initial values of all weights of the neural network should be sampled 
            from the NORMAL DISTRIBUTION with standard deviation of 0.01.
            for this, i found my "saver" :) - numpy.random.normal method, source 7.
            from method reference: ...random samples from a normal (Gaussian) distribution'''
            self.weights.append(np.random.normal(0, 0.01, (out, hidden))) #scale is 0.01 - parameter
            self.bias.append(np.random.normal(0, 0.01, (1, hidden)))      #is defined in lab task
            out = hidden                                                                 #so-called "sigma" deviation
            #adding last (final) weight and bias:
        self.weights.append(np.random.normal(0, 0.01, (out, output_size)))
        self.bias.append(np.random.normal(0, 0.01, (1, output_size)))

    #2 functions, basics i would rather say - sigmoid function and mean squared error(MSE) - standard implementation
    #(it is defined like... everywhere like this)
    def sigmaboi(self, x): return 1 / (1 + np.exp(-x))
    def MSE (self, y_true, y_pred): return np.mean((y_true - y_pred) ** 2)

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = self.sigmaboi(np.dot(x, self.weights[i]) + self.bias[i])
        return np.dot(x, self.weights[-1]) + self.bias[-1]


    def get_weights(self):
        weights = []
        for W, b in zip(self.weights, self.bias):
            weights.append(W.flatten())
            weights.append(b.flatten())
        return np.concatenate(weights)

    def set_weights(self, chromosome):
        offset = 0
        for i in range(len(self.weights)):
            W_shape = self.weights[i].shape
            b_shape = self.bias[i].shape
            self.weights[i] = chromosome[offset:offset + np.prod(W_shape)].reshape(W_shape)
            offset += np.prod(W_shape)
            self.bias[i] = chromosome[offset:offset + np.prod(b_shape)].reshape(b_shape)
            offset += np.prod(b_shape)


class GeneticAlgorithm:
    def __init__(self, args, X_train, y_train, X_test, y_test):
        self.args = args
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Determine neural network shape
        self.input_size = X_train.shape[1]
        self.output_size = 1
        self.hidden_sizes = [int(s) for s in args.nn.split('s') if s.isdigit()]

        # For population
        self.dummy_nn = NN(self.input_size, self.hidden_sizes, self.output_size)
        self.chromosome_size = len(self.dummy_nn.get_weights())
        self.population = self.initialize_population(args.popsize)

    def initialize_population(self, pop_size):
        return [np.random.normal(0, 0.01, self.chromosome_size) for _ in range(pop_size)]

    def fitness_function(self, chromosome):
        self.dummy_nn.set_weights(chromosome)
        y_pred = self.dummy_nn.forward(self.X_train)
        return -self.dummy_nn.MSE(self.y_train, y_pred)

    def select_parents(self, fitness):
        # Shift fitness to positive values to avoid issues with negative values
        fitness_shifted = fitness - np.min(fitness) + 1e-6
        probabilities = fitness_shifted / fitness_shifted.sum()
        size = max(0, len(self.population) - self.args.elitism)
        indices = np.random.choice(len(self.population), size=size, p=probabilities)
        return [self.population[i] for i in indices]

    def crossover(self, parent1, parent2):
        return (parent1 + parent2) / 2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.rand() < self.args.p:  # Mutation probability
                chromosome[i] += np.random.normal(0, self.args.K)  # Gaussian noise
        return chromosome

    def train(self):
        for generation in range(1, self.args.iter + 1):
            # Compute fitness for all chromosomes
            fitness = np.array([self.fitness_function(chromosome) for chromosome in self.population])

            # Elitism
            elite_indices = np.argsort(fitness)[-self.args.elitism:]
            next_population = [self.population[i] for i in elite_indices]

            # Selection and reproduction
            parents = self.select_parents(fitness)
            while len(next_population) < len(self.population):
                if len(parents) < 2:
                    next_population.append(self.population[np.random.randint(len(self.population))])
                else:
                    child = self.crossover(parents[-2], parents[-1])
                    next_population.append(child)
                    parents = parents[:-2]

            # Mutation
            next_population = [self.mutate(ch) for ch in next_population[:len(self.population)]]
            self.population = next_population

            # Log training progress
            if generation % 2000 == 0:
                best_idx = np.argmax(fitness)
                best_chromosome = self.population[best_idx]
                self.dummy_nn.set_weights(best_chromosome)
                train_error = -self.fitness_function(best_chromosome)
                print(f"[Train error {generation:05d}]: {train_error:.6f}")

        # Test final solution
        fitness = np.array([self.fitness_function(chromosome) for chromosome in self.population])
        best_idx = np.argmax(fitness)
        best_chromosome = self.population[best_idx]
        self.dummy_nn.set_weights(best_chromosome)
        test_error = self.dummy_nn.MSE(self.y_test, self.dummy_nn.forward(self.X_test))
        print(f"[Test error]: {test_error:.6f}")

parser = argparse.ArgumentParser()
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--nn")
parser.add_argument("--popsize", type=int)
parser.add_argument("--elitism", type=int)
parser.add_argument("--p", type=float)
parser.add_argument("--K", type=float)
parser.add_argument("--iter", type=int)
args = parser.parse_args()

X_train, y_train = readfile(args.train)
X_test, y_test = readfile(args.test)
ga = GeneticAlgorithm(args, X_train, y_train, X_test, y_test)
ga.train()


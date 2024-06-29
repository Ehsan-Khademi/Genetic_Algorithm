# import random
#
# import numpy as np
#
# # ---------------------------------------------------------project2 Ehsan Khademi
# # Distance matrix
# inputS=[]
# TspMatrix=[]
# # matrix = np.array([[0, 20, 10, 12],
# #                    [20, 0, 15, 11],
# #                    [10, 15, 0, 17],
# #                    [12, 11, 17, 0]])
# class ReadFromFile:
#     def __init__(self,filename):
#         self.filename=filename
#     def read_file(self):
#         file=open(self.filename,'r')
#         file.seek(0)
#         for line in file:
#             inputS=[eval(i)for i in line.rstrip('\n').split(' ')]
#             TspMatrix.append(inputS)
#         return TspMatrix
#
# rff=ReadFromFile('tspIput.txt')
# # matrix=np.array(rff.read_file())
# matrix=rff.read_file()
# # Genetic Algorithm Implementation
# class GeneticAlgorithm:
#     def __init__(self, population_size=100, generations=500,mutation=0.1):
#         self.population_size = population_size
#         self.generations = generations
#         self.population = self.initialpopulation(population_size)
#         self.mutation =mutation
#
#     def initialpopulation(self,size):
#         population = []
#         for _ in range(size):
#             individual = list(range(1,len(matrix)))
#             random.shuffle(individual)
#             individual.insert(0, 0)
#             individual.extend([0])
#             population.append(individual)
#         return population
#
#     def fitness(self,individual):
#         total_distance = 0
#         for i in range(len(individual)):
#             city1 = individual[i]
#             if i == len(individual) - 1:
#                 city2 = individual[0]
#             else:
#                 city2 = individual[i + 1]
#             total_distance += matrix[city1][city2]
#         return total_distance
#
#     # Selection Methods
#     def greedy_selection(self, population):
#         # print(population)
#         return min(population, key=self.fitness)
#
#     def roulette_wheel_selection(self, population):
#         fitness_scores = [self.fitness(p) for p in population]
#         total_fitness = sum(fitness_scores)
#         selection_probs = [f / total_fitness for f in fitness_scores]
#         # Normalize the probabilities
#         selection_probs = [float(i)/sum(selection_probs) for i in selection_probs]
#
#         return population[np.random.choice(len(population), p=selection_probs)]
#
#     # Mutation Methods
#     def single_point_mutation(self, path):
#         if (random.randint(0, 1) < self.mutation):
#             idx = random.randint(1, len(path) - 2)
#             path[idx], path[idx+1] = path[idx+1], path[idx]
#         return path
#
#     def double_point_mutation(self, path):
#         if(random.randint(0,1) <self.mutation):
#             idx1, idx2 = sorted(random.sample(range(1,len(path)-2), 2))
#             path[idx1], path[idx2] = path[idx2], path[idx1]
#         return path
#
#     # Combination Methods
#     def single_point_crossover(self, parent1, parent2):
#         point = random.randint(1, len(parent1) - 3)
#         child1 = np.concatenate((parent1[:point], parent2[point:])).tolist()
#         child2 = np.concatenate((parent2[:point], parent1[point:])).tolist()
#         return self.fix_duplicate(child1),self.fix_duplicate(child2)
#
#     def multipoint_crossover(self, parent1, parent2):
#         points = sorted(random.sample(range(1,len(parent1)-2), 2))
#         child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:])).tolist()
#         child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:])).tolist()
#         return self.fix_duplicate(child1),self.fix_duplicate(child2)
#
#     def fix_duplicate(self, child):
#         # child=np.array(child)
#         # unique, counts = np.unique(child, return_counts=True)
#         # duplicates = unique[counts > 1]
#         # missing = set(range(len(matrix))) - set(child)
#         # for dup in duplicates:
#         #     if len(missing)!=0:
#         #         child[np.where(child == dup)[0][1]] = missing.pop()
#         # return child.tolist()
#         child=set(child)
#
#         child=list(child)
#         for i in range(1,len(matrix)):
#             if i not in child:
#                 child.append(i)
#         child.append(0)
#
#         # print(child)
#         return child
#
#
#     def run(self):
#         for _ in range(self.generations):
#             # print(self.population)
#             new_population = []
#             fixSize=self.population_size//2
#             # self.population =list(self.population)
#             for _ in range(fixSize):
#                 # parent1 = self.roulette_wheel_selection(self.population)
#                 # self.population.remove(parent1)
#                 # parent2 = self.roulette_wheel_selection(self.population)
#                 # self.population.remove(parent2)
#                 parent1=self.greedy_selection(self.population)
#                 self.population.remove(parent1)
#                 parent2=self.greedy_selection(self.population)
#                 self.population.remove(parent2)
#
#
#                 child1,child2 = self.single_point_crossover(parent1, parent2)
#                 # child1,child2 = self.multipoint_crossover(parent1, parent2)
#
#                 child1 = self.single_point_mutation(child1)
#                 child2 = self.single_point_mutation(child2)
#                 # child1 = self.double_point_mutation(child1)
#                 # child2 = self.double_point_mutation(child2)
#
#                 # child= self.double_point_mutation(child)
#
#                 new_population.append(child1)
#                 new_population.append(child2)
#             while(len(self.population)!=0):
#                 new_population.append(self.population.pop())
#             self.population = new_population
#
#         return self.greedy_selection(self.population)
#         # return self.population
#         # return self.roulette_wheel_selection(self.population)
#         # return self.population
#
# # Running the Genetic Algorithm
#
# def main():
#     if __name__ == '__main__':
#         ga = GeneticAlgorithm()
#         best_path = ga.run()
#         formatted_path = '_'.join(map(str, best_path))
#         print("Best path found:", formatted_path)
#
# main()
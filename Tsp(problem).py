import random
import numpy as np




# ---------------------------------------------------------project2 Ehsan Khademi
# Distance matrix
inputS=[]
TspMatrix=[]
class ReadFromFile:
    def __init__(self,filename):
        self.filename=filename
    def read_file(self):
        file=open(self.filename,'r')
        file.seek(0)
        for line in file:
            inputS=[eval(i)for i in line.rstrip('\n').split(' ')]
            TspMatrix.append(inputS)
        return TspMatrix

rff=ReadFromFile('tspIput.txt')
# matrix=rff.read_file()
# ------------------------------------
# matrix=np.array(rff.read_file())
matrix= [[0, 10, 8, 9, 7],
[10, 0, 10, 5, 6],
[8, 10, 0, 8, 9],
[9, 5, 8, 0, 6],
[7, 6, 9, 6, 0]]


class Tsp:

    def __init__(self, populationSize=100, generations=500, mutation=.88):
        self.populationSize =populationSize
        self.generations =generations
        self.population =[]
        self.population =self.initialize_population()
        self.mutation =mutation


    # definitialize_population(self):
    #     self.population=[i.rermove(0) for i in self.population]
    #     self.population=[list(itertools.permutations(i))for i in self.population]
    #     self.population=[i.insert(0,0) for i in self.population]
    #     self.population.extend([i.append(0) for i in self.population])
    #     self.population.extend(random.sample(self.population,self.population_Size))
    # def finess(self,individual):
    #     for i in individual:

    def initialize_population(self):
        p=[]
        p=list(range(1,len(matrix)))
        self.shuffling(p)
        # print(self.population)
        for individual in self.population:
            self.insert_at_zero_first(individual)
            self.append_zero_to_end(individual)
        return self.population
    def shuffling(self,p):
        for _ in range(self.populationSize):
               random.shuffle(p)
               self.population.append(p)
               # print(self.population)

    def insert_at_zero_first(self,individual):
        individual.insert(0,0)
    def append_zero_to_end(self,individual):
        individual.append(0)

    def fitness(self,individual):
        total=0
        for i in range(len(individual)):
            if(i+1<=len(individual)-1):
                pos1,pos2=individual[i],individual[i+1]
                total+=matrix[pos1][pos2]

        return total

    def greedy_selection(self, population):
        # print(population)
        return min(population, key=self.fitness)

    def roulette_wheel_selection(self, population):
        fitness_scores = [self.fitness(p) for p in population]
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        # Normalize the probabilities
        selection_probs = [float(i)/sum(selection_probs) for i in selection_probs]

        return population[np.random.choice(len(population), p=selection_probs)]


    def single_point_mutation(self, path):
        if (random.randint(0, 1) < self.mutation):
            idx = random.randint(1, len(path) - 3)
            path[idx], path[idx + 1] = path[idx + 1], path[idx]
        return path

    def double_point_mutation(self, path):
        if (random.randint(0, 1) < self.mutation):
            idx1, idx2 = sorted(random.sample(range(1, len(path) - 2), 2))
            path[idx1], path[idx2] = path[idx2], path[idx1]
        return path

        # Combination Methods

    def single_point_crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 2)
        child1 = np.concatenate((parent1[:point], parent2[point:])).tolist()
        child2 = np.concatenate((parent2[:point], parent1[point:])).tolist()
        return self.fix_duplicate(child1), self.fix_duplicate(child2)

    def multipoint_crossover(self, parent1, parent2):
        points = sorted(random.sample(range(1, len(parent1) - 2), 2))
        child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:])).tolist()
        child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:])).tolist()
        return self.fix_duplicate(child1), self.fix_duplicate(child2)

    def fix_duplicate(self, child):
        child = set(child)

        child = list(child)
        for i in range(1, len(matrix)):
            if i not in child:
                child.append(i)
        child.append(0)
        return child

    def run(self):

        for _ in range(self.generations):
            # print(self.population)
            new_population = []
            fixSize = self.populationSize//2
            # self.population =list(self.population)
            for _ in range(fixSize):
                # parent1 = self.roulette_wheel_selection(self.population)
                # self.population.remove(parent1)
                # parent2 = self.roulette_wheel_selection(self.population)
                # self.population.remove(parent2)
                parent1 = self.greedy_selection(self.population)
                self.population.remove(parent1)
                parent2 = self.greedy_selection(self.population)
                self.population.remove(parent2)

                # child1, child2 = self.single_point_crossover(parent1, parent2)
                child1,child2 = self.multipoint_crossover(parent1, parent2)

                # child1 = self.single_point_mutation(child1)
                # child2 = self.single_point_mutation(child2)
                child1 = self.double_point_mutation(child1)
                child2 = self.double_point_mutation(child2)

                # child= self.double_point_mutation(child)

                new_population.append(child1)
                new_population.append(child2)
            while (len(self.population) != 0):
                new_population.append(self.population.pop())
            self.population = new_population

        return self.greedy_selection(self.population)
        # return self.population
        # return self.roulette_wheel_selection(self.population)
        # return self.population
    def findlengthofpath(self,totalpath):
        str=totalpath.split('_')
        count=0
        for i in range(len(str)-1):
            count+=matrix[int(str[i])][int(str[i+1])]
        print(f"length of best bath is:-------------> {count}")





# def main():
    # if __name__ == '__main__':
ga=Tsp()
best_path=ga.run()
formatted_path = '_'.join(map(str, best_path))
print("Best path found:", formatted_path)
ga.findlengthofpath(formatted_path)

# main()






# genetic_alg=GeneticAlg()
# print(genetic_alg.initialize_population())

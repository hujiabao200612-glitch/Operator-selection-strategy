import WTA
import random
from WTA_Visualizer import WTA_Monitor


class WtaProblem:
    def __init__(self, weapon, target, poolSize, iteration, mutationRate):
        self.weapon, self.target, self.poolSize = weapon, target, poolSize
        self.iteration, self.mutationRate = iteration, mutationRate
        self.object = WTA.Pool(self.poolSize, self.target, self.weapon, self.mutationRate)
        self.fitness = 0

    def Process(self, compare=False):
        if compare:
            trad_pool = WTA.Pool(self.poolSize, self.target, self.weapon, self.mutationRate, use_ucb=False)
            trad_pool.Create()
            trad_div = []
            for _ in range(self.iteration):
                trad_pool.Process()
                trad_div.append(trad_pool.diversity_history[-1])

            monitor = WTA_Monitor()
            self.object.Create()
            for i in range(self.iteration):
                self.object.Process(monitor=monitor, current_iter=i)

            monitor.show_all_reports(trad_pool.fitness_history, self.object.fitness_history, trad_div)
            self.fitness = self.object.theBest[-1].fitness
        else:
            self.object.Create()
            for _ in range(self.iteration): self.object.Process()
            if self.object.theBest: self.fitness = self.object.theBest[-1].fitness


class OperatorsPool:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.poolSize = 10
        self.pool = []
        self.mutationRate = 0.3
        self.theBest = []

    def Create(self):
        for _ in range(self.poolSize):
            c = OperatorsChromosome(self.weapon, self.target);
            c.Create();
            self.pool.append(c)
        if self.pool: self.theBest = [sorted(self.pool, key=lambda x: x.fitness)[-1]]

    def Process(self):
        children = []
        for i in range(0, len(self.pool), 2):
            if i + 1 < len(self.pool):
                c1, c2 = self.pool[i].Crossover(self.pool[i + 1]);
                children.extend([c1, c2])
        for c in children:
            if random.random() < self.mutationRate: c.Mutation()
            c.Fitness()
        self.pool.extend(children)
        self.pool = sorted(self.pool, key=lambda x: x.fitness, reverse=True)[:self.poolSize]
        if self.pool[0].fitness > self.theBest[-1].fitness: self.theBest.append(self.pool[0])


class OperatorsChromosome:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.chromosome = []
        self.fitness = -1
        self.best_wta_solution = []

    def Create(self):
        self.chromosome = [random.randint(0, 1) for _ in range(21)]
        self.Fitness()

    def Decode(self):
        ps = int(''.join(map(str, self.chromosome[:7])), 2) + 1
        it = int(''.join(map(str, self.chromosome[7:14])), 2) + 1
        mr = (int(''.join(map(str, self.chromosome[14:21])), 2) + 1) / 128.0
        return ps, it, mr

    def Fitness(self, show_plot=False):
        ps, it, mr = self.Decode()
        prob = WtaProblem(self.weapon, self.target, ps, it, mr)
        prob.Process(compare=show_plot)
        self.fitness = prob.fitness
        if prob.object.theBest: self.best_wta_solution = prob.object.theBest[-1].chromosome

    def Crossover(self, other):
        r = random.randint(1, 20)
        c1, c2 = OperatorsChromosome(self.weapon, self.target), OperatorsChromosome(self.weapon, self.target)
        c1.chromosome = self.chromosome[:r] + other.chromosome[r:];
        c2.chromosome = other.chromosome[:r] + self.chromosome[r:]
        return c1, c2

    def Mutation(self):
        idx = random.randint(0, 20);
        self.chromosome[idx] = 1 - self.chromosome[idx]
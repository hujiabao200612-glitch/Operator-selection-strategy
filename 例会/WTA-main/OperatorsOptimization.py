import WTA
import random

class WtaProblem:
    def __init__(self, weapon, target, poolSize, iteration, mutationRate):
        self.object = WTA.Pool(int(poolSize), int(target), int(weapon), mutationRate)
        self.iteration = int(iteration)

    def Run(self):
        self.object.Create()
        for _ in range(self.iteration):
            self.object.Process()
        return self.object.theBest[-1].fitness if self.object.theBest else 0

class OperatorsPool:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.thePool = []
        self.theBest = []

    def Create(self):
        for _ in range(5): # 初始种群
            c = OperatorsChromosome(self.weapon, self.target)
            c.Fitness()
            self.thePool.append(c)
        self.FindBest()

    def Process(self):
        new_gen = []
        # 杂交产生下一代
        parents = sorted(self.thePool, key=lambda x: x.fitness, reverse=True)
        for i in range(len(parents)//2):
            c1, c2 = parents[i].Crossover(parents[-1-i])
            c1.Fitness(); c2.Fitness()
            new_gen.extend([c1, c2])
        self.thePool = new_gen
        self.FindBest()

    def FindBest(self):
        if not self.thePool: return
        current_best = max(self.thePool, key=lambda x: x.fitness)
        if not self.theBest or current_best.fitness >= self.theBest[-1].fitness:
            self.theBest.append(current_best)

class OperatorsChromosome:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.chromosome = [random.randint(0, 1) for _ in range(21)]
        self.fitness = -1
        self.object = None 

    def Decimal(self):
        p = (int(''.join(map(str, self.chromosome[:7])), 2) % 64) + 10
        i = (int(''.join(map(str, self.chromosome[7:14])), 2) % 50) + 10
        m = (int(''.join(map(str, self.chromosome[14:21])), 2) + 1) / 256
        return p, i, m

    def Fitness(self):
        p, i, m = self.Decimal()
        prob = WtaProblem(self.weapon, self.target, p, i, m)
        self.fitness = prob.Run()
        self.object = prob.object # 这一步极其重要

    def Crossover(self, other):
        rnd = random.randint(1, 19)
        c1, c2 = OperatorsChromosome(self.weapon, self.target), OperatorsChromosome(self.weapon, self.target)
        c1.chromosome = self.chromosome[:rnd] + other.chromosome[rnd:]
        c2.chromosome = other.chromosome[:rnd] + self.chromosome[rnd:]
        return c1, c2
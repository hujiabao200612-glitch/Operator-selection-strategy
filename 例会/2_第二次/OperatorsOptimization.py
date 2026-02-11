import WTA
import random


class WtaProblem:
    def __init__(self, weapon, target, poolSize, iteration, mutationRate):
        self.weapon, self.target = weapon, target
        self.poolSize, self.mutationRate, self.iteration = poolSize, mutationRate, iteration
        self.object = WTA.Pool(self.poolSize, self.target,
                               self.weapon, self.mutationRate)
        dummy_pool = WTA.Pool(1, self.target, self.weapon, self.mutationRate)
        dummy_pool.theBest[-1].fitness = 0
        self.theBest = [dummy_pool]

    def Process(self):
        for i in range(self.iteration):
            self.object.Process(i, self.iteration)
        if self.object.theBest[-1].fitness > self.theBest[-1].theBest[-1].fitness:
            self.theBest.append(self.object)

    def Create(self):
        self.object.Create()


class OperatorsPool:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.poolSize, self.pool, self.children, self.mutationRate = 10, [], [], 0.3
        dummy = OperatorsChromosome(self.weapon, self.target)
        dummy.fitness = 0
        self.theBest = [dummy]

    def Process(self):
        self.Crossover()
        self.Mutation()
        for i in self.children:
            i.Fitness()
        self.pool.extend(self.children)
        self.children = []
        current_best = self.theBest[-1]
        b = False
        for i in self.pool:
            if i.fitness > current_best.fitness:
                current_best = i
                b = True
        if b:
            self.theBest.append(current_best)
        total = sum(i.fitness for i in self.pool)
        if total > 0:
            probs = [i.fitness / total for i in self.pool]
            selected = []
            for _ in range(self.poolSize):
                r, cum = random.random(), 0
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        selected.append(self.pool[idx])
                        break
            self.pool = selected

    def Create(self):
        for _ in range(self.poolSize):
            c = OperatorsChromosome(self.weapon, self.target)
            c.Create()
            self.pool.append(c)

    def Crossover(self):
        for i in range(0, len(self.pool)-1, 2):
            p1, p2 = self.pool[i], self.pool[i+1]
            rnd = random.randint(0, 20)
            c1, c2 = OperatorsChromosome(
                self.weapon, self.target), OperatorsChromosome(self.weapon, self.target)
            c1.chromosome = p1.chromosome[rnd:] + p2.chromosome[:rnd]
            c2.chromosome = p1.chromosome[:rnd] + p2.chromosome[rnd:]
            self.children.extend([c1, c2])

    def Mutation(self):
        for i in self.children:
            if random.random() < self.mutationRate:
                i.chromosome[random.randint(0, 20)] = random.randint(0, 1)


class OperatorsChromosome:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.chromosome, self.fitness = [], -1
        self.theBestChromosome, self.theBestChromosome_ref = [], None

    def Create(self):
        self.chromosome = [random.randint(0, 1) for _ in range(21)]
        self.Fitness()

    def Decimal(self):
        ps = int(''.join(map(str, self.chromosome[:7])), 2) + 1
        it = int(''.join(map(str, self.chromosome[7:14])), 2) + 1
        mr = (int(''.join(map(str, self.chromosome[14:20])), 2) + 1) / 128
        return ps, it, mr

    def Fitness(self):
        ps, it, mr = self.Decimal()
        prob = WtaProblem(self.weapon, self.target, ps, it, mr)
        prob.Create()
        prob.Process()
        self.fitness = prob.theBest[-1].theBest[-1].fitness
        self.theBestChromosome = prob.theBest[-1].theBest[-1].chromosome
        self.theBestChromosome_ref = prob.theBest[-1].theBest[-1]

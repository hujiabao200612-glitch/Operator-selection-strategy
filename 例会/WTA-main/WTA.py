import random
from AdaptiveSelector import UCBSelector

class Pool:
    def __init__(self, poolSize, numberofTarget, numberofWeapon, mutationRate):
        self.thePool, self.temporary = [], []
        self.poolSize, self.mutationRate = poolSize, mutationRate
        self.numberofTarget, self.numberofWeapon = numberofTarget, numberofWeapon
        self.history = [] 
        self.ops = ['RightShift', 'LeftShift', 'ReverseChromosome', 'ReversePiece', 'SwapPieces', 'ReverseHeadAndTail']
        self.selector = UCBSelector(self.ops)
        self.theBest = []

    def Create(self):
        for _ in range(self.poolSize):
            c = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            c.Create()
            self.thePool.append(c)
        self.FindBest()

    def Process(self):
        self.Crossover()
        self.Merge()
        self.FindBest()
        self.Selection()
        if self.theBest:
            self.history.append(self.theBest[-1].fitness)

    def FindBest(self):
        if not self.thePool: return
        current_best = self.thePool[0]
        for i in self.thePool:
            if i.fitness > current_best.fitness:
                current_best = i
        
        if not self.theBest or current_best.fitness > self.theBest[-1].fitness:
            new_best = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            new_best.chromosome = list(current_best.chromosome)
            new_best.fitness = current_best.fitness
            self.theBest.append(new_best)

    def Selection(self):
        total = sum(i.fitness for i in self.thePool)
        if total == 0: return
        probs = [i.fitness / total for i in self.thePool]
        selected = []
        for _ in range(self.poolSize):
            r, c = random.random(), 0
            for i, p in enumerate(probs):
                c += p
                if r <= c: selected.append(self.thePool[i]); break
        self.thePool, self.temporary = selected, []

    def Crossover(self):
        for i in self.thePool: self.temporary.append(i.Crossover(self.selector))

    def Merge(self):
        for i in self.temporary: self.thePool.append(i)

class Chromosome:
    def __init__(self, pool_obj, mutation, target, weapon):
        self.pool_obj = pool_obj
        self.mutationRate = mutation
        self.numberofTarget, self.numberofWeapon = target, weapon
        self.chromosome, self.fitness = [], -1
        self.weaponList = list(range(1, weapon + 1))

    def Fitness(self):
        self.fitness = sum(val * (idx + 1) for idx, val in enumerate(self.chromosome))

    def Crossover(self, selector):
        op_name = selector.select_operator()
        old_fit = self.fitness
        if op_name == 'RightShift': nc = self.RightShift()
        elif op_name == 'LeftShift': nc = self.LeftShift()
        elif op_name == 'ReverseChromosome': nc = self.ReverseChromosome()
        elif op_name == 'ReversePiece': nc = self.ReversePiece()
        elif op_name == 'SwapPieces': nc = self.SwapPieces()
        else: nc = self.ReverseHeadAndTail()
        nc.Mutation(); nc.Fitness()
        selector.update(op_name, max(0, nc.fitness - old_fit))
        return nc

    def RightShift(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        if len(c.chromosome) > 2:
            p = random.randint(0, len(c.chromosome)-1)
            val = c.chromosome.pop(p)
            c.chromosome.insert((p + 1) % len(c.chromosome), val)
        return c

    def LeftShift(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        if len(c.chromosome) > 2:
            p = random.randint(0, len(c.chromosome)-1)
            val = c.chromosome.pop(p)
            c.chromosome.insert((p - 1) % len(c.chromosome), val)
        return c

    def ReverseChromosome(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)[::-1]
        return c

    def ReversePiece(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        p1, p2 = sorted(random.sample(range(len(c.chromosome)), 2))
        c.chromosome[p1:p2] = c.chromosome[p1:p2][::-1]
        return c

    def SwapPieces(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        p1, p2 = random.sample(range(len(c.chromosome)), 2)
        c.chromosome[p1], c.chromosome[p2] = c.chromosome[p2], c.chromosome[p1]
        return c

    def ReverseHeadAndTail(self):
        c = Chromosome(self.pool_obj, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        if len(self.chromosome) > 2:
            c.chromosome = [self.chromosome[-1]] + self.chromosome[1:-1] + [self.chromosome[0]]
        else:
            c.chromosome = list(self.chromosome)
        return c

    def Mutation(self):
        if random.random() <= self.mutationRate:
            i1, i2 = random.sample(range(self.numberofTarget), 2)
            self.chromosome[i1], self.chromosome[i2] = self.chromosome[i2], self.chromosome[i1]

    def Create(self):
        w_list = list(range(1, self.numberofWeapon + 1))
        if self.numberofTarget > self.numberofWeapon:
            w_list += [0] * (self.numberofTarget - self.numberofWeapon)
        self.chromosome = random.sample(w_list, self.numberofTarget)
        self.Fitness()
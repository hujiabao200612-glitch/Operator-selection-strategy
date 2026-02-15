import random
import math
import UCB


class Pool:
    def __init__(self, poolSize, numberofTarget, numberofWeapon, mutationRate):
        self.thePool = []
        self.poolSize = poolSize
        self.numberofTarget = numberofTarget
        self.numberofWeapon = numberofWeapon
        self.temporary = []
        self.mutationRate = mutationRate
        self.ucb_controller = UCB.UCBOptimizer(n_actions=6, c=150.0)

        # --- 新增：用于存储每一代的最高适应度 ---
        self.fitness_history = []

        dummy = Chromosome(self, self.mutationRate,
                           self.numberofTarget, self.numberofWeapon)
        dummy.fitness = 0
        self.theBest = [dummy]

    def Process(self, current_iter, total_iter):
        self.Crossover(current_iter, total_iter)
        self.Merge()
        self.FindBest()
        self.Selection()
        self.ucb_controller.next_generation()

        # --- 新增：每迭代一代，就记录当前的最优适应度 ---
        self.fitness_history.append(self.theBest[-1].fitness)

    def Create(self):
        for _ in range(self.poolSize):
            chromosome = Chromosome(
                self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            chromosome.Create()
            self.thePool.append(chromosome)

    def FindBest(self):
        current_best = self.theBest[-1]
        b = False
        for i in self.thePool:
            if i.fitness > current_best.fitness:
                new_best = Chromosome(
                    self, i.mutationRate, i.numberofTarget, i.numberofWeapon)
                new_best.chromosome = list(i.chromosome)
                new_best.fitness = i.fitness
                current_best = new_best
                b = True
        if b:
            self.theBest.append(current_best)

    def Selection(self):
        total_fitness = sum(i.fitness for i in self.thePool)
        if total_fitness == 0:
            return
        probabilities = [i.fitness / total_fitness for i in self.thePool]
        selected = []
        for _ in range(self.poolSize):
            rand = random.random()
            cum = 0
            for i, p in enumerate(probabilities):
                cum += p
                if rand <= cum:
                    selected.append(self.thePool[i])
                    break
        self.thePool = selected
        self.temporary = []

    def Merge(self):
        self.thePool.extend(self.temporary)

    def Crossover(self, current_iter, total_iter):
        for i in self.thePool:
            self.temporary.append(i.Crossover(current_iter, total_iter))


class Chromosome:
    def __init__(self, pool_obj, mutation, numberofTarget, numberofWeapon):
        self.pool_obj = pool_obj
        self.chromosome = []
        self.fitness = 0
        self.mutationRate = mutation
        self.numberofWeapon = numberofWeapon
        self.numberofTarget = numberofTarget

    def Create(self):
        weapon_list = list(range(1, self.numberofWeapon + 1))
        if self.numberofTarget > self.numberofWeapon:
            for _ in range(self.numberofTarget - self.numberofWeapon):
                weapon_list.append(0)
        temp_list = list(weapon_list)
        for _ in range(self.numberofTarget):
            found = random.choice(temp_list)
            self.chromosome.append(found)
            temp_list.remove(found)
        self.Fitness()

    def Crossover(self, current_iter, total_iter):
        action = self.pool_obj.ucb_controller.select_action()
        # 根据 action 执行不同的算子（与之前逻辑一致）
        if action == 0:
            new = self.RightShift()
        elif action == 1:
            new = self.LeftShift()
        elif action == 2:
            new = self.ReverseChromosome()
        elif action == 3:
            new = self.ReversePiece()
        elif action == 4:
            new = self.SwapPieces()
        else:
            new = self.ReverseHeadAndTail()

        new.Mutation()
        new.Fitness()
        reward = new.fitness - self.fitness
        progress = current_iter / total_iter
        self.pool_obj.ucb_controller.update(action, reward, progress)
        return new

    def Fitness(self):
        # 适应度计算函数
        self.fitness = sum(val * (idx + 1)
                           for idx, val in enumerate(self.chromosome))

    def Mutation(self):
        if random.random() <= self.mutationRate:
            idx1, idx2 = random.sample(range(self.numberofTarget), 2)
            self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]

    def RightShift(self):
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        c = list(self.chromosome)
        if len(c) > 1:
            c = [c[-1]] + c[:-1]
        crossed.chromosome = c
        return crossed

    def LeftShift(self):
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        c = list(self.chromosome)
        if len(c) > 1:
            c = c[1:] + [c[0]]
        crossed.chromosome = c
        return crossed

    def ReverseChromosome(self):
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        crossed.chromosome = self.chromosome[::-1]
        return crossed

    def ReversePiece(self):
        p = random.randint(0, self.numberofTarget-2)
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        c = list(self.chromosome)
        c[p:p+2] = c[p:p+2][::-1]
        crossed.chromosome = c
        return crossed

    def SwapPieces(self):
        p1, p2 = random.sample(range(self.numberofTarget), 2)
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        c = list(self.chromosome)
        c[p1], c[p2] = c[p2], c[p1]
        crossed.chromosome = c
        return crossed

    def ReverseHeadAndTail(self):
        crossed = Chromosome(self.pool_obj, self.mutationRate,
                             self.numberofTarget, self.numberofWeapon)
        c = list(self.chromosome)
        if len(c) > 2:
            mid = c[1:-1]
            crossed.chromosome = [c[-1]] + mid + [c[0]]
        else:
            crossed.chromosome = c[::-1]
        return crossed

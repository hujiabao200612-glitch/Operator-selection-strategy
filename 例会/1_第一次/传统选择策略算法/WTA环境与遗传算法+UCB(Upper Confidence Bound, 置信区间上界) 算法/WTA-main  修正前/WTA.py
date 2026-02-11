import random
import math
import numpy as np


class Pool:
    def __init__(self, poolSize, numberofTarget, numberofWeapon, mutationRate, use_ucb=True):
        self.use_ucb = use_ucb
        self.fitness_history = []
        self.diversity_history = []  # 新增：记录每一代的多样性
        self.thePool = []
        self.poolSize = poolSize
        self.numberofTarget = numberofTarget
        self.numberofWeapon = numberofWeapon
        self.temporary = []
        self.mutationRate = mutationRate
        Chromosome.reset_ucb()
        dummy = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        dummy.fitness = 0
        self.theBest = [dummy]

    def calculate_diversity(self):
        """计算种群多样性：计算个体间基因的平均汉明距离"""
        if not self.thePool: return 0
        matrix = np.array([c.chromosome for c in self.thePool])
        # 计算每一位基因上不同方案的差异度
        diff_sum = 0
        for col in range(self.numberofTarget):
            unique_elements = len(np.unique(matrix[:, col]))
            diff_sum += (unique_elements / self.poolSize)
        return diff_sum / self.numberofTarget

    def Process(self, monitor=None, current_iter=0):
        self.Crossover()
        self.Merge()
        self.FindBest()
        self.Selection()

        # 记录数据
        self.fitness_history.append(self.theBest[-1].fitness)
        div = self.calculate_diversity()
        self.diversity_history.append(div)

        if monitor and self.use_ucb:
            monitor.update(current_iter, Chromosome.counts, div, Chromosome.last_rewards)

    def Create(self):
        for _ in range(self.poolSize):
            c = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            c.Create()
            self.thePool.append(c)

    def FindBest(self):
        current_best = self.theBest[-1]
        improved = False
        for i in self.thePool:
            if i.fitness > current_best.fitness:
                new_best = Chromosome(self, i.mutationRate, i.numberofTarget, i.numberofWeapon)
                new_best.chromosome = list(i.chromosome)
                new_best.fitness = i.fitness
                current_best = new_best
                improved = True
        if improved: self.theBest.append(current_best)

    def Selection(self):
        total_fitness = sum(i.fitness for i in self.thePool)
        if total_fitness == 0: return
        probs = [i.fitness / total_fitness for i in self.thePool]
        self.thePool = random.choices(self.thePool, weights=probs, k=self.poolSize)
        self.temporary = []

    def Merge(self):
        self.thePool.extend(self.temporary)
        self.temporary = []

    def Crossover(self):
        for i in self.thePool:
            self.temporary.append(i.Crossover())


class Chromosome:
    counts = [1] * 6
    rewards = [0.0] * 6
    total_calls = 6
    last_rewards = [[] for _ in range(6)]  # 新增：记录算子的单次收益分布

    @classmethod
    def reset_ucb(cls):
        cls.counts = [1] * 6
        cls.rewards = [0.0] * 6
        cls.total_calls = 6
        cls.last_rewards = [[] for _ in range(6)]

    def __init__(self, pool, mutation, numberofTarget, numberofWeapon):
        self.thePool = pool
        self.chromosome = []
        self.fitness = -1
        self.mutationRate = mutation
        self.numberofWeapon = numberofWeapon
        self.numberofTarget = numberofTarget

    def Create(self):
        weapon_list = list(range(1, self.numberofWeapon + 1))
        while len(weapon_list) < self.numberofTarget: weapon_list.append(0)
        temp_weapons = list(weapon_list)
        self.chromosome = random.sample(temp_weapons, self.numberofTarget)
        self.Fitness()

    def Crossover(self):
        if self.thePool.use_ucb:
            ucb_vals = [
                (self.rewards[i] / self.counts[i]) + 2.0 * math.sqrt(math.log(Chromosome.total_calls) / self.counts[i])
                for i in range(6)]
            selected_op = np.argmax(ucb_vals)
        else:
            selected_op = random.randint(0, 5)

        Chromosome.counts[selected_op] += 1
        Chromosome.total_calls += 1

        ops = [self.RightShift, self.LeftShift, self.ReverseChromosome, self.ReversePiece, self.SwapPieces,
               self.ReverseHeadAndTail]
        new = ops[selected_op]()
        new.Mutation()
        new.Fitness()

        # 计算奖励并存入分布
        reward = max(0, (new.fitness - self.fitness) / (self.fitness + 1e-6))
        if self.thePool.use_ucb:
            Chromosome.rewards[selected_op] += reward
            Chromosome.last_rewards[selected_op].append(reward)
        return new

    # --- 算子逻辑保持原样 ---
    def RightShift(self):
        p = random.randint(0, self.numberofTarget - 1);
        ln = random.randint(1, self.numberofTarget - p)
        s = random.randint(1, self.numberofTarget - 1);
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome);
        piece = c.chromosome[p:p + ln]
        for _ in range(ln): c.chromosome.pop(p)
        for i in range(len(piece)): c.chromosome.insert((p + i + s) % (len(c.chromosome) + 1), piece[i])
        return c

    def LeftShift(self):
        p = random.randint(0, self.numberofTarget - 1);
        ln = random.randint(1, self.numberofTarget - p)
        s = random.randint(1, self.numberofTarget - 1);
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome);
        piece = c.chromosome[p:p + ln]
        for _ in range(ln): c.chromosome.pop(p)
        for i in range(len(piece)): c.chromosome.insert((p + i - s) % (len(c.chromosome) + 1), piece[i])
        return c

    def ReverseChromosome(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = self.chromosome[::-1];
        return c

    def ReversePiece(self):
        p = random.randint(0, self.numberofTarget - 1);
        ln = random.randint(1, self.numberofTarget - p)
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome);
        c.chromosome[p:p + ln + 1] = c.chromosome[p:p + ln + 1][::-1];
        return c

    def SwapPieces(self):
        p = random.randint(1, self.numberofTarget - 1);
        ln = random.randint(1, self.numberofTarget - p)
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        h, m, t = self.chromosome[:p], self.chromosome[p:p + ln], self.chromosome[p + ln:]
        c.chromosome = t + m + h;
        return c

    def ReverseHeadAndTail(self):
        p = random.randint(1, self.numberofTarget - 1);
        ln = random.randint(1, self.numberofTarget - p)
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        h, m, t = self.chromosome[:p][::-1], self.chromosome[p:p + ln], self.chromosome[p + ln:][::-1]
        c.chromosome = h + m + t;
        return c

    def Mutation(self):
        if random.random() <= self.mutationRate:
            i1, i2 = random.sample(range(len(self.chromosome)), 2)
            self.chromosome[i1], self.chromosome[i2] = self.chromosome[i2], self.chromosome[i1]

    def Fitness(self):
        self.fitness = sum(i * (j + 1) for j, i in enumerate(self.chromosome))
import random
import math
import numpy as np


class Pool:
    def __init__(self, poolSize, numberofTarget, numberofWeapon, mutationRate, use_ucb=True):
        self.use_ucb = use_ucb
        self.poolSize = poolSize
        self.numberofTarget = numberofTarget
        self.numberofWeapon = numberofWeapon
        self.mutationRate = mutationRate
        self.initial_mutation_rate = mutationRate

        # 随机生成战场环境数据
        np.random.seed(42)
        self.P_matrix = np.random.uniform(0.4, 0.8, (numberofWeapon, numberofTarget))
        self.V_targets = np.random.randint(50, 200, size=numberofTarget)

        self.thePool = []
        self.temporary = []
        self.theBest = []
        self.fitness_history = []
        self.diversity_history = []
        self.no_improvement_count = 0
        Chromosome.reset_ucb()

    def calculate_diversity(self):
        if not self.thePool: return 0
        matrix = np.array([c.chromosome for c in self.thePool])
        diff_sum = 0
        for col in range(self.numberofWeapon):
            unique_elements = len(np.unique(matrix[:, col]))
            diff_sum += (unique_elements / self.poolSize)
        return diff_sum / self.numberofWeapon

    def Process(self, monitor=None, current_iter=0):
        div = self.calculate_diversity()
        # 动态变异率
        self.mutationRate = self.initial_mutation_rate * (1.5 if div < 0.2 else 1.0)

        self.Crossover()
        self.Merge()

        old_best = self.theBest[-1].fitness if self.theBest else 0
        self.FindBest()

        if self.theBest[-1].fitness <= old_best:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        # 毁灭与重建机制
        if self.no_improvement_count >= 15:
            self.Earthquake()
            self.no_improvement_count = 0

        self.Selection()
        self.fitness_history.append(self.theBest[-1].fitness)
        self.diversity_history.append(div)

        if monitor and self.use_ucb:
            monitor.update(current_iter, Chromosome.counts, div, Chromosome.last_rewards)

    def Earthquake(self):
        keep_num = max(1, int(self.poolSize * 0.2))
        self.thePool = sorted(self.thePool, key=lambda x: x.fitness, reverse=True)[:keep_num]
        for _ in range(self.poolSize - keep_num):
            c = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            c.Create()
            self.thePool.append(c)

    def Create(self):
        for _ in range(self.poolSize):
            c = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            c.Create()
            self.thePool.append(c)
        self.FindBest()

    def FindBest(self):
        if not self.thePool: return
        best_in_gen = max(self.thePool, key=lambda x: x.fitness)
        if not self.theBest or best_in_gen.fitness > self.theBest[-1].fitness:
            new_best = Chromosome(self, self.mutationRate, self.numberofTarget, self.numberofWeapon)
            new_best.chromosome = list(best_in_gen.chromosome)
            new_best.fitness = best_in_gen.fitness
            self.theBest.append(new_best)

    def Selection(self):
        self.thePool = sorted(self.thePool, key=lambda x: x.fitness, reverse=True)[:self.poolSize]

    def Merge(self):
        self.thePool.extend(self.temporary)
        self.temporary = []

    def Crossover(self):
        for i in self.thePool:
            self.temporary.append(i.Evolve())


class Chromosome:
    counts = [1] * 8
    rewards = [0.0] * 8
    total_calls = 8
    last_rewards = [[] for _ in range(8)]

    @classmethod
    def reset_ucb(cls):
        cls.counts = [1] * 8;
        cls.rewards = [0.0] * 8;
        cls.total_calls = 8
        cls.last_rewards = [[] for _ in range(8)]

    def __init__(self, pool, mutation, numberofTarget, numberofWeapon):
        self.thePool, self.mutationRate = pool, mutation
        self.numberofTarget, self.numberofWeapon = numberofTarget, numberofWeapon
        self.chromosome = []
        self.fitness = 0

    def Create(self):
        self.chromosome = [random.randint(0, self.numberofTarget - 1) for _ in range(self.numberofWeapon)]
        self.Fitness()

    def Fitness(self):
        survive_p = np.ones(self.numberofTarget)
        for w_idx, t_idx in enumerate(self.chromosome):
            survive_p[t_idx] *= (1 - self.thePool.P_matrix[w_idx][t_idx])
        expected_residual = np.sum(self.thePool.V_targets * survive_p)
        self.fitness = np.sum(self.thePool.V_targets) - expected_residual

    def Evolve(self):
        if self.thePool.use_ucb:
            ucb_vals = [
                (self.rewards[i] / self.counts[i]) + 1.2 * math.sqrt(math.log(Chromosome.total_calls) / self.counts[i])
                for i in range(8)]
            op_idx = np.argmax(ucb_vals)
        else:
            op_idx = random.randint(0, 7)

        ops = [self.RightShift, self.LeftShift, self.ReversePiece, self.SwapPieces,
               self.PointMutation, self.RandomReset, self.GreedyCorrection, self.LocalSearch]

        new = ops[op_idx]()
        new.Fitness()

        delta = new.fitness - self.fitness
        reward = math.log(1 + delta) if delta > 0 else (
            0.1 if new.fitness > (sum(c.fitness for c in self.thePool.thePool) / len(self.thePool.thePool)) else 0)

        if self.thePool.use_ucb:
            Chromosome.counts[op_idx] += 1
            Chromosome.total_calls += 1
            Chromosome.rewards[op_idx] += reward
            Chromosome.last_rewards[op_idx].append(reward)
        return new

    def GreedyCorrection(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        survive_p = np.ones(self.numberofTarget)
        for w_idx, t_idx in enumerate(c.chromosome):
            survive_p[t_idx] *= (1 - self.thePool.P_matrix[w_idx][t_idx])
        worst_t = np.argmax(self.thePool.V_targets * survive_p)
        c.chromosome[random.randint(0, self.numberofWeapon - 1)] = worst_t
        return c

    def LocalSearch(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        for idx in random.sample(range(self.numberofWeapon), min(3, self.numberofWeapon)):
            c.chromosome[idx] = random.randint(0, self.numberofTarget - 1)
        return c

    def PointMutation(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        if random.random() < self.mutationRate:
            c.chromosome[random.randint(0, self.numberofWeapon - 1)] = random.randint(0, self.numberofTarget - 1)
        return c

    def RandomReset(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        c.chromosome[random.randint(0, self.numberofWeapon - 1)] = random.randint(0, self.numberofTarget - 1)
        return c

    def RightShift(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = self.chromosome[1:] + self.chromosome[:1];
        return c

    def LeftShift(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = self.chromosome[-1:] + self.chromosome[:-1];
        return c

    def ReversePiece(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        p1, p2 = sorted(random.sample(range(self.numberofWeapon), 2))
        c.chromosome[p1:p2] = c.chromosome[p1:p2][::-1];
        return c

    def SwapPieces(self):
        c = Chromosome(self.thePool, self.mutationRate, self.numberofTarget, self.numberofWeapon)
        c.chromosome = list(self.chromosome)
        p1, p2 = random.sample(range(self.numberofWeapon), 2)
        c.chromosome[p1], c.chromosome[p2] = c.chromosome[p2], c.chromosome[p1];
        return c
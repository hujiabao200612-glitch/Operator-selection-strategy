import WTA
import random
import numpy as np
from WTA_Visualizer import WTA_Monitor


class WtaProblem:
    def __init__(self, weapon, target, poolSize, iteration, mutationRate):
        self.weapon, self.target, self.poolSize = weapon, target, poolSize
        self.iteration, self.mutationRate = iteration, mutationRate
        # 初始化 WTA 实例
        self.object = WTA.Pool(self.poolSize, self.target, self.weapon, self.mutationRate)
        self.fitness = 0  # 【修正】显式初始化 fitness 属性

    def Process(self, compare=False):
        if compare:
            # 运行传统随机模式作为对比
            trad_pool = WTA.Pool(self.poolSize, self.target, self.weapon, self.mutationRate, use_ucb=False)
            trad_pool.Create()
            trad_div = []
            for _ in range(self.iteration):
                trad_pool.Process()
                trad_div.append(trad_pool.diversity_history[-1])

            # 运行 UCB 智能自适应模式
            monitor = WTA_Monitor()
            self.object.Create()
            for i in range(self.iteration):
                self.object.Process(monitor=monitor, current_iter=i)

            # 弹出分析报告
            monitor.show_all_reports(trad_pool.fitness_history, self.object.fitness_history, trad_div)
            # 【修正】记录最终的适应度
            self.fitness = self.object.theBest[-1].fitness
        else:
            self.object.Create()
            for _ in range(self.iteration):
                self.object.Process()
            # 【修正】非对比模式也需要记录最终的适应度
            if self.object.theBest:
                self.fitness = self.object.theBest[-1].fitness


class OperatorsPool:
    def __init__(self, weapon, target):
        self.weapon, self.target = weapon, target
        self.poolSize = 10
        self.pool = []
        self.mutationRate = 0.3
        self.theBest = []

    def Create(self):
        for _ in range(self.poolSize):
            c = OperatorsChromosome(self.weapon, self.target)
            c.Create()
            self.pool.append(c)
        if self.pool:
            self.theBest = [sorted(self.pool, key=lambda x: x.fitness)[-1]]

    def Process(self):
        children = []
        for i in range(0, len(self.pool), 2):
            if i + 1 < len(self.pool):
                c1, c2 = self.pool[i].Crossover(self.pool[i + 1])
                children.extend([c1, c2])

        for c in children:
            if random.random() < self.mutationRate:
                c.Mutation()
            c.Fitness()

        self.pool.extend(children)
        self.pool = sorted(self.pool, key=lambda x: x.fitness, reverse=True)[:self.poolSize]

        if self.pool[0].fitness > self.theBest[-1].fitness:
            self.theBest.append(self.pool[0])


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
        # 参数映射：前7位 PoolSize (1-128)，中7位 Iteration (1-128)，后7位 MutationRate (0-1)
        ps = int(''.join(map(str, self.chromosome[:7])), 2) + 1
        it = int(''.join(map(str, self.chromosome[7:14])), 2) + 1
        mr = (int(''.join(map(str, self.chromosome[14:21])), 2) + 1) / 128.0
        return ps, it, mr

    def Fitness(self, show_plot=False):
        ps, it, mr = self.Decode()
        prob = WtaProblem(self.weapon, self.target, ps, it, mr)
        prob.Process(compare=show_plot)
        # 现在 prob.fitness 已经通过上面的修正被赋值了
        self.fitness = prob.fitness
        if prob.object.theBest:
            self.best_wta_solution = prob.object.theBest[-1].chromosome

    def Crossover(self, other):
        r = random.randint(1, 20)
        c1 = OperatorsChromosome(self.weapon, self.target)
        c2 = OperatorsChromosome(self.weapon, self.target)
        c1.chromosome = self.chromosome[:r] + other.chromosome[r:]
        c2.chromosome = other.chromosome[:r] + self.chromosome[r:]
        return c1, c2

    def Mutation(self):
        idx = random.randint(0, 20)
        self.chromosome[idx] = 1 - self.chromosome[idx]
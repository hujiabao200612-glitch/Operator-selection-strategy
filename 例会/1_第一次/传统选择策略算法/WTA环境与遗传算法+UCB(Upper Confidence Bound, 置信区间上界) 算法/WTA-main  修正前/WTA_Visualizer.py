import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'];
plt.rcParams['axes.unicode_minus'] = False


class WTA_Monitor:
    def __init__(self):
        self.op_history = {i: [] for i in range(1, 7)}
        self.div_history = []
        self.reward_dist = [[] for _ in range(6)]
        self.iterations = []

    def update(self, iteration, counts, diversity, rewards):
        self.iterations.append(iteration)
        total = sum(counts)
        for i in range(6): self.op_history[i + 1].append(counts[i] / total)
        self.div_history.append(diversity)
        for i in range(6): self.reward_dist[i].extend(rewards[i])

    def show_all_reports(self, trad_history, ucb_history, trad_div):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        labels = ['右移', '左移', '全反转', '片段反转', '片段交换', '首尾反转']

        # 1. 效能增长对比
        axs[0, 0].plot(trad_history, label='传统随机', color='gray', linestyle='--')
        axs[0, 0].plot(ucb_history, label='UCB智能', color='red', linewidth=2)
        axs[0, 0].set_title("效能增长对比 (Fitness)");
        axs[0, 0].legend()

        # 2. 算子选择占比
        for i in range(1, 7):
            axs[0, 1].plot(self.iterations, self.op_history[i], label=labels[i - 1])
        axs[0, 1].set_title("UCB 算子自适应趋势");
        axs[0, 1].legend()

        # 3. 种群多样性对比
        axs[1, 0].plot(trad_div, label='传统随机多样性', color='blue', alpha=0.5)
        axs[1, 0].plot(self.div_history, label='UCB智能多样性', color='green', linewidth=2)
        axs[1, 0].set_title("种群多样性对比 (Diversity)");
        axs[1, 0].legend()

        # 4. 算子单次收益箱线图
        axs[1, 1].boxplot(self.reward_dist, labels=labels)
        axs[1, 1].set_title("各算子单次进化收益分布")

        plt.tight_layout();
        plt.show()
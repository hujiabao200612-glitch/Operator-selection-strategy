import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'];
plt.rcParams['axes.unicode_minus'] = False


class WTA_Monitor:
    def __init__(self):
        self.op_history = {i: [] for i in range(1, 9)}
        self.div_history = []
        self.reward_dist = [[] for _ in range(8)]
        self.iterations = []

    def update(self, iteration, counts, diversity, rewards):
        self.iterations.append(iteration)
        # 修正：将记录改为占比，使右上角图表加和为 1
        total = sum(counts)
        for i in range(8):
            self.op_history[i + 1].append(counts[i] / total)

        self.div_history.append(diversity)
        for i in range(8):
            self.reward_dist[i].extend(rewards[i])

    def show_all_reports(self, trad_history, ucb_history, trad_div):
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        labels = ['右移', '左移', '片段反转', '交换', '单点变异', '随机重置', '贪心修正', '局部搜索']

        axs[0, 0].plot(trad_history, label='传统随机', color='gray', linestyle='--')
        axs[0, 0].plot(ucb_history, label='UCB智能', color='red', linewidth=2)
        axs[0, 0].set_title("效能增长对比 (Fitness)");
        axs[0, 0].legend()

        for i in range(1, 9):
            axs[0, 1].plot(self.iterations, self.op_history[i], label=labels[i - 1])
        axs[0, 1].set_title("UCB 算子选择概率趋势 (加和=1)");
        axs[0, 1].legend(loc='upper left', fontsize='small')

        axs[1, 0].plot(trad_div, label='传统随机多样性', color='blue', alpha=0.4)
        axs[1, 0].plot(self.div_history, label='UCB智能多样性', color='green', linewidth=2)
        axs[1, 0].set_title("种群多样性对比 (Diversity)");
        axs[1, 0].legend()

        cleaned_dist = [d if len(d) > 0 else [0] for d in self.reward_dist]
        axs[1, 1].boxplot(cleaned_dist, labels=labels)
        axs[1, 1].set_title("各算子单次进化收益分布")

        plt.tight_layout();
        plt.show()
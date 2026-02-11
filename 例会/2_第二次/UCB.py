import math
import random


class UCBOptimizer:
    def __init__(self, n_actions, c=50.0):
        self.n_actions = n_actions
        self.c = c
        self.counts = [0] * n_actions
        self.values = [0.0] * n_actions
        self.total_counts = 0
        self.iteration_history = []
        self._current_gen_counts = [0] * n_actions
        # --- 新增：用于动态调整 k 的历史数据 ---
        self.max_observed_reward = 0.1  # 防止除以零

    def select_action(self):
        """核心选择逻辑：天平称重"""
        for i in range(self.n_actions):
            if self.counts[i] == 0:
                self._current_gen_counts[i] += 1
                return i

        ucb_values = []
        for i in range(self.n_actions):
            bonus = self.c * \
                math.sqrt(math.log(self.total_counts + 1) /
                          (self.counts[i] + 0.1))
            ucb_values.append(self.values[i] + bonus)

        action = ucb_values.index(max(ucb_values))
        self._current_gen_counts[action] += 1
        return action

    def update(self, action, reward, progress):
        """核心反馈逻辑：引入动态 k 值"""
        self.counts[action] += 1
        self.total_counts += 1

        if reward > 0:
            # --- 动态 k 值调整逻辑 ---
            # 记录历史最大奖励。在大规模 (1000x1000) 下，reward 很大，k 变小；
            # 在小规模下，reward 小，k 变大。确保 Sigmoid 始终工作在敏感区。
            if reward > self.max_observed_reward:
                self.max_observed_reward = reward

            # 动态 k：使 k * reward 的最大值保持在合理范围内（例如 2~3）
            dynamic_k = 2.5 / self.max_observed_reward

            # --- 第一步：Sigmoid 映射 ---
            sigmoid_reward = 1 / (1 + math.exp(-dynamic_k * reward)) - 0.5

            # --- 第二步：计算后期权重 ---
            # 建议：1000 规模下不要用 e^8，太容易造成后期震荡。改用较温和的 e^2
            time_weight = math.exp(progress * 2.0)

            # 组合奖励
            adjusted_reward = sigmoid_reward * time_weight

            # --- 第三步：低保奖 ---
            if progress > 0.8:
                adjusted_reward += 0.5  # 降低低保分值，防止完全掩盖经验分
        else:
            # 负反馈也随进度温和加重
            adjusted_reward = -0.1 * (1 + progress)

        # 增量式更新均值
        n = self.counts[action]
        self.values[action] = ((n - 1) / n) * \
            self.values[action] + (1 / n) * adjusted_reward

    def next_generation(self):
        """代际交替：清空缓存，增加记忆持久度"""
        self.iteration_history.append(list(self._current_gen_counts))
        self._current_gen_counts = [0] * self.n_actions

        # 建议：将衰减系数从 0.8 提高到 0.95，有助于在 1000 规模下沉淀有效经验
        memory_factor = 0.95
        for i in range(self.n_actions):
            self.values[i] *= memory_factor
            self.counts[i] = int(self.counts[i] * memory_factor)
        self.total_counts = sum(self.counts)

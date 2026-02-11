import math

class UCBSelector:
    def __init__(self, operator_names, exploration_constant=2.0):
        self.operator_names = operator_names
        self.C = exploration_constant
        self.counts = {name: 0 for name in operator_names}
        self.rewards = {name: 0.0 for name in operator_names}
        self.total_steps = 0

    def select_operator(self):
        self.total_steps += 1
        for name in self.operator_names:
            if self.counts[name] == 0: return name
        
        best_op, max_ucb = None, -float('inf')
        for name in self.operator_names:
            avg_reward = self.rewards[name] / self.counts[name]
            exploration = self.C * math.sqrt(2 * math.log(self.total_steps) / self.counts[name])
            if (avg_reward + exploration) > max_ucb:
                max_ucb = avg_reward + exploration
                best_op = name
        return best_op

    def update(self, name, reward):
        self.counts[name] += 1
        self.rewards[name] += reward
import OperatorsOptimization as op
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def show_analysis_window(fitness_history, ucb_history):
    """
    此时两个 history 的长度都是内层迭代代数（如 128 代）
    """
    analysis_win = tk.Toplevel()
    analysis_win.title("UCB 性能与算子选择对齐分析")
    analysis_win.geometry("900x800")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    fig.tight_layout(pad=6.0)

    # 1. 适应度进化图 (此时 x 轴是 0 到 120+)
    ax1.plot(fitness_history, color='#1f77b4', linewidth=2, label='最优适应度进化')
    ax1.set_title("适应度进化全过程 (与下表代数同步)", fontsize=14)
    ax1.set_xlabel("内层迭代代数 (Generation)", fontsize=12)
    ax1.set_ylabel("适应度分值", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. 算子概率图
    operators = ['右移', '左移', '全反转', '片段反转', '片段交换', '头尾反转']
    colors = ['#e31a1c', '#33a02c', '#ff7f00', '#6a3d9a', '#1f78b4', '#b15928']

    prob_history = []
    for gen_counts in ucb_history:
        total = sum(gen_counts)
        prob_history.append(
            [count / total if total > 0 else 0 for count in gen_counts])

    x = range(len(prob_history))
    for i in range(6):
        y = [gen[i] for gen in prob_history]
        ax2.plot(x, y, label=operators[i], color=colors[i], linewidth=1.5)

    ax2.set_title("各代算子选择概率演变", fontsize=14)
    ax2.set_xlabel("内层迭代代数 (Generation)", fontsize=12)
    ax2.set_ylabel("选择概率", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    canvas = FigureCanvasTkAgg(fig, master=analysis_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def Fire():
    try:
        w_val, t_val = int(entry.get()), int(entry2.get())
    except:
        return

    a = op.OperatorsPool(w_val, t_val)
    a.Create()

    for i in range(10):  # 外层
        progress_bar['value'] = (i+1) * 10
        root.update_idletasks()
        a.Process()

    # --- 关键提取逻辑 ---
    try:
        # 1. 获取外层筛选出的“最佳参数个体”
        best_outer_chrom = a.theBest[-1]
        # 2. 从该个体的引用中提取它在运行时的详细历史
        detailed_fitness = best_outer_chrom.theBestChromosome_ref.pool_obj.fitness_history
        ucb_data = best_outer_chrom.theBestChromosome_ref.pool_obj.ucb_controller.iteration_history

        # 打开窗口
        show_analysis_window(detailed_fitness, ucb_data)
    except Exception as e:
        print(f"数据对齐失败: {e}")


# GUI 初始
root = tk.Tk()
root.geometry("300x300")
root.title("WTA 优化系统")

tk.Label(root, text="武器数量:").pack()
entry = ttk.Entry(root)
entry.pack()
tk.Label(root, text="目标数量:").pack()
entry2 = ttk.Entry(root)
entry2.pack()
progress_bar = ttk.Progressbar(root, length=200)
progress_bar.pack(pady=20)
tk.Button(root, text="开始运行并分析", command=lambda: threading.Thread(
    target=Fire, daemon=True).start()).pack()

root.mainloop()

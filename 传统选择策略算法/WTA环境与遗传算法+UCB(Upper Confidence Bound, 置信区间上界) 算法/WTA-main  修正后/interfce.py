import tkinter as tk
from tkinter import messagebox
import OperatorsOptimization


def Fire():
    try:
        # 从输入框获取用户自定义的数值
        w = int(weapon_entry.get())
        t = int(target_entry.get())

        if w <= 0 or t <= 0:
            messagebox.showwarning("输入错误", "武器数和目标数必须大于0")
            return

        res_label.config(text="算法运行中，请稍后...", fg="blue")
        root.update()  # 刷新界面显示

        # 运行外层参数优化（寻找最佳 PoolSize, Iteration, MutationRate）
        ops = OperatorsOptimization.OperatorsPool(w, t)
        ops.Create()
        for i in range(5):  # 外层迭代5代
            ops.Process()
            print(f"外层参数搜索进度: {i + 1}/5")

        best_chrom = ops.theBest[-1]

        # 弹出可视化分析图表（8算子分析）
        # 注意：程序会在这里暂停，直到你关闭弹出的 Matplotlib 窗口
        best_chrom.Fitness(show_plot=True)

        # 输出最终方案到控制台
        print("\n" + "=" * 30)
        print(f"--- 最终打击方案结果 (武器数:{w}, 目标数:{t}) ---")
        solution = best_chrom.best_wta_solution
        for weapon_idx, target_idx in enumerate(solution):
            print(f"武器 {weapon_idx + 1} -> 打击目标 {target_idx + 1}")
        print("=" * 30)

        # UI 界面反馈
        res_label.config(text=f"执行完毕！最优适应度: {best_chrom.fitness:.2f}", fg="green")
        messagebox.showinfo("完成", "优化计算已结束，详细方案请查看控制台输出。")

    except ValueError:
        messagebox.showerror("输入错误", "请输入有效的整数数字")


# --- GUI 界面布局 ---
root = tk.Tk()
root.title("WTA 智能决策系统 v2.0")
root.geometry("350x250")

# 武器数输入行 (Row 0)
tk.Label(root, text="武器数量 (W):").grid(row=0, column=0, padx=20, pady=10)
weapon_entry = tk.Entry(root)
weapon_entry.insert(0, "200")  # 默认值
weapon_entry.grid(row=0, column=1, padx=10, pady=10)

# 目标数输入行 (Row 1)
tk.Label(root, text="目标数量 (T):").grid(row=1, column=0, padx=20, pady=10)
target_entry = tk.Entry(root)
target_entry.insert(0, "150")  # 默认值
target_entry.grid(row=1, column=1, padx=10, pady=10)

# 执行按钮 (Row 2)
fire_btn = tk.Button(root, text="开始执行优化", command=Fire, bg="orange", width=20)
fire_btn.grid(row=2, columnspan=2, pady=20)

# 状态提示 (Row 3)
res_label = tk.Label(root, text="请设置参数后点击开始", font=("Arial", 10))
res_label.grid(row=3, columnspan=2, pady=10)

root.mainloop()
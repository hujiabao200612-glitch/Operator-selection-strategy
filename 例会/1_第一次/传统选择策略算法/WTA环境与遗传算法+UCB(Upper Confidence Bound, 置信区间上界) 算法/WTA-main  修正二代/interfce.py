import tkinter as tk
from tkinter import messagebox
import OperatorsOptimization


def Fire():
    try:
        w = int(weapon_entry.get())
        t = int(target_entry.get())
        res_label.config(text="正在搜索最优参数组合...", fg="blue");
        root.update()

        ops = OperatorsOptimization.OperatorsPool(w, t)
        ops.Create()
        for i in range(5):
            ops.Process()
            print(f"参数评估进度: {i + 1}/5")

        best_chrom = ops.theBest[-1]
        best_chrom.Fitness(show_plot=True)

        print("\n" + "=" * 40)
        print(f"最终方案 (W:{w}, T:{t})")
        for w_idx, t_idx in enumerate(best_chrom.best_wta_solution):
            print(f"武器 {w_idx + 1} -> 目标 {t_idx + 1}")
        print("=" * 40)

        res_label.config(text=f"完成！最优Fitness: {best_chrom.fitness:.2f}", fg="green")
        messagebox.showinfo("运行结束", "分析已弹出，详细方案见控制台。")
    except Exception as e:
        messagebox.showerror("运行错误", str(e))


root = tk.Tk();
root.title("WTA智能决策系统");
root.geometry("300x200")
tk.Label(root, text="武器数:").grid(row=0, column=0, pady=10)
weapon_entry = tk.Entry(root);
weapon_entry.insert(0, "150");
weapon_entry.grid(row=0, column=1)
tk.Label(root, text="目标数:").grid(row=1, column=0, pady=10)
target_entry = tk.Entry(root);
target_entry.insert(0, "150");
target_entry.grid(row=1, column=1)
tk.Button(root, text="开始优化", command=Fire, width=15, bg="lightblue").grid(row=2, columnspan=2, pady=10)
res_label = tk.Label(root, text="等待输入...");
res_label.grid(row=3, columnspan=2)
root.mainloop()
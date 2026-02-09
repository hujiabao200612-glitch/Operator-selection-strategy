import OperatorsOptimization as op
import tkinter as tk
from tkinter import ttk
import threading

root = tk.Tk();
root.geometry("380x250");
root.title("WTA 武器目标分配决策系统")
root.configure(bg="#d1d1d1")


def Fire():
    try:
        w, t = int(entry.get()), int(entry2.get())
    except:
        return
    a = op.OperatorsPool(w, t);
    a.Create()
    progress_bar.place(x=90, y=125)
    for i in range(10):
        progress_bar['value'] = (i + 1) * 10;
        root.update_idletasks();
        a.Process()

    # 最终报告
    best_chrom = a.theBest[-1]
    res_win = tk.Toplevel(root);
    res_win.title("方案结果")
    lb = tk.Listbox(res_win, width=40)
    for idx, weapon in enumerate(best_chrom.theBestWtaChromosome):
        lb.insert(tk.END, f"目标 {idx + 1}: 武器 {weapon}")
    lb.pack()

    # 这里会弹出四张对比图
    best_chrom.Fitness(show_plot=True)


# UI 布局
entry = ttk.Entry(root);
entry.place(x=160, y=20);
tk.Label(root, text="武器数:").place(x=40, y=20)
entry2 = ttk.Entry(root);
entry2.place(x=160, y=70);
tk.Label(root, text="目标数:").place(x=40, y=70)
progress_bar = ttk.Progressbar(root, length=200);
button = ttk.Button(root, text="执行并对比", command=Fire)
button.place(x=130, y=170);
root.mainloop()
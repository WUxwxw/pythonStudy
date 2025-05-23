import os
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from stl import mesh
from matplotlib import rcParams
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False


def read_prof_file(filename):
    """读取.prof文件，返回包含各参数的字典"""
    sections = {
        '(x': 'x', '(y': 'y', '(z': 'z',
        '(pressure': 'pressure', '(temperature': 'temperature'
    }
    current_section = None
    data = {'x': [], 'y': [], 'z': [], 'pressure': [], 'temperature': []}

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line in sections:
                current_section = sections[line]
            else:
                try:
                    val = float(line)
                    data[current_section].append(val)
                except (ValueError, KeyError) as e:
                    print(f"行 {line_num}: 忽略无效数据 - {str(e)}")
    return data


def read_field_file(filename, header_lines=1):
    """读取远场数据文件，返回处理后的numpy数组"""
    data = []
    with open(filename, 'r') as f:
        for _ in range(header_lines):
            next(f)

        for line_num, line in enumerate(f, start=header_lines + 1):
            line = line.strip().replace(',', '.').replace('%', '')
            if not line:
                continue
            for elem in line.split():
                try:
                    data.append(float(elem))
                except ValueError:
                    print(f"行 {line_num}: 忽略无效值 '{elem}'")
                    continue
    return np.array(data)


def visualize_data(inspeed, AOA, SS, frequency, folderPath):
    stlFile = f"Inspeed_{inspeed:.2f}_AOA_{AOA}_SS_{SS}.stl"
    fieldFile = f"Inspeed_{inspeed:.2f}_AOA_{AOA}_SS_{SS}_Farfields(f={frequency}).txt"
    profFile = f"Inspeed_{inspeed:.2f}_AOA_{AOA}_SS_{SS}.prof"

    stl_path = os.path.join(folderPath, stlFile)
    try:
        stl_mesh = mesh.Mesh.from_file(stl_path)
    except Exception as e:
        messagebox.showerror("错误", f"无法读取STL文件: {str(e)}")
        return

    rcsFolderPath = os.path.join(folderPath, 'rcs')
    field_path = os.path.join(rcsFolderPath, fieldFile)
    try:
        data = read_field_file(field_path, header_lines=1)
    except FileNotFoundError:
        messagebox.showerror("错误", f"远场文件未找到: {field_path}")
        return

    if len(data) % 8 != 0:
        messagebox.showerror("错误", f"数据不完整，期望能被8整除，实际长度: {len(data)}")
        return

    Xvct = data[0::8]  # Theta
    Yvct = data[1::8]  # Phi
    Zvct = data[2::8]  # E-field

    theta = np.deg2rad(Xvct.reshape(360, 181).T)
    phi = np.deg2rad(Yvct.reshape(360, 181).T)
    f_field = Zvct.reshape(360, 181).T

    xx = f_field * np.sin(theta) * np.cos(phi)
    yy = f_field * np.sin(theta) * np.sin(phi)
    zz = f_field * np.cos(theta)

    prof_path = os.path.join(folderPath, profFile)
    try:
        prof_data = read_prof_file(prof_path)
    except FileNotFoundError:
        messagebox.showerror("错误", f"prof文件未找到: {prof_path}")
        return

    x = np.array(prof_data['x'])
    y = np.array(prof_data['y'])
    z = np.array(prof_data['z'])
    pressure = np.array(prof_data['pressure'])
    temperature = np.array(prof_data['temperature'])

    fig = Figure(figsize=(10, 8), dpi=150)
    fig.suptitle("多物理场可视化分析", fontsize=14)
    elevdeg=60
    azimdeg=60
    #压力分布
    ax1 = fig.add_subplot(221, projection='3d')
    sc1 = ax1.scatter(x, y, z, c=pressure,
                      cmap='viridis',
                      s=15,
                      alpha=0.8)
    ax1.view_init(elev=elevdeg, azim=azimdeg)  # 设置视角
    ax1.set_title('流场：表面压力分布 (Pa)')
    fig.colorbar(sc1, ax=ax1, label='压力 (Pa)', shrink=0.6)
    #温度分布
    ax2 = fig.add_subplot(222, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=temperature,
                      cmap='hot',
                      s=15,
                      alpha=0.8)
    ax2.view_init(elev=elevdeg, azim=azimdeg)  # 设置视角
    ax2.set_title('热场：表面温度分布 (K)')
    fig.colorbar(sc2, ax=ax2, label='温度 (K)', shrink=0.6)
    #结构力场
    ax3 = fig.add_subplot(223, projection='3d')
    polygons = art3d.Poly3DCollection(stl_mesh.vectors,
                                      facecolors=[0.8, 0.8, 1.0],
                                      edgecolors=[0.4, 0.4, 0.4],
                                      linewidths=0.3,
                                      alpha=0.9)
    ax3.add_collection3d(polygons)
    ax3.auto_scale_xyz(stl_mesh.x.flatten(),
                       stl_mesh.y.flatten(),
                       stl_mesh.z.flatten())
    ax3.view_init(elev=elevdeg, azim=azimdeg)
    ax3.set_title('结构力场：形变后模型')
    ax3.set_axis_off()
    #电磁场分布
    norm = Normalize(vmin=f_field.min(), vmax=f_field.max())
    ax4 = fig.add_subplot(224, projection='3d')
    surf = ax4.plot_surface(xx, yy, zz,
                            facecolors=plt.cm.jet(f_field/f_field.max()),
                            rstride=2,
                            cstride=2,
                            antialiased=False)
    ax4.view_init(elev=elevdeg, azim=azimdeg)
    ax4.set_title('电磁场：远场电场分布 (V/m)')
    fig.colorbar(surf, ax=ax4, label='电场强度 (V/m)', shrink=0.6)
    surf.set_clim(vmin=f_field.min(), vmax=f_field.max())



    return fig


def generate_visualization():
    try:
        inspeed = float(inspeed_combobox.get())
        AOA = int(AOA_combobox.get())
        SS = int(SS_combobox.get())
        frequency = int(frequency_combobox.get())
        folderPath = folderPath_entry.get()

        fig = visualize_data(inspeed, AOA, SS, frequency, folderPath)

        for widget in canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except ValueError as e:
        messagebox.showerror("输入错误", f"请检查输入格式: {str(e)}")
    except Exception as e:
        messagebox.showerror("错误", f"发生错误: {str(e)}")


root = tk.Tk()
root.title("多物理场可视化工具")



# 生成选项值，并格式化为两位小数的字符串
inspeed_values = [f"{val:.2f}" for val in np.linspace(0.9, 1.5, 21)]
AOA_values = [f"{int(val)}" for val in np.linspace(-10, 11, 22)]  # 迎角和侧滑角是整数
SS_values = [f"{int(val)}" for val in np.linspace(-10, 11, 22)]
frequency_values = [f"{int(val)}" for val in np.linspace(200, 300, 11)]  # 频率是整数

# 创建下拉菜单
tk.Label(root, text="输入速度 (Ma):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
inspeed_combobox = ttk.Combobox(root, values=inspeed_values, width=10)
inspeed_combobox.grid(row=0, column=1, padx=5, pady=5)
inspeed_combobox.set("1.02")  # 设置默认值

tk.Label(root, text="迎角 (°):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
AOA_combobox = ttk.Combobox(root, values=AOA_values, width=10)
AOA_combobox.grid(row=1, column=1, padx=5, pady=5)
AOA_combobox.set("1")

tk.Label(root, text="侧滑角 (°):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
SS_combobox = ttk.Combobox(root, values=SS_values, width=10)
SS_combobox.grid(row=2, column=1, padx=5, pady=5)
SS_combobox.set("1")

tk.Label(root, text="电磁观察频率 (MHz):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
frequency_combobox = ttk.Combobox(root, values=frequency_values, width=10)
frequency_combobox.grid(row=3, column=1, padx=5, pady=5)
frequency_combobox.set("300")

# 文件夹路径输入
tk.Label(root, text="文件夹路径 (folderPath):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
folderPath_entry = tk.Entry(root, width=50)
folderPath_entry.grid(row=4, column=1, padx=5, pady=5)
folderPath_entry.insert(0, "E:\\WXW\\A_Project\\X_47_multiphysics\\3\\data102")

# 生成可视化按钮
generate_button = tk.Button(root, text="生成可视化", command=generate_visualization)
generate_button.grid(row=5, column=0, columnspan=2, pady=10)

# 画布框架
canvas_frame = tk.Frame(root)
canvas_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()